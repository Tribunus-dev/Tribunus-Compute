//! GGUF model import — ingest GGUF files into the ComputeImage compiler pipeline.
//!
//! Tribunus does NOT execute GGUF files directly. GGUF is an input format,
//! like safetensors or HuggingFace weights. The pipeline is:
//!
//!   raw model (GGUF/safetensors/HF) → validate → canonicalize
//!     → profile target hardware → compile → ComputeImage → serve
//!
//! This module extracts metadata, tensor names, quantization layout, tokenizer
//! config, and architecture properties from GGUF files, then feeds them into
//! the compile_sequential/compile_tensix pipeline to produce a ComputeImage.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Minimum GGUF version supported for import.
pub const MIN_GGUF_VERSION: u32 = 3;

/// GGUF metadata key constants for Tribunus ModelManifest extraction.
pub mod keys {
    pub const VOCAB_SIZE: &str = "llama.vocab_size";
    pub const HIDDEN_SIZE: &str = "llama.embedding_length";
    pub const INTERMEDIATE_SIZE: &str = "llama.feed_forward_length";
    pub const NUM_HIDDEN_LAYERS: &str = "llama.block_count";
    pub const NUM_ATTENTION_HEADS: &str = "llama.attention.head_count";
    pub const NUM_KV_HEADS: &str = "llama.attention.head_count_kv";
    pub const HEAD_DIM: &str = "llama.attention.head_dim";
    pub const MAX_SEQ_LEN: &str = "llama.context_length";
    pub const ROPE_THETA: &str = "llama.rope.freq_base";
    pub const NORM_EPS: &str = "llama.attention.layer_norm_rms_epsilon";
    pub const MODEL_TYPE: &str = "general.architecture";
    pub const QUANTIZATION_VERSION: &str = "general.quantization_version";
    pub const FILE_TYPE: &str = "general.file_type";
}

/// Results of importing a GGUF file into the compiler pipeline.
pub struct GgufImportResult {
    /// Model architecture config (feeds into `config::compile()`).
    pub model_config: crate::config::TextArchitecture,
    /// GGUF file path (for direct tensor read during compilation).
    pub source_path: std::path::PathBuf,
    /// Tensor names, shapes, dtypes, and byte offsets.
    pub tensor_inventory: Vec<GgufTensorMeta>,
    /// Tokenizer files (tokenizer.json or equivalent) extracted from the GGUF.
    pub tokenizer_path: Option<std::path::PathBuf>,
    /// Original GGUF metadata KV pairs (for diagnostics and provenance).
    pub metadata: Vec<(String, String)>,
}

/// Metadata for a single tensor in the GGUF file.
#[derive(Clone, Debug)]
pub struct GgufTensorMeta {
    pub name: String,
    pub dtype: String, // "f32", "f16", "q4_0", "q4_K_M", "q8_0", etc.
    pub shape: Vec<u32>,
    pub byte_offset: u64,
    pub byte_size: u64,
}

// ── GGUF binary value-type constants ───────────────────────────────────────

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;
const GGUF_TYPE_BF16: u32 = 13;

// ── GGML tensor dtype constants ────────────────────────────────────────────

/// GGML tensor type codes used in GGUF tensor info entries.
mod ggml_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    // 4 = GGML_TYPE_Q4_2 (removed)
    // 5 = GGML_TYPE_Q4_3 (removed)
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q8_1: u32 = 9;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const Q8_K: u32 = 15;
    pub const IQ1_S: u32 = 16;
    pub const IQ1_M: u32 = 17;
    pub const IQ2_XXS: u32 = 18;
    pub const IQ2_XS: u32 = 19;
    pub const IQ2_S: u32 = 20;
    pub const IQ2_M: u32 = 21;
    pub const IQ3_XXS: u32 = 22;
    pub const IQ3_XS: u32 = 23;
    pub const IQ3_S: u32 = 24;
    pub const IQ4_NL: u32 = 26;
    pub const IQ4_XS: u32 = 27;
    pub const BF16: u32 = 28;
}

/// Return a human-readable dtype name for a GGML type code.
fn ggml_dtype_name(typ: u32) -> &'static str {
    match typ {
        ggml_type::F32 => "f32",
        ggml_type::F16 => "f16",
        ggml_type::Q4_0 => "q4_0",
        ggml_type::Q4_1 => "q4_1",
        ggml_type::Q5_0 => "q5_0",
        ggml_type::Q5_1 => "q5_1",
        ggml_type::Q8_0 => "q8_0",
        ggml_type::Q8_1 => "q8_1",
        ggml_type::Q2_K => "q2_K",
        ggml_type::Q3_K => "q3_K",
        ggml_type::Q4_K => "q4_K",
        ggml_type::Q5_K => "q5_K",
        ggml_type::Q6_K => "q6_K",
        ggml_type::Q8_K => "q8_K",
        ggml_type::IQ1_S => "iq1_s",
        ggml_type::IQ1_M => "iq1_m",
        ggml_type::IQ2_XXS => "iq2_xxs",
        ggml_type::IQ2_XS => "iq2_xs",
        ggml_type::IQ2_S => "iq2_s",
        ggml_type::IQ2_M => "iq2_m",
        ggml_type::IQ3_XXS => "iq3_xxs",
        ggml_type::IQ3_XS => "iq3_xs",
        ggml_type::IQ3_S => "iq3_s",
        ggml_type::IQ4_NL => "iq4_nl",
        ggml_type::IQ4_XS => "iq4_xs",
        ggml_type::BF16 => "bf16",
        _ => "unknown",
    }
}

/// Return (block_size, type_size_in_bytes) for a GGML type.
/// block_size is how many elements are packed into one block of type_size bytes.
fn ggml_block_info(typ: u32) -> (u64, u64) {
    match typ {
        ggml_type::F32 => (1, 4),
        ggml_type::F16 | ggml_type::BF16 => (1, 2),
        ggml_type::Q4_0 => (32, 18),
        ggml_type::Q4_1 => (32, 20),
        ggml_type::Q5_0 => (32, 22),
        ggml_type::Q5_1 => (32, 24),
        ggml_type::Q8_0 => (32, 34),
        ggml_type::Q8_1 => (32, 40),
        ggml_type::Q2_K => (256, 72),
        ggml_type::Q3_K => (256, 104),
        ggml_type::Q4_K => (256, 144),
        ggml_type::Q5_K => (256, 176),
        ggml_type::Q6_K => (256, 208),
        ggml_type::Q8_K => (256, 272),
        // IQ types have non-standard block structures; approximate with
        // per-element sizing.  Users reading actual weight data should
        // consult the original ggml.h layout.
        ggml_type::IQ1_S => (256, 34 + 2),
        ggml_type::IQ1_M => (256, 50 + 2),
        ggml_type::IQ2_XXS => (256, 32 + 2 + 2),
        ggml_type::IQ2_XS => (256, 48 + 2 + 2),
        ggml_type::IQ2_S => (256, 64 + 4 + 2),
        ggml_type::IQ2_M => (256, 80 + 4 + 2),
        ggml_type::IQ3_XXS => (256, 72 + 4 + 2),
        ggml_type::IQ3_XS => (256, 88 + 4 + 2),
        ggml_type::IQ3_S => (256, 104 + 4 + 2),
        ggml_type::IQ4_NL => (32, 18),
        ggml_type::IQ4_XS => (256, 144 + 4 + 2),
        _ => (1, 1),
    }
}

/// Compute the byte size of a tensor given its dtype and shape.
fn ggml_tensor_byte_size(dtype: u32, shape: &[u32]) -> u64 {
    let total_elems: u64 = shape.iter().map(|&d| d as u64).product();
    if total_elems == 0 {
        return 0;
    }
    let (block_size, type_size) = ggml_block_info(dtype);
    let num_blocks = total_elems.div_ceil(block_size);
    num_blocks * type_size
}

// ── Binary reader helpers ──────────────────────────────────────────────────

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| format!("GGUF read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| format!("GGUF read u64: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("GGUF read string: {}", e))?;
    String::from_utf8(buf).map_err(|e| format!("Invalid UTF-8 in GGUF string: {}", e))
}

/// Parse a single GGUF value given its type tag, returning a string
/// representation.  Array values are serialised as comma-separated.
fn read_typed_value<R: Read>(r: &mut R, typ: u32) -> Result<String, String> {
    match typ {
        GGUF_TYPE_UINT8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read u8: {}", e))?;
            Ok(format!("{}", buf[0]))
        }
        GGUF_TYPE_INT8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read i8: {}", e))?;
            Ok(format!("{}", buf[0] as i8))
        }
        GGUF_TYPE_UINT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read u16: {}", e))?;
            Ok(format!("{}", u16::from_le_bytes(buf)))
        }
        GGUF_TYPE_INT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read i16: {}", e))?;
            Ok(format!("{}", i16::from_le_bytes(buf)))
        }
        GGUF_TYPE_UINT32 => {
            let v = read_u32_le(r)?;
            Ok(format!("{}", v))
        }
        GGUF_TYPE_INT32 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read i32: {}", e))?;
            Ok(format!("{}", i32::from_le_bytes(buf)))
        }
        GGUF_TYPE_FLOAT32 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read f32: {}", e))?;
            Ok(format!("{}", f32::from_le_bytes(buf)))
        }
        GGUF_TYPE_BOOL => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read bool: {}", e))?;
            Ok(if buf[0] == 0 {
                "false".into()
            } else {
                "true".into()
            })
        }
        GGUF_TYPE_STRING => read_string(r),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            let mut elems = Vec::with_capacity(count as usize);
            for _ in 0..count {
                elems.push(read_typed_value(r, elem_type)?);
            }
            Ok(format!("[{}]", elems.join(", ")))
        }
        GGUF_TYPE_UINT64 => {
            let v = read_u64_le(r)?;
            Ok(format!("{}", v))
        }
        GGUF_TYPE_INT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read i64: {}", e))?;
            Ok(format!("{}", i64::from_le_bytes(buf)))
        }
        GGUF_TYPE_FLOAT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read f64: {}", e))?;
            Ok(format!("{}", f64::from_le_bytes(buf)))
        }
        GGUF_TYPE_BF16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF read bf16: {}", e))?;
            // Represent as approximate f32 value (zero-extend bf16 mantissa)
            let bits = u16::from_le_bytes(buf) as u32;
            let f32_bits = bits << 16;
            let approx = f32::from_bits(f32_bits);
            Ok(format!("{}", approx))
        }
        _ => {
            // Unknown type — skip 4 bytes (the minimum value payload) as a
            // best-effort recovery, then return a placeholder.
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("GGUF unknown type {}: {}", typ, e))?;
            Ok(format!("<unknown_type_{}>", typ))
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Parse the GGUF header and return metadata + tensor inventory.
///
/// The ComputeImage compiler uses this to understand the model architecture
/// and tensor layout without loading the full weight data into memory.
/// Weight data is read on-demand during the actual compilation step.
pub fn parse_gguf_header(
    path: &Path,
) -> Result<(Vec<(String, String)>, Vec<GgufTensorMeta>), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("Cannot open GGUF file: {}", e))?;

    // ── Magic ───────────────────────────────────────────────────────────────
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("GGUF read magic: {}", e))?;
    if &magic != b"GGUF" {
        return Err(format!(
            "Not a GGUF file: magic={magic:?} expected b\"GGUF\""
        ));
    }

    // ── Version ─────────────────────────────────────────────────────────────
    let version = read_u32_le(&mut f)?;
    if version < 1 || version > 3 {
        return Err(format!(
            "Unsupported GGUF version {version} (supported: 1-3)"
        ));
    }

    // ── Tensor count ────────────────────────────────────────────────────────
    let tensor_count = if version >= 2 {
        read_u64_le(&mut f)?
    } else {
        read_u32_le(&mut f)? as u64
    };

    // ── Metadata KV count ───────────────────────────────────────────────────
    let metadata_kv_count = if version >= 2 {
        read_u64_le(&mut f)?
    } else {
        read_u32_le(&mut f)? as u64
    };

    // ── Metadata KV pairs ───────────────────────────────────────────────────
    let mut metadata: Vec<(String, String)> = Vec::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut f)?;
        let value_type = read_u32_le(&mut f)?;
        let value = read_typed_value(&mut f, value_type)?;
        metadata.push((key, value));
    }

    // ── Tensor infos ────────────────────────────────────────────────────────
    let mut tensors: Vec<GgufTensorMeta> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut f)?;
        let n_dims = read_u32_le(&mut f)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u32_le(&mut f)?);
        }
        let dtype = read_u32_le(&mut f)?;
        let byte_offset = read_u64_le(&mut f)?;

        let byte_size = ggml_tensor_byte_size(dtype, &dims);
        let dtype_name = ggml_dtype_name(dtype).to_string();

        tensors.push(GgufTensorMeta {
            name,
            dtype: dtype_name,
            shape: dims,
            byte_offset,
            byte_size,
        });
    }

    Ok((metadata, tensors))
}

// ── Metadata helpers ───────────────────────────────────────────────────────

/// Look up a metadata value by key and parse it as a u64.
fn meta_u64(metadata: &[(String, String)], key: &str) -> Option<u64> {
    let (_, val) = metadata.iter().find(|(k, _)| k == key)?;
    val.parse::<u64>().ok()
}

/// Look up a metadata value by key and parse it as a f64.
fn meta_f64(metadata: &[(String, String)], key: &str) -> Option<f64> {
    let (_, val) = metadata.iter().find(|(k, _)| k == key)?;
    val.parse::<f64>().ok()
}

/// Look up a metadata value by key and return it as a trimmed string.
fn meta_str<'a>(metadata: &'a [(String, String)], key: &str) -> Option<&'a str> {
    let (_, val) = metadata.iter().find(|(k, _)| k == key)?;
    Some(val.trim())
}

// ── import_gguf_model ──────────────────────────────────────────────────────

/// Import a GGUF model into the ComputeImage compilation pipeline.
///
/// This is the entry point for GGUF ingestion:
/// 1. Validates GGUF version and format
/// 2. Extracts architecture metadata (config, tokenizer, vocab)
/// 3. Builds a tensor inventory for the compiler
/// 4. Returns a GgufImportResult that feeds into compile_sequential()
///
/// The actual compilation happens when the result is passed to the
/// compute_image::compile pipeline.
pub fn import_gguf_model(path: &Path) -> Result<GgufImportResult, String> {
    let (metadata, tensor_inventory) = parse_gguf_header(path)?;

    let model_config = extract_architecture(&metadata)?;

    // Try to find a tokenizer alongside the GGUF file.
    let tokenizer_path = find_tokenizer(path);

    Ok(GgufImportResult {
        model_config,
        source_path: path.to_path_buf(),
        tensor_inventory,
        tokenizer_path,
        metadata,
    })
}

/// Extract a `TextArchitecture` from GGUF metadata KV pairs.
fn extract_architecture(
    metadata: &[(String, String)],
) -> Result<crate::config::TextArchitecture, String> {
    let hidden_size = meta_u64(metadata, keys::HIDDEN_SIZE).unwrap_or(4096) as u32;
    let num_attention_heads = meta_u64(metadata, keys::NUM_ATTENTION_HEADS).unwrap_or(32) as u32;
    let num_kv_heads =
        meta_u64(metadata, keys::NUM_KV_HEADS).unwrap_or(num_attention_heads as u64) as u32;
    let head_dim = meta_u64(metadata, keys::HEAD_DIM)
        .unwrap_or((hidden_size / num_attention_heads) as u64) as u32;
    let num_hidden_layers = meta_u64(metadata, keys::NUM_HIDDEN_LAYERS).unwrap_or(32) as u32;
    let vocab_size = meta_u64(metadata, keys::VOCAB_SIZE).unwrap_or(32000) as u32;
    let intermediate_size =
        meta_u64(metadata, keys::INTERMEDIATE_SIZE).unwrap_or(hidden_size as u64 * 4) as u32;
    let max_seq_len = meta_u64(metadata, keys::MAX_SEQ_LEN).unwrap_or(131072) as u32;
    let rms_norm_eps = meta_f64(metadata, keys::NORM_EPS).unwrap_or(1e-6);
    let rope_theta = meta_f64(metadata, keys::ROPE_THETA).unwrap_or(10000.0);
    let model_type = meta_str(metadata, keys::MODEL_TYPE)
        .unwrap_or("unknown")
        .to_string();

    let num_layers = num_hidden_layers as usize;
    let mut layer_types = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layer_types.push(crate::config::AttentionKind::SlidingAttention);
    }

    Ok(crate::config::TextArchitecture {
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        global_head_dim: None,
        num_global_key_value_heads: None,
        num_hidden_layers,
        vocab_size,
        sliding_window: 32768,
        max_position_embeddings: max_seq_len,
        rms_norm_eps,
        tie_word_embeddings: false,
        attention_k_eq_v: true,
        final_logit_softcapping: None,
        hidden_size_per_layer_input: 0,
        layer_types,
        rope_local: crate::config::RopeSpec {
            theta: rope_theta,
            rope_type: "default".into(),
            partial_rotary_factor: None,
        },
        rope_global: None,
        model_type,
        moe_config: None,
        diffusion_config: None,
    })
}

/// Search for a tokenizer file alongside the GGUF path.
fn find_tokenizer(gguf_path: &Path) -> Option<std::path::PathBuf> {
    let dir = gguf_path.parent().unwrap_or(Path::new("."));
    for name in &[
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "tokenizer_config.json",
    ] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ── gguf_to_manifest ───────────────────────────────────────────────────────

/// Build a Tribunus ModelManifest from GGUF metadata.
///
/// Maps GGUF metadata keys to Tribunus's internal config types
/// (TextArchitecture, QuantizationMeta, etc.) so the compiler can
/// work with the model without needing the original config.json.
pub fn gguf_to_manifest(
    metadata: &[(String, String)],
) -> Result<crate::config::ModelManifest, String> {
    // Build a JSON-like string from metadata for hashing (analogous to
    // config.rs hashing the raw config.json).
    let meta_json = metadata_to_json_string(metadata);
    let mut hasher = Sha256::new();
    hasher.update(meta_json.as_bytes());
    let config_hash = format!("{:x}", hasher.finalize());

    let model_type = meta_str(metadata, keys::MODEL_TYPE)
        .unwrap_or("unknown")
        .to_string();
    let quantization_version = meta_u64(metadata, keys::QUANTIZATION_VERSION);
    let file_type = meta_u64(metadata, keys::FILE_TYPE);

    Ok(crate::config::ModelManifest {
        config_path: String::new(),
        config_hash,
        model_type,
        has_text_config: true,
        has_vision_config: false,
        has_audio_config: false,
        has_quantization_metadata: quantization_version.is_some() || file_type.is_some(),
        quantization_bits: file_type.and_then(|ft| gguf_file_type_to_bits(ft as u32)),
        quantization_group_size: None, // GGUF doesn't always encode group_size in metadata
        quantization_mode: file_type.and_then(|ft| gguf_file_type_to_mode_str(ft as u32)),
        vision_config: None,
        audio_config: None,
        safetensors_shards: Vec::new(),
    })
}

/// Serialise metadata KVs into a deterministic JSON-like string for hashing.
fn metadata_to_json_string(metadata: &[(String, String)]) -> String {
    // Sort by key for determinism.
    let mut sorted: Vec<&(String, String)> = metadata.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut s = String::from('{');
    for (i, (k, v)) in sorted.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(k);
        s.push_str("\":\"");
        s.push_str(v);
        s.push('"');
    }
    s.push('}');
    s
}

/// Map GGUF file_type to an estimated bit width (for display / context).
fn gguf_file_type_to_bits(file_type: u32) -> Option<u32> {
    match file_type {
        0 | 1 => Some(16), // F32 or F16 (effectively unquantized)
        2 => Some(4),      // Q4_0
        3 => Some(4),      // Q4_1
        4 => Some(4),      // Q4_2 (removed but may appear)
        5 => Some(4),      // Q4_3 (removed)
        6 => Some(5),      // Q5_0
        7 => Some(5),      // Q5_1
        8 => Some(8),      // Q8_0
        9 => Some(8),      // Q8_1
        10 => Some(2),     // Q2_K
        11 => Some(3),     // Q3_K
        12 => Some(4),     // Q4_K
        13 => Some(5),     // Q5_K
        14 => Some(6),     // Q6_K
        15 => Some(8),     // Q8_K
        16 => Some(1),     // IQ1_S
        17 => Some(1),     // IQ1_M
        18 => Some(2),     // IQ2_XXS
        19 => Some(2),     // IQ2_XS
        20 => Some(2),     // IQ2_S
        21 => Some(2),     // IQ2_M
        22 => Some(3),     // IQ3_XXS
        23 => Some(3),     // IQ3_XS
        24 => Some(3),     // IQ3_S
        26 => Some(4),     // IQ4_NL
        27 => Some(4),     // IQ4_XS
        28 => Some(16),    // BF16
        _ => None,
    }
}

/// Map GGUF file_type to a quantization mode string.
fn gguf_file_type_to_mode_str(file_type: u32) -> Option<String> {
    match file_type {
        0 | 1 | 28 => None, // unquantized
        2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 => Some("Affine".into()),
        10 | 11 | 12 | 13 | 14 | 15 => Some("Affine".into()),
        16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 26 | 27 => Some("Affine".into()),
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gguf_header() {
        // Build a minimal valid GGUF v3 file in memory and write to a temp file.
        let dir = std::env::temp_dir();
        let path = dir.join("test_minimal.gguf");

        // GGUF v3 layout:
        // [0..4]  magic "GGUF"
        // [4..8]  version = 3 (u32 LE)
        // [8..16] tensor_count = 0 (u64 LE)
        // [16..24] metadata_kv_count = 1 (u64 LE)
        // [24..]  key: "test.key" | value: u32 = 42
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
                                                    // KV: key_len=8, "test.key", value_type=UINT32(4), value=42
        buf.extend_from_slice(&8u64.to_le_bytes()); // key_len
        buf.extend_from_slice(b"test.key"); // key
        buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = UINT32
        buf.extend_from_slice(&42u32.to_le_bytes()); // value

        std::fs::write(&path, &buf).expect("write test gguf");

        let result = parse_gguf_header(&path);
        assert!(
            result.is_ok(),
            "parse_gguf_header should succeed: {:?}",
            result.err()
        );
        let (metadata, tensors) = result.unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].0, "test.key");
        assert_eq!(metadata[0].1, "42");
        assert!(tensors.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_magic.gguf");

        // Write a file with wrong magic.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"NOTG"); // not GGUF
        buf.extend_from_slice(&3u32.to_le_bytes());
        std::fs::write(&path, &buf).expect("write bad gguf");

        let result = parse_gguf_header(&path);
        assert!(result.is_err(), "should reject bad magic");
        assert!(
            result.unwrap_err().contains("Not a GGUF file"),
            "error should mention bad magic"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_to_manifest() {
        let metadata = vec![
            ("general.architecture".to_string(), "llama".to_string()),
            ("llama.vocab_size".to_string(), "32000".to_string()),
            ("llama.embedding_length".to_string(), "4096".to_string()),
            ("llama.block_count".to_string(), "32".to_string()),
            ("general.file_type".to_string(), "2".to_string()), // Q4_0
        ];
        let manifest = gguf_to_manifest(&metadata).expect("gguf_to_manifest should succeed");
        assert_eq!(manifest.model_type, "llama");
        assert!(manifest.has_text_config);
        assert!(!manifest.has_vision_config);
        assert!(manifest.has_quantization_metadata);
        assert_eq!(manifest.quantization_bits, Some(4));
        assert!(!manifest.config_hash.is_empty());
        assert!(manifest.safetensors_shards.is_empty());
    }

    #[test]
    fn test_gguf_to_manifest_no_quant() {
        let metadata = vec![
            ("general.architecture".to_string(), "gemma2".to_string()),
            ("llama.vocab_size".to_string(), "256000".to_string()),
            ("llama.embedding_length".to_string(), "3584".to_string()),
        ];
        let manifest = gguf_to_manifest(&metadata).expect("gguf_to_manifest should succeed");
        assert_eq!(manifest.model_type, "gemma2");
        assert!(!manifest.has_quantization_metadata);
        assert!(manifest.quantization_bits.is_none());
    }

    #[test]
    fn test_parse_gguf_v1_format() {
        // GGUF v1 uses u32 for tensor_count and metadata_kv_count.
        let dir = std::env::temp_dir();
        let path = dir.join("test_v1.gguf");

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count (u32 in v1)
        buf.extend_from_slice(&2u32.to_le_bytes()); // metadata_kv_count (u32 in v1)

        // KV 1: "arch" = "test"
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"arch");
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"test");

        // KV 2: "layers" = 12
        buf.extend_from_slice(&6u64.to_le_bytes());
        buf.extend_from_slice(b"layers");
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&12u32.to_le_bytes());

        std::fs::write(&path, &buf).expect("write v1 gguf");

        let result = parse_gguf_header(&path);
        assert!(result.is_ok(), "v1 should parse: {:?}", result.err());
        let (metadata, tensors) = result.unwrap();
        assert_eq!(metadata.len(), 2);
        assert!(tensors.is_empty());

        std::fs::remove_file(&path).ok();
    }
}
