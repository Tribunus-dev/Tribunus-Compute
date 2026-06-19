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
    pub dtype: String,        // "f32", "f16", "q4_0", "q4_K_M", "q8_0", etc.
    pub shape: Vec<u32>,
    pub byte_offset: u64,
    pub byte_size: u64,
}

/// Parse the GGUF header and return metadata + tensor inventory.
///
/// The ComputeImage compiler uses this to understand the model architecture
/// and tensor layout without loading the full weight data into memory.
/// Weight data is read on-demand during the actual compilation step.
pub fn parse_gguf_header(
    path: &std::path::Path,
) -> Result<(Vec<(String, String)>, Vec<GgufTensorMeta>), String> {
    Err("GGUF header parser: stub — implement per-backend import pipeline".to_string())
}

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
pub fn import_gguf_model(
    path: &std::path::Path,
) -> Result<GgufImportResult, String> {
    Err("GGUF import: stub — implement per-backend import pipeline".to_string())
}

/// Build a Tribunus ModelManifest from GGUF metadata.
///
/// Maps GGUF metadata keys to Tribunus's internal config types
/// (TextArchitecture, QuantizationMeta, etc.) so the compiler can
/// work with the model without needing the original config.json.
pub fn gguf_to_manifest(
    metadata: &[(String, String)],
) -> Result<crate::config::ModelManifest, String> {
    Err("GGUF to manifest conversion: stub".to_string())
}
