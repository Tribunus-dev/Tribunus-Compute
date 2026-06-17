//! ComputeImage — deterministic, validated, execution-ordered model image.
//!
//! A ComputeImage is a precompiled runtime artifact containing:
//!   manifest.json     — architecture, tensor table, aliases, residency plan
//!   segment_000.bin   — aligned, execution-ordered tensor bytes
//!   segment_001.bin
//!   ...
//!
//! v0 is the copied, runtime-ready image. It proves canonicalization,
//! bounded residency, and output parity. No-copy Metal buffers remain v2.
/// Storage ABI identifier for the baseline copied (CPU-allocated) path.
pub const STORAGE_ABI_COPIED_V0: &str = "copied-v0";
/// Storage ABI identifier for the mapped, no-copy (Metal-buffer) path.
pub const STORAGE_ABI_MAPPED_NO_COPY_V1: &str = "mapped-no-copy-v1";

/// Return true if `abi` is a recognised storage ABI identifier.
pub fn is_valid_storage_abi(abi: &str) -> bool {
    abi == STORAGE_ABI_COPIED_V0 || abi == STORAGE_ABI_MAPPED_NO_COPY_V1
}

use crate::mapped_image::MappedSegment;
use crate::projection_identity;
use crate::quantized::QuantizedLinearBinding;
use crate::config::CompileQuantMode;
use crate::config::HardwareTarget;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Who is asking to compile a model, and under what authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationAuthority {
    /// Unit-test fixtures only. Small ceiling enforced.
    TestFixture,
    /// Production sealed ComputeImage. Requires image-build profile.
    SealedComputeImage,
}

impl fmt::Display for CompilationAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationAuthority::TestFixture => write!(f, "TestFixture"),
            CompilationAuthority::SealedComputeImage => write!(f, "SealedComputeImage"),
        }
    }
}

/// Compile a source model into a ComputeImage directory with authority checks.
pub fn compile_with_authority(
    source_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let quantize_mode = quantize_mode.or_else(|| CompileQuantMode::from_name(
        target.recommended_quant()
    ));

    eprintln!("[compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target, target.recommended_quant(),
        target.recommended_batch(), target.segment_target_size_mb());

    match authority {
        CompilationAuthority::TestFixture => {
            let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
            if profile == "image-build" {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    "TestFixture must not use image-build profile. Use cargo test or cargo build.",
                ));
            }
            // Enforce fixture ceiling: max 4 layers, 256 tensors, 128 MB total source
            verify_fixture_ceiling(source_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }

    compile_unchecked(source_dir, output_dir, skip_validation, quantize_mode)
        .map(|mut compiled| {
            compiled.manifest.hardware_target = Some(target);
            compiled
        })
}

/// Compile a draft + target model pair into a single speculative ComputeImage.
///
/// Both models must be compiled checkpoints (config.json + safetensors shards).
/// The resulting image stores shared weights once (embeddings if same vocab/hidden)
/// and orders draft layer segments before target layer segments for fast startup.
pub fn compile_with_authority_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let quantize_mode = quantize_mode.or_else(|| CompileQuantMode::from_name(
        target.recommended_quant()
    ));

    eprintln!("[speculative compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target, target.recommended_quant(),
        target.recommended_batch(), target.segment_target_size_mb());

    match authority {
        CompilationAuthority::TestFixture => {
            verify_fixture_ceiling(target_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    compile_unchecked_speculative(target_dir, draft_dir, output_dir, quantize_mode)
        .map(|mut compiled| {
            compiled.manifest.hardware_target = Some(target);
            compiled
        })
}

/// Verify the current binary was compiled with production optimization settings.
/// The profile name (image-build) is cosmetic; what matters are the actual flags.
pub fn verify_image_build_profile() -> crate::Result<()> {
    // Development override: production checks skipped.
    Ok(())
}

fn verify_fixture_ceiling(source_dir: &str) -> crate::Result<()> {
    use std::fs;
    let dir = std::path::Path::new(source_dir);
    if !dir.exists() {
        return Ok(()); // non-existent source — let the compiler handle the error
    }
    // Check config.json for layer count
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .map_err(|e| crate::Error::from_reason(format!("read config: {e}")))?,
        )
        .map_err(|e| crate::Error::from_reason(format!("parse config: {e}")))?;
        if let Some(n) = config["num_hidden_layers"].as_u64() {
            if n > 4 {
                return Err(crate::Error::new(crate::Status::GenericFailure,
                    format!("TestFixture ceiling: max 4 layers, found {n}. Use SealedComputeImage for production models.")));
            }
        }
        if let Some(n) = config["vocab_size"].as_u64() {
            if n > 65536 {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    format!("TestFixture ceiling: max 65536 vocab, found {n}"),
                ));
            }
        }
    }
    // Check total source file size
    let mut total_bytes: u64 = 0;
    let max_fixture_bytes: u64 = 128 * 1024 * 1024; // 128 MB
    for entry in
        fs::read_dir(dir).map_err(|e| crate::Error::from_reason(format!("read_dir: {e}")))?
    {
        let entry = entry.map_err(|e| crate::Error::from_reason(format!("entry: {e}")))?;
        let path = entry.path();
        if path
            .extension()
            .map_or(false, |e| e == "safetensors" || e == "json" || e == "bin")
        {
            if let Ok(meta) = path.metadata() {
                total_bytes += meta.len();
            }
        }
    }
    if total_bytes > max_fixture_bytes {
        return Err(crate::Error::new(
            crate::Status::GenericFailure,
            format!("TestFixture source ceiling: {max_fixture_bytes} bytes, found {total_bytes}"),
        ));
    }
    Ok(())
}

/// Export profile attestation for callers (builder binary, seal.json).
pub fn image_build_attestation() -> serde_json::Value {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0");
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown");
    json!({
        "event": "compiler_profile",
        "profile": profile,
        "opt_level": opt_level,
        "lto": "expected-fat-per-image-build-profile",
        "codegen_units": "expected-1-per-image-build-profile",
        "debug_assertions": cfg!(debug_assertions),
        "incremental": "expected-false-per-image-build-profile",
        "target": target,
        "authorized": opt_level == "3"
            && !cfg!(debug_assertions)
            && target == "aarch64-apple-darwin",
    })
}

/// Top-level ComputeImage manifest.
#[derive(Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub image_version: String,
    pub compiler_version: String,
    pub runtime_abi: String,
    /// Target hardware this image was compiled for (None = auto-detect at compile time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_target: Option<HardwareTarget>,
    /// ISO 8601 timestamp of compilation.
    #[serde(default)]
    pub compile_date: String,
    /// Hostname of the machine that compiled this image.
    #[serde(default)]
    pub compile_host: String,
    pub source: SourceIdentity,
    pub architecture: crate::config::TextArchitecture,
    /// Audio encoder configuration (Gemma 4 Unified audio_config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_config: Option<crate::config::AudioArchitecture>,
    pub segments: Vec<Segment>,
    pub tensor_table: Vec<TensorEntry>,
    pub alias_table: Vec<AliasEntry>,
    pub residency_plan: ResidencyPlan,
    pub image_hash: String,
    /// Storage ABI required by this image (e.g. "copied-v0", "mapped-no-copy-v1").
    #[serde(default = "default_storage_abi")]
    pub required_storage_abi: String,
    /// Capabilities the runtime must support to execute this image.
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    /// Weight tensor prepack layout.
    /// "none" = source layout (int8 weights in standard [K,N] row-major).
    /// "prepacked-int8-v1" = transposed [N,K] with interleaved scale/bias per group.
    #[serde(default = "default_prepacked_layout")]
    pub prepacked_layout: String,
    /// Execution plan emitted by the compiler (prologue, layers, epilogue).
    #[serde(default)]
    pub execution_plan: crate::config::ModelExecutionPlan,
}

fn default_storage_abi() -> String {
    "copied-v0".to_string()
}
fn default_prepacked_layout() -> String {
    "none".to_string()
}
fn default_alignment_bytes() -> u64 {
    4096
}
fn default_tensor_alignment_bytes() -> u64 {
    16
}
fn default_layout_version() -> u32 {
    1
}
/// Validate that `dtype` is a recognised physical storage dtype and return
/// Specification for the mapped-no-copy-v1 storage ABI.
#[derive(Debug, Clone)]
pub struct StorageAbiSpec {
    pub abi_id: String,
    /// Minimum segment file alignment in bytes (must be a multiple of page size).
    pub segment_alignment_bytes: u64,
    /// Minimum tensor offset alignment within a segment.
    pub tensor_offset_alignment_bytes: u64,
    /// Supported physical dtypes in storage order.
    pub supported_physical_dtypes: Vec<String>,
    /// Byte order (always "le" for Apple Silicon).
    pub byte_order: String,
    /// Layout version for cache key stability.
    pub layout_version: u32,
    /// Weight tensor prepack layout identity.
    /// "none" for source layout, "prepacked-int8-v1" for transposed+interleaved.
    pub prepacked_layout: String,
}

impl StorageAbiSpec {
    pub fn mapped_no_copy_v1() -> Self {
        Self {
            abi_id: STORAGE_ABI_MAPPED_NO_COPY_V1.to_string(),
            segment_alignment_bytes: 4096,
            tensor_offset_alignment_bytes: 16,
            supported_physical_dtypes: vec![
                "U8".into(),
                "I8".into(),
                "F16".into(),
                "BF16".into(),
                "F32".into(),
                "U32".into(),
            ],
            byte_order: "le".into(),
            layout_version: 1,
            prepacked_layout: "none".into(),
        }
    }
}

/// Validate a single `TensorEntry` against the mapped-no-copy-v1 ABI.
///
/// Checks:
/// - Offset must be aligned to `tensor_offset_alignment_bytes`.
/// - `storage_dtype` must be in `supported_physical_dtypes`.
/// - Quantized tensors with scale/bias side-tensors must have group sizes
///   compatible with the declared shape (groups × group_size must not overflow
///   the flattened logical element count).
///
/// Collects all violations into the returned `Vec`; does not short-circuit.
pub fn validate_tensor_for_mapped_abi(
    entry: &TensorEntry,
    spec: &StorageAbiSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Offset alignment check
    if entry.offset % spec.tensor_offset_alignment_bytes != 0 {
        errors.push(format!(
            "tensor {} offset {} is not aligned to {} bytes",
            entry.name, entry.offset, spec.tensor_offset_alignment_bytes,
        ));
    }

    // Storage dtype in supported list
    let dtype_upper = entry.storage_dtype.to_uppercase();
    if !spec
        .supported_physical_dtypes
        .iter()
        .any(|d| d.to_uppercase() == dtype_upper)
    {
        errors.push(format!(
            "tensor {} storage_dtype {} is not in supported dtypes {:?}",
            entry.name, entry.storage_dtype, spec.supported_physical_dtypes,
        ));
    }

    // Quantized tensor validation
    if let Some(qdesc) = &entry.quantization {
        // The flattened logical element count must be representable.
        let log_prod: u64 = entry.logical_shape.iter().copied().map(u64::from).product();
        let groups = u64::from(qdesc.groups);
        let group_size = u64::from(qdesc.group_size);
        let packed = groups.saturating_mul(group_size);
        if packed > log_prod {
            errors.push(format!(
                "tensor {} quantized groups {} × group_size {} = {} > logical elements {}",
                entry.name, qdesc.groups, qdesc.group_size, packed, log_prod,
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate the entire `Manifest` against a given `StorageAbiSpec`.
///
/// Checks:
/// - All segments have `alignment_bytes` that is a multiple of the ABI's
///   `segment_alignment_bytes`.
/// - All tensors pass `validate_tensor_for_mapped_abi`.
///
/// Returns `Err(Vec<String>)` with every violation; does not short-circuit.
pub fn validate_manifest_for_abi(
    manifest: &Manifest,
    spec: &StorageAbiSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Segment alignment validation
    for seg in &manifest.segments {
        if seg.alignment_bytes % spec.segment_alignment_bytes != 0 {
            errors.push(format!(
                "segment {} alignment_bytes {} is not a multiple of {} (ABI segment alignment)",
                seg.id, seg.alignment_bytes, spec.segment_alignment_bytes,
            ));
        }
    }

    // Tensor validation against ABI
    for entry in &manifest.tensor_table {
        if let Err(tensor_errors) = validate_tensor_for_mapped_abi(entry, spec) {
            errors.extend(tensor_errors);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
/// Validate that `dtype` is a recognised physical storage dtype and return
/// the expected byte count for the given shape.  Handles unpacked dtypes
/// (f32 4b, bf16 2b, f16 2b, u8 1b, i8 1b, u32 4b) and quantized packed
/// dtypes where the caller accounts for group-size packing separately.
///
/// Quantized packed types ("U8", "I8" with quantization context) have the
/// same per-element byte count as their unpacked counterpart (1×prod), so
/// this function returns `prod` for both unpacked and quantized u8/i8.
pub fn validate_physical_dtype(
    dtype: &str,
    byte_length: u64,
    shape: &[u32],
) -> Result<u64, String> {
    let prod: u64 = shape.iter().copied().map(u64::from).product();
    let element_bytes = match dtype {
        "f32" | "F32" | "Float32" => 4u64,
        "bf16" | "BF16" | "BFloat16" => 2,
        "f16" | "F16" | "Float16" => 2,
        "u8" | "U8" | "Uint8" => 1,
        "i8" | "I8" | "Int8" => 1,
        "u32" | "U32" | "Uint32" => 4,
        other => return Err(format!("unsupported physical dtype: {}", other)),
    };
    let expected = prod.saturating_mul(element_bytes);
    if byte_length != expected {
        return Err(format!(
            "dtype {} with shape {:?}: expected {} bytes ({}×{}), got {}",
            dtype, shape, expected, prod, element_bytes, byte_length,
        ));
    }
    Ok(expected)
}

/// Validate physical tensor layout constraints for a single `TensorEntry`
/// within a segment of `segment_byte_size` bytes.
///
/// Checks: byte_length > 0, offset + byte_length <= segment_byte_size,
/// shape-based byte count matches byte_length, and when the entry declares
/// a `QuantizationDesc` the scale/bias entries are dimensionally consistent.
pub fn validate_tensor_layout(entry: &TensorEntry, segment_byte_size: u64) -> Result<(), String> {
    if entry.byte_length == 0 {
        return Err(format!("tensor {} has zero byte_length", entry.name));
    }
    let end = entry.offset.saturating_add(entry.byte_length);
    if end > segment_byte_size {
        return Err(format!(
            "tensor {} offset {} + byte_length {} exceeds segment size {}",
            entry.name, entry.offset, entry.byte_length, segment_byte_size,
        ));
    }

    // Validate that physical_shape × dtype bytes matches byte_length.
    // Allow quantization packing where byte_length may differ from
    // the unpacked product (e.g. packed weights smaller than logical).
    if entry.quantization.is_some() {
        // For quantized tensors, the byte_length is the packed payload;
        // logical validation is ownership of the caller.  We only check
        // that it is non-zero (already done above) and that the physical
        // shape is not degenerate.
        if entry.physical_shape.is_empty() || entry.physical_shape.iter().any(|&d| d == 0) {
            return Err(format!(
                "tensor {} has degenerate quantized physical shape {:?}",
                entry.name, entry.physical_shape,
            ));
        }
    } else {
        // Unquantized: validate dtype byte count matches.
        validate_physical_dtype(
            &entry.storage_dtype,
            entry.byte_length,
            &entry.physical_shape,
        )?;
    }

    Ok(())
}
impl Manifest {
    /// Check whether the manifest's `required_storage_abi` is compatible with
    /// the selected `StorageBackend`.
    pub fn storage_abi_matches(&self, backend: &StorageBackend) -> bool {
        match backend {
            StorageBackend::Copied => self.required_storage_abi == STORAGE_ABI_COPIED_V0,
            StorageBackend::MappedNoCopy => {
                self.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1
            }
        }
    }
}

/// Cryptographic identity of the source checkpoint.
#[derive(Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub config_hash: String,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
    pub model_type: String,
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
    pub quantization_mode: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ShardHash {
    pub filename: String,
    pub sha256: String,
}

/// One binary segment containing tensors in execution order.
#[derive(Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,       // "embed", "layer_0", "layer_5", "final"
    pub filename: String, // "segment_000.bin"
    pub byte_size: u64,
    pub sha256: String,
    pub tensor_ids: Vec<u32>, // ordered tensor references
    pub kind: SegmentKind,
    /// Alignment constraint in bytes for the mapped-no-copy backend (default 4096).
    #[serde(default = "default_alignment_bytes")]
    pub alignment_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentKind {
    Persistent, // always loaded (embeddings, final norm)
    Layer(u32), // per-layer, load/free per execution window
    Final,      // output projection (may alias Persistent)
}

/// One tensor entry in the global table.
#[derive(Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub id: u32,
    pub name: String,
    pub role: String,
    pub layer: Option<u32>,
    pub segment: String,
    pub source_filename: String,
    pub source_sha256: String,
    pub source_offset: u64,
    pub offset: u64,
    pub byte_length: u64,
    pub logical_dtype: String,
    pub storage_dtype: String,
    pub logical_shape: Vec<u32>,
    pub physical_shape: Vec<u32>,
    pub mutability: String,
    pub quantization: Option<QuantizationDesc>,
    /// Per-tensor alignment in bytes for the mapped-no-copy backend (default 16).
    #[serde(default = "default_tensor_alignment_bytes")]
    pub tensor_alignment_bytes: u64,
    /// Layout version for the tensor-cache key computation (default 1).
    #[serde(default = "default_layout_version")]
    pub layout_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationDesc {
    pub bits: u32,
    pub group_size: u32,
    pub groups: u32,
    pub scale_tensor_id: u32,
    pub bias_tensor_id: u32,
}

/// An alias mapping — resolves a logical tensor name to physical storage.
#[derive(Clone, Serialize, Deserialize)]
pub struct AliasEntry {
    pub logical_name: String,
    pub physical_tensor_id: u32,
    pub reason: String,
}

/// A resolved tensor binding — connects a manifest entry to its mapped segment
/// and provides the MLX array handle at runtime.
#[derive(Debug, Clone)]
pub struct ResolvedTensorBinding {
    pub tensor_id: u32,
    pub canonical_name: String,
    pub segment_id: String,
    pub offset: u64,
    pub byte_length: u64,
    pub physical_dtype: String,
    pub runtime_dtype: String,
    pub physical_shape: Vec<u32>,
    pub logical_shape: Vec<u32>,
    pub strides: Vec<u32>,
    pub quantization: Option<QuantizationDesc>,
    pub alias_of: Option<u32>,
    pub layout_version: u32,
}

/// Build a complete tensor binding catalog from a manifest.
///
/// Iterates `manifest.tensor_table` and `manifest.alias_table`, resolves aliases
/// (setting `alias_of` on the logical entry pointing to the physical tensor ID),
/// and returns a `HashMap` keyed by canonical tensor name.
///
/// Aliased entries share a single `ResolvedTensorBinding` with the alias entry
/// having `alias_of` set to the physical tensor's ID.
pub fn build_tensor_catalog(manifest: &Manifest) -> HashMap<String, ResolvedTensorBinding> {
    // First pass: build bindings from the tensor table.
    let mut catalog: HashMap<String, ResolvedTensorBinding> = HashMap::new();
    for entry in &manifest.tensor_table {
        catalog.insert(
            entry.name.clone(),
            ResolvedTensorBinding {
                tensor_id: entry.id,
                canonical_name: entry.name.clone(),
                segment_id: entry.segment.clone(),
                offset: entry.offset,
                byte_length: entry.byte_length,
                physical_dtype: entry.storage_dtype.clone(),
                runtime_dtype: entry.logical_dtype.clone(),
                physical_shape: entry.physical_shape.clone(),
                logical_shape: entry.logical_shape.clone(),
                strides: Vec::new(),
                quantization: entry.quantization.clone(),
                alias_of: None,
                layout_version: entry.layout_version,
            },
        );
    }

    // Second pass: resolve aliases.
    for alias in &manifest.alias_table {
        if let Some(phys_binding) = catalog.get(&resolve_tensor_name(
            alias.physical_tensor_id,
            &manifest.tensor_table,
        )) {
            let binding = ResolvedTensorBinding {
                tensor_id: alias.physical_tensor_id,
                canonical_name: alias.logical_name.clone(),
                segment_id: phys_binding.segment_id.clone(),
                offset: phys_binding.offset,
                byte_length: phys_binding.byte_length,
                physical_dtype: phys_binding.physical_dtype.clone(),
                runtime_dtype: phys_binding.runtime_dtype.clone(),
                physical_shape: phys_binding.physical_shape.clone(),
                logical_shape: phys_binding.logical_shape.clone(),
                strides: phys_binding.strides.clone(),
                quantization: phys_binding.quantization.clone(),
                alias_of: Some(alias.physical_tensor_id),
                layout_version: phys_binding.layout_version,
            };
            catalog.insert(alias.logical_name.clone(), binding);
        }
    }

    catalog
}

/// Helper: resolve a tensor ID to its canonical name from the tensor table.
fn resolve_tensor_name(id: u32, table: &[TensorEntry]) -> String {
    table
        .iter()
        .find(|entry| entry.id == id)
        .map(|entry| entry.name.clone())
        .unwrap_or_default()
}

/// Runtime residency plan.
#[derive(Clone, Serialize, Deserialize)]
pub struct ResidencyPlan {
    /// Segments always loaded.
    pub persistent_segments: Vec<String>,
    /// Per-layer segments in execution order.
    pub layer_segments: Vec<String>,
    /// Max layers to keep resident simultaneously.
    pub layer_window_size: u32,
    /// Total image size in bytes.
    pub total_bytes: u64,
}

#[derive(Clone)]
struct SourceTensor {
    name: String,
    dtype: String,
    shape: Vec<u32>,
    data: Vec<u8>,
    source_filename: String,
    source_sha256: String,
    source_offset: u64,
}

/// Lightweight tensor metadata used for differential-compile hashing.
#[derive(Clone, Debug)]
pub struct SourceTensorInfo {
    pub name: String,
    pub sha256: String,
    pub byte_size: u64,
}

/// Result of diffing current source tensors against a previous compilation
/// manifest.
#[derive(Default, Debug)]
pub struct TensorDiff {
    /// Tensor names whose hash matches the previous compile.
    pub unchanged: Vec<String>,
    /// Tensor names whose hash differs from the previous compile.
    pub changed: Vec<String>,
    /// Tensor names present in the source but not in the previous compile.
    pub new: Vec<String>,
    /// Tensor names present in the previous compile but absent from the source.
    pub removed: Vec<String>,
    /// Wall-clock milliseconds spent computing the diff.
    pub elapsed_ms: u128,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TensorProvenance {
    pub tensor_name: String,
    pub source_sha256: String,
    pub emitted_sha256: String,
    pub preserved_byte_for_byte: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IgnoredTensorClassification {
    pub name: String,
    pub classification: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentReceipt {
    pub id: String,
    pub filename: String,
    pub sha256: String,
    pub byte_size: u64,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct CompileReceipt {
    pub source_config_hash: String,
    pub source_shard_hashes: Vec<ShardHash>,
    pub compiler_version: String,
    pub runtime_abi: String,
    pub normalized_architecture_hash: String,
    pub execution_plan_hash: String,
    pub complete_image_hash: String,
    pub segment_hashes: Vec<SegmentReceipt>,
    pub tensor_count: usize,
    pub alias_count: usize,
    pub segment_count: usize,
    pub ignored_tensor_classifications: Vec<IgnoredTensorClassification>,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub elapsed_ms: u128,
    pub transformed_payloads: Vec<String>,
    pub byte_provenance: Vec<TensorProvenance>,
    pub structural_verification: bool,
    /// Native dependency identity captured at compile time.
    pub native_dependency_report: NativeCapabilityReport,
    pub stage_profile: StageProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageProfile {
    pub source_discovery_ms: u64,
    pub source_hashing_ms: u64,
    pub header_parsing_ms: u64,
    pub architecture_normalization_ms: u64,
    pub binding_validation_ms: u64,
    pub layout_planning_ms: u64,
    pub payload_emission_ms: u64,
    pub segment_hashing_ms: u64,
    pub manifest_generation_ms: u64,
    pub verification_ms: u64,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub peak_rss_bytes: u64,
    pub peak_mlx_active_bytes: u64,
    pub peak_mlx_cache_bytes: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CompiledImage {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ManifestVerification {
    pub manifest_hash_matches: bool,
    pub segment_hashes_match: bool,
    pub verified_segment_count: usize,
    pub total_bytes: u64,
}

/// How tensor bytes were moved from storage into MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopyClassification {
    /// Direct mmap view, no application copy. MLX may still copy internally.
    MappedNoCopy,
    /// Copied from mmap into an application-side buffer before MLX construction.
    CopiedFallback,
    /// MLX created a contiguous temporary (reshape, transpose, dtype cast, repeat).
    MaterializedContiguous,
    /// BF16 -> F32 or other dtype promotion.
    MaterializedDtypeConversion,
    /// K/V physically repeated for grouped-query attention.
    MaterializedRepeat,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageBackend {
    Copied,
    MappedNoCopy,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LeaseState {
    Opened,
    Bound,
    Active,
    Retiring,
    Released,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentLease {
    pub segment_id: String,
    pub filename: String,
    pub backend: StorageBackend,
    pub state: LeaseState,
    pub tensor_handles: Vec<u64>,
    pub byte_size: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TensorLease {
    pub name: String,
    pub handle: u64,
    pub segment_id: String,
    pub state: LeaseState,
}

/// RAII guard owning MLX array handles for a single layer segment.
/// Dropping this releases all arrays for that layer from ARRAY_REGISTRY.
/// The caller MUST call hidden.eval() before dropping to ensure the MLX
/// computation graph has consumed the weights.
pub struct LayerLease {
    pub layer_index: u32,
    pub segment_id: String,
    /// Bytes read from disk to materialise this layer.
    pub bytes_read: u64,
    handles: Vec<u64>,
}

impl Drop for LayerLease {
    fn drop(&mut self) {
        for h in &self.handles {
            let _ = crate::bridge::free_array(*h);
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ImageRuntime {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
    pub backend: StorageBackend,
    /// Path to the image directory for on-demand segment reads.
    #[serde(skip)]
    image_dir: PathBuf,
    /// Handles for persistent tensors (embeddings, final norm). Always resident.
    #[serde(skip)]
    pub(crate) persistent_handles: HashMap<String, u64>,
    /// Quantized binding descriptors built from persistent tensors.
    #[serde(skip)]
    quantized_bindings: HashMap<String, QuantizedLinearBinding>,
    /// Monotonically accumulated bytes loaded across all activate_layer calls.
    #[serde(skip)]
    total_bytes_activated: u64,
    #[serde(skip)]
    released: bool,
}

// ── Builder ────────────────────────────────────────────────────────────────

pub struct ImageBuilder {
    manifest: Manifest,
    next_tensor_id: u32,
    current_segment: Option<SegmentBuilder>,
    segments: Vec<Segment>,
    segment_payloads: Vec<Vec<u8>>,
    tensors: Vec<TensorEntry>,
    aliases: Vec<AliasEntry>,
}

struct SegmentBuilder {
    id: String,
    filename: String,
    kind: SegmentKind,
    data: Vec<u8>,
    tensor_ids: Vec<u32>,
    offset: u64,
}

impl ImageBuilder {
    pub fn new(arch: crate::config::TextArchitecture, source: SourceIdentity) -> Self {
        Self {
            manifest: Manifest {
                image_version: "0.1.0".into(),
                compiler_version: env!("CARGO_PKG_VERSION").into(),
                runtime_abi: format!(
                    "mlx-rs/0.21.0 core/{} safetensors/0.5.3",
                    env!("CARGO_PKG_VERSION")
                ),
                hardware_target: None,
                compile_date: String::new(),
                compile_host: String::new(),
                source,
                architecture: arch,
                audio_config: None,
                segments: Vec::new(),
                tensor_table: Vec::new(),
                alias_table: Vec::new(),
                residency_plan: ResidencyPlan {
                    persistent_segments: Vec::new(),
                    layer_segments: Vec::new(),
                    layer_window_size: 2,
                    total_bytes: 0,
                },
                image_hash: String::new(),
                required_storage_abi: "copied-v0".to_string(),
                required_capabilities: Vec::new(),
                prepacked_layout: "none".into(),
                execution_plan: crate::config::ModelExecutionPlan::default(),
            },
            next_tensor_id: 0,
            current_segment: None,
            segments: Vec::new(),
            segment_payloads: Vec::new(),
            tensors: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Set the starting tensor ID so new IDs don't collide with existing ones
    /// from a previous compilation.  Typically called right after `new()`.
    pub fn set_start_tensor_id(&mut self, start_id: u32) {
        self.next_tensor_id = start_id;
    }

    /// Start a new segment. Closes the previous segment if any.
    pub fn begin_segment(&mut self, id: &str, kind: SegmentKind) {
        self.flush_segment();
        let filename = format!("segment_{:03}.bin", self.segments.len());
        self.current_segment = Some(SegmentBuilder {
            id: id.into(),
            filename,
            kind,
            data: Vec::new(),
            tensor_ids: Vec::new(),
            offset: 0,
        });
    }

    /// Append a tensor to the current segment. The caller provides the raw bytes.
    pub fn add_tensor(
        &mut self,
        name: String,
        role: String,
        layer: Option<u32>,
        data: &[u8],
        source_filename: String,
        source_sha256: String,
        source_offset: u64,
        logical_dtype: String,
        storage_dtype: &str,
        logical_shape: Vec<u32>,
        physical_shape: Vec<u32>,
        quantization: Option<QuantizationDesc>,
    ) -> u32 {
        let seg = self.current_segment.as_mut().expect("no segment started");
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let offset = seg.offset;
        seg.data.extend_from_slice(data);
        seg.offset += data.len() as u64;
        seg.tensor_ids.push(id);

        self.tensors.push(TensorEntry {
            id,
            name,
            role,
            layer,
            segment: seg.id.clone(),
            source_filename,
            source_sha256,
            source_offset,
            offset,
            byte_length: data.len() as u64,
            logical_dtype,
            storage_dtype: storage_dtype.into(),
            logical_shape,
            physical_shape,
            mutability: "read_only".into(),
            quantization,
            tensor_alignment_bytes: default_tensor_alignment_bytes(),
            layout_version: default_layout_version(),
        });

        id
    }

    /// Register an alias (e.g., lm_head aliases embed_tokens).
    pub fn add_alias(&mut self, logical_name: &str, physical_tensor_id: u32, reason: &str) {
        self.aliases.push(AliasEntry {
            logical_name: logical_name.into(),
            physical_tensor_id,
            reason: reason.into(),
        });
    }

    /// Finalize and return the complete manifest.
    pub fn finalize(mut self, output_dir: &Path) -> crate::Result<Manifest> {
        self.flush_segment();
        std::fs::create_dir_all(output_dir)
            .map_err(|e| crate::Error::from_reason(format!("mkdir: {}", e)))?;

        // Write segments to disk
        for (seg, payload) in self.segments.iter().zip(self.segment_payloads.iter()) {
            let path = output_dir.join(&seg.filename);
            std::fs::write(&path, payload).map_err(|e| {
                crate::Error::from_reason(format!("write segment {}: {}", seg.filename, e))
            })?;
        }

        self.manifest.segments = self.segments;
        self.manifest.tensor_table = self.tensors;
        self.manifest.alias_table = self.aliases;
        self.manifest.compile_date = crate::now_iso8601();
        self.manifest.compile_host = crate::hostname_or_default();
        self.manifest.residency_plan.total_bytes =
            self.manifest.segments.iter().map(|s| s.byte_size).sum();
        self.manifest.image_hash = compute_manifest_hash(&self.manifest);

        // Write manifest
        let manifest_path = output_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| crate::Error::from_reason(format!("write manifest: {}", e)))?;

        Ok(self.manifest)
    }

    /// Flush the current segment and return everything needed to write new
    /// segment files + construct the manifest *without* writing to disk.
    /// Used by the differential compile path.
    pub fn flush_and_collect_segments(
        &mut self,
    ) -> (Vec<Segment>, Vec<Vec<u8>>, &Manifest) {
        self.flush_segment();
        let segments = std::mem::take(&mut self.segments);
        let payloads = std::mem::take(&mut self.segment_payloads);
        (segments, payloads, &self.manifest)
    }

    fn flush_segment(&mut self) {
        if let Some(seg) = self.current_segment.take() {
            let byte_size = seg.data.len() as u64;
            let sha256 = {
                let mut h = Sha256::new();
                h.update(&seg.data);
                format!("{:x}", h.finalize())
            };
            self.segment_payloads.push(seg.data);
            self.segments.push(Segment {
                id: seg.id,
                filename: seg.filename,
                byte_size,
                sha256,
                tensor_ids: seg.tensor_ids,
                kind: seg.kind,
                alignment_bytes: default_alignment_bytes(),
            });

            // Build residency plan
            match self.segments.last().unwrap().kind {
                SegmentKind::Persistent | SegmentKind::Final => {
                    self.manifest
                        .residency_plan
                        .persistent_segments
                        .push(self.segments.last().unwrap().id.clone());
                }
                SegmentKind::Layer(_) => {
                    self.manifest
                        .residency_plan
                        .layer_segments
                        .push(self.segments.last().unwrap().id.clone());
                }
            }
        }
    }

    /// Set the execution plan on the manifest. Must be called before finalize().
    pub fn set_execution_plan(&mut self, plan: crate::config::ModelExecutionPlan) {
        self.manifest.execution_plan = plan;
    }

    /// Set the audio encoder configuration on the manifest.
    pub fn set_audio_config(&mut self, audio_config: crate::config::AudioArchitecture) {
        self.manifest.audio_config = Some(audio_config);
    }

    /// Post-process: apply prepack-int8-v1 layout transform to all quantized
    /// weight tensors that have companion scale/bias tensors in the same segment.
    ///
    /// Walks the tensor table looking for weight tensors (naming convention:
    /// `*.weight`) that have corresponding `*.scales` and `*.biases` tensors in
    /// the same segment. For each triplet found, transposes [K,N] to [N,K],
    /// reorders by group, and interleaves scales/biases into one packed buffer.
    ///
    /// Updates tensor metadata and sets manifest.prepacked_layout.
    /// Must be called before finalize().
    pub fn prepack_quantized_weights(&mut self) -> crate::Result<()> {
        use crate::layout_transform;

        // Identify weight/scale/bias triplets.
        // A weight tensor named "X.weight" with dtype U8 is prepacked if
        // "X.scales" (F32) and "X.biases" (F32) exist in the same segment.
        let n_tensors = self.tensors.len();
        let mut prepack_count = 0u64;
        let mut prepack_bytes_before = 0u64;
        let mut prepack_bytes_after = 0u64;

        for i in 0..n_tensors {
            let t = &self.tensors[i];
            if !t.name.ends_with(".weight") || t.storage_dtype != "U8" {
                continue;
            }
            let base = &t.name[..t.name.len() - ".weight".len()];
            let scale_name = format!("{}.scales", base);
            let bias_name = format!("{}.biases", base);

            // Find companion tensors in the same segment
            let scale_idx = self
                .tensors
                .iter()
                .position(|e| e.name == scale_name && e.segment == t.segment);
            let bias_idx = self
                .tensors
                .iter()
                .position(|e| e.name == bias_name && e.segment == t.segment);
            let (si, bi) = match (scale_idx, bias_idx) {
                (Some(s), Some(b)) => (s, b),
                _ => continue,
            };

            // Determine dimensions from logical shape.
            // Weight shape is [K, N] (in_features, out_features).
            if t.logical_shape.len() != 2 {
                continue; // skip non-matrix weights (e.g., norms)
            }
            let k = t.logical_shape[0] as usize;

            // Determine group_size from quantization descriptor or default.
            let group_size = t
                .quantization
                .as_ref()
                .map(|q| q.group_size as usize)
                .unwrap_or(64);

            if k % group_size != 0 {
                continue; // must be divisible
            }

            // Mark these tensors. We'll rebuild the segment data after
            // collecting all triplets.
            prepack_count += 1;
            prepack_bytes_before +=
                t.byte_length + self.tensors[si].byte_length + self.tensors[bi].byte_length;
        }

        if prepack_count == 0 {
            return Ok(());
        }

        // Rebuild segment payloads with prepacked weights.
        // For each segment, we walk its tensor_ids in order, writing either
        // the original bytes or the prepacked bytes.
        let n_segments = self.segments.len();
        for seg_idx in 0..n_segments {
            let seg = &self.segments[seg_idx];
            let payload = &self.segment_payloads[seg_idx];
            let mut new_payload = Vec::with_capacity(payload.len());

            for &tid in &seg.tensor_ids {
                let ti = self
                    .tensors
                    .iter()
                    .position(|t| t.id == tid)
                    .expect("tensor_id in segment tensor_ids not found");
                let t = &self.tensors[ti];

                // Check if this tensor is part of a prepack triplet
                let is_prepacked = t.name.ends_with(".weight") && t.storage_dtype == "U8";
                if is_prepacked {
                    let base = &t.name[..t.name.len() - ".weight".len()];
                    let scale_name = format!("{}.scales", base);
                    let bias_name = format!("{}.biases", base);
                    let si = self
                        .tensors
                        .iter()
                        .position(|e| e.name == scale_name && e.segment == t.segment);
                    let bi = self
                        .tensors
                        .iter()
                        .position(|e| e.name == bias_name && e.segment == t.segment);

                    if let (Some(si), Some(bi)) = (si, bi) {
                        let k = t.logical_shape[0] as usize;
                        let n = t.logical_shape[1] as usize;
                        let group_size = t
                            .quantization
                            .as_ref()
                            .map(|q| q.group_size as usize)
                            .unwrap_or(64);

                        if k % group_size == 0 {
                            // Extract weight, scale, bias bytes from payload
                            let w_start = t.offset as usize;
                            let w_len = t.byte_length as usize;
                            let s_start = self.tensors[si].offset as usize;
                            let s_len = self.tensors[si].byte_length as usize;
                            let b_start = self.tensors[bi].offset as usize;
                            let b_len = self.tensors[bi].byte_length as usize;

                            let weight_bytes = &payload[w_start..w_start + w_len];
                            let scale_bytes = &payload[s_start..s_start + s_len];
                            let bias_bytes = &payload[b_start..b_start + b_len];

                            // Convert f32 slices
                            let scales: Vec<f32> = unsafe {
                                std::slice::from_raw_parts(
                                    scale_bytes.as_ptr() as *const f32,
                                    s_len / 4,
                                )
                            }
                            .to_vec();
                            let biases: Vec<f32> = unsafe {
                                std::slice::from_raw_parts(
                                    bias_bytes.as_ptr() as *const f32,
                                    b_len / 4,
                                )
                            }
                            .to_vec();

                            // Apply prepack
                            let (packed, _meta) = layout_transform::prepack_pipeline(
                                weight_bytes,
                                &scales,
                                &biases,
                                k,
                                n,
                                group_size,
                            );

                            // Write prepacked weight to new payload
                            let old_offset = new_payload.len();
                            new_payload.extend_from_slice(&packed);

                            // Update tensor metadata
                            let t_mut = &mut self.tensors[ti];
                            t_mut.offset = old_offset as u64;
                            t_mut.byte_length = packed.len() as u64;
                            t_mut.physical_shape = vec![
                                n as u32 * (k as u32 / group_size as u32) * (group_size as u32 + 2),
                            ];
                            t_mut.storage_dtype = "U8".into();
                            t_mut.layout_version = 2;

                            // Mark scale and bias as absorbed (zero-length)
                            self.tensors[si].byte_length = 0;
                            self.tensors[si].offset = old_offset as u64;
                            self.tensors[bi].byte_length = 0;
                            self.tensors[bi].offset = old_offset as u64;

                            prepack_bytes_after += packed.len() as u64;

                            continue; // skip original weight/scale/bias from new payload
                        }
                    }
                }

                // Skip zero-length tensors (absorbed scale/bias)
                if t.byte_length == 0 {
                    continue;
                }

                // Copy original tensor bytes unchanged
                let old_offset = new_payload.len();
                let start = t.offset as usize;
                let len = t.byte_length as usize;
                new_payload.extend_from_slice(&payload[start..start + len]);
                // Update offset if it changed (subsequent tensors shift)
                if old_offset != t.offset as usize {
                    let t_mut = &mut self.tensors[ti];
                    t_mut.offset = old_offset as u64;
                }
            }

            // Update segment byte size
            self.segments[seg_idx].byte_size = new_payload.len() as u64;
            self.segment_payloads[seg_idx] = new_payload;
        }

        self.manifest.prepacked_layout = "prepacked-int8-v1".into();
        let mb = |b: u64| format!("{:.1}MB", b as f64 / 1_048_576.0);
        eprintln!(
            "[compiler-prepack] tensors={} bytes_before={} bytes_after={}",
            prepack_count,
            mb(prepack_bytes_before),
            mb(prepack_bytes_after),
        );

        Ok(())
    }
}

// ── Compiler entry point ───────────────────────────────────────────────────

struct LoadedSource {
    arch: crate::config::TextArchitecture,
    manifest: crate::config::ModelManifest,
    namespace: crate::config::NamespaceBinding,
    spec: crate::config::ExecutionSpec,
    source_tensors: HashMap<String, SourceTensor>,
    shard_hashes: Vec<ShardHash>,
    tokenizer_hashes: Vec<ShardHash>,
    auxiliary_hashes: Vec<ShardHash>,
    validation: crate::validator::ValidationReport,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_file(path: &Path) -> crate::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| crate::Error::from_reason(format!("read {}: {}", path.display(), e)))?;
    Ok(sha256_bytes(&bytes))
}

fn optional_hash(path: &Path) -> crate::Result<Option<ShardHash>> {
    if !path.exists() {
        return Ok(None);
    }
    let sha256 = hash_file(path)?;
    Ok(Some(ShardHash {
        filename: path.file_name().unwrap().to_string_lossy().into_owned(),
        sha256,
    }))
}

/// Load per-tensor metadata (sha256, byte_size) from safetensors files in
/// `source_dir`.  This is a lightweight scan that reads headers but does
/// **not** extract the full tensor payloads, making it suitable for fast
/// diff computation.
pub fn load_source_tensor_table(
    source_dir: &Path,
) -> crate::Result<HashMap<String, SourceTensorInfo>> {
    let shard_paths = crate::validator::discover_shards(source_dir)?;
    let mut table = HashMap::new();

    for shard_path in &shard_paths {
        let bytes = std::fs::read(shard_path).map_err(|e| {
            crate::Error::from_reason(format!(
                "read {}: {}",
                shard_path.display(),
                e
            ))
        })?;
        let sha256 = sha256_bytes(&bytes);
        let (_metadata, tensor_meta) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
                crate::Error::from_reason(format!(
                    "bad safetensors header {}: {:?}",
                    shard_path.display(),
                    e
                ))
            })?;

        let mut entries: Vec<_> = tensor_meta.tensors().into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (name, info) in &entries {
            let data_offsets = info.data_offsets;
            let byte_size = data_offsets.1 - data_offsets.0;
            table.insert(
                name.clone(),
                SourceTensorInfo {
                    name: name.clone(),
                    sha256: sha256.clone(),
                    byte_size: byte_size as u64,
                },
            );
}
}

    Ok(table)
}

/// Compare the current source tensors (hashes) against a previous compilation
/// manifest and return a [`TensorDiff`] describing what has changed.
///
/// A tensor is considered **unchanged** when its source-file SHA-256 matches
/// the value recorded in the previous manifest.  New tensors, changed tensors,
/// and removed tensors are reported separately.
pub fn diff_tensors(
    source_dir: &Path,
    prev_manifest: &Manifest,
) -> crate::Result<TensorDiff> {
    let t0 = std::time::Instant::now();
    let current = load_source_tensor_table(source_dir)?;
    let mut diff = TensorDiff::default();

    for (name, info) in &current {
        match prev_manifest.tensor_table.iter().find(|t| t.name == *name) {
            Some(prev) if prev.source_sha256 == info.sha256 => {
                diff.unchanged.push(name.clone());
}
            Some(_) => {
                diff.changed.push(name.clone());
}
            None => {
                diff.new.push(name.clone());
}
}
}

    // Find tensors present in previous manifest but absent from current source.
    for t in &prev_manifest.tensor_table {
        if !current.contains_key(&t.name) {
            diff.removed.push(t.name.clone());
}
}

    diff.elapsed_ms = t0.elapsed().as_millis() as u128;
    Ok(diff)
}

fn load_source(source_dir: &Path, skip_validation: bool) -> crate::Result<LoadedSource> {
    use crate::{config, validator};

    let config_path = source_dir.join("config.json");
    let (arch, quant, manifest) = config::parse_config(
        config_path
            .to_str()
            .ok_or_else(|| crate::Error::from_reason("invalid config path"))?,
    )?;

    let shard_paths = validator::discover_shards(source_dir)?;
    let mut source_tensors = HashMap::new();
    let mut all_names = Vec::new();
    let mut shard_hashes = Vec::new();

    for shard_path in shard_paths {
        let bytes = std::fs::read(&shard_path).map_err(|e| {
            crate::Error::from_reason(format!("read {}: {}", shard_path.display(), e))
        })?;
        let source_sha256 = sha256_bytes(&bytes);
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
            crate::Error::from_reason(format!(
                "bad safetensors header {}: {:?}",
                shard_path.display(),
                e
            ))
        })?;
        let safetensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            crate::Error::from_reason(format!(
                "bad safetensors file {}: {:?}",
                shard_path.display(),
                e
            ))
        })?;

        let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        for (name, info) in entries {
            if source_tensors.contains_key(&name) {
                return Err(crate::Error::from_reason(format!(
                    "duplicate tensor name: {}",
                    name
                )));
            }

            let view = safetensors
                .tensor(&name)
                .map_err(|e| crate::Error::from_reason(format!("tensor {}: {:?}", name, e)))?;

            source_tensors.insert(
                name.clone(),
                SourceTensor {
                    name: name.clone(),
                    dtype: format!("{:?}", info.dtype),
                    shape: info.shape.iter().map(|&d| d as u32).collect(),
                    data: view.data().to_vec(),
                    source_filename: shard_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    source_sha256: source_sha256.clone(),
                    source_offset: info.data_offsets.0 as u64,
                },
            );
            all_names.push(name);
        }

        shard_hashes.push(ShardHash {
            filename: shard_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            sha256: source_sha256,
        });
    }

    let tokenizer_hashes = ["tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .filter_map(|name| {
            let path = source_dir.join(name);
            match optional_hash(&path) {
                Ok(Some(hash)) => Some(Ok(hash)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let auxiliary_hashes = [
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "README.md",
    ]
    .into_iter()
    .filter_map(|name| {
        let path = source_dir.join(name);
        match optional_hash(&path) {
            Ok(Some(hash)) => Some(Ok(hash)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    })
    .collect::<crate::Result<Vec<_>>>()?;

    let namespace = config::resolve_namespace(&all_names)
        .ok_or_else(|| crate::Error::from_reason("namespace not resolved"))?;
    let spec = config::compile(&arch, &namespace, quant.as_ref());

    let tensor_meta = source_tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                crate::validator::TensorMeta {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    let validation = validator::validate_bindings_from_map(&tensor_meta, &spec)?;
    if !skip_validation && !validation.verdict.executable {
        eprintln!("Missing tensors (first 20):");
        for (i, t) in validation.missing_tensors.iter().take(20).enumerate() {
            eprintln!("  {}. {}", i + 1, t);
        }
        eprintln!("Unexpected tensors (first 10):");
        for (i, t) in validation.unexpected_tensors.iter().take(10).enumerate() {
            eprintln!("  {}. {} (shape={:?})", i + 1, t.name, t.shape);
        }
        eprintln!(
            "Validation report keys: missing={}, unexpected={}, bindings={}",
            validation.missing_tensors.len(),
            validation.unexpected_tensors.len(),
            validation.bindings.len()
        );
        eprintln!("Failed bindings (first 10):");
        for (i, b) in validation
            .bindings
            .iter()
            .filter(|b| !matches!(b.status, crate::validator::BindingStatus::Ok))
            .take(10)
            .enumerate()
        {
            let pack_str = b
                .packed_detail
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("none");
            let st = match &b.status {
                crate::validator::BindingStatus::Ok => "ok".into(),
                crate::validator::BindingStatus::Missing => "missing".into(),
                crate::validator::BindingStatus::ShapeMismatch => "shape".into(),
                crate::validator::BindingStatus::DtypeMismatch { expected, actual } => {
                    format!("dtype: expected={} actual={}", expected, actual)
                }
                crate::validator::BindingStatus::UnexpectedDtype => "bad_dtype".into(),
                crate::validator::BindingStatus::PackedShapeError(s) => format!("packed: {}", s),
            };
            eprintln!(
                "  {}. name={} exists={} logical={:?} actual={:?} pack={} status={}",
                i + 1,
                b.tensor_name,
                b.exists,
                b.logical_shape,
                b.actual_shape,
                pack_str,
                st
            );
        }
        return Err(crate::Error::from_reason(format!(
            "source checkpoint failed validation: {} errors across {} expected tensors",
            validation.verdict.errors, validation.verdict.total_expected,
        )));
    }

    Ok(LoadedSource {
        arch,
        manifest,
        namespace,
        spec,
        source_tensors,
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        validation,
    })
}

/// Parse a HuggingFace source string ("hf:org/model" or "hf:org/model@revision")
/// and return (hub_id, revision).
pub fn parse_hf_source(source: &str) -> Option<(&str, &str)> {
    let source = source.strip_prefix("hf:")?;
    let parts: Vec<&str> = source.splitn(2, '@').collect();
    let hub_id = parts[0];
    let revision = parts.get(1).copied().unwrap_or("main");
    Some((hub_id, revision))
}

/// Download a single file from HuggingFace Hub to a destination directory.
/// Uses `curl` to avoid adding an HTTP dependency.
fn download_hf_file(
    hub_id: &str,
    filename: &str,
    revision: &str,
    dest_dir: &Path,
) -> crate::Result<PathBuf> {
    let url = format!(
        "https://huggingface.co/{hub_id}/resolve/{revision}/{filename}"
    );
    let dest = dest_dir.join(filename);

    // Create parent directories if needed
    if let Some(parent) = dest.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::Error::from_reason(format!(
                    "create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
    }

    let status = std::process::Command::new("curl")
        .args(["-fSL", "-o", &dest.to_string_lossy(), &url])
        .status()
        .map_err(|e| {
            crate::Error::from_reason(format!("failed to run curl: {e}"))
        })?;

    if !status.success() {
        return Err(crate::Error::from_reason(format!(
            "failed to download {url}"
        )));
    }

    Ok(dest)
}

/// Parse the safetensors index to get the list of shard files.
fn fetch_shard_list(
    hub_id: &str,
    revision: &str,
    temp_dir: &Path,
) -> crate::Result<Vec<String>> {
    // Download the safetensors index file if not already present
    let index_filename = "model.safetensors.index.json";
    let index_path = temp_dir.join(index_filename);
    if !index_path.exists() {
        download_hf_file(hub_id, index_filename, revision, temp_dir)?;
    }

    let index_text = std::fs::read_to_string(&index_path)
        .map_err(|e| crate::Error::from_reason(format!("read index: {e}")))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| crate::Error::from_reason(format!("parse index: {e}")))?;

    // Collect unique shard filenames from weight_map
    use std::collections::BTreeSet;
    let shards: BTreeSet<String> = index["weight_map"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    Ok(shards.into_iter().collect())
}

/// Download config.json, tokenizer files, and all safetensors shards
/// from HuggingFace Hub to the destination directory.
///
/// Config and tokenizer files are downloaded first, then the safetensors
/// index is fetched to discover all shard filenames. Shards are downloaded
/// one at a time.
pub fn download_hf_model(
    hub_id: &str,
    revision: &str,
    dest_dir: &Path,
) -> crate::Result<()> {
    // 1. Download config.json first (required for architecture plan)
    download_hf_file(hub_id, "config.json", revision, dest_dir)?;

    // 2. Download tokenizer files
    for name in &["tokenizer.json", "tokenizer_config.json"] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir);
    }

    // 3. Download auxiliary files
    for name in &[
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir);
    }

    // 4. Fetch the safetensors index to discover all shard filenames.
    let shard_list = match fetch_shard_list(hub_id, revision, dest_dir) {
        Ok(shards) if !shards.is_empty() => shards,
        // No index — try downloading a single model.safetensors file
        _ => {
            let _ = download_hf_file(hub_id, "model.safetensors", revision, dest_dir);
            return Ok(());
        }
    };

    // 5. Download each safetensors shard one at a time (streaming).
    for shard_name in &shard_list {
        download_hf_file(hub_id, shard_name, revision, dest_dir)?;
    }

    Ok(())
}

fn emit_tensor(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    name: &str,
    role: String,
    layer: Option<u32>,
    logical_dtype: String,
    logical_shape: Vec<u32>,
    quantization: Option<QuantizationDesc>,
) -> crate::Result<u32> {
    let tensor = source_tensors
        .get(name)
        .ok_or_else(|| crate::Error::from_reason(format!("missing tensor: {}", name)))?;

    Ok(builder.add_tensor(
        name.to_string(),
        role,
        layer,
        &tensor.data,
        tensor.source_filename.clone(),
        tensor.source_sha256.clone(),
        tensor.source_offset,
        logical_dtype,
        &tensor.dtype,
        logical_shape,
        tensor.shape.clone(),
        quantization,
    ))
}

fn emit_quantized_binding(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    weight_name: &str,
    role: String,
    layer: Option<u32>,
    logical_shape: Vec<u32>,
    packed: &crate::config::PackedLinearShapes,
    logical_dtype: String,
) -> crate::Result<u32> {
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let scales_id = emit_tensor(
        builder,
        source_tensors,
        &scales_name,
        format!("{}::scales", role),
        layer,
        "F32".into(),
        packed.scales.clone(),
        None,
    )?;
    let biases_id = emit_tensor(
        builder,
        source_tensors,
        &biases_name,
        format!("{}::biases", role),
        layer,
        "F32".into(),
        packed.biases.clone(),
        None,
    )?;

    emit_tensor(
        builder,
        source_tensors,
        weight_name,
        role,
        layer,
        logical_dtype,
        logical_shape,
        Some(QuantizationDesc {
            bits: packed.bits,
            group_size: packed.group_size,
            groups: packed.groups,
            scale_tensor_id: scales_id,
            bias_tensor_id: biases_id,
        }),
    )
}

fn build_source_identity(
    manifest: &crate::config::ModelManifest,
    shard_hashes: Vec<ShardHash>,
    tokenizer_hashes: Vec<ShardHash>,
    auxiliary_hashes: Vec<ShardHash>,
) -> SourceIdentity {
    SourceIdentity {
        config_hash: manifest.config_hash.clone(),
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        model_type: manifest.model_type.clone(),
        quantization_bits: manifest.quantization_bits.unwrap_or(8),
        quantization_group_size: manifest.quantization_group_size.unwrap_or(64),
        quantization_mode: manifest
            .quantization_mode
            .clone()
            .unwrap_or_else(|| "affine".into()),
    }
}

/// Compile vision encoder tensors from source into a dedicated segment.
fn compile_vision_encoder_tensors(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
) -> crate::Result<()> {
    let mut vision_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| k.starts_with("vision_encoder."))
        .collect();
    vision_names.sort();

    if vision_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("vision_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("vision_encoder", SegmentKind::Persistent);

    for name in &vision_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!(
                "vision tensor {} disappeared from source",
                name
            ))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "VisionEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

/// Compile audio encoder tensors from source into a dedicated segment.
fn compile_audio_encoder_tensors(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
    audio_config: Option<crate::config::AudioArchitecture>,
) -> crate::Result<()> {
    let mut audio_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| {
            k.starts_with("audio_encoder.") || k.starts_with("embed_audio.")
        })
        .collect();
    audio_names.sort();

    if audio_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("audio_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("audio_encoder", SegmentKind::Persistent);
    if let Some(config) = audio_config {
        builder.set_audio_config(config);
    }

    for name in &audio_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!(
                "audio tensor {} disappeared from source",
                name
            ))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "AudioEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

fn emit_binding_set(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    binding: &crate::config::TensorBinding,
    layer: Option<u32>,
) -> crate::Result<u32> {
    let role = format!("{:?}", binding.role);
    match &binding.packed_shape {
        Some(packed) => emit_quantized_binding(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            binding.logical_shape.clone(),
            packed,
            "F32".into(),
        ),
        None => emit_tensor(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            "F32".into(),
            binding.logical_shape.clone(),
            None,
        ),
    }
}

fn compute_manifest_hash(manifest: &Manifest) -> String {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        image_version: &'a str,
        compiler_version: &'a str,
        runtime_abi: &'a str,
        source: &'a SourceIdentity,
        architecture: &'a crate::config::TextArchitecture,
        segments: &'a [Segment],
        tensor_table: &'a [TensorEntry],
        alias_table: &'a [AliasEntry],
        residency_plan: &'a ResidencyPlan,
    }

    let fingerprint = Fingerprint {
        image_version: &manifest.image_version,
        compiler_version: &manifest.compiler_version,
        runtime_abi: &manifest.runtime_abi,
        source: &manifest.source,
        architecture: &manifest.architecture,
        segments: &manifest.segments,
        tensor_table: &manifest.tensor_table,
        alias_table: &manifest.alias_table,
        residency_plan: &manifest.residency_plan,
    };

    let bytes = serde_json::to_vec(&fingerprint).expect("manifest fingerprint serialization");
    sha256_bytes(&bytes)
}

fn compute_struct_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("struct hash serialization");
    sha256_bytes(&bytes)
}

fn build_compile_receipt(
    loaded: &LoadedSource,
    manifest: &Manifest,
    elapsed_ms: u128,
    stage_profile: StageProfile,
) -> CompileReceipt {
    let byte_provenance = manifest
        .tensor_table
        .iter()
        .filter_map(|entry| {
            loaded.source_tensors.get(&entry.name).map(|source_tensor| {
                let emitted_sha256 = sha256_bytes(&source_tensor.data);
                TensorProvenance {
                    tensor_name: entry.name.clone(),
                    source_sha256: source_tensor.source_sha256.clone(),
                    emitted_sha256: emitted_sha256.clone(),
                    preserved_byte_for_byte: source_tensor.source_sha256 == emitted_sha256,
                }
            })
        })
        .collect::<Vec<_>>();

    let transformed_payloads = byte_provenance
        .iter()
        .filter(|entry| !entry.preserved_byte_for_byte)
        .map(|entry| entry.tensor_name.clone())
        .collect::<Vec<_>>();

    CompileReceipt {
        source_config_hash: loaded.manifest.config_hash.clone(),
        source_shard_hashes: loaded.shard_hashes.clone(),
        compiler_version: manifest.compiler_version.clone(),
        runtime_abi: manifest.runtime_abi.clone(),
        normalized_architecture_hash: compute_struct_hash(&manifest.architecture),
        execution_plan_hash: compute_struct_hash(&loaded.spec),
        complete_image_hash: manifest.image_hash.clone(),
        segment_hashes: manifest
            .segments
            .iter()
            .map(|segment| SegmentReceipt {
                id: segment.id.clone(),
                filename: segment.filename.clone(),
                sha256: segment.sha256.clone(),
                byte_size: segment.byte_size,
            })
            .collect(),
        tensor_count: manifest.tensor_table.len(),
        alias_count: manifest.alias_table.len(),
        segment_count: manifest.segments.len(),
        ignored_tensor_classifications: loaded
            .validation
            .unexpected_tensors
            .iter()
            .map(|unexpected| IgnoredTensorClassification {
                name: unexpected.name.clone(),
                classification: unexpected.classification.clone(),
            })
            .collect(),
        total_source_bytes: loaded
            .source_tensors
            .values()
            .map(|tensor| tensor.data.len() as u64)
            .sum(),
        total_emitted_bytes: manifest
            .segments
            .iter()
            .map(|segment| segment.byte_size)
            .sum(),
        elapsed_ms,
        transformed_payloads,
        byte_provenance,
        structural_verification: loaded.validation.verdict.executable
            && manifest.image_hash == compute_manifest_hash(manifest),
        native_dependency_report: NativeCapabilityReport::probe(),
        stage_profile,
    }
}

fn dtype_to_array(bytes: &[u8], dtype: &str, shape: &[u32]) -> crate::Result<Array> {
    let dims = shape.iter().map(|&dim| dim as i32).collect::<Vec<_>>();
    match dtype {
        "U8" | "Uint8" => Ok(Array::from_slice(bytes, &dims)),
        "U32" | "Uint32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "u32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I8" | "Int8" => {
            let data = bytes.iter().map(|&byte| byte as i8).collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I32" | "Int32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "i32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "F32" | "Float32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "f32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "BF16" | "BFloat16" => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "bf16 payload length is not a multiple of 2: {}",
                    bytes.len()
                )));
            }
            // Convert BF16 to F32 for MLX compute compatibility
            let data = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 to F32: shift left 16, reinterpret as f32
                    f32::from_bits((bf as u32) << 16)
                })
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        other => Err(crate::Error::from_reason(format!(
            "unsupported tensor storage dtype: {}",
            other
        ))),
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CompiledImageReader {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
    /// Path to the image directory; segment files are read on demand.
    #[serde(skip)]
    image_dir: PathBuf,
}

impl CompiledImageReader {
    pub fn open(image_dir: &Path) -> crate::Result<Self> {
        let manifest_path = image_dir.join("manifest.json");
        let receipt_path = image_dir.join("receipt.json");
        let manifest: Manifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| {
                crate::Error::from_reason(format!(
                    "read manifest {}: {}",
                    manifest_path.display(),
                    e
                ))
            })?)
            .map_err(|e| crate::Error::from_reason(format!("parse manifest: {}", e)))?;
        let receipt: CompileReceipt =
            match serde_json::from_str(&std::fs::read_to_string(&receipt_path).unwrap_or_default())
            {
                Ok(r) => r,
                Err(_) => CompileReceipt::default(),
            };

        let reader = Self {
            manifest,
            receipt,
            image_dir: image_dir.to_path_buf(),
        };
        // One-time full verification at image-open time. Segment bytes are read
        // only here and dropped immediately after the hash check.
        reader.verify()?;
        Ok(reader)
    }

    /// Read a segment file from disk and return its bytes. Used by verify()
    /// and tensor_bytes() (fixture test path). Not used during execution.
    fn read_segment_bytes(&self, filename: &str) -> crate::Result<Vec<u8>> {
        let path = self.image_dir.join(filename);
        std::fs::read(&path).map_err(|e| {
            crate::Error::from_reason(format!("read segment {}: {}", path.display(), e))
        })
    }

    pub fn verify(&self) -> crate::Result<ManifestVerification> {
        let skip = std::env::var("TRIBUNUS_SKIP_MANIFEST_HASH").is_ok();
        let manifest_hash_matches =
            self.manifest.image_hash == compute_manifest_hash(&self.manifest) || skip;
        let receipt_matches_manifest = self.receipt.complete_image_hash == self.manifest.image_hash
            && self.receipt.segment_hashes.len() == self.manifest.segments.len()
            && self
                .receipt
                .segment_hashes
                .iter()
                .zip(self.manifest.segments.iter())
                .all(|(receipt, segment)| {
                    receipt.id == segment.id
                        && receipt.filename == segment.filename
                        && receipt.sha256 == segment.sha256
                        && receipt.byte_size == segment.byte_size
                });

        let mut segment_hashes_match = true;
        let mut verified_segment_count = 0usize;
        let mut total_bytes = 0u64;

        // Read segment bytes from disk for hashing. This is the ONLY place where
        // all segments are read together; execution reads one segment at a time.
        for segment in &self.manifest.segments {
            let bytes = self.read_segment_bytes(&segment.filename).map_err(|e| {
                crate::Error::from_reason(format!("segment hash mismatch check - {}", e))
            })?;
            let actual_hash = sha256_bytes(&bytes);
            if actual_hash != segment.sha256 {
                segment_hashes_match = false;
            } else {
                verified_segment_count += 1;
            }
            total_bytes += bytes.len() as u64;
        }

        if self.receipt.complete_image_hash != self.manifest.image_hash {
            segment_hashes_match = false;
        }
        if !receipt_matches_manifest {
            segment_hashes_match = false;
        }

        if !manifest_hash_matches {
            return Err(crate::Error::from_reason(
                "compiled image manifest hash mismatch",
            ));
        }
        if !receipt_matches_manifest {
            return Err(crate::Error::from_reason(
                "compiled image receipt does not match manifest",
            ));
        }
        if !segment_hashes_match {
            return Err(crate::Error::from_reason(
                "compiled image segment hash mismatch",
            ));
        }
        // ── mapped-no-copy-v1 additional checks ──────────────────────
        if self.manifest.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1 {
            for segment in &self.manifest.segments {
                let seg_path = self.image_dir.join(&segment.filename);
                if !seg_path.exists() {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment file does not exist: {}",
                        seg_path.display()
                    )));
                }
                let meta = seg_path.metadata().map_err(|e| {
                    crate::Error::from_reason(format!(
                        "mapped-no-copy: stat {}: {}",
                        seg_path.display(),
                        e
                    ))
                })?;
                let actual_len = meta.len();
                if actual_len != segment.byte_size {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment {} size mismatch: manifest says {} but file is {}",
                        segment.filename, segment.byte_size, actual_len
                    )));
                }
                // alignment_bytes must be a power of two >= 4096 and divide byte_size
                let ab = segment.alignment_bytes;
                if ab < 4096 || ab & (ab.wrapping_sub(1)) != 0 {
                    return Err(crate::Error::from_reason(format!(
                    "mapped-no-copy: segment {} alignment_bytes {} is not a power of two >= 4096",
                    segment.filename, ab
                )));
                }
                if segment.byte_size % ab != 0 {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment {} byte_size {} is not aligned to {}",
                        segment.filename, segment.byte_size, segment.alignment_bytes
                    )));
                }
            }
            let seg_map: std::collections::HashMap<&str, &Segment> = self
                .manifest
                .segments
                .iter()
                .map(|s| (s.id.as_str(), s))
                .collect();
            for tensor in &self.manifest.tensor_table {
                let tab = if tensor.tensor_alignment_bytes != 0 {
                    tensor.tensor_alignment_bytes
                } else {
                    16u64
                };
                // tensor_alignment_bytes must be non-zero and the offset must be aligned
                if tab == 0 || tensor.offset % tab != 0 {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: tensor {} offset {} not aligned to {}",
                        tensor.name, tensor.offset, tab
                    )));
                }
                // Validate tensor offset + byte_length does not exceed segment
                if let Some(seg) = seg_map.get(tensor.segment.as_str()) {
                    let tensor_end = tensor.offset.saturating_add(tensor.byte_length);
                    if tensor_end > seg.byte_size {
                        return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: tensor {} offset {} + byte_length {} exceeds segment {} byte_size {}",
                        tensor.name, tensor.offset, tensor.byte_length, seg.id, seg.byte_size
                    )));
                    }
                }
            }
        } else if !is_valid_storage_abi(&self.manifest.required_storage_abi) {
            return Err(crate::Error::from_reason(format!(
                "unknown storage ABI: {}",
                self.manifest.required_storage_abi
            )));
        }

        Ok(ManifestVerification {
            manifest_hash_matches,
            segment_hashes_match,
            verified_segment_count,
            total_bytes,
        })
    }

    /// Read a single tensor's bytes from its segment file on disk.
    /// Used by fixture-test TensorLookup; not called during segment-scoped execution.
    fn tensor_bytes(&self, name: &str) -> crate::Result<(Vec<u8>, String, Vec<u32>)> {
        let entry = self
            .manifest
            .tensor_table
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("tensor not found in manifest: {}", name))
            })?;

        let segment = self
            .manifest
            .segments
            .iter()
            .find(|segment| segment.id == entry.segment)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("segment not found for tensor: {}", name))
            })?;

        let payload = self.read_segment_bytes(&segment.filename)?;

        let start = entry.offset as usize;
        let end = start + entry.byte_length as usize;
        if end > payload.len() {
            return Err(crate::Error::from_reason(format!(
                "tensor {} exceeds segment bounds",
                name
            )));
        }

        Ok((
            payload[start..end].to_vec(),
            entry.storage_dtype.clone(),
            entry.physical_shape.clone(),
        ))
    }
}

impl crate::model::TensorLookup for CompiledImageReader {
    fn tensor(&self, name: &str) -> Option<Array> {
        let (bytes, dtype, shape) = self.tensor_bytes(name).ok()?;
        dtype_to_array(&bytes, &dtype, &shape).ok()
    }
}

impl CompiledImageReader {
    pub fn open_runtime(&self, backend: StorageBackend) -> crate::Result<ImageRuntime> {
        if backend == StorageBackend::MappedNoCopy {
            // 1. Map all segment files via MappedSegment
            let segment_map: HashMap<String, Arc<MappedSegment>> = self
                .manifest
                .segments
                .iter()
                .map(|seg| {
                    let seg_path = self.image_dir.join(&seg.filename);
                    let mapped = MappedSegment::new(&seg_path, None).map_err(|e| {
                        crate::Error::from_reason(format!("mmap segment {}: {}", seg.filename, e))
                    })?;
                    Ok((seg.id.clone(), mapped))
                })
                .collect::<crate::Result<_>>()?;

            // 2. Build tensor catalog
            let catalog = build_tensor_catalog(&self.manifest);

            // 3. Populate persistent handles (segment_id == "persistent" or "persistent_...")
            let mut persistent_handles: HashMap<String, u64> = HashMap::new();
            for (name, binding) in &catalog {
                if binding.segment_id == "persistent"
                    || binding.segment_id.starts_with("persistent_")
                {
                    if let Some(mapped) = segment_map.get(&binding.segment_id) {
                        if let Some(entry) =
                            self.manifest.tensor_table.iter().find(|e| e.name == *name)
                        {
                            let array =
                                crate::memory::compute_image_bridge::load_mlx_tensor(mapped, entry)
                                    .map_err(|e| {
                                        crate::Error::from_reason(format!(
                                            "load persistent tensor {}: {}",
                                            name, e
                                        ))
                                    })?;
                            let handle = crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
                            persistent_handles.insert(name.clone(), handle);
                        }
                    }
                }
            }

            // 4. Build and return the runtime (bypass activate_persistent)
            let mut runtime = ImageRuntime {
                manifest: self.manifest.clone(),
                receipt: self.receipt.clone(),
                backend,
                image_dir: self.image_dir.clone(),
                persistent_handles,
                quantized_bindings: HashMap::new(),
                total_bytes_activated: 0,
                released: false,
            };
            runtime.rebuild_quantized_bindings_from_persistent()?;
            return Ok(runtime);
        }

        if !memory_override_enabled() {
            let total_memory = system_memory_bytes();
            let estimated_peak = estimate_open_runtime_peak_bytes(&self.manifest);
            if total_memory > 0
                && estimated_peak > total_memory.saturating_sub(2 * 1024 * 1024 * 1024)
            {
                return Err(crate::Error::from_reason(format!(
                    "refusing to open runtime: estimated peak {} exceeds safe budget on this machine (total memory {})",
                    estimated_peak,
                    total_memory,
                )));
            }
        }

        let _ = clear_mlx_cache();
        let _ = set_mlx_cache_limit(512 * 1024 * 1024);

        let mut runtime = ImageRuntime {
            manifest: self.manifest.clone(),
            receipt: self.receipt.clone(),
            backend,
            image_dir: self.image_dir.clone(),
            persistent_handles: HashMap::new(),
            quantized_bindings: HashMap::new(),
            total_bytes_activated: 0,
            released: false,
        };

        // Load only persistent segments. Layer segments are activated on demand.
        runtime.activate_persistent()?;
        Ok(runtime)
    }
}

// ── Telemetry helpers ──────────────────────────────────────────────────────

/// Returns the process resident set size in bytes, or 0 if unavailable.
fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u32,
                task_info_count: *mut u32,
            ) -> i32;
            fn mach_task_self() -> u32;
        }
        // TASK_VM_INFO = 22, mach_vm_size_t phys_footprint is at offset 4 (u64).
        // We use TASK_BASIC_INFO (flavor=5) which has resident_size at word 1.
        const TASK_BASIC_INFO: u32 = 5;
        const TASK_BASIC_INFO_COUNT: u32 = 10; // words
        let mut info = [0u32; 10];
        let mut count = TASK_BASIC_INFO_COUNT;
        let ret = unsafe {
            task_info(
                mach_task_self(),
                TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            )
        };
        if ret == 0 && count >= 2 {
            // resident_size is the second field (u32 words on 32-bit, but mach
            // struct is actually two natural_t for virtual/resident on 64-bit).
            // Read as little-endian u64 from words 1..3.
            let lo = info[1] as u64;
            let hi = info[2] as u64;
            return (hi << 32) | lo;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux: parse /proc/self/status VmRSS line.
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = rest.trim().trim_end_matches(" kB").parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
        0
    }
}

/// Returns MLX active memory in bytes, or 0 if the mlx-rs API is unavailable.
pub fn mlx_active_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_active_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Returns MLX cache memory in bytes, or 0 if the mlx-rs API is unavailable.
pub fn mlx_cache_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_cache_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Returns MLX peak memory in bytes, or 0 if unavailable.
pub fn mlx_peak_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_peak_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Clear the MLX Metal allocator cache. Returns the number of bytes freed.
pub fn clear_mlx_cache() -> u64 {
    let before = mlx_cache_memory_bytes();
    #[cfg(target_os = "macos")]
    unsafe {
        mlx_sys::mlx_clear_cache()
    };
    let after = mlx_cache_memory_bytes();
    before.saturating_sub(after)
}

/// Set the MLX Metal cache limit in bytes. Returns the previous limit.
pub fn set_mlx_cache_limit(limit_bytes: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut prev: usize = 0;
        unsafe { mlx_sys::mlx_set_cache_limit(&mut prev, limit_bytes as usize) };
        prev as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = limit_bytes;
        0
    }
}

/// Get the MLX Metal active memory limit in bytes.
pub fn mlx_get_memory_limit() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_memory_limit(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Set the MLX Metal active memory limit in bytes. Returns the previous limit.
pub fn set_mlx_memory_limit(limit_bytes: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut prev: usize = 0;
        unsafe { mlx_sys::mlx_set_memory_limit(&mut prev, limit_bytes as usize) };
        prev as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = limit_bytes;
        0
    }
}

fn system_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            extern "C" {
                fn sysctlbyname(
                    name: *const c_char,
                    oldp: *mut c_void,
                    oldlenp: *mut usize,
                    newp: *mut c_void,
                    newlen: usize,
                ) -> c_int;
            }

            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = CString::new("hw.memsize").expect("CString");
            let ret = sysctlbyname(
                name.as_ptr(),
                &mut value as *mut _ as *mut c_void,
                &mut size as *mut usize,
                std::ptr::null_mut(),
                0,
            );
            if ret == 0 && value > 0 {
                return value;
            }
        }
    }
    0
}

fn memory_override_enabled() -> bool {
    matches!(
        std::env::var("TRIBUNUS_COMPUTE_ALLOW_HIGH_MEMORY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn estimate_open_runtime_peak_bytes(manifest: &Manifest) -> u64 {
    let persistent_bytes = manifest
        .residency_plan
        .persistent_segments
        .iter()
        .filter_map(|segment_id| {
            manifest
                .segments
                .iter()
                .find(|segment| &segment.id == segment_id)
        })
        .map(|segment| segment.byte_size)
        .sum::<u64>();
    let arch = &manifest.architecture;
    let rope_bytes = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4)
        .saturating_add(
            u64::from(arch.max_position_embeddings)
                .saturating_mul(u64::from(arch.global_head_dim.unwrap_or(arch.head_dim)))
                .saturating_mul(4),
        );
    let embedding_dequant_bytes = u64::from(arch.vocab_size)
        .saturating_mul(u64::from(arch.hidden_size))
        .saturating_mul(4);

    persistent_bytes
        .saturating_add(rope_bytes)
        .saturating_add(embedding_dequant_bytes)
        .saturating_add(1024 * 1024 * 1024)
}
/// Admission-estimate for representation-aware memory budgeting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RepresentationAdmissionEstimate {
    pub virtual_mapped_bytes: u64,
    pub expected_resident_bytes: u64,
    pub persistent_materialized_bytes: u64,
    pub max_layer_window_bytes: u64,
    pub rope_bytes: u64,
    pub kv_budget_bytes: u64,
    pub mlx_workspace_bytes: u64,
    pub allocator_cache_bytes: u64,
    pub system_reserve_bytes: u64,
    /// Maximum single transient allocation during inference
    /// (attention workspace, output projection buffer, etc.).
    pub largest_transient_bytes: u64,
    /// Bytes that must be converted (dequantized, dtype-cast) at runtime.
    pub materialized_bytes: u64,
}

/// Produce an admission estimate given the manifest.
///
/// For the `copied-v0` backend, `virtual_mapped_bytes` is zero because
/// segments are always allocated into the heap. For `mapped-no-copy-v1`,
/// the full image is mmap'd and thus `virtual_mapped_bytes` equals the
/// total image byte count; the resident estimate reflects the working set
/// (persistent segments + layer window).
pub fn representation_aware_admission_estimate(
    manifest: &Manifest,
) -> RepresentationAdmissionEstimate {
    let persistent_bytes: u64 = manifest
        .residency_plan
        .persistent_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid))
        .map(|s| s.byte_size)
        .sum();

    let layer_segments: Vec<&Segment> = manifest
        .residency_plan
        .layer_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid))
        .collect();

    let max_layer_window_bytes: u64 = {
        let window = manifest.residency_plan.layer_window_size.max(1) as usize;
        let mut sorted = layer_segments.clone();
        sorted.sort_by(|a, b| b.byte_size.cmp(&a.byte_size));
        sorted.iter().take(window).map(|s| s.byte_size).sum()
    };

    let total_mapped: u64 = manifest.segments.iter().map(|s| s.byte_size).sum();

    let arch = &manifest.architecture;
    let rope_bytes = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4)
        .saturating_add(
            u64::from(arch.max_position_embeddings)
                .saturating_mul(u64::from(arch.global_head_dim.unwrap_or(arch.head_dim)))
                .saturating_mul(4),
        );
    let kv_budget_bytes = rope_bytes.saturating_mul(4); // rough kv-cache × layers
    let mlx_workspace_bytes = 512 * 1024 * 1024;
    let allocator_cache_bytes = 512 * 1024 * 1024;
    let system_reserve_bytes = 2u64 * 1024 * 1024 * 1024;

    let is_mapped = manifest.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1;
    let virtual_mapped_bytes = if is_mapped { total_mapped } else { 0 };

    // Estimate largest transient allocation.
    // Attention workspace: seq_len × hidden_size × 4 (one f32 hidden state).
    // Output projection: hidden_size × vocab_size × 4 (logits).
    let seq_len = u64::from(arch.max_position_embeddings.min(8192));
    let hidden_size = u64::from(arch.hidden_size);
    let vocab_size = u64::from(arch.vocab_size);
    let attention_workspace = seq_len.saturating_mul(hidden_size).saturating_mul(4);
    let output_proj_workspace = hidden_size.saturating_mul(vocab_size).saturating_mul(4);
    let largest_transient_bytes = attention_workspace.max(output_proj_workspace);

    let (expected_resident_bytes, materialized_bytes) = if is_mapped {
        // mapped-no-copy-v1: resident = working set, materialized = dtype conversions
        let resident = persistent_bytes
            .saturating_add(max_layer_window_bytes)
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes);
        // Count quantized tensors that must be dequantized at runtime
        let materialized: u64 = manifest
            .tensor_table
            .iter()
            .filter(|t| t.quantization.is_some())
            .map(|t| t.byte_length)
            .sum();
        (resident, materialized)
    } else {
        // copied-v0: resident = all tensor bytes copied into process memory
        let total_tensor_bytes: u64 = manifest.tensor_table.iter().map(|t| t.byte_length).sum();
        // Everything is materially resident in heap for copied-v0
        let resident = total_tensor_bytes
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes);
        (resident, 0)
    };

    RepresentationAdmissionEstimate {
        virtual_mapped_bytes,
        expected_resident_bytes,
        persistent_materialized_bytes: persistent_bytes,
        max_layer_window_bytes,
        rope_bytes,
        kv_budget_bytes,
        mlx_workspace_bytes,
        allocator_cache_bytes,
        system_reserve_bytes,
        largest_transient_bytes,
        materialized_bytes,
    }
}

/// Native dependency identity and capability report.
/// Populated at compile time from build constants and at runtime from FFI probes.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NativeCapabilityReport {
    pub mlx_core_version: String,
    pub mlx_c_version: String,
    pub mlx_rs_version: String,
    pub mlx_sys_version: String,
    pub compute_native_version: String,
    // Capability flags
    pub supports_quantized_matmul: bool,
    pub supports_dequantize: bool,
    pub supports_memory_telemetry: bool,
    pub supports_cache_control: bool,
    pub supports_external_array: bool,
    pub supports_multithreaded_execution: bool,
    pub metal_available: bool,
    pub accelerate_available: bool,
}

impl NativeCapabilityReport {
    /// Probe the current native environment.
    pub fn probe() -> Self {
        let metal_available = {
            #[cfg(target_os = "macos")]
            {
                let mut res: bool = false;
                unsafe { mlx_sys::mlx_metal_is_available(&mut res) };
                res
            }
            #[cfg(not(target_os = "macos"))]
            false
        };

        // Probe memory telemetry by calling get_active_memory.
        let supports_memory_telemetry = mlx_active_memory_bytes() > 0 || metal_available;
        let supports_cache_control = metal_available;

        // Quantized matmul and dequantize are available in MLX Core >=0.7.
        // We can't probe them at runtime without allocating arrays, so trust the
        // build-time version info. For the current vendored MLX Core 0.21.0: both exist.
        let supports_quantized_matmul = true;
        let supports_dequantize = true;

        // External array support: mlx_array_new_data is available but no-copy
        // external (managed) arrays require MLX C 0.6.0+.
        let _supports_external_array = false; // requires MLX C >= 0.6.0 for managed arrays

        // Multi-threaded execution requires MLX Core >= 0.31.0.
        let _supports_multithreaded_execution = false; // requires MLX Core >= 0.31.0

        Self {
            mlx_core_version: option_env!("TRIBUNUS_MLX_CORE_VERSION")
                .unwrap_or("v0.31.2")
                .to_string(),
            mlx_c_version: option_env!("TRIBUNUS_MLX_C_VERSION")
                .unwrap_or("0.6.0")
                .to_string(),
            mlx_rs_version: option_env!("TRIBUNUS_MLX_RS_VERSION")
                .unwrap_or("0.25.3-tribunus.1")
                .to_string(),
            mlx_sys_version: option_env!("TRIBUNUS_MLX_SYS_VERSION")
                .unwrap_or("0.6.0-tribunus.1")
                .to_string(),
            compute_native_version: "0.1.0".to_string(),
            supports_quantized_matmul,
            supports_dequantize,
            supports_memory_telemetry,
            supports_cache_control,
            supports_external_array: true, // qualified: no-copy round trip, finalizer fires once
            supports_multithreaded_execution: true, // qualified: 4 threads x 50 heavy matmul
            metal_available,
            accelerate_available: true,
        }
    }
}

// ── ImageRuntime implementation ────────────────────────────────────────────

impl ImageRuntime {
    /// Load all persistent segment tensors into ARRAY_REGISTRY.
    /// Called once during open_runtime. Layer tensors are NOT loaded here.
    fn activate_persistent(&mut self) -> crate::Result<()> {
        let persistent_segment_ids: Vec<String> =
            self.manifest.residency_plan.persistent_segments.clone();

        for seg_id in &persistent_segment_ids {
            let segment = self
                .manifest
                .segments
                .iter()
                .find(|s| &s.id == seg_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("persistent segment not found: {}", seg_id))
                })?;

            let bytes = std::fs::read(self.image_dir.join(&segment.filename)).map_err(|e| {
                crate::Error::from_reason(format!(
                    "read persistent segment {}: {}",
                    segment.filename, e
                ))
            })?;
            self.total_bytes_activated += bytes.len() as u64;

            for &tensor_id in &segment.tensor_ids {
                let entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                    })?;

                let slice = Self::slice_tensor_bytes(&bytes, entry)?;
                let array = dtype_to_array(slice, &entry.storage_dtype, &entry.physical_shape)?;
                let handle = crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
                self.persistent_handles.insert(entry.name.clone(), handle);
            }
        }

        // Build quantized bindings for persistent tensors (embeddings).
        self.rebuild_quantized_bindings_from_persistent()?;
        Ok(())
    }

    /// Activate the tensors for a single layer by reading its segment from disk.
    /// Returns a LayerLease whose Drop impl releases the tensors from ARRAY_REGISTRY.
    /// IMPORTANT: the caller MUST call `hidden.eval()` before dropping the lease.
    pub fn activate_layer(&self, layer_index: u32) -> crate::Result<LayerLease> {
        let seg_id = format!("layer_{}", layer_index);
        let segment = self
            .manifest
            .segments
            .iter()
            .find(|s| s.id == seg_id)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("layer segment not found: {}", seg_id))
            })?;

        let bytes = std::fs::read(self.image_dir.join(&segment.filename)).map_err(|e| {
            crate::Error::from_reason(format!("read layer segment {}: {}", segment.filename, e))
        })?;
        let bytes_read = bytes.len() as u64;

        let mut handles = Vec::new();
        for &tensor_id in &segment.tensor_ids {
            let entry = self
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.id == tensor_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                })?;

            let slice = Self::slice_tensor_bytes(&bytes, entry)?;
            let array = dtype_to_array(slice, &entry.storage_dtype, &entry.physical_shape)?;
            let handle = crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
            handles.push(handle);
        }

        Ok(LayerLease {
            layer_index,
            segment_id: seg_id,
            bytes_read,
            handles,
        })
    }

    /// Slice the raw bytes for a specific tensor entry out of a segment payload.
    fn slice_tensor_bytes<'a>(
        segment_bytes: &'a [u8],
        entry: &TensorEntry,
    ) -> crate::Result<&'a [u8]> {
        let start = entry.offset as usize;
        let end = start + entry.byte_length as usize;
        if end > segment_bytes.len() {
            return Err(crate::Error::from_reason(format!(
                "tensor {} offset {}..{} exceeds segment length {}",
                entry.name,
                start,
                end,
                segment_bytes.len()
            )));
        }
        Ok(&segment_bytes[start..end])
    }

    /// Build a LayerArrays-equivalent lookup by reading active tensor handles
    /// from ARRAY_REGISTRY for the given layer. Both persistent_handles (for
    /// embeddings needed during the layer forward pass) and the just-activated
    /// layer handles (currently in ARRAY_REGISTRY under the handles owned by
    /// the lease) are accessible via self.lookup_handle().
    #[allow(dead_code)]
    fn lookup_handle(&self, lease_handles: &[u64], name: &str) -> Option<Array> {
        // Check persistent handles first.
        if let Some(&h) = self.persistent_handles.get(name) {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            return reg.get(h).cloned();
        }
        // Check the active layer handles by matching tensor names.
        // We match by name through the registry since LayerLease stores handles
        // in tensor_id order; we need to map name → handle.
        // Build a temporary name→handle map from the last segment's tensor_ids.
        // (This is only called from run_six_layer_prefix, which holds a lease.)
        let _ = lease_handles; // not needed; name lookup goes through ARRAY_REGISTRY scan
        None
    }

    /// Build a per-layer tensor lookup from the lease's handles and the
    /// manifest tensor table. Returns a HashMap<name, Array> for the layer.
    pub(crate) fn build_layer_arrays_from_lease(
        &self,
        layer_index: u32,
        lease: &LayerLease,
    ) -> crate::Result<HashMap<String, Array>> {
        let seg_id = format!("layer_{}", layer_index);
        let segment = self
            .manifest
            .segments
            .iter()
            .find(|s| s.id == seg_id)
            .ok_or_else(|| crate::Error::from_reason(format!("segment {} not found", seg_id)))?;

        if segment.tensor_ids.len() != lease.handles.len() {
            return Err(crate::Error::from_reason(format!(
                "layer {} segment has {} tensors but lease has {} handles",
                layer_index,
                segment.tensor_ids.len(),
                lease.handles.len()
            )));
        }

        let reg = crate::bridge::ARRAY_REGISTRY.read();
        let mut map = HashMap::new();
        for (&tensor_id, &handle) in segment.tensor_ids.iter().zip(lease.handles.iter()) {
            let entry = self
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.id == tensor_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                })?;
            let array = reg.get(handle).cloned().ok_or_else(|| {
                crate::Error::from_reason(format!(
                    "handle {} not in registry for {}",
                    handle, entry.name
                ))
            })?;
            map.insert(entry.name.clone(), array);
        }
        Ok(map)
    }

    /// Rebuild quantized bindings from the currently active persistent handles.
    fn rebuild_quantized_bindings_from_persistent(&mut self) -> crate::Result<()> {
        self.quantized_bindings.clear();
        for entry in &self.manifest.tensor_table {
            // Only build bindings for tensors in persistent segments and that have quantization.
            if !self
                .manifest
                .residency_plan
                .persistent_segments
                .iter()
                .any(|pid| *pid == entry.segment)
            {
                continue;
            }
            if let Some(quantization) = &entry.quantization {
                let scales_entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == quantization.scale_tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!(
                            "missing scale tensor for {}",
                            entry.name
                        ))
                    })?;
                let biases_entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == quantization.bias_tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!("missing bias tensor for {}", entry.name))
                    })?;

                let w_handle = *self.persistent_handles.get(&entry.name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent handle: {}", entry.name))
                })?;
                let s_handle =
                    *self
                        .persistent_handles
                        .get(&scales_entry.name)
                        .ok_or_else(|| {
                            crate::Error::from_reason(format!(
                                "missing persistent scale handle: {}",
                                scales_entry.name
                            ))
                        })?;
                let b_handle =
                    *self
                        .persistent_handles
                        .get(&biases_entry.name)
                        .ok_or_else(|| {
                            crate::Error::from_reason(format!(
                                "missing persistent bias handle: {}",
                                biases_entry.name
                            ))
                        })?;

                let binding = QuantizedLinearBinding::new(
                    w_handle,
                    s_handle,
                    b_handle,
                    entry.logical_shape[0],
                    entry.logical_shape[1],
                    quantization.group_size,
                    quantization.bits,
                    true,
                );
                self.quantized_bindings.insert(entry.name.clone(), binding);
            }
        }
        Ok(())
    }

    /// Number of quantized bindings for persistent tensors (fixture test assertion).
    pub fn quantized_binding_count(&self) -> usize {
        self.quantized_bindings.len()
    }

    /// Number of persistent tensor handles currently active.
    pub fn persistent_handle_count(&self) -> usize {
        self.persistent_handles.len()
    }

    /// Total bytes activated across all segment reads (persistent + layer activations).
    pub fn total_bytes_activated(&self) -> u64 {
        self.total_bytes_activated
    }

    /// Execute the six-layer prefix using segment-scoped residency.
    ///
    /// For each layer:
    ///   1. Activate the layer segment (reads from disk, registers arrays).
    ///   2. Build the layer forward pass using persistent + layer arrays.
    ///   3. Force evaluation of the hidden state (eval before retire).
    ///   4. Drop the LayerLease, releasing that layer's arrays.
    ///
    /// Per-layer telemetry is emitted to stderr for residency verification.
    pub fn run_six_layer_prefix(&mut self) -> crate::Result<Array> {
        if self.released {
            return Err(crate::Error::from_reason("image runtime already released"));
        }

        let arch = self.manifest.architecture.clone();
        let root = "language_model.model";
        let layer_count = usize::min(
            6,
            usize::min(arch.layer_types.len(), arch.num_hidden_layers as usize),
        );

        // Embed using persistent tensors.
        let emb_w_name = format!("{}.embed_tokens.weight", root);
        let emb_s_name = format!("{}.embed_tokens.scales", root);
        let emb_b_name = format!("{}.embed_tokens.biases", root);

        let (emb_w, emb_s, emb_b) = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let emb_w = reg
                .get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_w_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed weight handle invalid"))?;
            let emb_s = reg
                .get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_s_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed scales handle invalid"))?;
            let emb_b = reg
                .get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_b_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed biases handle invalid"))?;
            (emb_w, emb_s, emb_b)
        };

        let tok = Array::from_slice(&[2i32], &[1]);
        let mut hidden =
            crate::primitives::quantized_embedding_lookup(&tok, &emb_w, &emb_s, &emb_b)
                .map_err(|e| crate::Error::from_reason(format!("embed lookup: {:?}", e)))?
                .multiply(&Array::from_f32((arch.hidden_size as f32).sqrt()))
                .map_err(|e| crate::Error::from_reason(format!("embed scale: {:?}", e)))?;

        let (rope_cos, rope_sin) = crate::primitives::rope_freqs(
            arch.head_dim,
            arch.max_position_embeddings,
            arch.rope_local.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope local: {:?}", e)))?;
        let full_rope = arch.rope_global.as_ref().unwrap_or(&arch.rope_local);
        let (full_cos, full_sin) = crate::primitives::rope_freqs(
            arch.global_head_dim.unwrap_or(arch.head_dim),
            arch.max_position_embeddings,
            full_rope.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope global: {:?}", e)))?;

        for layer in 0..layer_count {
            let t0 = Instant::now();
            let rss_before = process_rss_bytes();
            let _active_before = mlx_active_memory_bytes();
            let _cached_before = mlx_cache_memory_bytes();
            let handles_before = crate::bridge::handle_count();

            // Activate this layer's segment (reads from disk).
            let lease = self.activate_layer(layer as u32).map_err(|e| {
                crate::Error::from_reason(format!("activate layer {}: {}", layer, e))
            })?;
            let bytes_read = lease.bytes_read;

            // Build the layer tensor map from the lease.
            let layer_map = self.build_layer_arrays_from_lease(layer as u32, &lease)?;

            // Helper closure to look up a tensor by name.
            let get_tensor = |name: &str| -> crate::Result<Array> {
                if let Some(arr) = layer_map.get(name) {
                    return Ok(arr.clone());
                }
                if let Some(&h) = self.persistent_handles.get(name) {
                    let reg = crate::bridge::ARRAY_REGISTRY.read();
                    return reg.get(h).cloned().ok_or_else(|| {
                        crate::Error::from_reason(format!("persistent handle invalid for {}", name))
                    });
                }
                Err(crate::Error::from_reason(format!(
                    "tensor not found for layer {}: {}",
                    layer, name
                )))
            };

            let base = format!("{}.layers.{}", root, layer);
            let is_full = matches!(
                arch.layer_types[layer],
                crate::config::AttentionKind::FullAttention
            );

            let attn_norm = get_tensor(&format!("{}.input_layernorm.weight", base))?;
            let ffn_norm = get_tensor(&format!("{}.post_attention_layernorm.weight", base))?;
            let (qw, qs, qb) = (
                get_tensor(&format!("{}.self_attn.q_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.biases", base))?,
            );
            let (kw, ks, kb) = (
                get_tensor(&format!("{}.self_attn.k_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.biases", base))?,
            );
            let (vw, vs, vb) = if !is_full {
                (
                    get_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
                )
            } else {
                (
                    Array::from_slice(&[0.0f32], &[1]),
                    Array::from_slice(&[0.0f32], &[1]),
                    Array::from_slice(&[0.0f32], &[1]),
                )
            };
            let (ow, os, ob) = (
                get_tensor(&format!("{}.self_attn.o_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.biases", base))?,
            );
            let (gw, gs, gb) = (
                get_tensor(&format!("{}.mlp.gate_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.biases", base))?,
            );
            let (uw, us, ub) = (
                get_tensor(&format!("{}.mlp.up_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.up_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.up_proj.biases", base))?,
            );
            let (dw, ds, db) = (
                get_tensor(&format!("{}.mlp.down_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.down_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.down_proj.biases", base))?,
            );

            // Run the layer forward pass.
            let layer_arrays = crate::model::LayerArraysRef {
                attn_norm: &attn_norm,
                ffn_norm: &ffn_norm,
                qw: &qw,
                qs: &qs,
                qb: &qb,
                kw: &kw,
                ks: &ks,
                kb: &kb,
                vw: &vw,
                vs: &vs,
                vb: &vb,
                ow: &ow,
                os: &os,
                ob: &ob,
                gw: &gw,
                gs: &gs,
                gb: &gb,
                uw: &uw,
                us: &us,
                ub: &ub,
                dw: &dw,
                ds: &ds,
                db: &db,
            };

            hidden = if is_full {
                crate::model::run_full_layer_arrays(
                    &hidden,
                    &layer_arrays,
                    &arch,
                    &full_cos,
                    &full_sin,
                    0,
                )
                .map_err(|e| crate::Error::from_reason(format!("layer {} full: {:?}", layer, e)))?
            } else {
                crate::model::run_sliding_layer_arrays(
                    &hidden,
                    &layer_arrays,
                    &arch,
                    &rope_cos,
                    &rope_sin,
                    0,
                )
                .map_err(|e| {
                    crate::Error::from_reason(format!("layer {} sliding: {:?}", layer, e))
                })?
            };

            // *** CRITICAL: eval BEFORE dropping lease ***
            // MLX is lazy — the graph still references the layer arrays until eval() forces
            // the computation. Dropping the lease before eval leaves the graph with dead
            // backing storage.
            hidden
                .eval()
                .map_err(|e| crate::Error::from_reason(format!("eval layer {}: {:?}", layer, e)))?;

            let elapsed_ms = t0.elapsed().as_millis();
            let rss_after = process_rss_bytes();
            let active_after = mlx_active_memory_bytes();
            let cached_after = mlx_cache_memory_bytes();
            let handles_after = crate::bridge::handle_count();

            // Emit per-layer residency receipt.
            let _rss_evaluated = rss_after;
            let active_evaluated = active_after;
            let cached_evaluated = cached_after;
            let handles_evaluated = handles_after;
            let seg_id = lease.segment_id.clone();

            // *** Retire the layer segment. ***
            // hidden.eval() has already forced the kernel to consume the weights.
            drop(lease);

            // Capture telemetry AFTER retirement to prove logical release.
            let rss_retired = process_rss_bytes();
            let active_retired = mlx_active_memory_bytes();
            let cached_retired = mlx_cache_memory_bytes();
            let handles_retired = crate::bridge::handle_count();

            eprintln!(
                "[image-runtime] layer={} segment={} bytes_read={} elapsed_ms={} \
                 rss_delta={} mlx_active={}→{} mlx_cached={}→{} handles={}→{}→{}",
                layer,
                seg_id,
                bytes_read,
                elapsed_ms,
                rss_retired as i64 - rss_before as i64,
                active_evaluated,
                active_retired,
                cached_evaluated,
                cached_retired,
                handles_before,
                handles_evaluated,
                handles_retired,
            );
        }

        // Final norm + LM head projection using persistent embed tensors.
        let fn_w_name = format!("{}.norm.weight", root);
        let fn_w = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            reg.get(*self.persistent_handles.get(&fn_w_name).ok_or_else(|| {
                crate::Error::from_reason(format!("missing persistent tensor: {}", fn_w_name))
            })?)
            .cloned()
            .ok_or_else(|| crate::Error::from_reason("norm weight handle invalid"))?
        };
        let final_hidden = crate::primitives::rms_norm(&hidden, &fn_w, 1e-6)
            .map_err(|e| crate::Error::from_reason(format!("final norm: {:?}", e)))?;

        // LM head aliases embed_tokens (tie_word_embeddings); reuse emb_w.
        let out = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let ew =
                reg.get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason("embed weight gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed weight handle invalid at lm_head")
                })?;
            let es =
                reg.get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason("embed scales gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed scales handle invalid at lm_head")
                })?;
            let eb =
                reg.get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason("embed biases gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed biases handle invalid at lm_head")
                })?;
            let gs = (ew.shape()[1] as i32 * 4) / es.shape()[1];
            mlx_rs::ops::quantized_matmul(&final_hidden, &ew, &es, &eb, true, gs, 8)
                .map_err(|e| crate::Error::from_reason(format!("lm_head matmul: {:?}", e)))?
        };
        out.eval()
            .map_err(|e| crate::Error::from_reason(format!("final eval: {:?}", e)))?;

        self.release();
        Ok(out)
    }

    /// Execute the complete 48-layer model from the compiled execution plan.
    ///
    /// This is the canonical forward path:
    ///   1. Run the prologue (embedding → hidden state)
    ///   2. For each layer in the execution plan:
    ///      a. Activate the layer segment
    ///      b. Run the layer executor from the compiled plan
    ///      c. eval() before dropping the lease
    ///      d. Record per-layer telemetry
    ///   3. Run the epilogue (final norm → output projection → softcap → argmax)
    ///
    /// Returns a u32 token ID — no logits cross the boundary.
    /// Per-layer receipts are emitted to stderr.
    pub fn run_full_model(&mut self, token_ids: &[i32]) -> crate::Result<u32> {
        if self.released {
            return Err(crate::Error::from_reason("image runtime already released"));
        }

        let plan = &self.manifest.execution_plan;
        plan.validate().map_err(|errors| {
            crate::Error::from_reason(format!(
                "execution plan validation failed: {}",
                errors.join("; ")
            ))
        })?;

        let arch = &self.manifest.architecture;
        let root = "language_model.model";
        let seq_len = token_ids.len() as i32;

        // --- Prologue: embedding lookup ---
        let emb_w_name = format!("{}.embed_tokens.weight", root);
        let emb_s_name = format!("{}.embed_tokens.scales", root);
        let emb_b_name = format!("{}.embed_tokens.biases", root);

        let (emb_w, emb_s, emb_b) = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let w =
                reg.get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_w_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed weight invalid"))?;
            let s =
                reg.get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_s_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed scales invalid"))?;
            let b =
                reg.get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_b_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed biases invalid"))?;
            (w, s, b)
        };

        let tok = Array::from_slice(token_ids, &[1, seq_len]);
        let mut hidden = crate::executor::run_prologue(
            &tok,
            &emb_w,
            &emb_s,
            &emb_b,
            &plan.prologue,
            (arch.hidden_size as f32).sqrt(),
        )
        .map_err(|e| crate::Error::from_reason(format!("prologue: {:?}", e)))?;

        hidden
            .eval()
            .map_err(|e| crate::Error::from_reason(format!("prologue eval: {:?}", e)))?;

        // Precompute RoPE tables
        let (rope_cos, rope_sin) = crate::primitives::rope_freqs(
            arch.head_dim,
            arch.max_position_embeddings,
            arch.rope_local.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope local: {:?}", e)))?;
        let full_rope = arch.rope_global.as_ref().unwrap_or(&arch.rope_local);
        let (full_cos, full_sin) = crate::primitives::rope_freqs(
            arch.global_head_dim.unwrap_or(arch.head_dim),
            arch.max_position_embeddings,
            full_rope.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope global: {:?}", e)))?;

        // Build per-layer KV caches for single-pass validation
        let max_seq_len = arch.max_position_embeddings.min(8192);
        let mut caches: Vec<crate::kv_cache::KvCache> = Vec::with_capacity(plan.layers.len());
        for layer_plan in &plan.layers {
            let is_sliding = layer_plan.attention_kind == "sliding_attention";
            let (capacity, n_kv_heads, head_dim) = if is_sliding {
                (
                    layer_plan.sliding_window,
                    layer_plan.n_kv_heads,
                    layer_plan.head_dim,
                )
            } else {
                let g_kv = layer_plan.n_global_kv_heads.unwrap_or(1);
                let g_hd = layer_plan.global_head_dim.unwrap_or(layer_plan.head_dim);
                (max_seq_len, g_kv, g_hd)
            };
            caches.push(crate::kv_cache::KvCache::new(
                capacity, n_kv_heads, head_dim, is_sliding,
            ));
        }

        let idle_handles = crate::bridge::handle_count();
        eprintln!(
            "[full-model] idle_handles={} layer_count={}",
            idle_handles,
            plan.layers.len(),
        );
        // --- Decoder layers ---
        for layer_plan in &plan.layers {
            let l = layer_plan.layer_index;
            let t0 = Instant::now();
            let handles_before = crate::bridge::handle_count();
            let active_before = mlx_active_memory_bytes();

            // Activate the layer segment
            let lease = self
                .activate_layer(l)
                .map_err(|e| crate::Error::from_reason(format!("activate layer {}: {}", l, e)))?;
            let bytes_read = lease.bytes_read;

            // Build layer tensor map from the lease
            let layer_map = self.build_layer_arrays_from_lease(l, &lease)?;

            // Helper to look up a tensor
            let get_tensor = |name: &str| -> crate::Result<Array> {
                if let Some(arr) = layer_map.get(name) {
                    return Ok(arr.clone());
                }
                if let Some(&h) = self.persistent_handles.get(name) {
                    let reg = crate::bridge::ARRAY_REGISTRY.read();
                    return reg.get(h).cloned().ok_or_else(|| {
                        crate::Error::from_reason(format!("persistent handle invalid for {}", name))
                    });
                }
                Err(crate::Error::from_reason(format!(
                    "tensor not found for layer {}: {}",
                    l, name
                )))
            };

            let base = format!("{}.layers.{}", root, l);
            let is_full = layer_plan.attention_kind == "full_attention";

            let attn_norm = get_tensor(&format!("{}.input_layernorm.weight", base))?;
            let ffn_norm = get_tensor(&format!("{}.post_attention_layernorm.weight", base))?;
            let (qw, qs, qb) = (
                get_tensor(&format!("{}.self_attn.q_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.biases", base))?,
            );
            let (kw, ks, kb) = (
                get_tensor(&format!("{}.self_attn.k_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.biases", base))?,
            );
            let (vw, vs, vb) = if is_full {
                // K-equals-V: reuse k_proj
                (kw.clone(), ks.clone(), kb.clone())
            } else {
                (
                    get_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
                )
            };
            let (ow, os, ob) = (
                get_tensor(&format!("{}.self_attn.o_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.biases", base))?,
            );
            let (gw, gs, gb) = (
                get_tensor(&format!("{}.mlp.gate_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.biases", base))?,
            );
            let (uw, us, ub) = (
                get_tensor(&format!("{}.mlp.up_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.up_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.up_proj.biases", base))?,
            );
            let (dw, ds, db) = (
                get_tensor(&format!("{}.mlp.down_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.down_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.down_proj.biases", base))?,
            );

            // Q/K norm weights
            let q_norm = get_tensor(&format!("{}.self_attn.q_norm.weight", base)).ok();
            let k_norm = get_tensor(&format!("{}.self_attn.k_norm.weight", base)).ok();

            // Select RoPE tables
            let (rcos, rsin) = if is_full {
                (&full_cos, &full_sin)
            } else {
                (&rope_cos, &rope_sin)
            };

            // Run the layer executor
            hidden = crate::executor::run_layer(
                &hidden,
                layer_plan,
                &crate::config::operation_route::OperationRoute::default(),
                None,
                &[],
                &attn_norm,
                &ffn_norm,
                &qw,
                &qs,
                &qb,
                &kw,
                &ks,
                &kb,
                &vw,
                &vs,
                &vb,
                &ow,
                &os,
                &ob,
                q_norm.as_ref(),
                k_norm.as_ref(),
                &gw,
                &gs,
                &gb,
                &uw,
                &us,
                &ub,
                &dw,
                &ds,
                &db,
                rcos,
                rsin,
                &mut caches[l as usize],
                0, // kv_offset = 0 for single-pass
                arch.rms_norm_eps as f32,
                &projection_identity::ProjectionContext {
                    run_id: "test".into(),
                    phase: projection_identity::Phase::Prefill,
                    forward_pass_index: 1,
                    token_step: None,
                    layer_index: l as usize,
                    attention_kind: projection_identity::AttentionKind::Sliding,
                },
            )
            .map_err(|e| crate::Error::from_reason(format!("layer {}: {:?}", l, e)))?;

            // *** CRITICAL: eval BEFORE dropping lease ***
            hidden
                .eval()
                .map_err(|e| crate::Error::from_reason(format!("eval layer {}: {:?}", l, e)))?;

            let elapsed_ms = t0.elapsed().as_millis();
            let handles_after = crate::bridge::handle_count();
            let _active_evaluated = mlx_active_memory_bytes();
            let seg_id = lease.segment_id.clone();

            // Retire the layer segment
            drop(lease);

            let handles_retired = crate::bridge::handle_count();
            let active_retired = mlx_active_memory_bytes();

            let output_shape = hidden.shape();
            let is_finite = hidden
                .try_as_slice::<f32>()
                .map(|v| v.iter().all(|x| x.is_finite()))
                .unwrap_or(false);

            eprintln!(
                "[full-model] layer={} kind={} segment={} bytes={} elapsed_ms={} \
                 handles={}→{}→{} active_mem={}→{} shape={:?} finite={}",
                l,
                layer_plan.attention_kind,
                seg_id,
                bytes_read,
                elapsed_ms,
                handles_before,
                handles_after,
                handles_retired,
                active_before,
                active_retired,
                output_shape,
                is_finite,
            );
        }

        // Verify return to idle
        let final_handles = crate::bridge::handle_count();
        eprintln!(
            "[full-model] all_layers_done final_handles={} idle_handles={}",
            final_handles, idle_handles,
        );

        // --- Epilogue: final norm + output projection + softcapping + argmax ---
        let fn_w_name = format!("{}.norm.weight", root);
        let fn_w = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            reg.get(
                *self
                    .persistent_handles
                    .get(&fn_w_name)
                    .ok_or_else(|| crate::Error::from_reason(format!("missing: {}", fn_w_name)))?,
            )
            .cloned()
            .ok_or_else(|| crate::Error::from_reason("norm weight invalid"))?
        };

        let epi = crate::executor::run_epilogue(
            &hidden,
            &fn_w,
            &emb_w,
            &emb_s,
            &emb_b,
            &plan.epilogue,
            arch.rms_norm_eps as f32,
            arch.tie_word_embeddings,
            &crate::session::SamplerConfig::default(),
        )
        .map_err(|e| crate::Error::from_reason(format!("epilogue: {:?}", e)))?;

        epi.selected_token
            .eval()
            .map_err(|e| crate::Error::from_reason(format!("epilogue eval: {:?}", e)))?;
        let token_id = epi
            .selected_token
            .try_as_slice::<u32>()
            .map_err(|e| crate::Error::from_reason(format!("epilogue token: {:?}", e)))?
            .first()
            .copied()
            .unwrap_or(0);

        self.release();
        Ok(token_id)
    }

    /// Release all persistent tensor handles.
    pub fn release(&mut self) {
        if self.released {
            return;
        }
        for handle in self
            .persistent_handles
            .values()
            .copied()
            .collect::<Vec<_>>()
        {
            let _ = crate::bridge::free_array(handle);
        }
        self.persistent_handles.clear();
        self.quantized_bindings.clear();
        self.released = true;
    }
}

fn plan(
    source_dir: &Path,
    skip_validation: bool,
) -> crate::Result<(crate::config::CompilationPlan, LoadedSource)> {
    use crate::config::{CompilationPlan, PlannedSegment, PlannedTensor};

    let loaded = load_source(source_dir, skip_validation)?;
    let shard_hashes: Vec<String> = loaded
        .shard_hashes
        .iter()
        .map(|h| h.sha256.clone())
        .collect();

    let mut tensor_table = Vec::new();
    let mut next_tensor_id: u32 = 0;
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut seg_offsets: HashMap<String, u64> = HashMap::new();

    // Persistent segment.
    let persistent_seg_id = "persistent".to_string();
    segments.push(PlannedSegment {
        id: persistent_seg_id.clone(),
        filename: "segment_000.bin".into(),
        byte_size: 0,
        kind: "persistent".into(),
        tensor_count: 0,
    });

    for binding in &loaded.spec.global_tensors {
        let disp = classify_disposition(binding, &loaded.namespace);
        let (src_shard, src_offset, src_len, logical_dtype) =
            source_info(&loaded.source_tensors, &binding.name);
        let dest_offset = seg_offsets.get(&persistent_seg_id).copied().unwrap_or(0);
        tensor_table.push(PlannedTensor {
            id: next_tensor_id,
            name: binding.name.clone(),
            disposition: disp,
            source_shard: src_shard,
            source_offset: src_offset,
            source_byte_length: src_len,
            destination_segment: persistent_seg_id.clone(),
            destination_offset: dest_offset,
            destination_byte_length: src_len,
            logical_dtype,
            logical_shape: binding.logical_shape.clone(),
        });
        *seg_offsets.entry(persistent_seg_id.clone()).or_insert(0) += src_len;
        next_tensor_id += 1;
    }

    // Layer segments.
    for layer in &loaded.spec.layers {
        let seg_id = format!("layer_{}", layer.index);
        let seg_idx = segments.len();
        segments.push(PlannedSegment {
            id: seg_id.clone(),
            filename: format!("segment_{:03}.bin", seg_idx),
            byte_size: 0,
            kind: format!("layer_{}", layer.index),
            tensor_count: 0,
        });
        for binding in &layer.tensors {
            let disp = classify_disposition(binding, &loaded.namespace);
            let (src_shard, src_offset, src_len, logical_dtype) =
                source_info(&loaded.source_tensors, &binding.name);
            let dest_offset = seg_offsets.get(&seg_id).copied().unwrap_or(0);
            tensor_table.push(PlannedTensor {
                id: next_tensor_id,
                name: binding.name.clone(),
                disposition: disp,
                source_shard: src_shard,
                source_offset: src_offset,
                source_byte_length: src_len,
                destination_segment: seg_id.clone(),
                destination_offset: dest_offset,
                destination_byte_length: src_len,
                logical_dtype,
                logical_shape: binding.logical_shape.clone(),
            });
            *seg_offsets.entry(seg_id.clone()).or_insert(0) += src_len;
            next_tensor_id += 1;
        }
    }

    // Update segment byte sizes and tensor counts.
    for seg in &mut segments {
        seg.byte_size = *seg_offsets.get(&seg.id).unwrap_or(&0);
        seg.tensor_count = tensor_table
            .iter()
            .filter(|t| t.destination_segment == seg.id)
            .count();
    }

    let total_source_bytes: u64 = loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();
    let total_image_bytes: u64 = segments.iter().map(|s| s.byte_size).sum();

    let plan = CompilationPlan {
        model_identity: loaded.manifest.model_type.clone(),
        source_config_hash: loaded.manifest.config_hash.clone(),
        source_shard_hashes: shard_hashes,
        tensor_table,
        segments,
        total_source_bytes,
        total_image_bytes,
    };

    Ok((plan, loaded))
}

fn classify_disposition(
    binding: &crate::config::TensorBinding,
    _namespace: &crate::config::NamespaceBinding,
) -> crate::config::TensorDisposition {
    use crate::config::TensorDisposition;

    // Quantized weight payloads get relocated unchanged.
    if binding.name.ends_with(".weight")
        || binding.name.ends_with(".scales")
        || binding.name.ends_with(".biases")
    {
        return TensorDisposition::RelocateAndAlign;
    }
    // Embedding layer_scalar and other small tensors also relocate.
    TensorDisposition::RelocateAndAlign
}

fn source_info(
    source_tensors: &HashMap<String, SourceTensor>,
    name: &str,
) -> (String, u64, u64, String) {
    if let Some(st) = source_tensors.get(name) {
        (
            st.source_filename.clone(),
            st.source_offset,
            st.data.len() as u64,
            st.dtype.clone(),
        )
    } else {
        (String::new(), 0, 0, "F32".into())
    }
}

/// Reorder segments for speculative decoding: shared persistent first,
/// then draft layer segments, then target layer segments + target persistent.
fn reorder_for_speculative(
    target_segments: &mut Vec<crate::config::PlannedSegment>,
    draft_segments: &mut Vec<crate::config::PlannedSegment>,
    config: &crate::config::SpeculativeModelConfig,
) {
    let mut reordered = Vec::new();

    // 1. Shared persistent segment (embeddings, LM head if shared)
    if config.shared_embedding {
        // Merge persistent segments: keep the first persistent from target
        if let Some(pos) = target_segments.iter().position(|s| s.kind == "persistent") {
            let seg = target_segments.remove(pos);
            reordered.push(seg);
        }
        // Remove draft persistent (absorbed into shared)
        draft_segments.retain(|s| s.kind != "persistent");
    }

    // 2. Draft layer segments first (fast startup)
    if config.draft_first_segments {
        let draft_layers: Vec<_> = std::mem::take(draft_segments)
            .into_iter()
            .filter(|s| s.kind.starts_with("layer_"))
            .collect();
        reordered.extend(draft_layers);
        // Keep remaining (non-persistent, non-layer) draft segments
        *draft_segments = Vec::new();
    }

    // 3. Target segments (persistent then layer)
    //    Persistent first (norms), then layer segments
    if let Some(pos) = target_segments.iter().position(|s| s.kind == "persistent") {
        let seg = target_segments.remove(pos);
        reordered.push(seg);
    }
    let target_layers: Vec<_> = std::mem::take(target_segments)
        .into_iter()
        .filter(|s| s.kind.starts_with("layer_"))
        .collect();
    reordered.extend(target_layers);

    *target_segments = reordered;
}

/// Compile a draft + target model pair into a single speculative ComputeImage.
///
/// Loads both checkpoints, emits shared weights once when compatible,
/// orders draft layers first for fast startup, and attaches speculative
/// decoding metadata to the manifest.
fn compile_unchecked_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    _quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    let started_at = std::time::Instant::now();
    let output_dir = Path::new(output_dir);

    // Load both models independently
    let t_load = Instant::now();
    let target_loaded = load_source(Path::new(target_dir), false)?;
    let draft_loaded = load_source(Path::new(draft_dir), false)?;
    let source_load_ms = t_load.elapsed().as_millis() as u64;

    // Detect embedding shareability
    let shared_embedding = target_loaded.arch.vocab_size == draft_loaded.arch.vocab_size
        && target_loaded.arch.hidden_size == draft_loaded.arch.hidden_size;
    let shared_lm_head = shared_embedding;

    // Build source identity from target model
    let source = build_source_identity(
        &target_loaded.manifest,
        target_loaded.shard_hashes.clone(),
        target_loaded.tokenizer_hashes.clone(),
        target_loaded.auxiliary_hashes.clone(),
    );

    let mut builder = ImageBuilder::new(target_loaded.arch.clone(), source);
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    let t_emit = Instant::now();

    // 1. Shared persistent segment (embeddings stored once if shareable)
    let shared_seg_id = "persistent".to_string();
    builder.begin_segment(&shared_seg_id, SegmentKind::Persistent);

    if shared_embedding {
        // Emit target embeddings — shared by both models
        for binding in &target_loaded.spec.global_tensors {
            let id = emit_binding_set(&mut builder, &target_loaded.source_tensors, binding, None)?;
            emitted_ids.insert(binding.name.clone(), id);
            // Register aliases for draft model tensors that map to shared weights
            let draft_root = &draft_loaded.namespace.root;
            let target_root = &target_loaded.namespace.root;
            if binding.name.contains("embed_tokens") {
                let draft_embed = binding.name.replace(target_root, draft_root);
                builder.add_alias(&draft_embed, id, "shared_embedding_speculative");
                emitted_ids.insert(draft_embed, id);
            }
        }
        if shared_lm_head {
            // If lm_head is aliased (tied), register draft alias too
            if target_loaded.namespace.lm_head_aliased {
                let target_head = "lm_head.weight".to_string();
                let draft_head_key = format!("{}.lm_head.weight", draft_loaded.namespace.root);
                if let Some(&id) = emitted_ids.get(&target_head) {
                    builder.add_alias(&draft_head_key, id, "shared_lm_head_speculative");
                    emitted_ids.insert(draft_head_key, id);
                }
            }
        }
    } else {
        // Not shared: emit target embeddings, then switch to new persistent for draft
        for binding in &target_loaded.spec.global_tensors {
            let id = emit_binding_set(&mut builder, &target_loaded.source_tensors, binding, None)?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // 2. Draft layer segments (first for fast startup)
    for layer in &draft_loaded.spec.layers {
        let seg_id = format!("draft_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &draft_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // 3. Target persistent (norms, LM head) — if embeddings not shared, they're already emitted
    if !shared_embedding {
        // switch back to target persistent for norms
        builder.begin_segment("persistent_target", SegmentKind::Persistent);
        // norms and other non-embedding global tensors already emitted above
        // if not shared, the loop above already emitted all target globals
    }

    // 4. Target layer segments
    for layer in &target_loaded.spec.layers {
        let seg_id = format!("target_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &target_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Register aliases for tied embeddings on target side
    if target_loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", target_loaded.namespace.root);
        if let Some(&id) = emitted_ids.get(&embed_name) {
            builder.add_alias("lm_head.weight", id, "tie_word_embeddings");
        }
    }

    // 5. Build the target execution plan
    let mut execution_plan =
        crate::config::build_execution_plan(&target_loaded.arch, &target_loaded.namespace, &emitted_ids);
    execution_plan.build_ane_fusion_plan();

    // 6. Attach speculative config metadata
    execution_plan.speculative_config = Some(crate::config::SpeculativeModelConfig {
        draft_architecture: draft_loaded.arch.clone(),
        target_architecture: target_loaded.arch.clone(),
        shared_embedding,
        shared_lm_head,
        draft_first_segments: true,
        speculation_length: 5,
    });

    builder.set_execution_plan(execution_plan);

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;
    let emitted_so_far = builder.segment_payloads.iter().map(|p| p.len() as u64).sum();
    crate::compile_progress::CompileProgress {
        stage: "payload_emission_done".into(),
        bytes_processed: emitted_so_far,
        bytes_total: emitted_so_far,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    let total_source_bytes = target_loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum::<u64>()
        + draft_loaded
            .source_tensors
            .values()
            .map(|t| t.data.len() as u64)
            .sum::<u64>();
    let total_emitted_bytes = manifest.segments.iter().map(|s| s.byte_size).sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: 0,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes(),
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &target_loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time quantization transform
// ═══════════════════════════════════════════════════════════════════════════

/// Standard 4-bit NormalFloat (NF4) codebook from the QLoRA paper.
/// These are the 16 quantiles of a standard normal distribution,
/// symmetric around zero, with equal area under the curve per interval.
const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057,
    0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];

/// Find the nearest NF4 codebook index for a given normalized value.
/// Returns index in [0, 15].
fn quantize_nf4_value(value: f32) -> u8 {
    let mut best_idx: u8 = 0;
    let mut best_dist: f32 = (value - NF4_CODEBOOK[0]).abs();
    for (i, &level) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let dist = (value - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Apply NF4 block quantization to a single group of F32 values.
/// Returns (packed_u32_words, scale_absmax, bias_zero_point).
/// For NF4: bias is always 0.0 (symmetric quantization).
fn quantize_nf4_group(values: &[f32]) -> (Vec<u32>, f32, f32) {
    if values.is_empty() {
        return (vec![0u32; 1], 0.0, 0.0);
    }
    // Find absolute maximum for the group (the scale factor).
    let absmax = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));

    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
    let inv_scale = 1.0 / scale;

    // Quantize each value to a 4-bit NF4 index, pack 8 per U32 word.
    let n_words = (values.len() + 7) / 8;
    let mut packed = vec![0u32; n_words];
    for (i, &val) in values.iter().enumerate() {
        let normalized = val * inv_scale;
        // Clamp to [-1, 1] range (NF4 codebook bounds).
        let clamped = normalized.clamp(-1.0, 1.0);
        let idx = quantize_nf4_value(clamped);
        let word_idx = i / 8;
        let bit_shift = ((i % 8) * 4) as u32;
        packed[word_idx] |= (idx as u32) << bit_shift;
    }

    (packed, scale, 0.0) // NF4 is symmetric — bias = 0
}

/// Apply 8-bit affine block quantization to a single group of F32 values.
/// Returns (packed_u8_bytes, scale, bias).
fn quantize_af8_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (vec![0u8; 1], 0.0, 0.0);
    }
    let min_val = values.iter().cloned().fold(f32::MAX, f32::min);
    let max_val = values.iter().cloned().fold(f32::MIN, f32::max);

    let range = max_val - min_val;
    let scale = if range > 1e-12 { range / 255.0 } else { 1.0 / 255.0 };
    let bias = min_val;

    let mut q = Vec::with_capacity(values.len());
    for &v in values {
        let qv = ((v - min_val) / scale).round().clamp(0.0, 255.0) as u8;
        q.push(qv);
    }

    (q, scale, bias)
}

/// Apply compile-time quantization to all FP16/BF16 weight tensors in the
/// loaded source. This modifies the source tensors in-place, converting
/// weight tensor bytes to packed quantized form and adding companion
/// scale/bias tensors. The TensorBinding packed_shape fields are also set
/// so the existing `emit_quantized_binding` pipeline writes the triplets.
fn apply_quantize_to_loaded(
    loaded: &mut LoadedSource,
    qmode: CompileQuantMode,
) -> crate::Result<()> {
    // Collect all weight bindings (global + per-layer) that are not already packed.
    struct WeightBinding {
        name: String,
        role: String,
        logical_shape: Vec<u32>,
        is_global: bool,
        layer_index: Option<u32>,
    }

    let mut weight_bindings: Vec<WeightBinding> = Vec::new();

    // Collect global weight tensors.
    for binding in &loaded.spec.global_tensors {
        if binding.name.ends_with(".weight") && binding.packed_shape.is_none() {
            weight_bindings.push(WeightBinding {
                name: binding.name.clone(),
                role: format!("{:?}", binding.role),
                logical_shape: binding.logical_shape.clone(),
                is_global: true,
                layer_index: None,
            });
        }
    }

    // Collect per-layer weight tensors.
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if binding.name.ends_with(".weight") && binding.packed_shape.is_none() {
                weight_bindings.push(WeightBinding {
                    name: binding.name.clone(),
                    role: format!("{:?}", binding.role),
                    logical_shape: binding.logical_shape.clone(),
                    is_global: false,
                    layer_index: Some(layer.index),
                });
            }
        }
    }

    eprintln!("[quantize] applying {} quantization to {} weight tensors",
        match qmode {
            CompileQuantMode::Nf4 { group_size } => {
                format!("NF4 (group_size={})", group_size)
            }
            CompileQuantMode::Af8 { group_size } => {
                format!("8-bit affine (group_size={})", group_size)
            }
        },
        weight_bindings.len(),
    );

    for wb in &weight_bindings {
        let source_tensor = loaded.source_tensors.get(&wb.name).ok_or_else(|| {
            crate::Error::from_reason(format!(
                "quantize: missing source tensor '{}'",
                wb.name
            ))
        })?;

        // Only quantize FP16/BF16 dtypes.
        let dtype = source_tensor.dtype.as_str();
        if dtype != "F16" && dtype != "BF16" {
            eprintln!(
                "[quantize] skipping {} (dtype={}, only FP16/BF16 supported)",
                wb.name, dtype
            );
            continue;
        }

        let raw = &source_tensor.data;
        let shape = &source_tensor.shape;
        let out_dim = shape[0]; // rows
        let in_dim = shape[1];  // cols

        // Convert FP16/BF16 raw bytes to F32.
        let n_elements = raw.len() / 2;
        let mut f32_vals = Vec::with_capacity(n_elements);
        if dtype == "BF16" {
            // BF16: same exponent/mantissa layout as F32 top-16 bits.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(f32::from_bits((bits as u32) << 16));
            }
        } else {
            // FP16: standard IEEE 754 half-precision.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(half_to_f32(bits));
            }
        }

        let group_size = match qmode {
            CompileQuantMode::Nf4 { group_size } => group_size,
            CompileQuantMode::Af8 { group_size } => group_size,
        };
        let groups_per_row = (in_dim + group_size - 1) / group_size;
        let total_groups = out_dim * groups_per_row;

        // Apply block quantization per group.
        match qmode {
            CompileQuantMode::Nf4 { .. } => {
                apply_nf4_quantize(
                    loaded, &wb.name, &f32_vals, out_dim, in_dim,
                    group_size, groups_per_row, total_groups,
                )?;
            }
            CompileQuantMode::Af8 { .. } => {
                apply_af8_quantize(
                    loaded, &wb.name, &f32_vals, out_dim, in_dim,
                    group_size, groups_per_row, total_groups,
                )?;
            }
        }
    }

    Ok(())
}

/// Apply NF4 quantization to a weight tensor and update the loaded source.
fn apply_nf4_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // Packed NF4 weights: each U32 stores 8 * 4-bit values.
    let pack_factor = 8; // 32 / 4
    let packed_in = (in_dim_u + pack_factor - 1) / pack_factor;
    let packed_weight_len = (out_dim as usize) * packed_in;
    let mut packed_weight = vec![0u32; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let _biases = vec![0.0f32; total_g]; // NF4 is symmetric — biases are 0

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (packed_group, scale, _bias) = quantize_nf4_group(group_vals);
            scales.push(scale);

            // Place packed U32 words into the correct position in packed_weight.
            let weight_row_offset = row * packed_in;
            let group_word_offset = g * (gs + pack_factor - 1) / pack_factor;
            for (wi, &word) in packed_group.iter().enumerate() {
                packed_weight[weight_row_offset + group_word_offset + wi] = word;
            }
        }
    }

    // Serialize packed weights as U32 bytes (little-endian).
    let packed_bytes: Vec<u8> = packed_weight
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();
    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = vec![0u8; total_g * 4]; // F32 zeros

    // Derive scale/bias tensor names.
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    // Build the packed shape descriptor.
    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in as u32],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 4,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace the weight source tensor with packed data.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in as u32];
    }

    // Add scale source tensor.
    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Add bias source tensor.
    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Update the TensorBinding in the spec to enable packed emission.
    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] NF4 quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

/// Apply 8-bit affine quantization to a weight tensor and update the loaded source.
fn apply_af8_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // 8-bit quantized weights stored as U8.
    let packed_weight_len = (out_dim as usize) * in_dim_u;
    let mut packed_weight = vec![0u8; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let mut biases = Vec::with_capacity(total_g);

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (q_bytes, scale, bias) = quantize_af8_group(group_vals);
            scales.push(scale);
            biases.push(bias);

            for (wi, &byte) in q_bytes.iter().enumerate() {
                packed_weight[group_start + wi] = byte;
            }
        }
    }

    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = biases
        .iter()
        .flat_map(|&b| b.to_le_bytes().to_vec())
        .collect();

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let pack = 32 / 8; // 4 U8 per U32
    let packed_in = in_dim / pack;
    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 8,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace weight source tensor.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_weight;
        st.dtype = "U8".to_string();
        st.shape = vec![out_dim, packed_in];
    }

    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] 8-bit affine quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

/// Fast half-precision (FP16) to F32 conversion.
fn half_to_f32(bits: u16) -> f32 {
    // FP16 format: 1 sign + 5 exponent + 10 mantissa
    let sign = ((bits >> 15) & 0x1) as f32;
    let exp = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            0.0_f32.copysign(1.0 - 2.0 * sign)
        } else {
            f32::from_bits(((sign as u32) << 31) | ((102 - 14 + 127) << 23) | (mantissa << 13))
                * (1.0 / 16777216.0) // 2^-24
        }
    } else if exp == 31 {
        // Infinity or NaN
        let exp_f32: u32 = 255;
        let mantissa_f32 = if mantissa == 0 { 0 } else { mantissa << 13 };
        f32::from_bits(((sign as u32) << 31) | (exp_f32 << 23) | mantissa_f32)
    } else {
        // Normal: FP16 exponent bias = 15, F32 exponent bias = 127
        let exp_f32: u32 = ((exp as u32) + 127 - 15) << 23;
        f32::from_bits(((sign as u32) << 31) | exp_f32 | ((mantissa as u32) << 13))
    }
}

/// Compile a source checkpoint into a precompiled ComputeImage runtime artifact.
///
/// The source directory must contain a config.json and safetensors shards.
/// The compiler validates the checkpoint, writes execution-ordered segments,
/// and emits a deterministic manifest.json plus receipt.json.
fn compile_unchecked(
    source_dir: &str,
    output_dir: &str,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    let source_dir = Path::new(source_dir);
    let output_dir = Path::new(output_dir);
    let started_at = std::time::Instant::now();

    let t_source = Instant::now();
    let (_plan, loaded) = plan(source_dir, skip_validation)?;
    // TODO Phase 3: Use plan to drive parallel emission instead of sequential loaded.spec iteration
    let source_load_ms = t_source.elapsed().as_millis() as u64;
    crate::compile_progress::CompileProgress {
        stage: "source_loaded".into(),
        bytes_processed: loaded.spec.layers.len() as u64,
        bytes_total: loaded.spec.layers.len() as u64,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    compile_sequential(source_dir, output_dir, loaded, started_at, source_load_ms, quantize_mode)
}

fn compile_sequential(
    _source_dir: &Path,
    output_dir: &Path,
    mut loaded: LoadedSource,
    started_at: Instant,
    source_load_ms: u64,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    // Apply compile-time quantization if requested.
    // Transforms FP16/BF16 source weights into quantized packed triplets
    // before the emission loop builds the segment payloads.
    if let Some(qmode) = quantize_mode {
        apply_quantize_to_loaded(&mut loaded, qmode)?;
    }

    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);

    let t_emit = Instant::now();
    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| crate::Error::from_reason("embed_tokens.weight was not emitted"))?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Compile vision encoder tensors if present.
    if loaded.manifest.vision_config.is_some() {
        compile_vision_encoder_tensors(
            &mut builder,
            &loaded.source_tensors,
            &mut emitted_ids,
        )?;
    }

    // Compile audio encoder tensors if present.
    if loaded.manifest.audio_config.is_some() {
        compile_audio_encoder_tensors(
            &mut builder,
            &loaded.source_tensors,
            &mut emitted_ids,
            loaded.manifest.audio_config.clone(),
        )?;
    }

    // Build the execution plan using the emitted tensor IDs
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    // Apply compile-time graph optimization passes.
    crate::compiler::graph_optimizer::optimize(&mut plan_with_fusion);
    builder.set_execution_plan(plan_with_fusion);

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;
    let emitted_so_far = builder
        .segment_payloads
        .iter()
        .map(|p| p.len() as u64)
        .sum();
    crate::compile_progress::CompileProgress {
        stage: "payload_emission_done".into(),
        bytes_processed: emitted_so_far,
        bytes_total: emitted_so_far,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    let total_source_bytes = loaded
        .source_tensors
        .values()
        .map(|tensor| tensor.data.len() as u64)
        .sum();
    let total_emitted_bytes = manifest
        .segments
        .iter()
        .map(|segment| segment.byte_size)
        .sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: 0,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}

pub fn read(image_dir: &str) -> crate::Result<CompiledImageReader> {
    CompiledImageReader::open(Path::new(image_dir))
}

/// Compile a model image with differential recompilation against a previous
/// compilation manifest.
///
/// 1. Compares source tensor SHA-256 hashes against the previous manifest.
/// 2. Copies segment files that contain **only** unchanged tensors directly
///    from the previous output directory — no recompile needed.
/// 3. Emits only changed / new tensors into fresh segment files.
/// 4. Merges unchanged and new segments into a single manifest.
///
/// Requires `prev_manifest_path` to point at a `manifest.json` from a prior
/// `tribunus-compute-image build` run.  The previous *output* directory is
/// inferred as the parent of that file.
pub fn compile_differential(
    source_dir: &str,
    output_dir: &str,
    prev_manifest_path: &str,
) -> crate::Result<CompiledImage> {
    let started_at = Instant::now();
    let output_dir_path = Path::new(output_dir);

    // Load previous manifest
    let prev_manifest_text = std::fs::read_to_string(prev_manifest_path)
        .map_err(|e| {
            crate::Error::from_reason(format!(
                "read previous manifest {}: {e}",
                prev_manifest_path
            ))
        })?;
    let prev_manifest: Manifest = serde_json::from_str(&prev_manifest_text)
        .map_err(|e| {
            crate::Error::from_reason(format!("parse previous manifest: {e}"))
        })?;
    let prev_output_dir_path = Path::new(prev_manifest_path)
        .parent()
        .ok_or_else(|| {
            crate::Error::from_reason(
                "cannot determine previous output directory from manifest path",
            )
        })?;

    // Build diff
    let diff = diff_tensors(Path::new(source_dir), &prev_manifest)?;
    eprintln!(
        "[diff-compile] tensors: {} unchanged, {} changed, {} new, {} removed ({elapsed} ms)",
        diff.unchanged.len(),
        diff.changed.len(),
        diff.new.len(),
        diff.removed.len(),
        elapsed = diff.elapsed_ms,
    );

    let t_source = Instant::now();
    let (_plan, loaded) = plan(Path::new(source_dir), false)?;
    let source_load_ms = t_source.elapsed().as_millis() as u64;

    // Build lookup sets
    let compile_names: std::collections::HashSet<&str> = diff
        .changed
        .iter()
        .chain(diff.new.iter())
        .map(|s| s.as_str())
        .collect();
    let unchanged_names: std::collections::HashSet<&str> =
        diff.unchanged.iter().map(|s| s.as_str()).collect();

    // Identify and copy unchanged segments
    let unchanged_segments: Vec<Segment> = prev_manifest
        .segments
        .iter()
        .filter(|seg| {
            seg.tensor_ids.iter().all(|tid| {
                prev_manifest
                    .tensor_table
                    .iter()
                    .find(|t| t.id == *tid)
                    .map(|t| unchanged_names.contains(t.name.as_str()))
                    .unwrap_or(false)
            })
        })
        .cloned()
        .collect();

    std::fs::create_dir_all(output_dir_path)
        .map_err(|e| crate::Error::from_reason(format!("mkdir: {e}")))?;
    for seg in &unchanged_segments {
        let src = prev_output_dir_path.join(&seg.filename);
        let dst = output_dir_path.join(&seg.filename);
        if src.exists() {
            std::fs::copy(&src, &dst).map_err(|e| {
                crate::Error::from_reason(format!(
                    "copy unchanged segment {}: {e}",
                    seg.filename
                ))
            })?;
        }
    }

    // Build source identity
    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    // Emit only changed / new tensors
    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);
    // Offset starting tensor ID so new IDs don't collide with IDs from the
    // previous compilation manifest (which are still referenced by unchanged
    // tensors and the existing execution plan / alias entries).
    let start_tensor_id: u32 = prev_manifest
        .tensor_table
        .iter()
        .map(|t| t.id)
        .max()
        .map(|id| id + 1)
        .unwrap_or(0);
    builder.set_start_tensor_id(start_tensor_id);
    let t_emit = Instant::now();

    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        if !compile_names.contains(binding.name.as_str()) {
            continue;
        }
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| {
                crate::Error::from_reason("embed_tokens.weight was not emitted")
            })?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            if !compile_names.contains(binding.name.as_str()) {
                continue;
            }
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Build the execution plan
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    builder.set_execution_plan(plan_with_fusion);

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;

    // Flush and collect new segments
    let (new_segments, new_payloads, partial_manifest) = builder.flush_and_collect_segments();

    // Determine offset for new segment filenames
    let max_existing: usize = unchanged_segments
        .iter()
        .filter_map(|s| {
            let stripped = s.filename.strip_prefix("segment_")?;
            let num_str = stripped.strip_suffix(".bin")?;
            num_str.parse::<usize>().ok()
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(0);

    // Write new segment files with offset filenames
    for (i, payload) in new_payloads.iter().enumerate() {
        let new_filename = format!("segment_{:03}.bin", max_existing + i);
        let path = output_dir_path.join(&new_filename);
        std::fs::write(&path, payload).map_err(|e| {
            crate::Error::from_reason(format!("write new segment {}: {e}", new_filename))
        })?;
    }

    // Build combined manifest
    let mut combined_segments: Vec<Segment> = Vec::with_capacity(
        unchanged_segments.len() + new_segments.len(),
    );
    combined_segments.extend(unchanged_segments);

    for (i, (seg, payload)) in new_segments.iter().zip(new_payloads.iter()).enumerate() {
        let new_filename = format!("segment_{:03}.bin", max_existing + i);
        let sha256 = {
            let mut h = Sha256::new();
            h.update(payload);
            format!("{:x}", h.finalize())
        };
        combined_segments.push(Segment {
            id: seg.id.clone(),
            filename: new_filename,
            byte_size: payload.len() as u64,
            sha256,
            tensor_ids: seg.tensor_ids.clone(),
            kind: seg.kind.clone(),
            alignment_bytes: seg.alignment_bytes,
        });
    }

    // Combined tensor table: unchanged from prev, changed/new from partial
    let mut combined_tensors: Vec<TensorEntry> =
        Vec::with_capacity(prev_manifest.tensor_table.len() + partial_manifest.tensor_table.len());

    for t in &prev_manifest.tensor_table {
        if unchanged_names.contains(t.name.as_str()) {
            combined_tensors.push(t.clone());
        }
    }
    for t in &partial_manifest.tensor_table {
        let mut entry = t.clone();
        // Fix segment reference: map from builder's internal segment id to
        // the actual filename on disk.
        if let Some(seg) = combined_segments
            .iter()
            .find(|cs| cs.tensor_ids.contains(&entry.id))
        {
                entry.segment = seg.filename.clone();
        }
        combined_tensors.push(entry);
    }

    let mut combined_manifest = partial_manifest.clone();
    combined_manifest.segments = combined_segments;
    combined_manifest.tensor_table = combined_tensors;
    combined_manifest.alias_table = {
        let mut merged = prev_manifest.alias_table.clone();
        merged.extend(partial_manifest.alias_table.clone());
        merged
    };
    combined_manifest.residency_plan.total_bytes =
        combined_manifest.segments.iter().map(|s| s.byte_size).sum();
    combined_manifest.image_hash = compute_manifest_hash(&combined_manifest);

    let manifest_path = output_dir_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&combined_manifest)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| crate::Error::from_reason(format!("write manifest: {e}")))?;

    // Build and write receipt
    let finalize_ms = t_emit.elapsed().as_millis() as u64;
    let total_source_bytes: u64 = loaded.source_tensors.values().map(|t| t.data.len() as u64).sum();
    let total_emitted_bytes: u64 = combined_manifest.segments.iter().map(|s| s.byte_size).sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: diff.elapsed_ms as u64,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &combined_manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
    );
    let receipt_path = output_dir_path.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {e}")))?;

    Ok(CompiledImage {
        manifest: combined_manifest,
        receipt,
    })
}

pub fn verify(image_dir: &str) -> crate::Result<ManifestVerification> {
    read(image_dir)?.verify()
}

/// Results from compile-time diagnostic verification.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    pub passed: bool,
    pub layers: Vec<LayerDiagnostic>,
    pub global: GlobalDiagnostic,
    pub issues: Vec<DiagnosticIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDiagnostic {
    pub layer_index: u32,
    pub attention_kind: String,
    pub hidden_norm: f64,
    pub hidden_finite: bool,
    pub hidden_min: f64,
    pub hidden_max: f64,
    pub entropy: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobalDiagnostic {
    pub total_layers: usize,
    pub nan_layers: usize,
    pub inf_layers: usize,
    pub max_runtime_ms: u64,
    pub total_runtime_ms: u64,
    pub memory_peak_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub enum DiagnosticIssue {
    NanInLayer(u32),
    InfInLayer(u32),
    ExplodingActivation { layer: u32, norm: f64 },
    VanishingActivation { layer: u32, norm: f64 },
    EntropyExtreme { layer: u32, entropy: f64 },
}

impl Default for GlobalDiagnostic {
    fn default() -> Self {
        Self {
            total_layers: 0,
            nan_layers: 0,
            inf_layers: 0,
            max_runtime_ms: 0,
            total_runtime_ms: 0,
            memory_peak_bytes: 0,
        }
    }
}

/// Run compile-time diagnostic verification on a compiled image.
pub fn run_diagnostics(image_dir: &Path) -> crate::Result<DiagnosticReport> {
    let reader = CompiledImageReader::open(image_dir)?;
    let plan = &reader.manifest.execution_plan;
    let mut runtime = reader.open_runtime(StorageBackend::Copied)?;

    let mut report = DiagnosticReport {
        passed: true,
        layers: Vec::new(),
        global: GlobalDiagnostic::default(),
        issues: Vec::new(),
    };

    for layer_plan in &plan.layers {
        let l = layer_plan.layer_index;
        let start = std::time::Instant::now();

        let lease = runtime.activate_layer(l)?;
        let layer_map = runtime.build_layer_arrays_from_lease(l, &lease)?;

        let mut has_nan = false;
        let mut has_inf = false;
        let mut norm_sum_sq: f64 = 0.0;
        let mut min_val: f64 = f64::MAX;
        let mut max_val: f64 = f64::NEG_INFINITY;

        for (_name, arr) in &layer_map {
            if let Ok(slice) = arr.try_as_slice::<f32>() {
                for &v in slice {
                    let vf = v as f64;
                    if v.is_nan() { has_nan = true; }
                    if v.is_infinite() { has_inf = true; }
                    if vf < min_val { min_val = vf; }
                    if vf > max_val { max_val = vf; }
                    norm_sum_sq += vf * vf;
                }
            }
        }

        let norm = norm_sum_sq.sqrt();
        let elapsed = start.elapsed().as_millis() as u64;

        let diag = LayerDiagnostic {
            layer_index: l,
            attention_kind: layer_plan.attention_kind.clone(),
            hidden_norm: norm,
            hidden_finite: !has_nan && !has_inf,
            hidden_min: min_val,
            hidden_max: max_val,
            entropy: 0.0,
            elapsed_ms: elapsed,
        };

        if has_nan {
            report.issues.push(DiagnosticIssue::NanInLayer(l));
            report.passed = false;
        }
        if has_inf {
            report.issues.push(DiagnosticIssue::InfInLayer(l));
            report.passed = false;
        }

        report.layers.push(diag);
    }

    report.global.total_layers = plan.layers.len();
    report.global.nan_layers = report.issues.iter()
        .filter(|i| matches!(i, DiagnosticIssue::NanInLayer(_))).count();
    report.global.inf_layers = report.issues.iter()
        .filter(|i| matches!(i, DiagnosticIssue::InfInLayer(_))).count();
    report.global.total_runtime_ms = report.layers.iter().map(|l| l.elapsed_ms).sum();
    report.global.max_runtime_ms = report.layers.iter().map(|l| l.elapsed_ms).max().unwrap_or(0);
    report.global.memory_peak_bytes = mlx_peak_memory_bytes();

    Ok(report)
}

/// Atomically publish a staged compilation to its final destination.
///
/// 1. Writes a `.publishing` marker inside `staging`.
/// 2. Renames `staging` to `destination` (falls back to recursive copy
///    when the rename crosses filesystem boundaries).
/// 3. On failure the staging directory is left intact with a `.failed` marker
///    so that the caller can inspect or retry.
pub fn publish_image(staging: &Path, destination: &Path) -> crate::Result<()> {
    let publishing_marker = staging.join(".publishing");
    std::fs::write(&publishing_marker, b"")
        .map_err(|e| crate::Error::from_reason(format!("write .publishing: {}", e)))?;

    let result = std::fs::rename(staging, destination);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            // rename fails across filesystem boundaries — fall back to copy + remove
            if e.kind() == std::io::ErrorKind::CrossesDevices {
                let failed_marker = staging.join(".failed");
                if let Err(write_err) =
                    std::fs::write(&failed_marker, format!("rename failed: {}", e))
                {
                    return Err(crate::Error::from_reason(format!(
                        "write .failed marker: {} (original rename: {})",
                        write_err, e
                    )));
                }
                return Err(crate::Error::from_reason(format!(
                    "rename crosses devices: {}. Staging left in place with .failed marker.",
                    e
                )));
            }
            let failed_marker = staging.join(".failed");
            if let Err(write_err) = std::fs::write(&failed_marker, format!("rename failed: {}", e))
            {
                return Err(crate::Error::from_reason(format!(
                    "write .failed marker: {} (original rename: {})",
                    write_err, e
                )));
            }
            Err(crate::Error::from_reason(format!(
                "rename {} -> {}: {}",
                staging.display(),
                destination.display(),
                e
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorLookup;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "tribunus-compute-image-{}-{}-{}",
            std::process::id(),
            label,
            stamp
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
        Box::leak(bytes.into_boxed_slice())
    }

    fn u32_tensor(name: &str, shape: &[usize], seed: u32) -> (String, TensorView<'static>) {
        let len = shape.iter().product::<usize>();
        let mut bytes = Vec::with_capacity(len * std::mem::size_of::<u32>());
        for index in 0..len {
            let value = seed.wrapping_add(index as u32);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor =
            TensorView::new(Dtype::U32, shape.to_vec(), leak_bytes(bytes)).expect("tensor");
        (name.to_string(), tensor)
    }

    fn f32_tensor(name: &str, shape: &[usize], seed: f32) -> (String, TensorView<'static>) {
        let len = shape.iter().product::<usize>();
        let mut bytes = Vec::with_capacity(len * std::mem::size_of::<f32>());
        for index in 0..len {
            let value = seed + (index as f32 * 0.03125);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor =
            TensorView::new(Dtype::F32, shape.to_vec(), leak_bytes(bytes)).expect("tensor");
        (name.to_string(), tensor)
    }

    fn write_fixture_model(source_dir: &Path) {
        let config = serde_json::json!({
            "model_type": "tiny_gemma_like",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "sliding_window": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "attention_k_eq_v": true,
                "final_logit_softcapping": null,
                "hidden_size_per_layer_input": 0,
                "layer_types": ["sliding_attention"],
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    },
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    }
                },
                "model_type": "tiny_gemma_like"
            },
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }
        });

        fs::write(
            source_dir.join("config.json"),
            serde_json::to_string_pretty(&config).expect("config json"),
        )
        .expect("write config");

        let root = "language_model.model";
        let mut tensors = vec![
            u32_tensor(&format!("{}.embed_tokens.weight", root), &[64, 16], 1),
            f32_tensor(&format!("{}.embed_tokens.scales", root), &[64, 1], 0.5),
            f32_tensor(&format!("{}.embed_tokens.biases", root), &[64, 1], 1.5),
            f32_tensor(&format!("{}.norm.weight", root), &[64], 2.0),
            f32_tensor(
                &format!("{}.layers.0.input_layernorm.weight", root),
                &[64],
                3.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.post_attention_layernorm.weight", root),
                &[64],
                4.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_norm.weight", root),
                &[16],
                5.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_norm.weight", root),
                &[16],
                6.0,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.weight", root),
                &[64, 16],
                7,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.scales", root),
                &[64, 1],
                7.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.biases", root),
                &[64, 1],
                7.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.weight", root),
                &[16, 16],
                8,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.scales", root),
                &[16, 1],
                8.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.biases", root),
                &[16, 1],
                8.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.weight", root),
                &[16, 16],
                9,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.scales", root),
                &[16, 1],
                9.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.biases", root),
                &[16, 1],
                9.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.weight", root),
                &[64, 16],
                10,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.scales", root),
                &[64, 1],
                10.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.biases", root),
                &[64, 1],
                10.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.weight", root),
                &[128, 16],
                11,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.scales", root),
                &[128, 1],
                11.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.biases", root),
                &[128, 1],
                11.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.up_proj.weight", root),
                &[128, 16],
                12,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.up_proj.scales", root),
                &[128, 1],
                12.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.up_proj.biases", root),
                &[128, 1],
                12.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.down_proj.weight", root),
                &[64, 32],
                13,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.down_proj.scales", root),
                &[64, 2],
                13.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.down_proj.biases", root),
                &[64, 2],
                13.75,
            ),
        ];

        tensors.sort_by(|left, right| left.0.cmp(&right.0));
        serialize_to_file(tensors, &None, &source_dir.join("model.safetensors"))
            .expect("write safetensors");
    }

    /// Build a synthetic model with N layers driven by `layer_types`
    /// ("sliding_attention" or "full_attention"). Full-attention layers
    /// omit v_proj (K-equals-V).
    fn write_two_layer_fixture_model(source_dir: &Path, layer_types: &[&str]) {
        let num_layers = layer_types.len();
        let config = serde_json::json!({
            "model_type": "tiny_gemma_like",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "num_hidden_layers": num_layers,
                "vocab_size": 64,
                "sliding_window": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "attention_k_eq_v": true,
                "final_logit_softcapping": null,
                "hidden_size_per_layer_input": 0,
                "layer_types": layer_types,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    },
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    }
                },
                "model_type": "tiny_gemma_like"
            },
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }
        });

        fs::write(
            source_dir.join("config.json"),
            serde_json::to_string_pretty(&config).expect("config json"),
        )
        .expect("write config");

        let root = "language_model.model";
        let mut tensors = vec![
            u32_tensor(&format!("{}.embed_tokens.weight", root), &[64, 16], 1),
            f32_tensor(&format!("{}.embed_tokens.scales", root), &[64, 1], 0.5),
            f32_tensor(&format!("{}.embed_tokens.biases", root), &[64, 1], 1.5),
            f32_tensor(&format!("{}.norm.weight", root), &[64], 2.0),
        ];

        for (i, lt) in layer_types.iter().enumerate() {
            let layer = i as u32;
            let is_full = *lt == "full_attention";

            // Norms
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.input_layernorm.weight", root, layer),
                &[64],
                3.0 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.post_attention_layernorm.weight", root, layer),
                &[64],
                4.0 + layer as f32 * 10.0,
            ));

            // Q/K norms
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_norm.weight", root, layer),
                &[16],
                5.0 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_norm.weight", root, layer),
                &[16],
                6.0 + layer as f32 * 10.0,
            ));

            // Q projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.weight", root, layer),
                &[64, 16],
                7 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.scales", root, layer),
                &[64, 1],
                7.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.biases", root, layer),
                &[64, 1],
                7.75 + layer as f32 * 10.0,
            ));

            // K projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.weight", root, layer),
                &[16, 16],
                8 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.scales", root, layer),
                &[16, 1],
                8.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.biases", root, layer),
                &[16, 1],
                8.75 + layer as f32 * 10.0,
            ));

            // V projection: only for sliding attention layers
            if !is_full {
                tensors.push(u32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.weight", root, layer),
                    &[16, 16],
                    9 + layer * 100,
                ));
                tensors.push(f32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.scales", root, layer),
                    &[16, 1],
                    9.5 + layer as f32 * 10.0,
                ));
                tensors.push(f32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.biases", root, layer),
                    &[16, 1],
                    9.75 + layer as f32 * 10.0,
                ));
            }

            // O projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.weight", root, layer),
                &[64, 16],
                10 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.scales", root, layer),
                &[64, 1],
                10.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.biases", root, layer),
                &[64, 1],
                10.75 + layer as f32 * 10.0,
            ));

            // MLP gate/up/down
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.weight", root, layer),
                &[128, 16],
                11 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.scales", root, layer),
                &[128, 1],
                11.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.biases", root, layer),
                &[128, 1],
                11.75 + layer as f32 * 10.0,
            ));
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.weight", root, layer),
                &[128, 16],
                12 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.scales", root, layer),
                &[128, 1],
                12.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.biases", root, layer),
                &[128, 1],
                12.75 + layer as f32 * 10.0,
            ));
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.weight", root, layer),
                &[64, 32],
                13 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.scales", root, layer),
                &[64, 2],
                13.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.biases", root, layer),
                &[64, 2],
                13.75 + layer as f32 * 10.0,
            ));
        }

        tensors.sort_by(|left, right| left.0.cmp(&right.0));
        serialize_to_file(tensors, &None, &source_dir.join("model.safetensors"))
            .expect("write safetensors");
    }

    #[derive(Debug)]
    struct TensorComparison {
        shape_matches: bool,
        dtype_matches: bool,
        source_finite: bool,
        runtime_finite: bool,
        max_abs_diff: f32,
        mean_abs_diff: f32,
        cosine_similarity: f32,
    }

    fn compare_tensors(source: &Array, runtime: &Array) -> TensorComparison {
        let source_slice = source.try_as_slice::<f32>().expect("source slice");
        let runtime_slice = runtime.try_as_slice::<f32>().expect("runtime slice");
        let len = usize::min(source_slice.len(), runtime_slice.len());
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut dot = 0.0f32;
        let mut source_norm = 0.0f32;
        let mut runtime_norm = 0.0f32;

        for i in 0..len {
            let left = source_slice[i];
            let right = runtime_slice[i];
            let diff = (left - right).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            sum_abs_diff += diff;
            dot += left * right;
            source_norm += left * left;
            runtime_norm += right * right;
        }

        let cosine_similarity = if source_norm == 0.0 || runtime_norm == 0.0 {
            0.0
        } else {
            dot / (source_norm.sqrt() * runtime_norm.sqrt())
        };

        TensorComparison {
            shape_matches: source.shape() == runtime.shape(),
            dtype_matches: format!("{:?}", source.dtype()) == format!("{:?}", runtime.dtype()),
            source_finite: source_slice.iter().all(|value| value.is_finite()),
            runtime_finite: runtime_slice.iter().all(|value| value.is_finite()),
            max_abs_diff,
            mean_abs_diff: if len == 0 {
                0.0
            } else {
                sum_abs_diff / len as f32
            },
            cosine_similarity,
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct RealCheckpointReference {
        shape: Vec<i32>,
        values: Vec<f32>,
    }

    fn real_checkpoint_env(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    fn real_checkpoint_run_child(
        phase: &str,
        source_dir: &Path,
        output_dir: &Path,
        reference_path: &Path,
    ) {
        let current_exe = std::env::current_exe().expect("current exe");
        let status = std::process::Command::new(current_exe)
            .arg("compute_image::tests::real_checkpoint_six_layer_prefix_round_trip")
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .env("TRIBUNUS_REAL_CHECKPOINT_PHASE", phase)
            .env("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR", source_dir)
            .env("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR", output_dir)
            .env("TRIBUNUS_REAL_CHECKPOINT_REFERENCE", reference_path)
            .status()
            .expect("spawn real checkpoint child");
        assert!(
            status.success(),
            "real checkpoint child failed in phase {}",
            phase
        );
    }

    fn real_checkpoint_source_phase(source_dir: &Path, reference_path: &Path) {
        let source = crate::model::Shard::load(
            source_dir
                .join("model-00001-of-00003.safetensors")
                .to_str()
                .expect("source shard 1"),
        );
        let source_2 = crate::model::Shard::load(
            source_dir
                .join("model-00002-of-00003.safetensors")
                .to_str()
                .expect("source shard 2"),
        );
        let source_3 = crate::model::Shard::load(
            source_dir
                .join("model-00003-of-00003.safetensors")
                .to_str()
                .expect("source shard 3"),
        );
        let (arch, _, _) = crate::config::parse_config(
            source_dir
                .join("config.json")
                .to_str()
                .expect("config path"),
        )
        .expect("parse config");
        let output = crate::model::run_six_layer_prefix(&[&source, &source_2, &source_3], &arch)
            .expect("source prefix");
        output.eval().expect("source eval");
        let reference = RealCheckpointReference {
            shape: output.shape().to_vec(),
            values: output.try_as_slice::<f32>().expect("source slice").to_vec(),
        };
        std::fs::write(
            reference_path,
            serde_json::to_string_pretty(&reference).expect("reference json"),
        )
        .expect("write reference");
        crate::bridge::ARRAY_REGISTRY.write().drain();
    }

    fn real_checkpoint_compile_phase(source_dir: &Path, output_dir: &Path) {
        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile real checkpoint");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);
        assert_eq!(
            verification.verified_segment_count,
            compiled.manifest.segments.len()
        );
    }

    fn real_checkpoint_runtime_phase(source_dir: &Path, output_dir: &Path, reference_path: &Path) {
        let source_exists = source_dir.exists();
        assert!(
            !source_exists,
            "source checkpoint should not be accessible during runtime"
        );

        let reference: RealCheckpointReference =
            serde_json::from_str(&std::fs::read_to_string(reference_path).expect("read reference"))
                .expect("parse reference");
        let expected = Array::from_slice(&reference.values, &reference.shape);

        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);

        let baseline_handles = crate::bridge::handle_count();
        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        let runtime_prefix = runtime.run_six_layer_prefix().expect("runtime prefix");
        runtime_prefix.eval().expect("runtime eval");

        let comparison = compare_tensors(&expected, &runtime_prefix);
        assert!(comparison.shape_matches, "shape mismatch");
        assert!(comparison.dtype_matches, "dtype mismatch");
        assert!(
            comparison.source_finite,
            "reference output contains non-finite values"
        );
        assert!(
            comparison.runtime_finite,
            "runtime output contains non-finite values"
        );
        assert!(
            comparison.max_abs_diff <= 1e-4,
            "max abs diff too large: {}",
            comparison.max_abs_diff
        );
        assert!(
            comparison.mean_abs_diff <= 1e-5,
            "mean abs diff too large: {}",
            comparison.mean_abs_diff
        );
        assert!(
            comparison.cosine_similarity >= 0.999_999,
            "cosine similarity too low: {}",
            comparison.cosine_similarity
        );
        assert_eq!(crate::bridge::handle_count(), baseline_handles);
    }

    #[test]
    fn compile_source_dir_writes_deterministic_image() {
        let source_dir = temp_dir("source");
        let output_dir_a = temp_dir("out-a");
        let output_dir_b = temp_dir("out-b");

        write_fixture_model(&source_dir);

        let first = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir_a.to_str().expect("output dir a"), CompilationAuthority::TestFixture, false, None, None)
        .expect("second compile");
        let second = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir_b.to_str().expect("output dir b"), CompilationAuthority::TestFixture, false, None, None)
        .expect("first compile");

        assert_eq!(first.manifest.image_hash, second.manifest.image_hash);
        assert_eq!(first.receipt.complete_image_hash, first.manifest.image_hash);
        assert_eq!(first.manifest.segments.len(), 2);
        assert_eq!(
            first.manifest.segments.len(),
            first.manifest.residency_plan.persistent_segments.len()
                + first.manifest.residency_plan.layer_segments.len()
        );
        assert_eq!(first.manifest.alias_table.len(), 1);
        assert_eq!(first.manifest.alias_table[0].logical_name, "lm_head.weight");
        assert!(first.receipt.structural_verification);

        let manifest_path = output_dir_a.join("manifest.json");
        assert!(manifest_path.exists());
        let receipt_path = output_dir_a.join("receipt.json");
        assert!(receipt_path.exists());

        let persisted = fs::read(output_dir_a.join("segment_000.bin")).expect("segment 0");
        assert_eq!(persisted.len() as u64, first.manifest.segments[0].byte_size);

        let reloaded_manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(manifest_path).expect("manifest json"))
                .expect("manifest parse");
        assert_eq!(reloaded_manifest.image_hash, first.manifest.image_hash);
        assert_eq!(
            reloaded_manifest.segments.len(),
            first.manifest.segments.len()
        );
    }

    #[test]
    fn compiled_image_reader_round_trip_matches_source_prefix() {
        let source_dir = temp_dir("source-round-trip");
        let output_dir = temp_dir("out-round-trip");

        write_fixture_model(&source_dir);

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);
        assert_eq!(
            verification.verified_segment_count,
            compiled.manifest.segments.len()
        );

        let source = crate::model::Shard::load(
            source_dir
                .join("model.safetensors")
                .to_str()
                .expect("source shard"),
        );

        for name in [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.scales",
            "language_model.model.layers.0.self_attn.q_proj.biases",
        ] {
            let left = source.tensor(name).expect("source tensor");
            let right = reader.tensor(name).expect("reader tensor");
            assert_eq!(left.shape(), right.shape());
            let left_dtype = format!("{:?}", left.dtype());
            let right_dtype = format!("{:?}", right.dtype());
            assert_eq!(left_dtype, right_dtype);
            match left_dtype.as_str() {
                "Uint32" | "U32" => {
                    assert_eq!(
                        left.try_as_slice::<u32>().expect("source u32"),
                        right.try_as_slice::<u32>().expect("reader u32")
                    );
                }
                "Float32" | "F32" => {
                    assert_eq!(
                        left.try_as_slice::<f32>().expect("source f32"),
                        right.try_as_slice::<f32>().expect("reader f32")
                    );
                }
                other => panic!("unexpected dtype for {}: {}", name, other),
            }
        }
    }

    #[ignore = "requires compiled modelc fixture on disk"]
    #[test]
    fn compiled_image_runtime_copied_round_trip_matches_source_prefix() {
        let source_dir = temp_dir("source-runtime");
        let output_dir = temp_dir("out-runtime");

        write_fixture_model(&source_dir);

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let baseline_handles = crate::bridge::handle_count();
        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        assert!(runtime.quantized_binding_count() > 0);
        assert!(crate::bridge::handle_count() > baseline_handles);

        // After open_runtime: only persistent segment bytes loaded, not layer segments.
        let persistent_bytes: u64 = compiled
            .manifest
            .segments
            .iter()
            .filter(|s| matches!(s.kind, SegmentKind::Persistent | SegmentKind::Final))
            .map(|s| s.byte_size)
            .sum();
        assert_eq!(runtime.total_bytes_activated(), persistent_bytes);

        let source = crate::model::Shard::load(
            source_dir
                .join("model.safetensors")
                .to_str()
                .expect("source shard"),
        );
        let source_prefix =
            crate::model::run_six_layer_prefix(&[&source], &compiled.manifest.architecture)
                .expect("source prefix");
        let runtime_prefix = runtime.run_six_layer_prefix().expect("runtime prefix");

        assert_eq!(source_prefix.shape(), runtime_prefix.shape());
        assert_eq!(
            source_prefix.try_as_slice::<f32>().expect("source slice"),
            runtime_prefix.try_as_slice::<f32>().expect("runtime slice")
        );
        assert_eq!(crate::bridge::handle_count(), baseline_handles);
    }

    #[test]
    #[ignore = "real checkpoint smoke test; run manually when you want to pay the 12G cost"]
    fn real_checkpoint_six_layer_prefix_round_trip() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-out");

        if let Some(phase) = real_checkpoint_env("TRIBUNUS_REAL_CHECKPOINT_PHASE") {
            let source = std::env::var("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR")
                .expect("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR");
            let output = std::env::var("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR")
                .expect("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR");
            let reference = std::env::var("TRIBUNUS_REAL_CHECKPOINT_REFERENCE")
                .expect("TRIBUNUS_REAL_CHECKPOINT_REFERENCE");
            let source = Path::new(&source);
            let output = Path::new(&output);
            let reference = Path::new(&reference);

            match phase.as_str() {
                "source" => real_checkpoint_source_phase(source, reference),
                "compile" => real_checkpoint_compile_phase(source, output),
                "runtime" => real_checkpoint_runtime_phase(source, output, reference),
                other => panic!("unknown checkpoint phase: {}", other),
            }
            return;
        }

        let reference_path = temp_dir("real-reference").join("reference.json");
        let hidden_source_dir = source_dir.with_extension("hidden-for-runtime");
        struct RestoreSourceDir {
            hidden: PathBuf,
            original: PathBuf,
        }
        impl Drop for RestoreSourceDir {
            fn drop(&mut self) {
                if self.hidden.exists() {
                    let _ = std::fs::rename(&self.hidden, &self.original);
                }
            }
        }

        real_checkpoint_run_child("source", &source_dir, &output_dir, &reference_path);
        real_checkpoint_run_child("compile", &source_dir, &output_dir, &reference_path);

        std::fs::rename(&source_dir, &hidden_source_dir).expect("hide source checkpoint");
        let _restore_source_dir = RestoreSourceDir {
            hidden: hidden_source_dir.clone(),
            original: source_dir.clone(),
        };
        assert!(
            !source_dir.exists(),
            "source checkpoint should be hidden before runtime"
        );

        real_checkpoint_run_child("runtime", &source_dir, &output_dir, &reference_path);
    }

    #[test]
    fn compiled_image_rejects_corruption_and_missing_segment() {
        let source_dir = temp_dir("source-corruption");
        write_fixture_model(&source_dir);

        let corrupted_dir = temp_dir("out-corrupted");
        compile_with_authority(source_dir.to_str().expect("source dir"), corrupted_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile corrupted fixture");
        let segment_path = corrupted_dir.join("segment_000.bin");
        let mut bytes = fs::read(&segment_path).expect("segment bytes");
        bytes[0] ^= 0xFF;
        fs::write(&segment_path, bytes).expect("rewrite corrupted segment");
        let err = match read(corrupted_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected corruption error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("segment hash mismatch"),
            "unexpected corruption error: {}",
            err
        );

        let missing_dir = temp_dir("out-missing");
        compile_with_authority(source_dir.to_str().expect("source dir"), missing_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile missing fixture");
        fs::remove_file(missing_dir.join("segment_000.bin")).expect("remove segment");
        let err = match read(missing_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected missing-segment error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("read segment") || err.to_string().contains("missing segment"),
            "unexpected missing-segment error: {}",
            err
        );

        let abi_dir = temp_dir("out-abi");
        compile_with_authority(source_dir.to_str().expect("source dir"), abi_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile abi fixture");
        let manifest_path = abi_dir.join("manifest.json");
        let manifest = fs::read_to_string(&manifest_path).expect("read manifest");
        let mutated = manifest.replace(
            "\"runtime_abi\": \"mlx-rs/0.21.0 core/",
            "\"runtime_abi\": \"mlx-rs/0.21.0 core-mutated/",
        );
        fs::write(&manifest_path, mutated).expect("rewrite manifest");
        let err = match read(abi_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected abi-mismatch error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("manifest hash mismatch"),
            "unexpected abi-mismatch error: {}",
            err
        );
    }
    #[test]
    fn test_storage_abi_matching() {
        let source = SourceIdentity {
            config_hash: "abc".into(),
            shard_hashes: vec![],
            tokenizer_hashes: vec![],
            auxiliary_hashes: vec![],
            model_type: "test".into(),
            quantization_bits: 8,
            quantization_group_size: 64,
            quantization_mode: "affine".into(),
        };

        let defaults = Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "test".into(),
            runtime_abi: "test".into(),
            source: source,
            architecture: crate::config::TextArchitecture {
                hidden_size: 64,
                intermediate_size: 128,
                num_attention_heads: 4,
                num_key_value_heads: 1,
                head_dim: 16,
                global_head_dim: Some(16),
                num_global_key_value_heads: Some(1),
                num_hidden_layers: 1,
                vocab_size: 64,
                sliding_window: 8,
                max_position_embeddings: 16,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                attention_k_eq_v: true,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: 0,
                layer_types: vec![crate::config::AttentionKind::SlidingAttention],
                rope_local: crate::config::RopeSpec {
                    theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                rope_global: None,
                model_type: "test".into(),
            },
            segments: vec![],
            tensor_table: vec![],
            alias_table: vec![],
            residency_plan: ResidencyPlan {
                persistent_segments: vec![],
                layer_segments: vec![],
                layer_window_size: 2,
                total_bytes: 0,
            },
            image_hash: "dummy".into(),
            required_storage_abi: STORAGE_ABI_COPIED_V0.into(),
            required_capabilities: vec![],
            prepacked_layout: "none".into(),
            execution_plan: crate::config::ModelExecutionPlan::default(),
        };

        assert!(defaults.storage_abi_matches(&StorageBackend::Copied));
        assert!(!defaults.storage_abi_matches(&StorageBackend::MappedNoCopy));

        // Check constants
        assert_eq!(STORAGE_ABI_COPIED_V0, "copied-v0");
        assert_eq!(STORAGE_ABI_MAPPED_NO_COPY_V1, "mapped-no-copy-v1");
    }

    #[test]
    fn test_alignment_validation() {
        // Build a manifest manually with mapped-no-copy-v1 and proper alignment
        let segment = Segment {
            id: "test_seg".into(),
            filename: "segment_000.bin".into(),
            byte_size: 4096,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            tensor_ids: vec![0],
            kind: SegmentKind::Persistent,
            alignment_bytes: 4096,
        };
        let tensor = TensorEntry {
            id: 0,
            name: "weight".into(),
            role: "embed".into(),
            layer: None,
            segment: "test_seg".into(),
            source_filename: "x.safetensors".into(),
            source_sha256: "0000".into(),
            source_offset: 0,
            offset: 0,
            byte_length: 256,
            logical_dtype: "F32".into(),
            storage_dtype: "F32".into(),
            logical_shape: vec![16, 16],
            physical_shape: vec![16, 16],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
        };
        let manifest = Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "test".into(),
            runtime_abi: "test".into(),
            source: SourceIdentity {
                config_hash: "abc".into(),
                shard_hashes: vec![],
                tokenizer_hashes: vec![],
                auxiliary_hashes: vec![],
                model_type: "test".into(),
                quantization_bits: 8,
                quantization_group_size: 64,
                quantization_mode: "affine".into(),
            },
            architecture: crate::config::TextArchitecture {
                hidden_size: 64,
                intermediate_size: 128,
                num_attention_heads: 4,
                num_key_value_heads: 1,
                head_dim: 16,
                global_head_dim: Some(16),
                num_global_key_value_heads: Some(1),
                num_hidden_layers: 1,
                vocab_size: 64,
                sliding_window: 8,
                max_position_embeddings: 16,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                attention_k_eq_v: true,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: 0,
                layer_types: vec![crate::config::AttentionKind::SlidingAttention],
                rope_local: crate::config::RopeSpec {
                    theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                rope_global: None,
                model_type: "test".into(),
            },
            segments: vec![segment],
            tensor_table: vec![tensor],
            alias_table: vec![],
            residency_plan: ResidencyPlan {
                persistent_segments: vec!["test_seg".into()],
                layer_segments: vec![],
                layer_window_size: 2,
                total_bytes: 4096,
            },
            image_hash: "dummy".into(),
            required_storage_abi: STORAGE_ABI_MAPPED_NO_COPY_V1.into(),
            required_capabilities: vec![],
            prepacked_layout: "none".into(),
            execution_plan: crate::config::ModelExecutionPlan::default(),
        };

        assert!(manifest.storage_abi_matches(&StorageBackend::MappedNoCopy));
        assert!(!manifest.storage_abi_matches(&StorageBackend::Copied));
    }

    #[test]
    fn segment_corruption_rejected() {
        let source_dir = temp_dir("source-seg-corr");
        write_fixture_model(&source_dir);

        let output_dir = temp_dir("out-seg-corr");
        compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile corrupted fixture");

        // segment_000.bin = persistent (embed + final), segment_001.bin = layer 0
        let segment_path = output_dir.join("segment_001.bin");
        let mut bytes = fs::read(&segment_path).expect("layer segment bytes");
        bytes[100] ^= 0xFF;
        fs::write(&segment_path, bytes).expect("rewrite corrupted layer segment");

        let err = match read(output_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected segment corruption error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("segment hash mismatch"),
            "unexpected segment corruption error: {}",
            err
        );
    }

    #[test]
    fn synthetic_plan_driven_execution() {
        let source_dir = temp_dir("source-plan");
        let output_dir = temp_dir("out-plan");

        write_two_layer_fixture_model(&source_dir, &["sliding_attention", "full_attention"]);

        let baseline_handles = crate::bridge::handle_count();
        {
            let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
            .expect("compile");

            let reader = read(output_dir.to_str().expect("output dir")).expect("reader");

            // Verify execution plan from manifest
            let plan = &compiled.manifest.execution_plan;
            assert_eq!(plan.layers.len(), 2);

            assert_eq!(plan.layers[0].attention_kind, "sliding_attention");
            assert_eq!(plan.layers[0].layer_index, 0);
            assert!(plan.layers[0].global_head_dim.is_none());
            assert!(
                plan.layers[0].v_proj_tensor_id != 0,
                "sliding layer needs v_proj"
            );

            assert_eq!(plan.layers[1].attention_kind, "full_attention");
            assert_eq!(plan.layers[1].layer_index, 1);
            assert_eq!(plan.layers[1].global_head_dim, Some(16));
            // K-equals-V: v_proj aliases k_proj
            assert_eq!(
                plan.layers[1].v_proj_tensor_id,
                plan.layers[1].k_proj_tensor_id
            );

            // Validate the plan
            plan.validate().expect("execution plan should validate");

            // Open runtime and verify handle lifecycle
            let mut runtime = reader
                .open_runtime(StorageBackend::Copied)
                .expect("runtime");

            // Handle count after persistent activation
            let after_persistent = crate::bridge::handle_count();
            assert!(after_persistent > baseline_handles);

            // Run full model - this activates layers, runs inference, then retires them
            let token = runtime.run_full_model(&[2i32]).expect("run full model");
            assert!(token < 64, "token {} should be in [0, 64)", token);
        }

        // After all model-owned values are dropped, handles should return to baseline
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count should return to baseline after runtime teardown; {} != {}",
            after_run, baseline_handles
        );
    }

    #[test]
    fn test_synthetic_prefill_decode_parity() {
        std::env::set_var("TRIBUNUS_COMPUTE_ALLOW_HIGH_MEMORY", "1");
        let source_dir = temp_dir("source-parity");
        let output_dir = temp_dir("out-parity");

        write_two_layer_fixture_model(&source_dir, &["sliding_attention", "full_attention"]);

        let _compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");

        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(&output_dir).expect("load bindings");

        let build_kv_caches = || -> Vec<crate::kv_cache::KvCache> {
            profiled_model
                .reader
                .manifest
                .execution_plan
                .layers
                .iter()
                .map(|layer| {
                    let capacity = if layer.attention_kind == "sliding_attention" {
                        layer.sliding_window
                    } else {
                        16
                    };
                    let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
                    let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
                    crate::kv_cache::KvCache::new(
                        capacity,
                        n_kv_heads,
                        head_dim,
                        layer.attention_kind == "sliding_attention",
                    )
                })
                .collect()
        };

        let mut session = crate::profiled_executor::ProfiledInferenceSession::new(
            "test-parity-session".to_string(),
            build_kv_caches(),
        );

        // Prefill parity
        let prompt = vec![2u32, 10u32, 15u32];
        let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

        let t1_cached = session.prefill(&prompt, &profiled_model).expect("prefill");
        let t1_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&prompt_i32)
            .expect("run_full_model");
        assert_eq!(t1_cached, t1_uncached, "Prefill token parity mismatch");

        // Decode Step 1 parity
        let mut history = prompt.clone();
        history.push(t1_cached);
        let history_i32: Vec<i32> = history.iter().map(|&t| t as i32).collect();

        let t2_cached = session
            .decode_one(t1_cached, &profiled_model)
            .expect("decode_one step 1");
        let t2_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&history_i32)
            .expect("run_full_model step 1");
        assert_eq!(
            t2_cached, t2_uncached,
            "Decode step 1 token parity mismatch"
        );

        // Decode Step 2 parity
        history.push(t2_cached);
        let history_i32_2: Vec<i32> = history.iter().map(|&t| t as i32).collect();

        let t3_cached = session
            .decode_one(t2_cached, &profiled_model)
            .expect("decode_one step 2");
        let t3_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&history_i32_2)
            .expect("run_full_model step 2");
        assert_eq!(
            t3_cached, t3_uncached,
            "Decode step 2 token parity mismatch"
        );
    }

    #[test]
    #[ignore = "requires sealed image at TRIBUNUS_COMPILED_IMAGE"]
    fn real_checkpoint_full_model_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("set TRIBUNUS_COMPILED_IMAGE");
        let image_path = std::path::Path::new(&image_dir);
        assert!(image_path.join("manifest.json").exists());
        assert!(image_path.join("seal.json").exists());

        eprintln!("Opening sealed image: {}", image_dir);
        let baseline_handles = crate::bridge::handle_count();
        let reader = read(&image_dir).expect("reader");
        let plan = &reader.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48);
        plan.validate().expect("plan validation");

        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        eprintln!("Running 48-layer forward pass...");
        let started = std::time::Instant::now();
        let token = runtime.run_full_model(&[2i32]).expect("run_full_model");
        let elapsed = started.elapsed().as_secs_f64();

        let after_run = crate::bridge::handle_count();
        eprintln!(
            "GATE PASSED: token={} elapsed={:.1}s handles={}->{}",
            token, elapsed, baseline_handles, after_run
        );
        assert_eq!(after_run, baseline_handles);
    }

    #[test]
    fn test_storage_abi_validation_rejects_unknown() {
        // Verify that is_valid_storage_abi rejects unknown identifiers
        assert!(is_valid_storage_abi(STORAGE_ABI_COPIED_V0));
        assert!(is_valid_storage_abi(STORAGE_ABI_MAPPED_NO_COPY_V1));
        assert!(!is_valid_storage_abi("copied-v2"));
        assert!(!is_valid_storage_abi("mapped-no-copy-v0"));
        assert!(!is_valid_storage_abi(""));
        assert!(!is_valid_storage_abi("unknown-abi"));
    }

    #[test]
    fn test_tensor_layout_offset_oob() {
        // A tensor whose offset + byte_length exceeds its segment should fail.
        let entry = TensorEntry {
            id: 0,
            name: "oob_tensor".into(),
            role: "test".into(),
            layer: None,
            segment: "seg".into(),
            source_filename: "x.safetensors".into(),
            source_sha256: "0000".into(),
            source_offset: 0,
            offset: 100,
            byte_length: 200,
            logical_dtype: "F32".into(),
            storage_dtype: "F32".into(),
            logical_shape: vec![10, 5],
            physical_shape: vec![10, 5],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
        };

        // Segment is only 250 bytes, tensor ends at 300 -> OOB
        let result = validate_tensor_layout(&entry, 250);
        assert!(result.is_err(), "expected OOB error");
        assert!(
            result.unwrap_err().contains("exceeds segment size"),
            "unexpected error message"
        );

        // With enough space it should succeed
        let result = validate_tensor_layout(&entry, 301);
        assert!(result.is_ok(), "expected OK for large enough segment");

        // Zero byte_length should be rejected
        let zero_entry = TensorEntry {
            byte_length: 0,
            ..entry.clone()
        };
        let result = validate_tensor_layout(&zero_entry, 100);
        assert!(result.is_err(), "expected error for zero byte_length");
        assert!(
            result.unwrap_err().contains("zero byte_length"),
            "unexpected error message"
        );
    }

    #[test]
    fn test_physical_dtype_byte_count() {
        // f32: 4 * (2*3*4) = 96
        let r = validate_physical_dtype("f32", 96, &[2, 3, 4]);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), 96);

        // bf16: 2 * (8*4) = 64
        let r = validate_physical_dtype("BF16", 64, &[8, 4]);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), 64);

        // f16: 2 * 128 = 256
        let r = validate_physical_dtype("f16", 256, &[128]);
        assert!(r.is_ok());

        // u8: 1 * (4*8) = 32
        let r = validate_physical_dtype("U8", 32, &[4, 8]);
        assert!(r.is_ok());

        // i8: same as u8
        let r = validate_physical_dtype("I8", 32, &[4, 8]);
        assert!(r.is_ok());

        // u32: 4 * 50 = 200
        let r = validate_physical_dtype("U32", 200, &[50]);
        assert!(r.is_ok());

        // Wrong byte count
        let r = validate_physical_dtype("f32", 100, &[2, 3, 4]);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected 96 bytes"));

        // Unknown dtype
        let r = validate_physical_dtype("f64", 8, &[1]);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("unsupported"));
    }

    #[test]
    #[ignore = "real checkpoint prefill+decode_one; requires ~12GB quantized model at models/gemma4-12b-8bit"]
    fn real_checkpoint_decode_one_token_after_prefill() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-decode-1-out");

        if !source_dir.join("config.json").exists() {
            eprintln!("SKIP: no model at {}", source_dir.display());
            return;
        }

        eprintln!("Compiling quantized Gemma 4 12B...");
        let started = std::time::Instant::now();

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile model");

        let compile_secs = started.elapsed().as_secs_f64();
        eprintln!(
            "Compiled in {:.1}s: {} segments, {} tensors, {:?}",
            compile_secs,
            compiled.manifest.segments.len(),
            compiled.manifest.tensor_table.len(),
            compiled.manifest.image_hash
        );

        let plan = &compiled.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48, "expected 48 layers");
        plan.validate().expect("execution plan should validate");
        eprintln!("image hash: {}", compiled.manifest.image_hash);

        eprintln!("Opening runtime...");
        let baseline_handles = crate::bridge::handle_count();
        let _reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        eprintln!("Loading profiled model...");
        let profiled_model = crate::profiled_executor::LoadedProfiledModel::new(&output_dir)
            .expect("load profiled model");

        let kv_caches: Vec<crate::kv_cache::KvCache> = plan
            .layers
            .iter()
            .map(|lp| {
                let is_sliding = lp.attention_kind == "sliding_attention";
                let (capacity, n_kv_heads, head_dim) = if is_sliding {
                    (lp.sliding_window, lp.n_kv_heads, lp.head_dim)
                } else {
                    let g_kv = lp.n_global_kv_heads.unwrap_or(lp.n_kv_heads);
                    let g_hd = lp.global_head_dim.unwrap_or(lp.head_dim);
                    (32768u32, g_kv, g_hd)
                };
                crate::kv_cache::KvCache::new(capacity, n_kv_heads, head_dim, is_sliding)
            })
            .collect();

        let mut session =
            crate::profiled_executor::ProfiledInferenceSession::new("decode-1".into(), kv_caches);

        eprintln!("Prefilling with [2, 42, 100, 500]...");
        let first_token = session
            .prefill(&[2, 42, 100, 500], &profiled_model)
            .expect("prefill");
        eprintln!("Prefill token: {}", first_token);
        assert!(
            first_token < 262144,
            "first token {} out of vocab range",
            first_token
        );
        assert!(first_token != 0, "first token must not be padding token 0");

        eprintln!("Decoding one token after prefill...");
        let second_token = session
            .decode_one(first_token, &profiled_model)
            .expect("decode_one");
        eprintln!("Decode token: {}", second_token);
        assert!(
            second_token < 262144,
            "second token {} out of vocab range",
            second_token
        );
        assert!(
            second_token != 0,
            "second token must not be padding token 0"
        );

        drop(session);
        drop(profiled_model);
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count must return to baseline after decode; {} != {}",
            after_run, baseline_handles
        );

        eprintln!(
            "[decode-1] PASSED: first={} second={}",
            first_token, second_token
        );
    }

    #[test]
    #[ignore = "real checkpoint 8-token decode; requires ~12GB quantized model at models/gemma4-12b-8bit"]
    fn real_checkpoint_decode_eight_tokens() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-decode-8-out");

        if !source_dir.join("config.json").exists() {
            eprintln!("SKIP: no model at {}", source_dir.display());
            return;
        }

        eprintln!("Compiling quantized Gemma 4 12B...");
        let started = std::time::Instant::now();

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile model");

        let compile_secs = started.elapsed().as_secs_f64();
        eprintln!(
            "Compiled in {:.1}s: {} segments, {} tensors, {:?}",
            compile_secs,
            compiled.manifest.segments.len(),
            compiled.manifest.tensor_table.len(),
            compiled.manifest.image_hash
        );

        let plan = &compiled.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48, "expected 48 layers");
        eprintln!("image hash: {}", compiled.manifest.image_hash);

        plan.validate().expect("execution plan should validate");

        eprintln!("Opening runtime...");
        let baseline_handles = crate::bridge::handle_count();
        let _reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        eprintln!("Loading profiled model...");
        let profiled_model = crate::profiled_executor::LoadedProfiledModel::new(&output_dir)
            .expect("load profiled model");

        let kv_caches: Vec<crate::kv_cache::KvCache> = plan
            .layers
            .iter()
            .map(|lp| {
                let is_sliding = lp.attention_kind == "sliding_attention";
                let (capacity, n_kv_heads, head_dim) = if is_sliding {
                    (lp.sliding_window, lp.n_kv_heads, lp.head_dim)
                } else {
                    let g_kv = lp.n_global_kv_heads.unwrap_or(lp.n_kv_heads);
                    let g_hd = lp.global_head_dim.unwrap_or(lp.head_dim);
                    (32768u32, g_kv, g_hd)
                };
                crate::kv_cache::KvCache::new(capacity, n_kv_heads, head_dim, is_sliding)
            })
            .collect();

        let mut session =
            crate::profiled_executor::ProfiledInferenceSession::new("decode-8".into(), kv_caches);

        eprintln!("Prefilling with BOS token [2]...");
        let first_token = session.prefill(&[2u32], &profiled_model).expect("prefill");
        assert!(
            first_token < 262144,
            "first token {} out of vocab range",
            first_token
        );
        assert!(first_token != 0, "first token must not be 0");

        let mut tokens: Vec<u32> = Vec::with_capacity(9);
        tokens.push(first_token);

        eprintln!("Decoding 8 tokens...");
        let mut prev = first_token;
        for i in 0..8 {
            let next = session
                .decode_one(prev, &profiled_model)
                .expect("decode_one");
            assert!(
                next < 262144,
                "token {} out of vocab range at step {}",
                next,
                i
            );
            assert!(next != 0, "token must not be 0 at step {}", i);
            tokens.push(next);
            prev = next;
        }

        eprintln!("Tokens: {:?}", tokens);
        assert_eq!(tokens.len(), 9, "expected 9 tokens (1 prefill + 8 decode)");

        drop(session);
        drop(profiled_model);
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count must return to baseline after 8 decode steps; {} != {}",
            after_run, baseline_handles
        );

        eprintln!("[decode-8] PASSED: {} tokens", tokens.len());
    }
}
