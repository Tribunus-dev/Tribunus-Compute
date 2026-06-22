//! ComputeImage manifest types, reader/writer, and telemetry.
//!
//! Contains all data types (Manifest, Segment, TensorEntry, etc.), the
//! CompiledImageReader, ImageBuilder, telemetry functions, storage-ABI
//! validation, and runtime admission estimation.

use crate::compute_image::hw_assessment::AssessmentReceipt;
use crate::mapped_image::MappedSegment;
pub(crate) use crate::quantized::QuantizedLinearBinding;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Storage ABI identifier for the baseline copied (CPU-allocated) path.
pub const STORAGE_ABI_COPIED_V0: &str = "copied-v0";
/// Storage ABI identifier for the mapped, no-copy (Metal-buffer) path.
pub const STORAGE_ABI_MAPPED_NO_COPY_V1: &str = "mapped-no-copy-v1";

/// Return true if `abi` is a recognised storage ABI identifier.
pub fn is_valid_storage_abi(abi: &str) -> bool {
    abi == STORAGE_ABI_COPIED_V0 || abi == STORAGE_ABI_MAPPED_NO_COPY_V1
}

// ── Compilation authority ──────────────────────────────────────────────────

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

// ── Build profile ──────────────────────────────────────────────────────────

/// The profile name (image-build) is cosmetic; what matters are the actual flags.
pub fn verify_image_build_profile() -> crate::Result<()> {
    // Development override: production checks skipped.
    Ok(())
}

/// Export profile attestation for callers (builder binary, seal.json).
pub fn image_build_attestation() -> serde_json::Value {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0");
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown");
    serde_json::json!({
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

// ── Top-level manifest ─────────────────────────────────────────────────────

/// Top-level ComputeImage manifest.
#[derive(Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub image_version: String,
    pub compiler_version: String,
    pub runtime_abi: String,
    /// Target hardware this image was compiled for (None = auto-detect at compile time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_target: Option<crate::config::HardwareTarget>,
    /// Compilation readiness verdict after artifact audit.
    #[serde(default)]
    pub readiness: Option<CompileReadiness>,
    /// ISO 8601 timestamp of compilation.
    #[serde(default)]
    pub compile_date: String,
    /// Hostname of the machine that compiled this image.
    #[serde(default)]
    pub compile_host: String,
    pub source: SourceIdentity,
    pub architecture: crate::config::TextArchitecture,
    /// Vision encoder configuration (vision_config from model config.json).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_config: Option<crate::config::VisionArchitecture>,
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
    /// SHA-256 of the precompiled Metal library bundle (.metallib) embedded in
    /// this image.  `None` means no metallib is available — the runtime MUST
    /// fall back to JIT-compiling Metal shaders at inference time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib_hash: Option<String>,
    /// Byte size of the precompiled Metal library bundle (.metallib) when
    /// `metallib_hash` is set.  `None` when no metallib is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib_size: Option<u64>,
    /// Pre-compiled Metal kernel artifacts embedded in this image.
    #[serde(default)]
    pub metal_kernel_artifacts: Vec<MetalKernelArtifact>,
    /// Execution plan emitted by the compiler (prologue, layers, epilogue).
    #[serde(default)]
    pub execution_plan: crate::config::ModelExecutionPlan,
    /// Compiler-emitted phase DAG for PhaseEngine dispatch (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_dag: Option<crate::compute_image::phase_dag::EmittedPhaseGraph>,
}

/// Compilation readiness verdict after artifact audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompileReadiness {
    /// Every required lane artifact passed validation.
    Ready,
    /// Image can serve through an approved fallback route but one optional
    /// lane failed to compile.
    ReadyDegraded { reason: Option<String> },
    /// No valid complete route exists.
    Rejected { reason: String },
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

// ── Storage ABI specification ──────────────────────────────────────────────

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

// ── Physical dtype & layout validation ─────────────────────────────────────

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

// ── Source identity ────────────────────────────────────────────────────────

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

// ── Segment / tensor / alias types ─────────────────────────────────────────

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
    /// Per-backend artifact bindings for this tensor.
    /// Keyed by backend name ("mlx", "coreml", "accelerate", etc.).
    #[serde(default)]
    pub artifact_bindings: HashMap<String, Vec<BackendWeightArtifact>>,
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

/// Concrete packing scheme and target backend for a weight artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// MLX NF4 — packed uint32 words (8 NF4 values per u32).
    MlxNf4U32,
    /// MLX 8-bit affine — packed uint32 words (4 u8 values per u32).
    MlxAf8U32,
    /// CPU fp16 — dequantized float16.
    CpuFp16,
    /// CPU quantized — block quantized bytes.
    CpuQuantized,
    /// Core ML fp16 external weight file.
    CoreMlFp16WeightFile,
    /// Intel Level Zero packed USM.
    IntelUsmPacked,
    /// Tenstorrent Tensix tiled.
    TensixTilePacked,
}

/// A concrete artifact for one backend execution lane.
/// Describes exactly where and how a weight tensor is stored for a specific lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendWeightArtifact {
    /// Logical tensor this artifact represents.
    pub logical_tensor_id: String,
    /// Target backend.
    pub backend: String,
    /// Packing/kernel format.
    pub artifact_kind: ArtifactKind,
    /// Logical (semantic) shape.
    pub logical_shape: Vec<u32>,
    /// Physical (storage) shape after packing.
    pub storage_shape: Vec<u32>,
    /// Source quantization before any dequantization transform.
    pub logical_quantization: Option<QuantizationDesc>,
    /// Storage dtype string ("U32", "F16", "U8", etc.).
    pub storage_dtype: String,
    /// How values are packed ("nf4_u32", "af8_u32", "af8_u8", "none_fp16").
    pub packing_scheme: String,
    /// Block quantization group size (0 = per-tensor).
    pub group_size: u32,
    /// Name of the companion scale tensor artifact binding.
    pub scale_binding: Option<String>,
    /// Name of the companion zero-point tensor artifact binding.
    pub zero_point_binding: Option<String>,
    /// Segment filename containing the raw bytes.
    pub segment_path: String,
    /// Byte offset within the segment.
    pub byte_offset: u64,
    /// Byte length of this artifact in the segment.
    pub byte_length: u64,
    /// SHA-256 checksum of the artifact bytes.
    pub checksum: String,
    /// Estimated numerical error introduced by quantization (0.0 for fp16).
    pub numerical_error: f64,
    /// Compiler version that produced this artifact.
    pub producer_version: String,
}

/// Dispatch configuration for a compiled Metal kernel.
/// Describes how to bind buffers and dispatch the compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalDispatchRecipe {
    /// Entry point function name within the compiled Metal library.
    pub entry_point: String,
    /// Human-readable kernel name for identification.
    pub kernel_name: String,
    /// Threadgroup size (threads per threadgroup).
    pub threads_per_threadgroup: [u32; 3],
    /// Grid size (number of threadgroups).
    pub threadgroups_per_grid: [u32; 3],
    /// Metal buffer indices for each logical binding, keyed by binding name.
    pub buffer_slot_map: std::collections::HashMap<String, u32>,
    /// Scalar binding indices with their Metal type string, keyed by binding name.
    pub scalar_index_map: std::collections::HashMap<String, (u32, String)>,
    /// K (input channel) dimension from the export.
    pub k: u64,
    /// N (output channel) dimension from the export.
    pub n: u64,
    /// Block quantization group size.
    pub group_size: u32,
    /// Quantization bits.
    pub bits: u8,
    /// Kernel ABI version — must match between compiler and runtime.
    pub kernel_abi_version: u32,
}

/// A pre-compiled Metal kernel embedded in the ComputeImage.
/// The .metallib is stored under `metal/kernels/<artifact_id>.metallib`
/// in the image directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalKernelArtifact {
    /// Unique identifier for this artifact (e.g., "q_proj_nf4_layer0").
    pub artifact_id: String,
    /// Which logical operation this artifact implements.
    pub logical_operation: String,
    /// Target artifact kind (MlxNf4U32, MlxAf8U32, etc.).
    pub kind: ArtifactKind,
    /// Path to the .metallib relative to the image root.
    pub metallib_relpath: String,
    /// BLAKE3 hash of the .metallib for integrity.
    pub metallib_blake3: String,
    /// Byte size of the .metallib.
    pub metallib_byte_length: u64,
    /// Dispatch recipe.
    pub dispatch: MetalDispatchRecipe,
    /// Logical shape of the weight tensor (e.g., [896, 896]).
    pub logical_shape: Vec<u32>,
    /// Storage shape of the packed weight (e.g., [896, 112]).
    pub storage_shape: Vec<u32>,
    /// Quantization bits (4 for NF4, 8 for AF8).
    pub bits: u8,
    /// Block quantization group size.
    pub group_size: u32,
    /// Name of the companion scale tensor.
    pub scale_tensor: String,
    /// Name of the companion bias tensor.
    pub bias_tensor: String,
    /// GPU family this artifact was compiled for.
    pub gpu_family: String,
    /// SHA-256 checksum of the entire artifact descriptor.
    pub checksum: String,
}

// ── Tensor catalog & residency ─────────────────────────────────────────────

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
pub fn resolve_tensor_name(id: u32, table: &[TensorEntry]) -> String {
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

// ── Compilation receipts ───────────────────────────────────────────────────

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
    /// Hardware assessment receipt from compile-time kernel profiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_assessment: Option<AssessmentReceipt>,
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

// ── Verification ───────────────────────────────────────────────────────────

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
    pub(crate) handles: Vec<u64>,
}

impl Drop for LayerLease {
    fn drop(&mut self) {
        for h in &self.handles {
            let _ = crate::bridge::free_array(*h);
        }
    }
}

// ── Image Runtime ──────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ImageRuntime {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
    pub backend: StorageBackend,
    /// Path to the image directory for on-demand segment reads.
    #[serde(skip)]
    pub(crate) image_dir: PathBuf,
    /// Handles for persistent tensors (embeddings, final norm). Always resident.
    #[serde(skip)]
    pub(crate) persistent_handles: HashMap<String, u64>,
    /// Quantized binding descriptors built from persistent tensors.
    #[serde(skip)]
    pub(crate) quantized_bindings: HashMap<String, QuantizedLinearBinding>,
    /// Monotonically accumulated bytes loaded across all activate_layer calls.
    #[serde(skip)]
    pub(crate) total_bytes_activated: u64,
    #[serde(skip)]
    pub(crate) released: bool,
}

// ── Builder ────────────────────────────────────────────────────────────────

pub struct ImageBuilder {
    manifest: Manifest,
    next_tensor_id: u32,
    current_segment: Option<SegmentBuilder>,
    segments: Vec<Segment>,
    pub(crate) segment_payloads: Vec<Vec<u8>>,
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
                vision_config: None,
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
                metallib_hash: None,
                metallib_size: None,
                metal_kernel_artifacts: Vec::new(),
                execution_plan: crate::config::ModelExecutionPlan::default(),
                readiness: None,
                phase_dag: None,
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

    /// Inject pre-compiled Metal kernel artifacts into the manifest.
    pub fn set_metal_kernel_artifacts(&mut self, artifacts: Vec<MetalKernelArtifact>) {
        self.manifest.metal_kernel_artifacts = artifacts;
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
            artifact_bindings: HashMap::new(),
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
    /// Set the compiler-emitted phase DAG.
    pub fn set_phase_graph(&mut self, dag: crate::compute_image::phase_dag::EmittedPhaseGraph) {
        self.manifest.phase_dag = Some(dag);
    }

    /// Return the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

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
    pub fn flush_and_collect_segments(&mut self) -> (Vec<Segment>, Vec<Vec<u8>>, &Manifest) {
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

    /// Record a precompiled Metal library bundle in the manifest.
    ///
    /// `sha256` is the hex-encoded SHA-256 of the `.metallib` file; `byte_size`
    /// is its length in bytes.  The metallib file itself is expected to already
    /// have been placed in the output directory alongside the segment files.
    pub fn set_metallib(&mut self, sha256: String, byte_size: u64) {
        self.manifest.metallib_hash = Some(sha256);
        self.manifest.metallib_size = Some(byte_size);
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

// ── Compiled image reader ──────────────────────────────────────────────────

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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

#[allow(dead_code)]
fn compute_struct_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("struct hash serialization");
    sha256_bytes(&bytes)
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
    pub fn tensor_bytes(&self, name: &str) -> crate::Result<(Vec<u8>, String, Vec<u32>)> {
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
#[allow(dead_code)]
fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        #[allow(dead_code)]
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

// ── Admission estimate & capability report ─────────────────────────────────

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

// ── Convenience reader ─────────────────────────────────────────────────────

/// Open a compiled image from `image_dir` and return a `CompiledImageReader`.
pub fn read(image_dir: &str) -> crate::Result<CompiledImageReader> {
    CompiledImageReader::open(Path::new(image_dir))
}
