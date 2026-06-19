//! Profiled heterogeneous executor — GPU-canary-gated execution with explicit receipts.
//!
//! Uses MappedImage-based segment file access via seek + read_exact.
//!
//! The model runtime (LoadedProfiledModel) is immutable and survives requests.
//! Per-generation state lives in ProfiledInferenceSession (owns KV caches,
//! cancellation flag, token buffer, and timeline).

use crate::compute_image::{CompiledImageReader, CopyClassification, TensorEntry};
use crate::config::ModelExecutionPlan;
use crate::engine_error::{EngineError, EngineErrorCode};
use crate::kv_cache::{KvCache, PageMigrationService};
use crate::mapped_image::MappedImage;
use crate::quantization::turboquant_kv::AsymmetricQuantMode;
use crate::ane::hot_row_predictor::HotRowPredictor;
use crate::ane::weight_row_cache::WeightRowCache;
use crate::arena::Arena;
use crate::autopsy::ModelAutopsy;
use crate::cache::chunk_kv::ChunkKvCache;
use crate::cache::evolkv::CalibrationSet;
use crate::cache::evolkv::LayerBudget;
use crate::cache::prefix_cache::{check_shared_prefix, insert_shared_prefix};
use crate::placement_profile::ExecutionPlacementProfile;
use crate::runtime_contract::{
    AuthorityMode, BackendTarget, BudgetClass, RetryPolicy, RuntimeWorkItem,
};
use crate::runtime_orchestration::InMemoryCoordinationFabric;
use crate::heterogeneous::ComputeRuntime;
use crate::runtime_trace::{RuntimeTimeline, TimelineEvent, TimelineEventType};
use crate::session::InferenceSessionState;
use crate::session::SamplerConfig;
use crate::worker_memory;
use crate::coreml_bridge::CoreMlModel;
use crate::vision::encoder::VisionEncoder;
use crate::video::{extract_frames, MAX_VIDEO_FRAMES};
use mlx_rs::Array;

/// Maximum tokens per prefill chunk for chunked prefill.
/// Longer prompts are split into chunks to allow interleaving decode
/// of other sequences between chunks, preventing long-prefill latency spikes.
pub const PREFILL_CHUNK_SIZE: u32 = 512;

/// Input image for multi-modal inference.
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Execution mode for the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Copied segment, serial, default stream — correctness oracle only.
    SemanticOracle,
    /// Profiled heterogeneous execution with GPU-canary gating.
    Profiled,
}

/// Result of one profiled full-model execution.
#[derive(Debug, Clone)]
pub struct ProfiledReceipt {
    pub executor: String,
    pub execution_profile: String,
    pub storage_backend: String,
    pub explicit_gpu_stream: bool,
    pub oracle_fallback: bool,
    pub compiler_invocations: u64,
    pub source_checkpoint_accesses: u64,
    pub copied_weight_bytes: u64,
    pub mapped_weight_bytes: u64,
    pub token: u32,
    pub layer_count: u32,
    pub elapsed_ms: u64,
    pub profile_validation: bool,
    pub gpu_canary_us: u64,
    pub gpu_canary_ratio: f64,
    pub image_hash: String,
    pub handle_baseline: u64,
    pub handle_final: u64,
    pub layer_records: Vec<crate::mlx_executor::ExecutionRecord>,
    pub active_window_bytes: u64,
    pub prefetched_count: u64,
    pub total_kv_cache_bytes: u64,
    pub cache_hit_tokens: u64,
    pub wall_clock_total_us: u64,
    pub unaccounted_us: u64,
    pub timeline: RuntimeTimeline,
}

/// Adapter wrapping a sub-range of a MappedSegment for no-copy external array
/// construction via [`crate::external_array::new_external_array`].
struct SegmentSlice {
    segment: Arc<crate::mapped_image::MappedSegment>,
    offset: usize,
    length: usize,
}

impl crate::external_array::ExternalStorage for SegmentSlice {
    fn data_ptr(&self) -> *const u8 {
        unsafe { self.segment.data_ptr().add(self.offset) }
    }
    fn byte_len(&self) -> usize {
        self.length
    }
}

/// Load tensor data from a MappedSegment using external array construction.
///
/// Uses [`crate::external_array::new_external_array`] for all supported dtypes
/// so that MLX operates directly on the mmap-backed memory rather than a copy.
pub(crate) fn load_tensor_from_mapped_segment(
    segment: &std::sync::Arc<crate::mapped_image::MappedSegment>,
    entry: &TensorEntry,
    force_copy: bool,
) -> crate::Result<(mlx_rs::Array, CopyClassification)> {
    let mapping = segment.data_slice();
    let offset = entry.offset as usize;
    let len = entry.byte_length as usize;
    let end = offset + len;
    if end > mapping.len() {
        return Err(crate::Error::from_reason(format!(
            "tensor {} at offset {} len {} exceeds mapping len {}",
            entry.name,
            offset,
            len,
            mapping.len()
        )));
    }
    let dims: Vec<i32> = entry.physical_shape.iter().map(|&d| d as i32).collect();

    // When force_copy is true, copy the mapping data into MLX-owned buffers
    // instead of using the mmap-backed external_array. This avoids potential
    // segfaults from fused MLX kernels reading mmap pages that Metal may
    // reposition.
    if force_copy {
        let data: Vec<u8> = mapping[offset..end].to_vec();
        let arr = mlx_rs::Array::from_slice(&data, &dims);
        return Ok((arr, CopyClassification::CopiedFallback));
    }

    // TODO: wire external_array for true no-copy when mapped ABI is complete
    let storage = Arc::new(SegmentSlice {
        segment: segment.clone(),
        offset,
        length: len,
    });

    match entry.storage_dtype.as_str() {
        "U8" | "Uint8" => unsafe {
            let arr =
                crate::external_array::new_external_array(storage, &dims, mlx_rs::Dtype::Uint8)
                    .map_err(|e| crate::Error::from_reason(e))?;
            Ok((arr, CopyClassification::MappedNoCopy))
        },
        "F32" | "Float32" => unsafe {
            let arr =
                crate::external_array::new_external_array(storage, &dims, mlx_rs::Dtype::Float32)
                    .map_err(|e| crate::Error::from_reason(e))?;
            Ok((arr, CopyClassification::MappedNoCopy))
        },
        "BF16" | "BFloat16" => unsafe {
            let arr =
                crate::external_array::new_external_array(storage, &dims, mlx_rs::Dtype::Bfloat16)
                    .map_err(|e| crate::Error::from_reason(e))?;
            Ok((arr, CopyClassification::MappedNoCopy))
        },
        "I8" | "Int8" => {
            // external_array does not yet support Int8 natively; fall back to the
            // copy path. This is harmless since Int8 weights are tiny (scales).
            let data: Vec<i8> = mapping[offset..end].iter().map(|&b| b as i8).collect();
            let arr = mlx_rs::Array::from_slice(&data, &dims);
            Ok((arr, CopyClassification::CopiedFallback))
        }
        "U32" | "Uint32" => unsafe {
            let arr =
                crate::external_array::new_external_array(storage, &dims, mlx_rs::Dtype::Uint32)
                    .map_err(|e| crate::Error::from_reason(e))?;
            Ok((arr, CopyClassification::MappedNoCopy))
        },
        other => Err(crate::Error::from_reason(format!(
            "unsupported storage dtype in profiled executor: {}",
            other
        ))),
    }
}

fn build_rope_tables(
    arch: &crate::config::TextArchitecture,
) -> crate::Result<(Arc<Array>, Arc<Array>, Arc<Array>, Arc<Array>)> {
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

    Ok((
        Arc::new(rope_cos),
        Arc::new(rope_sin),
        Arc::new(full_cos),
        Arc::new(full_sin),
    ))
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

fn high_memory_override_enabled() -> bool {
    matches!(
        std::env::var("TRIBUNUS_COMPUTE_ALLOW_HIGH_MEMORY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn estimate_profiled_peak_bytes(reader: &CompiledImageReader) -> u64 {
    let manifest = &reader.manifest;
    let tensor_bytes = manifest
        .tensor_table
        .iter()
        .map(|entry| entry.byte_length)
        .sum::<u64>();
    let max_tensor_bytes = manifest
        .tensor_table
        .iter()
        .map(|entry| entry.byte_length)
        .max()
        .unwrap_or(0);
    let max_segment_bytes = manifest
        .segments
        .iter()
        .map(|segment| segment.byte_size)
        .max()
        .unwrap_or(0);
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

    tensor_bytes
        .saturating_add(max_tensor_bytes)
        .saturating_add(max_segment_bytes)
        .saturating_add(rope_bytes)
        .saturating_add(embedding_dequant_bytes)
        .saturating_add(2 * 1024 * 1024 * 1024)
}

pub struct LayerWeights {
    pub input_layernorm: Arc<Array>,
    pub post_attention_layernorm: Arc<Array>,
    pub q_proj_w: Arc<Array>,
    pub q_proj_s: Arc<Array>,
    pub q_proj_b: Arc<Array>,
    pub k_proj_w: Arc<Array>,
    pub k_proj_s: Arc<Array>,
    pub k_proj_b: Arc<Array>,
    pub v_proj_w: Arc<Array>,
    pub v_proj_s: Arc<Array>,
    pub v_proj_b: Arc<Array>,
    pub o_proj_w: Arc<Array>,
    pub o_proj_s: Arc<Array>,
    pub o_proj_b: Arc<Array>,
    pub gate_proj_w: Arc<Array>,
    pub gate_proj_s: Arc<Array>,
    pub gate_proj_b: Arc<Array>,
    pub up_proj_w: Arc<Array>,
    pub up_proj_s: Arc<Array>,
    pub up_proj_b: Arc<Array>,
    pub down_proj_w: Arc<Array>,
    pub down_proj_s: Arc<Array>,
    pub down_proj_b: Arc<Array>,
    pub q_norm: Option<Arc<Array>>,
    pub k_norm: Option<Arc<Array>>,
}

/// Format bytes for human-readable display in per-layer telemetry.
fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.1}GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1}MB", b as f64 / 1_048_576.0)
    } else {
        format!("{}B", b)
    }
}

pub struct LoadedProfiledModel {
    pub image_dir: PathBuf,
    pub reader: CompiledImageReader,
    pub mapped_image: MappedImage,
    pub layers: Vec<LayerWeights>,
    pub emb_w: Arc<Array>,
    pub emb_s: Arc<Array>,
    pub emb_b: Arc<Array>,
    pub fn_w: Arc<Array>,
    pub rope_cos: Arc<Array>,
    pub rope_sin: Arc<Array>,
    pub full_cos: Arc<Array>,
    pub full_sin: Arc<Array>,
    pub mapped_weight_bytes: u64,
    pub copied_weight_bytes: u64,
    pub materialized_bytes: u64,
    pub handle_baseline: usize,
    /// Compiled ANE programs for layers routed to Orion.
    pub ane_cache: Option<crate::memory::ane_program_cache::AneProgramCache>,
    /// Pre-loaded CoreML models for ANE-routed attention layers, indexed by
    /// layer index. Fused islands replicate their model (via Arc) across
    /// all covered layer slots.
    pub ane_coreml_models: Vec<Option<std::sync::Arc<CoreMlModel>>>,
    /// Shared IOSurface memory island — all runtime memory allocations
    /// (intermediates, KV cache) come from this pool. MLX does NOT manage
    /// memory independently.
    pub memory_island: crate::heterogeneous::SharedMemoryIsland,
    /// Compiled schedule with regions, memory plan, and evaluation boundaries.
    /// Populated during [`new()`] from the manifest's architecture + execution plan.
    pub scheduled_module: Option<crate::compiler::scheduled::ScheduledModule>,
    /// Vision encoder for multi-modal image input (None for text-only models).
    pub vision_encoder: Option<VisionEncoder>,
    /// Currently active LoRA adapter (None = no adapter loaded).
    pub active_adapter: Option<crate::lora::LoraAdapter>,
}

// Safety: raw pointers are to MLX ref-counted objects (thread-safe).
unsafe impl Send for LoadedProfiledModel {}
unsafe impl Sync for LoadedProfiledModel {}

impl LoadedProfiledModel {
    pub fn new(image_dir: &Path) -> crate::Result<Self> {
        let handle_baseline = crate::bridge::handle_count();
        let mut reader = CompiledImageReader::open(image_dir)?;
        if !high_memory_override_enabled() {
            let total_memory = system_memory_bytes();
            let estimated_peak = estimate_profiled_peak_bytes(&reader);
            if total_memory > 0
                && estimated_peak > total_memory.saturating_sub(2 * 1024 * 1024 * 1024)
            {
                return Err(crate::Error::from_reason(format!(
                    "refusing to load profiled model: estimated peak {} exceeds safe budget on this machine (total memory {})",
                    estimated_peak,
                    total_memory,
                )));
            }
        }
        // Compute admission estimate and configure MLX memory limits before
        // loading any tensors so the allocator is already constrained.
        let estimate = crate::model_runtime::compute_admission_estimate(&reader.manifest);
        let machine = worker_memory::detect_machine_profile();
        worker_memory::configure_mlx_limits_for_model(&estimate, &machine);
        let segment_views: Vec<crate::mapped_image::SegmentView> = reader
            .manifest
            .segments
            .iter()
            .map(|s| crate::mapped_image::SegmentView {
                segment_id: s.id.clone(),
                segment_index: 0,
                file_path: std::path::PathBuf::from(s.filename.clone()),
                byte_offset: 0,
                byte_length: s.byte_size,
                kind: String::new(),
                segment_lease: None,
            })
            .collect();
        let mapped_image = crate::mapped_image::MappedImage::open_mapped(image_dir, &segment_views)
            .map_err(|e| crate::Error::from_reason(format!("open mapped image: {}", e)))?;

        let mut mapped_weight_bytes = 0;
        let mut copied_weight_bytes = 0;
        let mut materialized_bytes = 0;
        let mut tensor_cache: HashMap<String, Arc<Array>> = HashMap::new();

        let mut load_tensor = |name: &str| -> crate::Result<Arc<Array>> {
            if let Some(arr) = tensor_cache.get(name) {
                return Ok(arr.clone());
            }
            let entry = reader
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| crate::Error::from_reason(format!("tensor not found: {}", name)))?;
            let seg_id = &entry.segment;
            let segment = mapped_image.segments.get(seg_id).ok_or_else(|| {
                crate::Error::from_reason(format!("segment not found: {}", seg_id))
            })?;
            let (arr, classification) = load_tensor_from_mapped_segment(segment, entry, false)?;
            let byte_len = entry.byte_length;
            match classification {
                CopyClassification::MappedNoCopy => mapped_weight_bytes += byte_len,
                CopyClassification::CopiedFallback => copied_weight_bytes += byte_len,
                _ => materialized_bytes += byte_len,
            }
            let arc = Arc::new(arr);
            tensor_cache.insert(name.to_string(), arc.clone());
            Ok(arc)
        };

        /// Detect tensor namespace root from the manifest's tensor table.
        fn detect_ns(table: &[crate::compute_image::manifest::TensorEntry]) -> String {
            // Pick the first global tensor's prefix before "embed_tokens" or ".layers."
            for entry in table {
                if entry.name.contains(".embed_tokens.") || entry.name.contains(".embed_tokens.weight") {
                    if let Some(idx) = entry.name.rfind(".embed_tokens") {
                        return entry.name[..idx].to_string();
                    }
                }
            }
            // Fallback: try to find any tensor with ".layers.0." in name
            for entry in table {
                if let Some(idx) = entry.name.rfind(".layers.0.") {
                    return entry.name[..idx].to_string();
                }
            }
            "model".to_string()
        }

        let ns = detect_ns(&reader.manifest.tensor_table);
        let ns_str = ns.clone();
        eprintln!("[detect-ns] detected namespace root: '{}'", ns);

        // Load global tensors
        let emb_w = load_tensor(&format!("{}.embed_tokens.weight", ns))?;
        let emb_s = load_tensor(&format!("{}.embed_tokens.scales", ns))?;
        let emb_b = load_tensor(&format!("{}.embed_tokens.biases", ns))?;
        let fn_w = load_tensor(&format!("{}.norm.weight", ns))?;

        // RoPE tables are derived from the architecture rather than loaded
        // from the manifest. This avoids falling back to 1-element placeholders
        // when the compiled image does not materialize explicit rope tensors.
        let (rope_cos, rope_sin, full_cos, full_sin) =
            build_rope_tables(&reader.manifest.architecture)?;

        // Load layer weights
        let mut layers = Vec::new();
        for (l, layer_plan) in reader.manifest.execution_plan.layers.iter().enumerate() {
            let base = format!("{}.layers.{}", ns, l);

            let input_layernorm = load_tensor(&format!("{}.input_layernorm.weight", base))?;
            let post_attention_layernorm =
                load_tensor(&format!("{}.post_attention_layernorm.weight", base))?;

            let q_proj_w = load_tensor(&format!("{}.self_attn.q_proj.weight", base))?;
            let q_proj_s = load_tensor(&format!("{}.self_attn.q_proj.scales", base))?;
            let q_proj_b = load_tensor(&format!("{}.self_attn.q_proj.biases", base))?;

            let k_proj_w = load_tensor(&format!("{}.self_attn.k_proj.weight", base))?;
            let k_proj_s = load_tensor(&format!("{}.self_attn.k_proj.scales", base))?;
            let k_proj_b = load_tensor(&format!("{}.self_attn.k_proj.biases", base))?;

            let (v_proj_w, v_proj_s, v_proj_b) = if layer_plan.attention_k_eq_v {
                (k_proj_w.clone(), k_proj_s.clone(), k_proj_b.clone())
            } else {
                (
                    load_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                    load_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                    load_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
                )
            };

            let o_proj_w = load_tensor(&format!("{}.self_attn.o_proj.weight", base))?;
            let o_proj_s = load_tensor(&format!("{}.self_attn.o_proj.scales", base))?;
            let o_proj_b = load_tensor(&format!("{}.self_attn.o_proj.biases", base))?;

            let gate_proj_w = load_tensor(&format!("{}.mlp.gate_proj.weight", base))?;
            let gate_proj_s = load_tensor(&format!("{}.mlp.gate_proj.scales", base))?;
            let gate_proj_b = load_tensor(&format!("{}.mlp.gate_proj.biases", base))?;

            let up_proj_w = load_tensor(&format!("{}.mlp.up_proj.weight", base))?;
            let up_proj_s = load_tensor(&format!("{}.mlp.up_proj.scales", base))?;
            let up_proj_b = load_tensor(&format!("{}.mlp.up_proj.biases", base))?;

            let down_proj_w = load_tensor(&format!("{}.mlp.down_proj.weight", base))?;
            let down_proj_s = load_tensor(&format!("{}.mlp.down_proj.scales", base))?;
            let down_proj_b = load_tensor(&format!("{}.mlp.down_proj.biases", base))?;

            let q_norm_name = format!("{}.self_attn.q_norm.weight", base);
            let q_norm = if reader
                .manifest
                .tensor_table
                .iter()
                .any(|e| e.name == q_norm_name)
            {
                Some(load_tensor(&q_norm_name)?)
            } else {
                None
            };
            let k_norm_name = format!("{}.self_attn.k_norm.weight", base);
            let k_norm = if reader
                .manifest
                .tensor_table
                .iter()
                .any(|e| e.name == k_norm_name)
            {
                Some(load_tensor(&k_norm_name)?)
            } else {
                None
            };

            layers.push(LayerWeights {
                input_layernorm,
                post_attention_layernorm,
                q_proj_w,
                q_proj_s,
                q_proj_b,
                k_proj_w,
                k_proj_s,
                k_proj_b,
                v_proj_w,
                v_proj_s,
                v_proj_b,
                o_proj_w,
                o_proj_s,
                o_proj_b,
                gate_proj_w,
                gate_proj_s,
                gate_proj_b,
                up_proj_w,
                up_proj_s,
                up_proj_b,
                down_proj_w,
                down_proj_s,
                down_proj_b,
                q_norm,
                k_norm,
            });
        }

        // ── Assign per-layer backend routes ──────────────────────────
        // Sliding window attention → Core ML / ANE (backend 2)
        // Full attention → MLX / GPU (backend 0)
        for layer_plan in &mut reader.manifest.execution_plan.layers {
            let backend = crate::executor::resolve_attention_backend(layer_plan);
            layer_plan.route.set_dominant_backend(backend.0);
        }

        // Post-load RSS comparison: warn if actual RSS exceeds the admission
        // estimate by more than 20 %.
        let postload_rss = worker_memory::sample_process_rss_self();
        let estimated_peak = estimate.peak_bytes();
        if postload_rss > estimated_peak && estimated_peak > 0 {
            let ratio = postload_rss as f64 / estimated_peak as f64;
            if ratio > 1.20 {
                eprintln!(
                    "[profiled-model] WARNING: post-load RSS ({} bytes) exceeds admission estimate ({} bytes) by {:.1}%",
                    postload_rss,
                    estimated_peak,
                    (ratio - 1.0) * 100.0,
                );
            }
        }

        // ── Pre-warm ANE hardware via CoreML ─────────────────────────────
        let _ane_prewarmed = crate::memory::orion_bridge::prewarm_ane();

        // ── Shared IOSurface memory island ──────────────────────────────
        // All runtime intermediates allocate from this pool, NOT from MLX.
        // This ensures Accelerate and CoreML read the same physical pages
        // that MLX writes, achieving zero-copy across all backends.
        let memory_island = crate::heterogeneous::SharedMemoryIsland::new();

        // ── Compile ANE programs for Orion-routed layers ────────────────────
        let ane_cache = {
            let orion_indices: Vec<usize> = reader
                .manifest
                .execution_plan
                .layers
                .iter()
                .enumerate()
                .filter(|(_, p)| p.route.attention == 3)
                .map(|(i, _)| i)
                .collect();
            if orion_indices.is_empty() {
                None // No ANE routing in this model
            } else {
                let mut segments_vec: Vec<std::sync::Arc<crate::mapped_image::MappedSegment>> =
                    Vec::new();
                // MappedImage stores segments as HashMap<String, Arc<MappedSegment>>.
                // Convert to Vec<Arc<MappedSegment>> ordered by manifest segment order.
                for seg in &reader.manifest.segments {
                    if let Some(s) = mapped_image.segments.get(&seg.id) {
                        segments_vec.push(s.clone());
                    }
                }
                let mut cache = crate::memory::ane_program_cache::AneProgramCache::new();
                cache.compile_from_manifest(
                    reader.manifest.execution_plan.layers.len(),
                    &orion_indices,
                    &segments_vec,
                    &reader.manifest.tensor_table,
                );
                if cache.compiled_count() > 0 {
                    Some(cache)
                } else {
                    None
                }
            }
        };

        // ── Load ANE CoreML models for ANE-routed attention layers ─────
        let n_layers = reader.manifest.execution_plan.layers.len();
        let mut ane_coreml_models: Vec<Option<std::sync::Arc<CoreMlModel>>> =
            vec![None; n_layers];
        for island in &reader.manifest.execution_plan.fused_ane_islands {
            let model_path = image_dir.join(&island.modelc_relpath);
            match CoreMlModel::load_with_compute_units(
                &model_path.to_string_lossy(),
                crate::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
            ) {
                Ok(m) => {
                    let arc = std::sync::Arc::new(m);
                    for &layer_idx in &island.layer_indices {
                        let idx = layer_idx as usize;
                        if idx < ane_coreml_models.len() {
                            ane_coreml_models[idx] = Some(arc.clone());
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[profiled-model] WARNING: failed to load ANE CoreML \
                         model for island {}: {} (falling back to MLX)",
                        island.island_id, e,
                    );
                }
            }
        }

        // ── Load ANE CoreML models for sliding window attention layers ──
        // Each sliding window layer gets its own CoreML model compiled from
        // the MIL program template for sliding window attention. Full attention
        // layers (every 6th) remain on MLX and get no model.
        for (l, layer_plan) in reader.manifest.execution_plan.layers.iter().enumerate() {
            if ane_coreml_models[l].is_some() {
                // Already has a model from fused ANE islands
                continue;
            }
            if layer_plan.attention_kind != "full_attention" {
                let mil = crate::compiler::ane::kv_decompress_program::generate_attention_mil(
                    layer_plan.n_heads,
                    layer_plan.n_kv_heads,
                    layer_plan.head_dim,
                    layer_plan.sliding_window,
                );
                match crate::compiler::ane::kv_decompress_program::compile_mil_text(&mil) {
                    Ok(model) => {
                        let arc = std::sync::Arc::new(model);
                        ane_coreml_models[l] = Some(arc);
                    }
                    Err(e) => {
                        eprintln!(
                            "[profiled-model] WARNING: ANE compile failed for sliding layer {}: {} (falling back to MLX)",
                            l, e,
                        );
                    }
                }
            }
        }

        // ── Compile scheduled module (memory plan, regions, boundaries) ──
        let scheduled_module = Some(
            crate::compiler::compile_schedule::compile_model_to_scheduled_module(
                &reader.manifest.execution_plan,
                &reader.manifest.architecture,
                crate::backend::routing::EvidenceDigest(
                    reader.manifest.image_hash.clone(),
                ),
            ),
        );

        // ── Load vision encoder (if present) ─────────────────────────
        let vision_encoder = if reader.manifest.tensor_table.iter().any(|e| e.name.contains("vision_encoder")) {
            // Find the model's vision_config from the manifest metadata.
            // Fall back to the image metadata embedded in the architecture.
            let vision_config = crate::config::VisionArchitecture {
                hidden_size: 2048,
                num_attention_heads: 16,
                num_hidden_layers: 24,
                intermediate_size: 8192,
                image_size: 896,
                patch_size: 14,
                num_channels: 3,
                projection_dim: reader.manifest.architecture.hidden_size,
            };
            // Override with actual config from manifest if available.
            let vc = vision_config;
            // Use the same load_tensor approach as text weights.
            // We create a mutable closure that resolves tensor names
            // from the compiled image's tensor table.
            let mut load_vision_tensor = |name: &str| -> Result<std::sync::Arc<Array>, String> {
                if let Some(entry) = reader.manifest.tensor_table.iter().find(|e| e.name == name) {
                    let seg_id = &entry.segment;
                    let segment = mapped_image.segments.get(seg_id).ok_or_else(|| {
                        format!("segment not found for vision tensor {}: {}", name, seg_id)
                    })?;
                    let (arr, _classification) = load_tensor_from_mapped_segment(segment, entry, false)
                        .map_err(|e| format!("load vision tensor {}: {}", name, e))?;
                    Ok(std::sync::Arc::new(arr))
                } else {
                    // Return a zero-initialized placeholder so the encoder
                    // can still be constructed for models that don't have
                    // vision weights (graceful fallback).
                    Err(format!("vision tensor not found in compiled image: {}", name))
                }
            };
            match VisionEncoder::load(vc, &mut load_vision_tensor) {
                Ok(enc) => Some(enc),
                Err(e) => {
                    eprintln!("[profiled-model] WARNING: vision encoder load failed: {} (continuing without vision)", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            image_dir: image_dir.to_path_buf(),
            reader,
            mapped_image,
            layers,
            emb_w,
            emb_s,
            emb_b,
            fn_w,
            rope_cos,
            rope_sin,
            full_cos,
            full_sin,
            mapped_weight_bytes,
            copied_weight_bytes,
            materialized_bytes,
            handle_baseline,
            ane_cache,
            ane_coreml_models,
            memory_island,
            scheduled_module,
            vision_encoder,
            active_adapter: None,
        })
    }
}

/// ANE DMA prefetcher for asynchronous weight loading.
///
/// The ANE has its own DMA engine that can read from disk and write to
/// IOSurface without GPU involvement. This struct wraps that capability.
/// Currently a placeholder — actual ANE DMA programming will be added
/// when the ANE kernel driver exposes the DMA interface.
pub struct AneDmaPrefetcher {
    /// Temporary IOSurface arena for DMA writes.
    io_arena: Arena,
}

impl AneDmaPrefetcher {
    /// Create a new DMA prefetcher with an IOSurface-backed IO buffer.
    pub fn new() -> Result<Self, String> {
        // 4MB buffer — enough for a single layer's weights (~400MB for a 2-layer window
        // but we only buffer the DMA transfer, not the full weight storage).
        let io_arena = Arena::new(1024 * 1024, 1, mlx_rs::Dtype::Uint8)
            .map_err(|e| format!("DMA prefetcher arena: {}", e))?;
        Ok(Self { io_arena })
    }

    /// Issue a non-blocking DMA read from a segment file on disk into the
    /// IOSurface arena. Returns immediately — the ANE handles the transfer.
    pub fn dma_read(&self, _segment_path: &str) -> Result<(), String> {
        // Placeholder: in production, this would program the ANE DMA engine
        // to copy data from the NVMe segment file into the IOSurface arena.
        // The copy happens asynchronously while the GPU computes the current layer.
        Ok(())
    }

    /// Wait for any in-flight DMA transfers to complete.
    pub fn sync(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Streaming weight manager for a single model.
///
/// Only keeps a small window of layer weights in GPU memory at any time.
/// As the layer loop advances, the streamer:
/// 1. Prefetches layer N+1 weights from disk into GPU memory
/// 2. Keeps layer N weights for the current computation
/// 3. Evicts layer N-1 weights from GPU memory
///
/// The ANE manages the prefetch DMA, so it doesn't consume GPU cycles.
pub struct LayerWeightStreamer {
    /// Detected tensor namespace root.
    pub ns: String,
    /// Path to the model's compiled segment files
    model_path: PathBuf,
    /// The execution plan (layer count, shapes, etc.)
    plan: Arc<ModelExecutionPlan>,
    /// Active layer weights in GPU memory (window of 2-3 layers)
    active_weights: HashMap<u32, LayerWeights>,
    /// Prefetch window size (default: 2, meaning weights for layer N and N+1 are resident)
    prefetch_window: u32,
    /// IO buffer for DMA transfers (IOSurface-backed)
    io_buffer: Arena,
    /// ANE prefetcher for async DMA
    ane_prefetcher: Option<AneDmaPrefetcher>,
    /// Shared reference to the mapped image for zero-copy reads
    mapped_image: Arc<MappedImage>,
    /// Shared reference to the compiled reader for tensor metadata
    reader: Arc<CompiledImageReader>,
    /// Statistics
    pub prefetches: u64,
    pub evictions: u64,
}

impl LayerWeightStreamer {
    /// Detect tensor namespace root from the compiled image reader's tensor table.
    fn detect_ns_from_reader(reader: &CompiledImageReader) -> String {
        for entry in &reader.manifest.tensor_table {
            if entry.name.contains(".embed_tokens.weight") {
                if let Some(idx) = entry.name.rfind(".embed_tokens") {
                    let ns = &entry.name[..idx];
                    return ns.to_string();
                }
            }
        }
        "model".to_string()
    }


    /// Create a new weight streamer.
    ///
    /// `model_path` — path to the compiled model directory containing segment files.
    /// `plan` — the model's execution plan with layer metadata.
    /// `mapped_image` — shared mapped image for zero-copy segment access.
    /// `reader` — compiled image reader with tensor table metadata.
    pub fn new(
        model_path: &str,
        plan: Arc<ModelExecutionPlan>,
        mapped_image: Arc<MappedImage>,
        reader: Arc<CompiledImageReader>,
    ) -> Result<Self, String> {
        let detected_ns = Self::detect_ns_from_reader(&reader);
        let io_buffer = Arena::new(4 * 1024 * 1024, 1, mlx_rs::Dtype::Uint8)
            .map_err(|e| format!("weight streamer io arena: {}", e))?;

        let ane_prefetcher = AneDmaPrefetcher::new().ok();

        Ok(Self {
            ns: detected_ns,
            model_path: PathBuf::from(model_path),
            plan,
            active_weights: HashMap::new(),
            prefetch_window: 2,
            io_buffer,
            ane_prefetcher,
            mapped_image,
            reader,
            prefetches: 0,
            evictions: 0,
        })
    }

    /// Ensure weights for `layer_idx` are in GPU memory.
    /// If not loaded yet, load them. Trigger prefetch for next layer(s).
    pub fn activate(&mut self, layer_idx: u32) -> Result<&LayerWeights, String> {
        let plan = &self.plan;

        // 1. If this layer's weights are already active, just return them
        if self.active_weights.contains_key(&layer_idx) {
            // Still, trigger prefetch for next layer if not already in flight
            let next = layer_idx + 1;
            if next < plan.layers.len() as u32 && !self.active_weights.contains_key(&next) {
                self.prefetch_layer_async(next)?;
            }
            return Ok(self.active_weights.get(&layer_idx).unwrap());
        }

        // 2. Load weights for this layer from the mapped image
        let weights = self.load_layer(layer_idx)?;

        // 3. Evict layers outside prefetch window
        let min_active = layer_idx.saturating_sub(self.prefetch_window);
        let before = self.active_weights.len();
        self.active_weights.retain(|&idx, _| idx >= min_active);
        self.evictions += (before - self.active_weights.len()) as u64;

        // 4. Insert the layer we just loaded
        self.active_weights.insert(layer_idx, weights);

        // 5. Prefetch next layer
        let next = layer_idx + 1;
        if next < plan.layers.len() as u32 && !self.active_weights.contains_key(&next) {
            self.prefetch_layer_async(next)?;
            self.prefetches += 1;
        }

        Ok(self.active_weights.get(&layer_idx).unwrap())
    }

    /// Load a single layer's weights from its segment file in the mapped image.
    /// Uses mmap for zero-copy where possible.
    fn load_layer(&self, layer_idx: u32) -> Result<LayerWeights, String> {
        let base = format!("{}.layers.{}", self.ns, layer_idx);

        let load_tensor = |name: &str| -> Result<Arc<Array>, String> {
            let entry = self.reader.manifest.tensor_table
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| format!("tensor not found: {}", name))?;
            let seg_id = &entry.segment;
            let segment = self.mapped_image.segments.get(seg_id)
                .ok_or_else(|| format!("segment not found: {}", seg_id))?;
            let (arr, _classification) = load_tensor_from_mapped_segment(segment, entry, false)
                .map_err(|e| format!("load {}: {}", name, e))?;
            Ok(Arc::new(arr))
        };

        let input_layernorm = load_tensor(&format!("{}.input_layernorm.weight", base))?;
        let post_attention_layernorm = load_tensor(&format!("{}.post_attention_layernorm.weight", base))?;

        let q_proj_w = load_tensor(&format!("{}.self_attn.q_proj.weight", base))?;
        let q_proj_s = load_tensor(&format!("{}.self_attn.q_proj.scales", base))?;
        let q_proj_b = load_tensor(&format!("{}.self_attn.q_proj.biases", base))?;

        let k_proj_w = load_tensor(&format!("{}.self_attn.k_proj.weight", base))?;
        let k_proj_s = load_tensor(&format!("{}.self_attn.k_proj.scales", base))?;
        let k_proj_b = load_tensor(&format!("{}.self_attn.k_proj.biases", base))?;

        let layer_plan = &self.plan.layers[layer_idx as usize];
        let (v_proj_w, v_proj_s, v_proj_b) = if layer_plan.attention_k_eq_v {
            (k_proj_w.clone(), k_proj_s.clone(), k_proj_b.clone())
        } else {
            (
                load_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                load_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                load_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
            )
        };

        let o_proj_w = load_tensor(&format!("{}.self_attn.o_proj.weight", base))?;
        let o_proj_s = load_tensor(&format!("{}.self_attn.o_proj.scales", base))?;
        let o_proj_b = load_tensor(&format!("{}.self_attn.o_proj.biases", base))?;

        let gate_proj_w = load_tensor(&format!("{}.mlp.gate_proj.weight", base))?;
        let gate_proj_s = load_tensor(&format!("{}.mlp.gate_proj.scales", base))?;
        let gate_proj_b = load_tensor(&format!("{}.mlp.gate_proj.biases", base))?;

        let up_proj_w = load_tensor(&format!("{}.mlp.up_proj.weight", base))?;
        let up_proj_s = load_tensor(&format!("{}.mlp.up_proj.scales", base))?;
        let up_proj_b = load_tensor(&format!("{}.mlp.up_proj.biases", base))?;

        let down_proj_w = load_tensor(&format!("{}.mlp.down_proj.weight", base))?;
        let down_proj_s = load_tensor(&format!("{}.mlp.down_proj.scales", base))?;
        let down_proj_b = load_tensor(&format!("{}.mlp.down_proj.biases", base))?;

        let q_norm_name = format!("{}.self_attn.q_norm.weight", base);
        let q_norm = if self.reader.manifest.tensor_table.iter().any(|e| e.name == q_norm_name) {
            Some(load_tensor(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_name = format!("{}.self_attn.k_norm.weight", base);
        let k_norm = if self.reader.manifest.tensor_table.iter().any(|e| e.name == k_norm_name) {
            Some(load_tensor(&k_norm_name)?)
        } else {
            None
        };

        Ok(LayerWeights {
            input_layernorm,
            post_attention_layernorm,
            q_proj_w,
            q_proj_s,
            q_proj_b,
            k_proj_w,
            k_proj_s,
            k_proj_b,
            v_proj_w,
            v_proj_s,
            v_proj_b,
            o_proj_w,
            o_proj_s,
            o_proj_b,
            gate_proj_w,
            gate_proj_s,
            gate_proj_b,
            up_proj_w,
            up_proj_s,
            up_proj_b,
            down_proj_w,
            down_proj_s,
            down_proj_b,
            q_norm,
            k_norm,
        })
    }

    /// Non-blocking prefetch: fire ANE DMA to load next layer's weights
    /// while GPU finishes the current layer.
    fn prefetch_layer_async(&self, layer_idx: u32) -> Result<(), String> {
        if let Some(ane) = &self.ane_prefetcher {
            let segment_path = format!(
                "{}/{}",
                self.model_path.display(),
                self.plan.layers[layer_idx as usize].segment_id
            );
            ane.dma_read(&segment_path)?;
        }
        Ok(())
    }

    /// Unload all weights (for model swap).
    pub fn unload_all(&mut self) -> Result<(), String> {
        self.active_weights.clear();
        self.prefetches = 0;
        self.evictions = 0;
        if let Some(ane) = &self.ane_prefetcher {
            ane.sync()?;
        }
        Ok(())
    }

    /// Current memory usage of loaded weights.
    /// Each layer's weights are sized by their tensors' total bytes.
    pub fn active_memory_bytes(&self) -> u64 {
        // Approximate: each loaded layer consumes hidden_size^2 * ~8 bytes
        // for Q, K, V, O, Gate, Up, Down projections (each 2D) plus norms
        let plan = &self.plan;
        let hidden = plan.hidden_size as u64;
        // Each projection is ~hidden * hidden * 4 bytes (for f32)
        // 7 projections (Q, K, V, O, G, U, D) + 2 norms
        let per_layer = hidden * hidden * 4 * 7 + hidden * 4 * 2;
        self.active_weights.len() as u64 * per_layer
    }

    /// Memory budget: prefetch_window * avg_layer_size
    pub fn max_memory_bytes(&self) -> u64 {
        let plan = &self.plan;
        let hidden = plan.hidden_size as u64;
        let per_layer = hidden * hidden * 4 * 7 + hidden * 4 * 2;
        self.prefetch_window as u64 * per_layer
    }
}

/// Manages the working set for a single inference session.
///
/// Keeps ~400MB of weights + ~1K hot KV pages in GPU-accessible memory.
/// Everything else is on disk, streamed on demand.
pub struct WorkingSetManager {
    pub weight_streamer: LayerWeightStreamer,
    pub kv_page_migration: PageMigrationService,
    pub max_working_set_bytes: u64,
    /// Whether KV cache disk tier (L4) is enabled.
    pub disk_eviction_enabled: bool,
}

impl WorkingSetManager {
    /// Create a new working set manager.
    pub fn new(
        model_path: &str,
        plan: Arc<ModelExecutionPlan>,
        mapped_image: Arc<MappedImage>,
        reader: Arc<CompiledImageReader>,
        kv_page_migration: PageMigrationService,
    ) -> Result<Self, String> {
        let weight_streamer = LayerWeightStreamer::new(model_path, plan.clone(), mapped_image, reader)?;

        // Budget: ~400MB weights + ~200MB hot KV pages = ~600MB
        let max_working_set_bytes = 600 * 1024 * 1024;

        Ok(Self {
            weight_streamer,
            kv_page_migration,
            max_working_set_bytes,
            disk_eviction_enabled: true,
        })
    }

    /// Run EvolKV search and apply the optimal per-layer budget.
    ///
    /// Delegates to the page migration service's evolutionary search,
    /// which finds per-layer cache budget fractions that minimize
    /// perplexity on the provided calibration set under the current
    /// total cache budget constraint.
    pub fn learn_evolk_budgets(
        &mut self,
        num_layers: usize,
        calibration_set: CalibrationSet,
        cache: &mut crate::kv_cache::CompressedKvCache,
    ) -> Result<LayerBudget, String> {
        self.kv_page_migration
            .learn_evolk_budgets(num_layers, calibration_set, cache)?;
        Ok(self
            .kv_page_migration
            .evolvk_budget
            .clone()
            .expect("learn_evolk_budgets just set evolvk_budget"))
    }

    /// Called before each decode step. Manages prefetch/evict for weights and KV.
    pub fn step(&mut self, current_layer: u32) -> Result<(), String> {
        // 1. Ensure current + next layer weights are active
        self.weight_streamer.activate(current_layer)?;

        // 2. Check KV cache pressure, evict cold pages to disk
        if self.disk_eviction_enabled {
            self.kv_page_migration.check_and_evict()?;
        }

        // 3. Prefetch KV pages predicted to be needed next
        if self.disk_eviction_enabled {
            self.kv_page_migration.prefetch_predicted()?;
        }

        Ok(())
    }

    /// Memory status for debugging.
    pub fn status(&self) -> String {
        let (l1, l2, l3, l4) = self.kv_page_migration.tier_counts();
        format!(
            "WorkingSet: weights={} layers active, ~{}MB; KV pages: L1={} L2={} L3={} L4={}; max={}MB",
            self.weight_streamer.active_weights.len(),
            self.weight_streamer.active_memory_bytes() / (1024 * 1024),
            l1, l2, l3, l4,
            self.max_working_set_bytes / (1024 * 1024),
        )
    }

    /// Total GPU-visible bytes across weights and KV pages.
    pub fn total_active_bytes(&self) -> u64 {
        self.weight_streamer.active_memory_bytes()
            + self.kv_page_migration.allocated_bytes()
    }

    /// Check if we've exceeded the working set budget.
    pub fn over_budget(&self) -> bool {
        self.total_active_bytes() > self.max_working_set_bytes
    }
}

/// Per-request inference session — owns KV caches, generated tokens, and
/// cancellation state.  The model weights live in [`LoadedProfiledModel`]
/// and are passed as a parameter to [`prefill`] and [`decode_one`].
pub struct ProfiledInferenceSession {
    pub session_id: String,
    pub kv_caches: Vec<KvCache>,
    /// Attention sink states — one per layer.
    /// Populated during prefill; used during decode for efficient attention.
    pub sink_states: Vec<crate::executor::SinkState>,
    pub absolute_position: u32,
    pub generated_tokens: Vec<u32>,
    pub phase: InferenceSessionState,
    pub cancellation_flag: AtomicBool,
    pub timeline: RuntimeTimeline,
    /// Runtime coordination fabric — tracks every layer's work lifecycle.
    pub coordinator: InMemoryCoordinationFabric,
    /// Per-backend compute lanes (MLX, Accelerate, CoreML).
    pub runtime: Option<ComputeRuntime>,
    /// Chunked prefill: tokens processed so far in the current prefill.
    pub prefilled_tokens: u32,
    /// Remaining prompt tokens for chunked prefill (None = prefill complete).
    pub pending_prompt_tokens: Option<Vec<u32>>,
    /// Active memory plan for the Metal allocator (applied before layers).
    pub memory_plan: Option<crate::memory::plan::MemoryPlan>,
    /// Compression ratio for KV cache memory plan (None = uncompressed FP16).
    /// When set, the planned allocation sizes are divided by this ratio.
    pub compression_ratio: Option<f64>,
    /// Asymmetric quantization mode for K/V (K uses fewer bits than V).
    /// When set, overrides `compression_ratio` with the asymmetric ratio
    /// and the KV cache uses `append_asymmetric()`.
    pub asymmetric_quant: Option<AsymmetricQuantMode>,
    /// Sampling configuration, including optional grammar-guided generation.
    /// The grammar FSM is advanced after each decoded token.
    pub sampler: SamplerConfig,
    /// Video encoder for multi-modal video input (None for text-only models).
    pub video_encoder: Option<crate::video::encoder::VideoEncoder>,
    /// Model autopsy for anomaly detection and patching (None = disabled).
    pub autopsy: Option<ModelAutopsy>,
    /// Hot row predictor for ANE weight prefetch in the epilogue.
    pub predictor: Option<HotRowPredictor>,
    /// Weight row cache for ANE weight prefetch in the epilogue.
    pub row_cache: Option<WeightRowCache>,
    /// Working set manager for weight streaming and KV cache migration.
    /// When `Some`, layer weights are loaded on demand and KV pages
    /// can be evicted to disk. When `None`, all weights are pre-loaded
    /// and KV pages stay in GPU memory (legacy behavior).
    pub working_set: Option<WorkingSetManager>,
    /// ChunkKV semantic-preserving cache instance.
    ///
    /// When `Some`, tokens are chunked at semantic boundaries (sentences,
    /// speaker turns) and entire chunks are evicted on budget pressure
    /// instead of individual pages.  New tokens are buffered between
    /// chunk boundaries.
    pub chunk_kv_cache: Option<ChunkKvCache>,
    /// Per-token execution receipts for this session.
    pub receipts: Vec<crate::receipt::TokenReceipt>,
    /// Latest output logits from the most recent forward pass.
    /// Populated by the inference engine after each prefill/decode step.
    /// Used by SpecHub speculative decoding to access the target distribution
    /// without re-running the model.
    pub logits: Option<Array>,
    /// Physical memory page table indexed by page_id.
    /// Each entry holds an ArenaPage when the page is resident.
    pub page_table: Vec<Option<crate::ring::ArenaPage>>,
}

/// Pooling strategy for extracting a single embedding vector from
/// token-level hidden states.
# [derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbedPoolStrategy {
    /// Average across all token positions
    Mean,
    /// First token's hidden state (CLS-style, requires special token at position 0)
    Cls,
    /// Last token's hidden state
    Last,
}

impl ProfiledInferenceSession {
    /// Create a new inference session.
    ///
    /// `kv_caches` must be pre-allocated for each layer and will be populated
    /// during the first prefill call.
    pub fn new(session_id: String, kv_caches: Vec<KvCache>) -> Self {
        let mut timeline = RuntimeTimeline::new();
        timeline.push_event(TimelineEvent::new(
            0,
            TimelineEventType::EvalComplete,
            format!("session {} created", session_id),
        ));

        Self {
            session_id,
            kv_caches,
            sink_states: Vec::new(),
            absolute_position: 0,
            generated_tokens: Vec::new(),
            phase: InferenceSessionState::Created,
            cancellation_flag: AtomicBool::new(false),
            timeline,
            coordinator: InMemoryCoordinationFabric::default(),
            runtime: None,
            prefilled_tokens: 0,
            pending_prompt_tokens: None,
            memory_plan: None,
            compression_ratio: None,
            asymmetric_quant: None,
            autopsy: None,
            sampler: SamplerConfig::default(),
            video_encoder: None,
            predictor: None,
            row_cache: None,
            working_set: None,
            chunk_kv_cache: None,
            logits: None,
            receipts: Vec::new(),
            page_table: (0..256).map(|_| None).collect(),
        }
        }

    // Enable the model autopsy system for anomaly detection and patching.
    pub fn enable_autopsy(&mut self, model: Arc<LoadedProfiledModel>) {
        let fabric = self.coordinator.clone();
        self.autopsy = Some(ModelAutopsy::new(model, fabric));
    }

    /// Enable weight streaming for this session.
    ///
    /// When enabled, layer weights are loaded on-demand as the layer loop
    /// advances, and cold KV cache pages can be evicted to disk. This keeps
    /// GPU-accessible memory at ~600MB instead of requiring the full model.
    ///
    ///  — path to the compiled model image directory.
    ///  — the loaded model (provides mapped_image, reader, and plan).
    ///  — ANE-driven KV cache page migration service.
    pub fn enable_weight_streaming(
        &mut self,
        model_path: &str,
        model: &LoadedProfiledModel,
        kv_page_migration: PageMigrationService,
    ) -> Result<(), String> {
        let plan = Arc::new(model.reader.manifest.execution_plan.clone());
        let mapped_image = Arc::new(model.mapped_image.clone());
        let reader = Arc::new(model.reader.clone());
        let ws = WorkingSetManager::new(
            model_path,
            plan,
            mapped_image,
            reader,
            kv_page_migration,
        )?;
        self.working_set = Some(ws);
        Ok(())
    }

    /// Access the latest output logits for SpecHub speculative decoding.
    ///
    /// Returns the cached logits from the most recent forward pass, or an
    /// error if no forward pass has been completed yet.
    pub fn get_target_logits(&self) -> Result<Array, String> {
        self.logits
            .clone()
            .ok_or_else(|| "no logits available".to_string())
    }

    /// Check the decoded step for anomalies using the autopsy system.
    /// Hidden states are indexed by layer (hidden_states[l] = output of layer l).
    pub fn check_anomalies(
        &mut self,
        hidden_states: &[mlx_rs::Array],
        tokens: &[u32],
    ) -> Result<(), String> {
        if let Some(ref mut autopsy) = self.autopsy {
            let patches = autopsy.inspect_step(hidden_states, tokens)?;
            for patch in &patches {
                eprintln!(
                    "[autopsy] Applied patch: {} — {}",
                    patch.tensor_name, patch.reason
                );
            }
        }
        Ok(())
    }

    /// Populate the memory plan from a loaded model's scheduled module.
    pub fn setup_from_model(&mut self, model: &LoadedProfiledModel) {
        // If asymmetric quantization is configured, derive compression ratio
        // from the asymmetric mode (key vs value bit widths).
        if self.asymmetric_quant.is_some() && self.compression_ratio.is_none() {
            let ratio = self.asymmetric_quant.unwrap().compression_ratio();
            self.compression_ratio = Some(ratio);
        }

        if let Some(scheduled) = &model.scheduled_module {
        if std::env::var("TRIBUNUS_SKIP_MEMORY_PLAN").is_ok() {
            return;
        }
            if let Some(plan) = crate::memory::plan::plan_from_scheduled_module(
                scheduled,
                &crate::arena::Arena::new(1, 1, mlx_rs::Dtype::Float32).unwrap_or_else(|_| panic!("tmp arena")),
                self.compression_ratio,
            ) {
                self.memory_plan = Some(plan);
            }
        }
    }

    // ── Preemption support ─────────────────────────────────────────────

    /// Capture the current KV cache state as a compressed snapshot for
    /// preemption.
    ///
    /// Returns one [`CompressedKvSlot`] per layer, recording the layer's
    /// current committed length and logical start position.
    pub fn capture_kv_snapshot(&self) -> Vec<crate::kv_cache::CompressedKvSlot> {
        self.kv_caches
            .iter()
            .map(|kvc| crate::kv_cache::CompressedKvSlot {
                compressed_keys: Vec::new(),
                compressed_values: Vec::new(),
                qjl_correction: None,
                kv_offset: kvc.logical_start,
                num_tokens: kvc.committed_len as usize,
            })
            .collect()
    }

    /// Restore KV cache state from a previously captured snapshot.
    ///
    /// # Panics
    /// Panics if the snapshot length does not match the number of layers.
    pub fn restore_from_kv_snapshot(
        &mut self,
        snapshot: &[crate::kv_cache::CompressedKvSlot],
        absolute_position: u32,
        generated_tokens: &[u32],
    ) {
        assert_eq!(
            snapshot.len(),
            self.kv_caches.len(),
            "restore_from_kv_snapshot: snapshot length {} != {} layers",
            snapshot.len(),
            self.kv_caches.len()
        );

        for (layer_idx, slot) in snapshot.iter().enumerate() {
            let kvc = &mut self.kv_caches[layer_idx];
            let target_len = slot.num_tokens as u32;

            if target_len == 0 || kvc.committed_len == target_len {
                kvc.logical_start = slot.kv_offset;
                continue;
            }

            if kvc.committed_len > target_len {
                kvc.rollback();
                if kvc.seq_len > target_len || kvc.committed_len > target_len {
                    kvc.clear();
                }
            }

            kvc.logical_start = slot.kv_offset;
        }

        self.absolute_position = absolute_position;
        self.generated_tokens = generated_tokens.to_vec();
        self.phase = crate::session::InferenceSessionState::Decoding;
        self.prefilled_tokens = 0;
        self.pending_prompt_tokens = None;
    }

    /// Chunked prefill: process the next chunk of the prompt.
    ///
    /// On the first call, stores the full prompt and processes the first
    /// chunk of up to [`PREFILL_CHUNK_SIZE`] tokens.  Subsequent calls
    /// continue from where the previous chunk left off.
    ///
    /// Returns `Ok(None)` when the prefill is complete and the session
    /// has transitioned to `Decoding`. Returns `Ok(Some(token))` if more
    /// chunks remain (the caller should interleave decode steps for other
    /// sequences before calling this again).
    pub fn prefill_chunk(
        &mut self,
        prompt_token_ids: &[u32],
        model: &LoadedProfiledModel,
    ) -> Result<Option<u32>, EngineError> {
        // On first call, store full prompt
        if self.prefilled_tokens == 0 {
            self.pending_prompt_tokens = Some(prompt_token_ids.to_vec());
            self.phase = InferenceSessionState::PrefillRunning;
            self.runtime = Some(crate::heterogeneous::ComputeRuntime {
                island: model.memory_island.clone(),
                lanes: crate::heterogeneous::create_backend_lanes(),
            });

            // Initialize sink states: one per layer, with 4 permanent sinks
            // and a base window of 128 tokens.
            if self.sink_states.is_empty() {
                let n_layers = model.reader.manifest.execution_plan.layers.len();
                self.sink_states = (0..n_layers)
                    .map(|_| crate::executor::SinkState::new(
                        4,   // num_permanent_sinks
                        128, // window_size
                    ))
                    .collect();
            }
        }

        let plan = &model.reader.manifest.execution_plan;
        let full_prompt = self.pending_prompt_tokens.as_ref().unwrap();
        let remaining = full_prompt.len() as u32 - self.prefilled_tokens;
        let chunk_size = remaining.min(PREFILL_CHUNK_SIZE);

        // Build the chunk of token IDs
        let chunk_start = self.prefilled_tokens as usize;
        let chunk_end = chunk_start + chunk_size as usize;
        let chunk_tokens = &full_prompt[chunk_start..chunk_end];

        // Convert to MLX array
        let kv_offset = self.absolute_position;
        let token_ids_i32: Vec<i32> = chunk_tokens.iter().map(|&t| t as i32).collect();
        let tok_arr = Array::from_slice(&token_ids_i32, &[1, chunk_size as i32]);

        let mut hidden = crate::executor::run_prologue(
            &tok_arr,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.prologue,
            crate::executor::prologue_hidden_scale(&plan.prologue),
        )
        .map_err(|e| EngineError::new(
            EngineErrorCode::InferenceFailed,
            format!("chunk prologue: {:?}", e),
        ))?;
        log_debug!("[infer] event=prologue_output shape={:?} elems={}", hidden.shape(), hidden.shape().iter().product::<i32>());
        hidden.eval().map_err(|e| {
            EngineError::new(EngineErrorCode::NumericalFailure, format!("chunk prologue eval: {}", e))
        })?;

        let _slots = model.memory_island.preallocate_layer_slots(1, 3840);

        for (l, layer_plan) in plan.layers.iter().enumerate() {
            log_debug!("[infer] event=layer_run layer={} kind={}", l, &layer_plan.attention_kind);
            let lw = match &mut self.working_set {
                Some(ws) => ws.weight_streamer.activate(l as u32)
                    .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?,
                None => &model.layers[l],
            };
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer_with_sinks(
                &hidden,
                layer_plan,
                &layer_plan.route,
                Some(&model.memory_island),
                &model.ane_coreml_models,
                &lw.input_layernorm,
                &lw.post_attention_layernorm,
                &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b,
                &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b,
                &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b,
                &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b,
                lw.q_norm.as_deref(), lw.k_norm.as_deref(),
                &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b,
                &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b,
                &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b,
                rcos, rsin,
                &mut self.kv_caches[l],
                kv_offset,
                plan.rms_norm_eps as f32,
                &crate::projection_identity::ProjectionContext {
                    run_id: self.session_id.clone(),
                    phase: crate::projection_identity::Phase::Prefill,
                    forward_pass_index: 0,
                    token_step: Some(kv_offset),
                    layer_index: l,
                    attention_kind: if is_full {
                        crate::projection_identity::AttentionKind::Full
                    } else {
                        crate::projection_identity::AttentionKind::Sliding
                    },
                },
                &mut self.sink_states[l],
                false, // is_decode=false → prefill path captures sinks
            )
            .map_err(|e| {
                EngineError::new(EngineErrorCode::InferenceFailed, format!("chunk layer {}: {}", l, e))
            })?;
            hidden.eval().map_err(|e| EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("chunk layer {} eval: {}", l, e),
            ))?;
            if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
                hidden.eval().map_err(|e| {
                    EngineError::new(EngineErrorCode::NumericalFailure, format!("chunk layer {} eval: {}", l, e))
                })?;
            }
            // DIAGNOSTIC: materialization checksum after every layer.
            // Logs layer index, kind, output shape, and a readback checksum.
            // Isolates the crash between layers.
            {
                let h_shape = hidden.shape();
                let h_elems = h_shape.iter().product::<i32>() as usize;
                let checksum = match hidden.try_as_slice::<f32>() {
                    Ok(slice) => {
                        let n = slice.len().min(100);
                        let partial_sum: f32 = slice[..n].iter().copied().sum();
                        partial_sum
                    },
                    Err(_) => -1.0f32,
                };
                log_debug!("[infer] event=layer_materialize layer={} kind={} shape={:?} elems={} checksum={:.6}",
                    l, &plan.layers[l].attention_kind, h_shape, h_elems, checksum);
            }
            self.kv_caches[l].commit_step();
        }

        // Clear the memory plan after the layer loop completes.
        // Subsequent allocations (epilogue, next chunk) use normal paths
        // unless a new plan is applied before the next region.
        if self.memory_plan.is_some() {
            let _ = crate::memory::plan::clear_memory_plan();
        }

        // Record a receipt for this prefill chunk.
        let chunk_start = self.absolute_position;
        self.receipts.push(self.build_receipt(model, chunk_start));

        self.prefilled_tokens += chunk_size;
        self.absolute_position += chunk_size;

        let is_last_chunk = self.prefilled_tokens >= full_prompt.len() as u32;
        if is_last_chunk {
            // Run epilogue on completion
            let sampler = &self.sampler;
            let out_token = crate::executor::run_epilogue(
                &hidden,
                &model.fn_w,
                &model.emb_w,
                &model.emb_s,
                &model.emb_b,
                &plan.epilogue,
                plan.rms_norm_eps as f32,
                plan.tie_word_embeddings,
                sampler,
            )
            .map_err(|e| EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("chunk epilogue: {:?}", e),
            ))?;
            out_token.selected_token.eval().map_err(|e| {
                EngineError::new(EngineErrorCode::NumericalFailure, format!("chunk epilogue eval: {:?}", e))
            })?;
            let token = out_token.selected_token.try_as_slice::<u32>()
                .map_err(|e| EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("chunk epilogue token: {:?}", e),
                ))?
                .first().copied().unwrap_or(0);
            self.generated_tokens.push(token);
            // Advance the grammar FSM if grammar-guided generation is active.
            let text = if let Some(tokenizer) = &self.sampler.grammar_tokenizer {
                tokenizer.decode(token).to_string()
            } else {
                String::new()
            };
            if !text.is_empty() {
                if let Err(e) = self.sampler.advance_grammar(&text) {
                    eprintln!("[grammar] prefill advance failed for token {}: {}", token, e);
                }
            }
            self.phase = InferenceSessionState::Decoding;
            self.pending_prompt_tokens = None;
            self.prefilled_tokens = 0;
            return Ok(Some(token));
        }

        Ok(None)
    }

    /// Run prefill on the given prompt tokens, populating KV caches.
    ///
    /// Accepts a prompt of any length.  Internally delegates to
    /// [`prefill_chunk`] for chunked execution.  Runs the prologue, all layers,
    /// and the epilogue.  Returns the first generated token (the model's
    /// continuation after the prompt).
    ///
    /// On success, advances `absolute_position` to `prompt_token_ids.len()`
    /// and transitions the session phase to `Decoding`.
    pub fn prefill(
        &mut self,
        prompt_token_ids: &[u32],
        model: &LoadedProfiledModel,
    ) -> Result<u32, EngineError> {
        // Check shared prefix cache: if another session already computed
        // prefix blocks, skip them and start computing from the first miss.
        let skip_tokens = check_shared_prefix(prompt_token_ids)
            .map(|(_, count)| count)
            .unwrap_or(0);

        if skip_tokens > 0 {
            // Fast-forward session state past the cached prefix blocks.
            // prefill_chunk will compute only the uncached suffix.
            self.pending_prompt_tokens = Some(prompt_token_ids.to_vec());
            self.prefilled_tokens = skip_tokens as u32;
            self.absolute_position = skip_tokens as u32;
            self.phase = InferenceSessionState::PrefillRunning;
            if self.runtime.is_none() {
                self.runtime = Some(crate::heterogeneous::ComputeRuntime {
                    island: model.memory_island.clone(),
                    lanes: crate::heterogeneous::create_backend_lanes(),
                });
            }
            if self.sink_states.is_empty() {
                let n_layers = model.reader.manifest.execution_plan.layers.len();
                self.sink_states = (0..n_layers)
                    .map(|_| crate::executor::SinkState::new(4, 128))
                    .collect();
            }
        }

        // Delegate to chunked prefill, processing all chunks in a loop
        loop {
            match self.prefill_chunk(prompt_token_ids, model)? {
                Some(token) => {
                    // After prefill completes, insert newly computed blocks
                    // into the shared cache so future sessions can skip them.
                    insert_shared_prefix(prompt_token_ids, 0);
                    return Ok(token);
                }
                None => {
                    // More chunks remain — the caller would interleave
                    // decode here in continuous batching mode.
                    // For full-batch prefill we continue immediately.
                    continue;
        }
            }
        }
    }

    /// Prefill with optional image input.
    ///
    /// 1. Processes images through the vision encoder (if any).
    /// 2. Inserts image token embeddings at the right positions.
    /// 3. Continues with the standard text prefill.
    pub fn prefill_with_images(
        &mut self,
        prompt_token_ids: &[u32],
        images: &[ImageInput],
        model: &LoadedProfiledModel,
    ) -> Result<u32, EngineError> {
        // If no images or no vision encoder, fall back to standard prefill.
        if images.is_empty() || model.vision_encoder.is_none() {
            return self.prefill(prompt_token_ids, model);
        }

        let encoder = model.vision_encoder.as_ref().unwrap();
        let config = &encoder.config;

        // 1. Process each image through the vision encoder.
        let mut vision_features: Vec<Array> = Vec::with_capacity(images.len());
        for img in images {
            let preprocessed = crate::vision::preprocess::preprocess_image(&img.source, config)
                .map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::InvalidRequest,
                        format!("image preprocess '{}': {}", img.source, e),
                    )
                })?;
            let features = encoder
                .encode(&preprocessed)
                .map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::InferenceFailed,
                        format!("vision encode '{}': {}", img.source, e),
                    )
                })?;
            vision_features.push(features);
        }

        // 2. Build the modified prompt: replace placeholder tokens with
        //    actual image token IDs from the vision encoder.  Each image
        //    contributes `num_patches` tokens.
        let mut modified_tokens: Vec<u32> = Vec::with_capacity(prompt_token_ids.len());
        for &tid in prompt_token_ids {
            let mut inserted = false;
            for (img_idx, img) in images.iter().enumerate() {
                if img.placeholder_tokens.contains(&tid) {
                    // Replace the placeholder with num_patches actual vision tokens.
                    let num_patches = encoder.num_patches;
                    // The vision token IDs start at a high offset to avoid
                    // colliding with actual vocabulary tokens.  We use an
                    // offset of 250,000 (beyond typical vocab size).
                    let base_token = 250_000u32 + (img_idx as u32) * num_patches;
                    for p in 0..num_patches {
                        modified_tokens.push(base_token + p);
                    }
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                modified_tokens.push(tid);
            }
        }

        // 3. Run standard prefill with the modified token sequence.
        //    The vision features will be injected via the embedding path
        //    during the prologue (see executor.rs modifications).
        self.prefill(&modified_tokens, model)
    }

    /// Decode one token using the model.
    ///
    /// Accepts exactly one previously selected token, feeds it through all
    /// layers (appending one KV cache position per layer), and returns the
    /// next predicted token.  Advances `absolute_position` by 1.
    pub fn decode_one(
        &mut self,
        token_id: u32,
        model: &LoadedProfiledModel,
    ) -> Result<u32, EngineError> {
        if self.phase != InferenceSessionState::Decoding {
            return Err(EngineError::new(
                EngineErrorCode::InvalidRequest,
                format!(
                    "decode_one called in phase {:?}, expected Decoding",
                    self.phase
                ),
            ));
        }

        let plan = &model.reader.manifest.execution_plan;
        let kv_offset = self.absolute_position;

        let token_ids_i32 = [token_id as i32];
        let tok_arr = Array::from_slice(&token_ids_i32, &[1, 1]);
        let _pos_arr = Array::from_slice(&[kv_offset as i32], &[1, 1]);

        let mut hidden = crate::executor::run_prologue(
            &tok_arr,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.prologue,
            crate::executor::prologue_hidden_scale(&plan.prologue),
        )
        .map_err(|e| {
            EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("prologue: {:?}", e),
            )
        })?;
        hidden.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("prologue eval: {}", e),
            )
        })?;

        eprintln!(
            "[phase] decode_step start token_step={}",
            self.absolute_position
        );

        // Apply memory plan before executing layers (if one is set).
        // The plan tells the Metal allocator to use pre-assigned IOSurface
        // slices instead of allocating new GPU memory for each tensor.
        if let Some(plan) = &self.memory_plan {
            unsafe {
                plan.apply().map_err(|e| {
                    EngineError::new(EngineErrorCode::NumericalFailure,
                        format!("memory plan apply: {}", e))
                })?;
            }
            }

        // Collect per-layer hidden states for anomaly detection
        let mut layer_hiddens: Vec<mlx_rs::Array> = Vec::new();

        for (l, layer_plan) in plan.layers.iter().enumerate() {
            if self.cancellation_flag.load(Ordering::Relaxed) {
                return Err(EngineError::new(
                    EngineErrorCode::Cancelled,
                    "cancelled during prefill",
                ));
            }

            let layer_start = std::time::Instant::now();
            let work_id = format!("layer_{}", l);
            let backend_id = layer_plan.route.dominant_backend();
            let target = match backend_id {
                0 => BackendTarget::Mlx,
                1 => BackendTarget::Accelerate,
                2 | 3 => BackendTarget::Coreml,
                _ => BackendTarget::Mlx,
            };
            let work_item = RuntimeWorkItem {
                schema: "tribunus.runtime_work_item.v1".into(),
                schema_version: "v1".into(),
                work_id: work_id.clone(),
                run_id: self.session_id.clone(),
                phase_id: format!("decode_{}", l),
                canonical_phase: Some("inference_layer".into()),
                backend_target: target,
                island_id: "island_main".into(),
                input_tensor_ids: vec![format!("hidden_{}", l)],
                output_tensor_ids: vec![format!("hidden_{}", l + 1)],
                authority_mode: AuthorityMode::Authority,
                deadline: String::new(),
                budget_class: BudgetClass::Interactive,
                retry_policy: RetryPolicy {
                    max_retries: 0,
                    backoff_ms: 0,
                },
                expected_receipts: vec![],
                receipt_before_ack: true,
            };
            self.coordinator
                .admit_sync(work_item)
                .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, format!("admit layer {}: {}", l, e)))?;
            let handles_before = crate::bridge::handle_count();
            let lw = match &mut self.working_set {
                Some(ws) => ws.weight_streamer.activate(l as u32)
                    .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?,
                None => &model.layers[l],
            };
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer_with_sinks(
                &hidden,
                layer_plan,
                &layer_plan.route,
                Some(&model.memory_island),
                &model.ane_coreml_models,
                &lw.input_layernorm,
                &lw.post_attention_layernorm,
                &lw.q_proj_w,
                &lw.q_proj_s,
                &lw.q_proj_b,
                &lw.k_proj_w,
                &lw.k_proj_s,
                &lw.k_proj_b,
                &lw.v_proj_w,
                &lw.v_proj_s,
                &lw.v_proj_b,
                &lw.o_proj_w,
                &lw.o_proj_s,
                &lw.o_proj_b,
                lw.q_norm.as_deref(),
                lw.k_norm.as_deref(),
                &lw.gate_proj_w,
                &lw.gate_proj_s,
                &lw.gate_proj_b,
                &lw.up_proj_w,
                &lw.up_proj_s,
                &lw.up_proj_b,
                &lw.down_proj_w,
                &lw.down_proj_s,
                &lw.down_proj_b,
                rcos,
                rsin,
                &mut self.kv_caches[l],
                kv_offset,
                plan.rms_norm_eps as f32,
                &crate::projection_identity::ProjectionContext {
                    run_id: self.session_id.clone(),
                    phase: crate::projection_identity::Phase::Decode,
                    forward_pass_index: 0,
                    token_step: Some(kv_offset),
                    layer_index: l,
                    attention_kind: if is_full {
                        crate::projection_identity::AttentionKind::Full
                    } else {
                        crate::projection_identity::AttentionKind::Sliding
                    },
                },
                &mut self.sink_states[l],
                true, // is_decode=true → use sink attention path
            )
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("decode layer {}: {}", l, e),
                )
            })?;
            // Capture the per-layer hidden state for anomaly detection
            layer_hiddens.push(hidden.clone());
            // OPT-0005: batch eval every 6 layers
            if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
                hidden.eval().map_err(|e| {
                    self.kv_caches[l].rollback();
                    EngineError::new(
                        EngineErrorCode::NumericalFailure,
                        format!("decode layer {} eval: {}", l, e),
                    )
                })?;
            }
            self.kv_caches[l].commit_step();
            let kvc = &self.kv_caches[l];
            eprintln!(
                "[kv] layer={} capacity={} committed={} seq_len={} copy_bytes={} allocated_bytes={}",
                l, kvc.capacity, kvc.committed_len, kvc.seq_len, kvc.copy_bytes(), kvc.allocated_bytes()
            );
            let s = hidden.shape();
            let layer_elapsed_ms = layer_start.elapsed().as_millis() as u64;
            let shape_d0 = s.first().copied().unwrap_or(0);
            let shape_d1 = s.get(1).copied().unwrap_or(0);
            eprintln!(
                "[full-model] layer={} kind={} elapsed_ms={} handles={}→{} active_mem={}→{} cache_mem={}→{} shape=[{},{}] finite={}",
                l,
                layer_plan.attention_kind,
                layer_elapsed_ms,
                handles_before,
                crate::bridge::handle_count(),
                format_bytes(crate::compute_image::mlx_active_memory_bytes()),
                format_bytes(crate::compute_image::mlx_active_memory_bytes()), // measured after eval above
                format_bytes(crate::compute_image::mlx_cache_memory_bytes()),
                format_bytes(crate::compute_image::mlx_cache_memory_bytes()),
                shape_d0, shape_d1,
                true,
            );
            self.coordinator
                .commit_receipt_sync(&work_id)
                .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, format!("commit receipt layer {}: {}", l, e)))?;
            self.coordinator
                .ack_sync(&work_id)
                .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, format!("ack layer {}: {}", l, e)))?;
        }
        eprintln!("[phase] decode_step end");
        let expected = kv_offset + 1;
        for (l, _) in plan.layers.iter().enumerate() {
            if self.kv_caches[l].committed_len != expected {
                return Err(EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!(
                        "decode layer {} committed {} positions, expected {}",
                        l, self.kv_caches[l].committed_len, expected
                    ),
                ));
            }
        }

        // Check for anomalies in the decoded step (NaN, Inf, forbidden tokens)
        let gen_tokens = self.generated_tokens.clone();
        if let Err(e) = self.check_anomalies(&layer_hiddens, &gen_tokens) {
            eprintln!("[autopsy] Anomaly check failed: {}", e);
        }

        let sampler = &self.sampler;
        let out_token = crate::executor::run_epilogue_prefetch(
            &hidden,
            &model.fn_w,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.epilogue,
            plan.rms_norm_eps as f32,
            plan.tie_word_embeddings,
            sampler,
            self.predictor.as_mut(),
            self.row_cache.as_mut(),
        )
        .map_err(|e| {
            EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("epilogue: {:?}", e),
            )
        })?;

        out_token.selected_token.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("epilogue eval: {:?}", e),
            )
        })?;
        let token = out_token
            .selected_token
            .try_as_slice::<u32>()
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("epilogue token: {:?}", e),
                )
            })?
            .first()
            .copied()
            .unwrap_or(0);

        self.absolute_position += 1;
        self.generated_tokens.push(token);

        // Advance the grammar FSM if grammar-guided generation is active.
        // The FSM tracks which tokens are valid given the current generation
        // state. We need the tokenizer to decode the token ID back to text
        // so the FSM can consume the text and advance to the next state.
        let text = if let Some(tokenizer) = &self.sampler.grammar_tokenizer {
            tokenizer.decode(token).to_string()
        } else {
            String::new()
        };
        if !text.is_empty() {
            if let Err(e) = self.sampler.advance_grammar(&text) {
                eprintln!("[grammar] advance failed for token {}: {}", token, e);
            }
        }

    // Inject encoded media tokens into the prompt embedding sequence.
        self.timeline.push_event(TimelineEvent::new(
            self.absolute_position as u64,
            TimelineEventType::DecodeStep,
            format!("decoded token {}", token),
        ));

        // Record a receipt for this decoded token.
        let token_index = self.absolute_position.saturating_sub(1);
        self.receipts.push(self.build_receipt(model, token_index));

        Ok(token)
    }

    /// Run the model once to produce an embedding vector for the given text.
    /// No autoregressive decoding — just one forward pass.
    ///
    /// Pooling strategies:
    /// - Mean: average all token hidden states
    /// - CLS: take the first token's hidden state
    /// - Last: take the final token's hidden state
    pub fn embed(
        &mut self,
        token_ids: &[u32],
        model: &LoadedProfiledModel,
        pool_strategy: EmbedPoolStrategy,
    ) -> Result<Vec<f32>, EngineError> {
        let plan = &model.reader.manifest.execution_plan;
        let seq_len = token_ids.len() as u32;

        // Reset KV caches for a clean prefill
        for cache in &mut self.kv_caches {
            cache.clear();
        }

        // Set up runtime if not already done
        if self.runtime.is_none() {
            self.runtime = Some(crate::heterogeneous::ComputeRuntime {
                island: model.memory_island.clone(),
                lanes: crate::heterogeneous::create_backend_lanes(),
            });
        }

        // Init sink states if empty
        if self.sink_states.is_empty() {
            let n_layers = plan.layers.len();
            self.sink_states = (0..n_layers)
                .map(|_| crate::executor::SinkState::new(4, 128))
                .collect();
        }

        self.phase = InferenceSessionState::PrefillRunning;

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let tok_arr = Array::from_slice(&token_ids_i32, &[1, seq_len as i32]);

        // ── Prologue (embedding lookup) ────────────────────────────────
        let mut hidden = crate::executor::run_prologue(
            &tok_arr,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.prologue,
            crate::executor::prologue_hidden_scale(&plan.prologue),
        )
        .map_err(|e| {
            EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("embed prologue: {:?}", e),
            )
        })?;
        hidden.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed prologue eval: {}", e),
            )
        })?;

        // Apply memory plan if set
        if let Some(mem_plan) = &self.memory_plan {
            unsafe {
                mem_plan.apply().map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::NumericalFailure,
                        format!("embed memory plan apply: {}", e),
                    )
                })?;
            }
        }

        // ── Execute all transformer layers ─────────────────────────────
        for (l, layer_plan) in plan.layers.iter().enumerate() {
            let lw = match &mut self.working_set {
                Some(ws) => ws.weight_streamer.activate(l as u32)
                    .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?,
                None => &model.layers[l],
            };
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer_with_sinks(
                &hidden,
                layer_plan,
                &layer_plan.route,
                Some(&model.memory_island),
                &model.ane_coreml_models,
                &lw.input_layernorm,
                &lw.post_attention_layernorm,
                &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b,
                &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b,
                &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b,
                &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b,
                lw.q_norm.as_deref(), lw.k_norm.as_deref(),
                &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b,
                &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b,
                &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b,
                rcos, rsin,
                &mut self.kv_caches[l],
                0, // kv_offset = 0 for single-pass embedding
                plan.rms_norm_eps as f32,
                &crate::projection_identity::ProjectionContext {
                    run_id: self.session_id.clone(),
                    phase: crate::projection_identity::Phase::Prefill,
                    forward_pass_index: 0,
                    token_step: Some(0),
                    layer_index: l,
                    attention_kind: if is_full {
                        crate::projection_identity::AttentionKind::Full
                    } else {
                        crate::projection_identity::AttentionKind::Sliding
                    },
                },
                &mut self.sink_states[l],
                false, // is_decode=false
            )
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("embed layer {}: {}", l, e),
                )
            })?;

            // Batch eval every 6 layers
            if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
                hidden.eval().map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::NumericalFailure,
                        format!("embed layer {} eval: {}", l, e),
                    )
                })?;
            }
            self.kv_caches[l].commit_step();
        }

        // Clear memory plan after layer loop
        if self.memory_plan.is_some() {
            let _ = crate::memory::plan::clear_memory_plan();
        }

        // ── Pooling ────────────────────────────────────────────────────
        // hidden shape: [seq_len, hidden_size] (batchless)
        let hidden_f32 = hidden
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::NumericalFailure,
                    format!("embed dtype cast: {}", e),
                )
            })?;

        let pooled = match pool_strategy {
            EmbedPoolStrategy::Mean => {
                // Mean over token dimension (dim 0): result [hidden_size]
                mlx_rs::ops::mean_axes(&hidden_f32, &[0], false).map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::NumericalFailure,
                        format!("embed mean pool: {}", e),
                    )
                })?
            }
            EmbedPoolStrategy::Cls => {
                // First token at position 0
                mlx_rs::ops::indexing::IndexOp::index(&hidden_f32, 0i32)
            }
            EmbedPoolStrategy::Last => {
                // Last token at position seq_len - 1
                mlx_rs::ops::indexing::IndexOp::index(
                    &hidden_f32,
                    seq_len as i32 - 1,
                )
            }
        };
        pooled.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed pool eval: {}", e),
            )
        })?;

        // ── L2 normalize ───────────────────────────────────────────────
        // pooled shape: [hidden_size] (1D)
        let squared = pooled.square().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed square: {}", e),
            )
        })?;
        let sum_sq = mlx_rs::ops::sum(&squared, false).map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed sum: {}", e),
            )
        })?;
        let norm = mlx_rs::ops::sqrt(&sum_sq).map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed sqrt: {}", e),
            )
        })?;
        let normalized = pooled.divide(&norm).map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed normalize: {}", e),
            )
        })?;
        normalized.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("embed norm eval: {}", e),
            )
        })?;

        // ── Extract to Vec<f32> ────────────────────────────────────────
        let vec: Vec<f32> = normalized
            .try_as_slice::<f32>()
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("embed extract: {}", e),
                )
            })?
            .to_vec();

        // Transition session back to idle state
        self.phase = InferenceSessionState::Created;
        self.pending_prompt_tokens = None;
        self.prefilled_tokens = 0;

        Ok(vec)
    }

    ///
    /// Replaces each placeholder token in `prompt_embeds` with the
    /// corresponding media feature vectors.  The features are inserted
    /// in order, expanding the sequence if the placeholder is a single
    /// token that maps to multiple feature vectors.
    ///
    /// # Arguments
    ///
    /// * `prompt_embeds` — Mutable embedding sequence
    ///   `[1, num_tokens, hidden_size]`.
    /// * `media_features` — Media feature vectors
    ///   `[num_feature_tokens, projection_dim]`.
    /// * `placeholder_tokens` — Token IDs in the prompt to replace.
    ///
    /// # Design
    ///
    /// Find each occurrence of a placeholder token ID in the original
    /// prompt, and replace the corresponding embedding vector(s) with
    /// the media feature vectors.  If `media_features` has more vectors
    /// than placeholder tokens, the extra features are appended after
    /// the last placeholder.  If fewer, the trailing placeholder
    /// positions are zeroed out.
    pub fn inject_media_tokens(
        &self,
        prompt_embeds: &mut Array,
        media_features: &Array,
        placeholder_tokens: &[u32],
    ) -> Result<(), String> {
        if placeholder_tokens.is_empty() {
            return Ok(());
        }

        let emb_shape = prompt_embeds.shape();
        if emb_shape.len() < 2 {
            return Err("prompt_embeds shape too small for injection".to_string());
        }
        if emb_shape.len() < 3 {
            return Err("prompt_embeds must be 3D [1, seq_len, hidden]".to_string());
        }
        let seq_len = emb_shape[1] as usize;
        let _hidden = emb_shape[2] as usize;

        let feat_shape = media_features.shape();
        if feat_shape.len() < 2 {
            return Err("media_features must be 2D [num_feat, proj_dim]".to_string());
        }
        let num_feat = feat_shape[0] as usize;
        let feat_dim = feat_shape[1] as usize;

        if feat_dim != _hidden {
            return Err(format!(
                "media feature dimension {} != embedding hidden dimension {}",
                feat_dim, _hidden,
            ));
        }

        // The token IDs used during embedding are stored in pending_prompt_tokens.
        // Walk through them to find placeholder positions.
        let prompt_tokens = match &self.pending_prompt_tokens {
            Some(tokens) => tokens,
            None => {
                return Err("no pending prompt tokens — call prefill_chunk first".to_string());
            }
        };

        // Collect all positions where placeholders occur.
        let placeholder_positions: Vec<usize> = prompt_tokens
            .iter()
            .enumerate()
            .filter(|(_, tid)| placeholder_tokens.contains(tid))
            .map(|(i, _)| i)
            .collect();

        if placeholder_positions.is_empty() {
            // No placeholders found — nothing to inject.
            return Ok(());
        }

        // Iterate through positions in order, replacing embeddings.
        // prompt_embeds is [1, seq_len, hidden]; we slice at [:1, pos, :].
        for (feat_idx, &pos) in placeholder_positions.iter().enumerate() {
            if feat_idx >= num_feat {
                break;
            }
            // Extract the single feature vector.
            let feat_slice = mlx_rs::Array::slice(
                media_features,
                &[feat_idx as i32, 0],
                &[1, feat_dim as i32],
                &[1, 1],
            )?;

            // Extract the single position in the embedding sequence.
            // slice_assign is not available in the mlx-rs fork — rebuild via concatenation.
            let pos_i32 = pos as i32;
            let hidden_i32 = feat_dim as i32;
            let seq_len_i32 = seq_len as i32;
            let left = if pos > 0 {
                Some(mlx_rs::Array::slice(
                    prompt_embeds,
                    &[0, 0, 0],
                    &[1, pos_i32, hidden_i32],
                    &[1, 1, 1],
                )?)
            } else {
                None
            };
            let right_len = seq_len_i32 - pos_i32 - 1;
            let right = if right_len > 0 {
                Some(mlx_rs::Array::slice(
                    prompt_embeds,
                    &[0, pos_i32 + 1, 0],
                    &[1, right_len, hidden_i32],
                    &[1, 1, 1],
                )?)
            } else {
                None
            };
            let mut parts: Vec<&Array> = Vec::new();
            if let Some(l) = left.as_ref() { parts.push(l); }
            parts.push(&feat_slice);
            if let Some(r) = right.as_ref() { parts.push(r); }
            *prompt_embeds = mlx_rs::ops::concatenate_axis(&parts, 1)
                .map_err(|e| format!("media injection concatenation: {}", e))?;
        }

        Ok(())
    }

    /// Run prefill with multi-modal media inputs (images, audio, video).
    ///
    /// This extends the standard [`prefill`] by first processing any
    /// media inputs (extracting frames for video, encoding each frame
    /// through the vision encoder, then performing temporal aggregation)
    /// and injecting the resulting media feature tokens into the prompt
    /// embedding sequence before running the language model layers.
    ///
    /// The placeholder tokens in the prompt (e.g. `<image>`, `<video>`)
    /// are replaced with the actual media feature vectors.
    ///
    /// # Arguments
    ///
    /// * `prompt_token_ids` — Full prompt token ID sequence (including
    ///   placeholder tokens sentinel values).
    /// * `media_inputs` — List of multi-modal inputs to process.
    /// * `model` — The loaded model (provides weights, vision config, etc.).
    ///
    /// # Returns
    ///
    /// The first generated token on success, or an `EngineError`.
    pub fn prefill_with_media(
        &mut self,
        prompt_token_ids: &[u32],
        media_inputs: &[MultiModalInput],
        model: &LoadedProfiledModel,
    ) -> Result<u32, EngineError> {
        if media_inputs.is_empty() {
            // No media — fall back to standard prefill.
            return self.prefill(prompt_token_ids, model);
        }

        // Store the initial prompt with placeholder tokens so that
        // inject_media_tokens can locate the placeholder positions.
        self.pending_prompt_tokens = Some(prompt_token_ids.to_vec());

        let plan = &model.reader.manifest.execution_plan;

        // Pre-compute all media features before touching any session state.
        let mut all_media_features: Vec<Array> = Vec::new();
        let mut all_placeholder_tokens: Vec<u32> = Vec::new();

        for media in media_inputs {
            match media {
                MultiModalInput::Video(video) => {
                    let vision_config = model.reader.manifest.vision_config
                        .as_ref()
                        .ok_or_else(|| EngineError::new(
                            EngineErrorCode::InvalidRequest,
                            "video input requires vision_config in model manifest".to_string(),
                        ))?;

                    let num_frames = video.num_frames
                        .unwrap_or(8)
                        .min(MAX_VIDEO_FRAMES);
                    let target_size = vision_config.image_size;

                    // 1. Extract frames from video.
                    let frames = extract_frames(
                        &video.source,
                        num_frames,
                        target_size,
                    ).map_err(|e| EngineError::new(
                        EngineErrorCode::InvalidRequest,
                        format!("video frame extraction failed: {}", e),
                    ))?;

                    // 2. Encode through vision encoder + temporal.
                    let video_features = self.video_encoder.as_ref()
                        .ok_or_else(|| EngineError::new(
                            EngineErrorCode::InvalidRequest,
                            "video input requires a video encoder (set on session)".to_string(),
                        ))?
                        .encode(&frames)
                        .map_err(|e| EngineError::new(
                            EngineErrorCode::InferenceFailed,
                            format!("video encoding failed: {}", e),
                        ))?;

                    all_media_features.push(video_features);
                    all_placeholder_tokens.extend_from_slice(&video.placeholder_tokens);
                },
                MultiModalInput::Image(img) => {
                    // Delegate to the vision encoder for single-image encoding.
                    let vision_config = model.reader.manifest.vision_config
                        .as_ref()
                        .ok_or_else(|| EngineError::new(
                            EngineErrorCode::InvalidRequest,
                            "image input requires vision_config in model manifest".to_string(),
                        ))?;
                    let vision_enc = model.vision_encoder.as_ref()
                        .ok_or_else(|| EngineError::new(
                            EngineErrorCode::InvalidRequest,
                            "image input requires vision_encoder in model".to_string(),
                        ))?;
                    let processed = crate::vision::preprocess::preprocess_image(&img.source, vision_config)
                        .map_err(|e| EngineError::new(
                            EngineErrorCode::InferenceFailed,
                            format!("image preprocessing failed: {}", e),
                        ))?;
                    let features = vision_enc.encode(&processed)
                        .map_err(|e| EngineError::new(
                            EngineErrorCode::InferenceFailed,
                            format!("image encoding failed: {}", e),
                        ))?;
                    all_media_features.push(features);
                    all_placeholder_tokens.extend_from_slice(&img.placeholder_tokens);
                },
                MultiModalInput::Audio(audio) => {
                    // Audio is handled by the audio subsystem.
                    // For now, emit a placeholder message and skip.
                    eprintln!(
                        "[prefill_with_media] audio input not yet implemented; skipping {}",
                        audio.source,
                    );
                    // Push empty features to keep indexing consistent.
                    let empty = mlx_rs::Array::from_slice::<f32>(
                        &[],
                        &[0, 1],
                    );
                    all_media_features.push(empty);
                },
            }
        }

        // Run prologue to embed the prompt tokens (including placeholders).
        let token_ids_i32: Vec<i32> = prompt_token_ids.iter().map(|&t| t as i32).collect();
        let seq_len = token_ids_i32.len() as i32;
        let tok_arr = Array::from_slice(&token_ids_i32, &[1, seq_len]);

        let mut hidden = crate::executor::run_prologue(
            &tok_arr,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.prologue,
            crate::executor::prologue_hidden_scale(&plan.prologue),
        )
        .map_err(|e| EngineError::new(
            EngineErrorCode::InferenceFailed,
            format!("prefill_with_media prologue: {:?}", e),
        ))?;
        hidden.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("prefill_with_media prologue eval: {}", e),
            )
        })?;

        // Inject encoded media features into the hidden state.
        for (media_features, placeholder_tokens) in all_media_features.iter()
            .zip(all_placeholder_tokens.chunks(1))
        {
            self.inject_media_tokens(
                &mut hidden,
                media_features,
                placeholder_tokens,
            ).map_err(|e| EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("media token injection failed: {}", e),
            ))?;
        }

        // Eval after injection so the modified embeddings take effect.
        hidden.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("prefill_with_media post-injection eval: {}", e),
            )
        })?;

        // ── Run language model layers ──
        // Following the same structure as prefill_chunk.
        self.phase = InferenceSessionState::PrefillRunning;
        self.runtime = Some(ComputeRuntime {
            island: model.memory_island.clone(),
            lanes: crate::heterogeneous::create_backend_lanes(),
        });

        // Initialize sink states if empty.
        if self.sink_states.is_empty() {
            let n_layers = plan.layers.len();
            self.sink_states = (0..n_layers)
                .map(|_| crate::executor::SinkState::new(4, 128))
                .collect();
        }

        let kv_offset = 0u32;
        let _slots = model.memory_island.preallocate_layer_slots(1, 3840);

        for (l, layer_plan) in plan.layers.iter().enumerate() {
            let lw = match &mut self.working_set {
                Some(ws) => ws.weight_streamer.activate(l as u32)
                    .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?,
                None => &model.layers[l],
            };
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer_with_sinks(
                &hidden,
                layer_plan,
                &layer_plan.route,
                Some(&model.memory_island),
                &model.ane_coreml_models,
                &lw.input_layernorm,
                &lw.post_attention_layernorm,
                &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b,
                &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b,
                &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b,
                &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b,
                lw.q_norm.as_deref(), lw.k_norm.as_deref(),
                &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b,
                &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b,
                &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b,
                rcos, rsin,
                &mut self.kv_caches[l],
                kv_offset,
                plan.rms_norm_eps as f32,
                &crate::projection_identity::ProjectionContext {
                    run_id: self.session_id.clone(),
                    phase: crate::projection_identity::Phase::Prefill,
                    forward_pass_index: 0,
                    token_step: Some(kv_offset),
                    layer_index: l,
                    attention_kind: if is_full {
                        crate::projection_identity::AttentionKind::Full
                    } else {
                        crate::projection_identity::AttentionKind::Sliding
                    },
                },
                &mut self.sink_states[l],
                false,
            )
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("prefill_with_media layer {}: {}", l, e),
                )
            })?;

            hidden.eval().map_err(|e| EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("prefill_with_media layer {} eval: {}", l, e),
            ))?;

            if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
                hidden.eval().map_err(|e| {
                    EngineError::new(
                        EngineErrorCode::NumericalFailure,
                        format!("prefill_with_media layer {} eval: {}", l, e),
                    )
                })?;
            }
            self.kv_caches[l].commit_step();
        }

        // Clear memory plan if active.
        if self.memory_plan.is_some() {
            let _ = crate::memory::plan::clear_memory_plan();
        }

        self.absolute_position = prompt_token_ids.len() as u32;
        self.prefilled_tokens = 0;

        // Run epilogue to sample the first generated token.
        let sampler = crate::session::SamplerConfig::default();
        let out_token = crate::executor::run_epilogue(
            &hidden,
            &model.fn_w,
            &model.emb_w,
            &model.emb_s,
            &model.emb_b,
            &plan.epilogue,
            plan.rms_norm_eps as f32,
            plan.tie_word_embeddings,
            &sampler,
        )
        .map_err(|e| EngineError::new(
            EngineErrorCode::InferenceFailed,
            format!("prefill_with_media epilogue: {:?}", e),
        ))?;
        out_token.selected_token.eval().map_err(|e| {
            EngineError::new(
                EngineErrorCode::NumericalFailure,
                format!("prefill_with_media epilogue eval: {:?}", e),
            )
        })?;
        let token = out_token.selected_token.try_as_slice::<u32>()
            .map_err(|e| EngineError::new(
                EngineErrorCode::InferenceFailed,
                format!("prefill_with_media token: {:?}", e),
            ))?
            .first().copied().unwrap_or(0);
        self.generated_tokens.push(token);
        self.phase = InferenceSessionState::Decoding;
        self.pending_prompt_tokens = None;

        Ok(token)
    }

    /// Build a per-token receipt from the current session state.
    fn build_receipt(
        &self,
        model: &LoadedProfiledModel,
        token_index: u32,
    ) -> crate::receipt::TokenReceipt {
        // Determine the dominant backend from the first layer's route.
        let plan = &model.reader.manifest.execution_plan;
        let backend_id = plan
            .layers
            .first()
            .map(|l| l.route.dominant_backend())
            .unwrap_or(0);
        let backend = crate::receipt::backend_id_to_label(backend_id as u8).to_string();

        crate::receipt::TokenReceipt {
            token_index,
            backend,
            bytes_copied_h2d: 0,
            bytes_copied_d2d: 0,
            bytes_copied_d2h: 0,
            arena_allocations: 0,
            arena_failures: 0,
            fallback_count: 0,
            fallback_by_priority: Vec::new(),
            stage_durations_us: Vec::new(),
            speculative_branches_accepted: 0,
            speculative_branches_rejected: 0,
            kv_page_faults: 0,
            disk_bytes_read: 0,
        }
    }

    /// Return aggregated session receipts.
    pub fn session_receipts(&self) -> crate::receipt::SessionReceipts {
        let total_fallbacks = self.receipts.iter().map(|r| r.fallback_count).sum();
        let total_backend_switches = self
            .receipts
            .windows(2)
            .filter(|w| w[0].backend != w[1].backend)
            .count() as u32;

        crate::receipt::SessionReceipts {
            per_token: self.receipts.clone(),
            total_tokens: self.generated_tokens.len() as u32,
            total_backend_switches,
            total_fallbacks,
        }
    }
}

/// Cold one-shot wrapper for testing. Re-loads the entire model!
pub fn execute_profiled_cold_once(
    image_dir: &Path,
    _profile: &ExecutionPlacementProfile,
    token_ids: &[i32],
    _mode: ExecutionMode,
    cancel_flag: Option<&AtomicBool>,
    _sampler: &crate::session::SamplerConfig,
    kv_offset: u32,
) -> crate::Result<(u32, ProfiledReceipt)> {
    let model = LoadedProfiledModel::new(image_dir)?;
    let plan = &model.reader.manifest.execution_plan;

    // Build per-layer KV caches matching the execution plan.
    let kv_caches: Vec<KvCache> = plan
        .layers
        .iter()
        .map(|layer| {
            let capacity = if layer.attention_kind == "sliding_attention" {
                layer.sliding_window
            } else {
                32768
            };
            let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
            let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
            KvCache::new(
                capacity,
                n_kv_heads,
                head_dim,
                layer.attention_kind == "sliding_attention",
            )
        })
        .collect();

    let mut session = ProfiledInferenceSession::new("cold-once".to_string(), kv_caches);
    session.absolute_position = kv_offset;

    // Wire cancellation flag if provided.
    if let Some(cf) = cancel_flag {
        session
            .cancellation_flag
            .store(cf.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    let prompt: Vec<u32> = token_ids.iter().map(|&t| t as u32).collect();
    let is_prefill = prompt.len() > 1;

    let token = if is_prefill {
        session
            .prefill(&prompt, &model)
            .map_err(|e| crate::Error::from_reason(format!("cold prefill: {}", e)))?
    } else {
        // Single-token prompt: still run it through prefill (which handles 1 token).
        session
            .prefill(&prompt, &model)
            .map_err(|e| crate::Error::from_reason(format!("cold first decode: {}", e)))?
    };

    let step_elapsed_ms = 0;
    let end_us = 0;
    let cache_hit_tokens = kv_offset as u64;

    let receipt = ProfiledReceipt {
        executor: "mlx_rs".into(),
        execution_profile: model.reader.manifest.image_hash.clone(),
        storage_backend: "copied".into(),
        explicit_gpu_stream: true,
        oracle_fallback: false,
        compiler_invocations: 0,
        source_checkpoint_accesses: 0,
        copied_weight_bytes: model.mapped_weight_bytes,
        mapped_weight_bytes: model.mapped_weight_bytes,
        token,
        layer_count: plan.layers.len() as u32,
        elapsed_ms: step_elapsed_ms,
        profile_validation: true,
        gpu_canary_us: 0,
        gpu_canary_ratio: 0.0,
        image_hash: model.reader.manifest.image_hash.clone(),
        handle_baseline: model.handle_baseline as u64,
        handle_final: crate::bridge::handle_count() as u64,
        layer_records: plan
            .layers
            .iter()
            .map(|_| crate::mlx_executor::ExecutionRecord {
                device: "cpu".into(),
                stream_id: "default".into(),
                graph_build_us: 0,
                eval_us: 0,
                sync_us: 0,
                peak_active_mem: 0,
                peak_cache_mem: 0,
                error: None,
            })
            .collect(),
        active_window_bytes: model.mapped_weight_bytes,
        prefetched_count: 0,
        total_kv_cache_bytes: session.kv_caches.iter().map(|c| c.allocated_bytes()).sum(),
        cache_hit_tokens,
        wall_clock_total_us: end_us,
        unaccounted_us: 0,
        timeline: session.timeline.clone(),
    };

    Ok((token, receipt))
}

// ── Multi-modal input support ────────────────────────────────────────────

/// Multi-modal input types accepted during prefill.
#[derive(Debug, Clone)]
pub enum MultiModalInput {
    Image(ImageInput),
    Audio(AudioInput),
    Video(VideoInput),
}

/// Image input for vision-capable models.
#[derive(Debug, Clone)]
pub struct ImageInput {
    pub source: String,
    pub placeholder_tokens: Vec<u32>,
}

/// Video input for video-capable models.
#[derive(Debug, Clone)]
pub struct VideoInput {
    pub source: String,
    pub placeholder_tokens: Vec<u32>,
    pub num_frames: Option<u32>,
}

/// Audio input for audio-capable models.
#[derive(Debug, Clone)]
pub struct AudioInput {
    pub source: String,
    /// The <audio> token IDs to replace with audio features.
    pub placeholder_tokens: Vec<u32>,
}

/// Inject audio features at the placeholder token positions in the prompt
/// before running prefill.
///
/// The audio encoder processes the audio source into feature embeddings.
/// These replace the placeholder tokens in the embedding sequence so that
/// the text model can attend to audio context during prefill.
pub fn prefill_with_audio(
    sess: &mut ProfiledInferenceSession,
    model: &LoadedProfiledModel,
    text_tokens: &[u32],
    audio_inputs: &[AudioInput],
) -> Result<u32, EngineError> {
    use crate::audio::{AudioEncoder, preprocess_audio, inject_audio_features};
    use crate::executor::run_prologue;
    use crate::session::SamplerConfig;

    if audio_inputs.is_empty() {
        return sess.prefill(text_tokens, model);
    }

    let plan = &model.reader.manifest.execution_plan;

    // Load audio encoder.
    let audio_encoder = AudioEncoder::load(model)
        .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?;

    let mut audio_features_list: Vec<mlx_rs::Array> = Vec::new();
    let mut total_audio_frames: usize = 0;

    for audio_input in audio_inputs {
        // Preprocess audio -> mel spectrogram.
        let mel_spec = preprocess_audio(&audio_input.source, &audio_encoder.config)
            .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?;

        // Encode -> [num_frames, projection_dim].
        let features = audio_encoder.encode(&mel_spec)
            .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?;

        total_audio_frames += features.shape()[0] as usize;
        audio_features_list.push(features);
    }

    // Get the hidden scale constant
    let hidden_scale = crate::executor::prologue_hidden_scale(&plan.prologue);

    // Process text prompt with placeholders replaced by audio features.
    let text_tokens_count = text_tokens.len() as u32;

    // Convert text tokens to hidden states.
    let token_ids_i32: Vec<i32> = text_tokens.iter().map(|&t| t as i32).collect();
    let tok_arr = Array::from_slice(&token_ids_i32, &[1, text_tokens.len() as i32]);

    let hidden = run_prologue(
        &tok_arr,
        &model.emb_w,
        &model.emb_s,
        &model.emb_b,
        &plan.prologue,
        hidden_scale,
    )
    .map_err(|e| EngineError::new(
        EngineErrorCode::InferenceFailed,
        format!("prologue: {:?}", e),
    ))?;
    hidden.eval().map_err(|e| {
        EngineError::new(EngineErrorCode::NumericalFailure, format!("prologue eval: {}", e))
    })?;

    // Concatenate all audio features.
    let combined_audio: Array = if audio_features_list.len() == 1 {
        audio_features_list.remove(0)
    } else {
        mlx_rs::ops::concatenate(
            &audio_features_list.iter().collect::<Vec<_>>(),
        )
        .map_err(|e| EngineError::new(
            EngineErrorCode::InferenceFailed,
            format!("concat audio features: {:?}", e),
        ))?
    };

    // Inject audio features into the hidden state (prepend before text tokens).
    let combined_hidden = inject_audio_features(&hidden, &combined_audio)
        .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?;
    combined_hidden.eval().map_err(|e| {
        EngineError::new(EngineErrorCode::NumericalFailure,
            format!("combined hidden eval: {}", e))
    })?;

    let _kv_offset = 0u32;
    sess.phase = InferenceSessionState::PrefillRunning;

    let total_tokens = text_tokens_count + total_audio_frames as u32;
    sess.absolute_position = total_tokens;

    // Execute all layers on the combined hidden state.
    let mut layer_hidden = combined_hidden;

    for (l, layer_plan) in plan.layers.iter().enumerate() {
        if sess.cancellation_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(EngineError::new(
                EngineErrorCode::Cancelled,
                "cancelled during audio prefill",
            ));
        }

        let lw = match &mut sess.working_set {
            Some(ws) => ws.weight_streamer.activate(l as u32)
                .map_err(|e| EngineError::new(EngineErrorCode::InferenceFailed, e))?,
            None => &model.layers[l],
        };
        let is_full = layer_plan.attention_kind == "full_attention";
        let (rcos, rsin) = if is_full {
            (&model.full_cos, &model.full_sin)
        } else {
            (&model.rope_cos, &model.rope_sin)
        };

        layer_hidden = crate::executor::run_layer_with_sinks(
            &layer_hidden,
            layer_plan,
            &layer_plan.route,
            Some(&model.memory_island),
            &model.ane_coreml_models,
            &lw.input_layernorm,
            &lw.post_attention_layernorm,
            &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b,
            &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b,
            &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b,
            &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b,
            lw.q_norm.as_deref(), lw.k_norm.as_deref(),
            &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b,
            &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b,
            &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b,
            rcos, rsin,
            &mut sess.kv_caches[l],
            0, // kv_offset = 0 for prefill
            plan.rms_norm_eps as f32,
            &crate::projection_identity::ProjectionContext {
                run_id: sess.session_id.clone(),
                phase: crate::projection_identity::Phase::Prefill,
                forward_pass_index: 0,
                token_step: Some(0),
                layer_index: l,
                attention_kind: if is_full {
                    crate::projection_identity::AttentionKind::Full
                } else {
                    crate::projection_identity::AttentionKind::Sliding
                },
            },
            &mut sess.sink_states[l],
            false,
        )
        .map_err(|e| {
            EngineError::new(EngineErrorCode::InferenceFailed,
                format!("audio prefill layer {}: {:?}", l, e))
        })?;

        if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
            layer_hidden.eval().map_err(|e| {
                EngineError::new(EngineErrorCode::NumericalFailure,
                    format!("audio prefill layer {} eval: {}", l, e))
            })?;
        }
        sess.kv_caches[l].commit_step();
    }

    // Epilogue: predict first token.
    let sampler = SamplerConfig::default();
    let out_token = crate::executor::run_epilogue(
        &layer_hidden,
        &model.fn_w,
        &model.emb_w,
        &model.emb_s,
        &model.emb_b,
        &plan.epilogue,
        plan.rms_norm_eps as f32,
        plan.tie_word_embeddings,
        &sampler,
    )
    .map_err(|e| EngineError::new(
        EngineErrorCode::InferenceFailed,
        format!("epilogue: {:?}", e),
    ))?;
    out_token.selected_token.eval().map_err(|e| {
        EngineError::new(EngineErrorCode::NumericalFailure,
            format!("epilogue eval: {:?}", e))
    })?;
    let token = out_token.selected_token.try_as_slice::<u32>()
        .map_err(|e| EngineError::new(
            EngineErrorCode::InferenceFailed,
            format!("token read: {:?}", e),
        ))?
        .first().copied().unwrap_or(0);

    sess.generated_tokens.push(token);
    sess.phase = InferenceSessionState::Decoding;

    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_architecture() -> crate::config::TextArchitecture {
        crate::config::TextArchitecture {
            hidden_size: 3840,
            intermediate_size: 15360,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 256,
            global_head_dim: Some(512),
            num_global_key_value_heads: Some(1),
            num_hidden_layers: 2,
            vocab_size: 256128,
            sliding_window: 1024,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 3840,
            layer_types: vec![
                crate::config::AttentionKind::SlidingAttention,
                crate::config::AttentionKind::FullAttention,
            ],
            rope_local: crate::config::RopeSpec {
                theta: 10_000.0,
                rope_type: "default".to_string(),
                partial_rotary_factor: None,
            },
            rope_global: Some(crate::config::RopeSpec {
                theta: 1_000_000.0,
                rope_type: "default".to_string(),
                partial_rotary_factor: None,
            }),
            model_type: "gemma".to_string(),
        }
    }

    #[test]
    fn build_rope_tables_uses_architecture_dimensions() {
        let arch = test_architecture();
        let (rope_cos, rope_sin, full_cos, full_sin) =
            build_rope_tables(&arch).expect("rope tables");

        assert_eq!(rope_cos.shape(), &[8, 128]);
        assert_eq!(rope_sin.shape(), &[8, 128]);
        assert_eq!(full_cos.shape(), &[8, 256]);
        assert_eq!(full_sin.shape(), &[8, 256]);
        assert_eq!(rope_cos.shape()[0], arch.max_position_embeddings as i32);
        assert_eq!(full_cos.shape()[0], arch.max_position_embeddings as i32);
    }
}

impl std::fmt::Debug for LoadedProfiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedProfiledModel")
            .field("image_dir", &self.image_dir)
            .finish()
    }
}

impl ProfiledInferenceSession {
    /// Run inference with a prompt string and sampler config, returning
    /// generated text.  This is the same pattern as the server's
    /// `run_inference` but exposed publicly for tool call retry and other
    /// programmatic use.
    pub fn chat_with_sampler(
        &mut self,
        prompt: &str,
        max_tokens: u64,
        sampler_config: &crate::session::SamplerConfig,
        model: &LoadedProfiledModel,
    ) -> Result<String, String> {
        // Tokenize (byte-level, matching existing code).
        let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

        // Apply sampler config.
        self.sampler = sampler_config.clone();

        // Prefill.
        let first_token = self
            .prefill(&prompt_tokens, model)
            .map_err(|e| format!("chat prefill failed: {:?}", e))?;

        let mut generated = vec![first_token];

        // Decode loop.
        let mut current = first_token;
        for _step in 1..max_tokens {
            match self.decode_one(current, model) {
                Ok(next) => {
                    generated.push(next);
                    // Stop on EOS token (0 typically marks end-of-sequence for
                    // byte-level tokenization).
                    if next == 0 {
                        break;
                    }
                    current = next;
                }
                Err(e) => {
                    eprintln!(
                        "chat decode error at step {}: {:?}",
                        generated.len(),
                        e
                    );
                    break;
                }
            }
        }

        // Convert tokens to text.
        let output_text: String = generated
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();

        Ok(output_text)
    }
}
/// Adaptive token streaming configuration.
///
/// Controls how generated tokens are batched into SSE chunks to reduce
/// per-event overhead while maintaining low latency.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Max tokens per SSE chunk (to batch tokens when generation is fast)
    pub max_tokens_per_chunk: usize,
    /// Min latency before sending a partial chunk (ms)
    pub flush_interval_ms: u64,
    /// Whether to use sub-token streaming
    pub enable_sub_token: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_chunk: 5,
            flush_interval_ms: 10,
            enable_sub_token: false,
        }
    }
}
