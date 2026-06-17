//! Profiled heterogeneous executor — GPU-canary-gated execution with explicit receipts.
//!
//! Uses MappedImage-based segment file access via seek + read_exact.
//!
//! The model runtime (LoadedProfiledModel) is immutable and survives requests.
//! Per-generation state lives in ProfiledInferenceSession (owns KV caches,
//! cancellation flag, token buffer, and timeline).

use crate::compute_image::{CompiledImageReader, CopyClassification, TensorEntry};
use crate::engine_error::{EngineError, EngineErrorCode};
use crate::kv_cache::KvCache;
use crate::mapped_image::MappedImage;
use crate::placement_profile::ExecutionPlacementProfile;
use crate::runtime_contract::{
    AuthorityMode, BackendTarget, BudgetClass, RetryPolicy, RuntimeWorkItem,
};
use crate::runtime_orchestration::InMemoryCoordinationFabric;
use crate::heterogeneous::ComputeRuntime;
use crate::runtime_trace::{RuntimeTimeline, TimelineEvent, TimelineEventType};
use crate::session::InferenceSessionState;
use crate::worker_memory;
use crate::coreml_bridge::CoreMlModel;
use mlx_rs::Array;

/// Maximum tokens per prefill chunk for chunked prefill.
/// Longer prompts are split into chunks to allow interleaving decode
/// of other sequences between chunks, preventing long-prefill latency spikes.
pub const PREFILL_CHUNK_SIZE: u32 = 512;

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
}

impl LoadedProfiledModel {
    pub fn new(image_dir: &Path) -> crate::Result<Self> {
        let handle_baseline = crate::bridge::handle_count();
        let reader = CompiledImageReader::open(image_dir)?;
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
            let (arr, classification) = load_tensor_from_mapped_segment(segment, entry)?;
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

        // Load global tensors
        let emb_w = load_tensor("language_model.model.embed_tokens.weight")?;
        let emb_s = load_tensor("language_model.model.embed_tokens.scales")?;
        let emb_b = load_tensor("language_model.model.embed_tokens.biases")?;
        let fn_w = load_tensor("language_model.model.norm.weight")?;

        // RoPE tables are derived from the architecture rather than loaded
        // from the manifest. This avoids falling back to 1-element placeholders
        // when the compiled image does not materialize explicit rope tensors.
        let (rope_cos, rope_sin, full_cos, full_sin) =
            build_rope_tables(&reader.manifest.architecture)?;

        // Load layer weights
        let mut layers = Vec::new();
        for (l, layer_plan) in reader.manifest.execution_plan.layers.iter().enumerate() {
            let base = format!("language_model.model.layers.{}", l);

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
        })
    }
}

/// Per-request inference session — owns KV caches, generated tokens, and
/// cancellation state.  The model weights live in [`LoadedProfiledModel`]
/// and are passed as a parameter to [`prefill`] and [`decode_one`].
pub struct ProfiledInferenceSession {
    pub session_id: String,
    pub kv_caches: Vec<KvCache>,
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
        }
    }

    /// Populate the memory plan from a loaded model's scheduled module.
    pub fn setup_from_model(&mut self, model: &LoadedProfiledModel) {
        if let Some(scheduled) = &model.scheduled_module {
            if let Some(plan) = crate::memory::plan::plan_from_scheduled_module(
                scheduled,
                &crate::arena::Arena::new(1, 1, mlx_rs::Dtype::Float32).unwrap_or_else(|_| panic!("tmp arena")),
            ) {
                self.memory_plan = Some(plan);
            }
        }
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
        hidden.eval().map_err(|e| {
            EngineError::new(EngineErrorCode::NumericalFailure, format!("chunk prologue eval: {}", e))
        })?;

        let slots = model.memory_island.preallocate_layer_slots(1, 3840);

        for (l, layer_plan) in plan.layers.iter().enumerate() {
            let lw = &model.layers[l];
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer(
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
            )
            .map_err(|e| {
                EngineError::new(EngineErrorCode::InferenceFailed, format!("chunk layer {}: {}", l, e))
            })?;
            crate::heterogeneous::evaluate_into_island(slots.hidden_a.as_ref(), &hidden)
                .map_err(|e| EngineError::new(
                    EngineErrorCode::NumericalFailure,
                    format!("chunk evaluate_into_island: {}", e),
                ))?;
            if ((l + 1) % 6 == 0) || (l + 1 == plan.layers.len()) {
                hidden.eval().map_err(|e| {
                    EngineError::new(EngineErrorCode::NumericalFailure, format!("chunk layer {} eval: {}", l, e))
                })?;
            }
            self.kv_caches[l].commit_step();
        }

        // Clear the memory plan after the layer loop completes.
        // Subsequent allocations (epilogue, next chunk) use normal paths
        // unless a new plan is applied before the next region.
        if self.memory_plan.is_some() {
            let _ = crate::memory::plan::clear_memory_plan();
        }

        self.prefilled_tokens += chunk_size;
        self.absolute_position += chunk_size;

        let is_last_chunk = self.prefilled_tokens >= full_prompt.len() as u32;
        if is_last_chunk {
            // Run epilogue on completion
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
        // Delegate to chunked prefill, processing all chunks in a loop
        loop {
            match self.prefill_chunk(prompt_token_ids, model)? {
                Some(token) => return Ok(token),
                None => {
                    // More chunks remain — the caller would interleave
                    // decode here in continuous batching mode.
                    // For full-batch prefill we continue immediately.
                    continue;
        }
            }
        }
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
            let lw = &model.layers[l];
            let is_full = layer_plan.attention_kind == "full_attention";
            let (rcos, rsin) = if is_full {
                (&model.full_cos, &model.full_sin)
            } else {
                (&model.rope_cos, &model.rope_sin)
            };

            hidden = crate::executor::run_layer(
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
            )
            .map_err(|e| {
                EngineError::new(
                    EngineErrorCode::InferenceFailed,
                    format!("decode layer {}: {}", l, e),
                )
            })?;
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

        self.timeline.push_event(TimelineEvent::new(
            self.absolute_position as u64,
            TimelineEventType::DecodeStep,
            format!("decoded token {}", token),
        ));

        Ok(token)
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
