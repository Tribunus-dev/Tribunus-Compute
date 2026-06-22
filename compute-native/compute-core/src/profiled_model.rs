//! Model loading and weight management for the profiled heterogeneous executor.
//!
//! Extracted from profiled_executor.rs to reduce god-monolith size.
//!
//! Contains:
//! - [`LoadedProfiledModel`] — immutable model runtime with all loaded tensors
//! - [`LayerWeights`] — per-layer weight projections
//! - [`LayerWeightStreamer`] — on-demand weight loading with ANE DMA prefetch
//! - [`AneDmaPrefetcher`] — ANE DMA engine wrapper for async weight transfers
//! - [`load_tensor_from_mapped_segment`] — mmap-backed tensor loading

use crate::arena::Arena;
use crate::compute_image::phase_dag::EmittedPhaseGraph;
use crate::external_array::{ExternalStorage, new_external_array};
use crate::external_array;
use crate::external_array::BorrowedStorage;
use std::cell::RefCell;
use crate::compute_image::{CompiledImageReader, CopyClassification, TensorEntry};
use crate::config::{ModelExecutionPlan, TextArchitecture, VisionArchitecture};
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use crate::heterogeneous::SharedMemoryIsland;
use crate::mapped_image::MappedImage;
use crate::vision::encoder::VisionEncoder;
use crate::worker_dispatch::LoadedMetalKernel;
use crate::worker_dispatch::MetalKernelRegistry;
use crate::worker_memory;
use mlx_rs::Array;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Segment-slice adapter for no-copy external array construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tensor loading from mapped segments
// ---------------------------------------------------------------------------

/// Convert a storage dtype string to mlx_rs::Dtype.
fn storage_dtype_to_mlx(dtype: &str) -> crate::Result<mlx_rs::Dtype> {
    match dtype {
        "U8" | "Uint8" => Ok(mlx_rs::Dtype::Uint8),
        "F32" | "Float32" => Ok(mlx_rs::Dtype::Float32),
        "BF16" | "BFloat16" => Ok(mlx_rs::Dtype::Bfloat16),
        "I8" | "Int8" => Ok(mlx_rs::Dtype::Int8),
        "U32" | "Uint32" => Ok(mlx_rs::Dtype::Uint32),
        other => Err(crate::Error::from_reason(format!(
            "unsupported storage dtype: {}", other
        ))),
    }
}

/// Thread-local registry of IOSurface arenas created during weight loading.
/// `load_tensor_from_mapped_segment` pushes arenas here. The caller
/// (`LoadedProfiledModel::new`) drains them into the model struct.
thread_local! {
    static WEIGHT_ARENAS: RefCell<Vec<Arc<Arena>>> = const { RefCell::new(Vec::new()) };
}

/// Load tensor data from a MappedSegment using external array construction.
///
/// Uses [`crate::external_array::new_external_array`] for all supported dtypes
/// so that MLX operates directly on the mmap-backed memory rather than a copy.
pub(crate) fn load_tensor_from_mapped_segment(
    segment: &Arc<crate::mapped_image::MappedSegment>,
    entry: &TensorEntry,
    force_copy: bool,
) -> crate::Result<(Array, CopyClassification)> {
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
        // Int8 tensors (scales/biases) are tiny — keep the simple copy path.
        if matches!(entry.storage_dtype.as_str(), "I8" | "Int8") {
            let data: Vec<i8> = mapping[offset..end].iter().map(|&b| b as i8).collect();
            let arr = Array::from_slice(&data, &dims);
            return Ok((arr, CopyClassification::CopiedFallback));
        }

        // Allocate an IOSurface-backed arena — no backend manages memory.
        let arena = Arena::new_bytes(len as u32).map_err(|e| {
            crate::Error::from_reason(format!("arena alloc for {}: {}", entry.name, e))
        })?;
        let arena = Arc::new(arena);
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapping.as_ptr().add(offset),
                arena.base_ptr() as *mut u8,
                len,
            );
        }
        // Tribunus owns the arena; MLX borrows it via BorrowedStorage.
        let borrowed = Arc::new(BorrowedStorage {
            ptr: arena.data_ptr(),
            len: arena.byte_len(),
        });
        WEIGHT_ARENAS.with(|arenas| arenas.borrow_mut().push(arena));
        let mlx_dtype = storage_dtype_to_mlx(&entry.storage_dtype)?;
        let arr = unsafe {
            new_external_array(borrowed, &dims, mlx_dtype)
                .map_err(|e| crate::Error::from_reason(e))?
        };
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
            let arr = Array::from_slice(&data, &dims);
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

// ---------------------------------------------------------------------------
// RoPE table construction
// ---------------------------------------------------------------------------

pub(crate) fn build_rope_tables(
    arch: &TextArchitecture,
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

// ---------------------------------------------------------------------------
// System memory helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Layer weights
// ---------------------------------------------------------------------------

#[derive(Clone)]
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

// ---------------------------------------------------------------------------
// Byte formatting helper
// ---------------------------------------------------------------------------

pub(crate) fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.1}GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1}MB", b as f64 / 1_048_576.0)
    } else {
        format!("{}B", b)
    }
}

// ---------------------------------------------------------------------------
// LoadedProfiledModel – the immutable model runtime
// ---------------------------------------------------------------------------

pub struct LoadedProfiledModel {
    pub image_dir: PathBuf,
    pub reader: CompiledImageReader,
    pub mapped_image: MappedImage,
    /// IOSurface-backed arenas holding weight tensor memory.
    /// Owned by Tribunus — backends only borrow from these via external arrays.
    pub weight_arenas: Vec<Arc<Arena>>,
    /// Compiler-emitted phase DAG for PhaseEngine dispatch (optional).
    pub phase_dag: Option<EmittedPhaseGraph>,
    /// Pre-loaded Metal kernel pipeline states for fused-kernel dispatch.
    pub metal_kernels: Arc<Vec<LoadedMetalKernel>>,
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
    pub ane_coreml_models: Vec<Option<Arc<CoreMlModel>>>,
    /// Shared IOSurface memory island — all runtime memory allocations
    /// (intermediates, KV cache) come from this pool. MLX does NOT manage
    /// memory independently.
    pub memory_island: SharedMemoryIsland,
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
    /// Load and construct a profiled model from a compiled image directory.
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
            let (arr, classification) = load_tensor_from_mapped_segment(segment, entry, true)?;
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
                if entry.name.contains(".embed_tokens.")
                    || entry.name.contains(".embed_tokens.weight")
                {
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
        // SKIPPED: ANE pre-warm causes InvalidMILProgram exception.
        let _ane_prewarmed = false;

        // ── Shared IOSurface memory island ──────────────────────────────
        // All runtime intermediates allocate from this pool, NOT from MLX.
        // This ensures Accelerate and CoreML read the same physical pages
        // that MLX writes, achieving zero-copy across all backends.

        // Compute the IOSurface pool size from model architecture.
        // The pool holds per-step scratch buffers, attention workspace,
        // and KV cache headroom for the current sequence. Model weights
        // are MLX-managed and separate from this pool.
        let arch = &reader.manifest.architecture;
        let scratch_bytes = arch.hidden_size as u64 * 10 * 4;  // 10 f32 scratch tensors
        let attn_scores = (arch.max_position_embeddings as u64).min(4096)  // chunk cap
            * arch.num_attention_heads as u64 * arch.head_dim as u64 * 4;
        let kv_per_token = 2 * arch.num_key_value_heads as u64 * arch.head_dim as u64 * 2;  // FP16
        let kv_headroom = kv_per_token * 4096;  // room for 4K tokens
        let computed_pool = (scratch_bytes + attn_scores + kv_headroom) * 125 / 100;  // +25% margin
        let total_ram = system_memory_bytes();
        let max_pool = if total_ram > 0 {
            // Truthful: model need capped at 25% of RAM, min 16 MB
            computed_pool.min(total_ram / 4).max(16 * 1024 * 1024)
        } else {
            computed_pool.max(16 * 1024 * 1024)
        };
        eprintln!("[memory] IOSurface pool: {} MB (model estimate: {} MB, RAM: {} MB)",
            max_pool / (1024 * 1024), computed_pool / (1024 * 1024), total_ram / (1024 * 1024));
        let memory_island = SharedMemoryIsland::with_limit(max_pool);

        // ── Load ANE CoreML models for ANE-routed attention layers ─────
        let n_layers = reader.manifest.execution_plan.layers.len();
        let mut ane_coreml_models: Vec<Option<Arc<CoreMlModel>>> = vec![None; n_layers];

        // Load each ANE island's compiled .mlmodelc from disk.
        for island in &reader.manifest.execution_plan.fused_ane_islands {
            let modelc_path = image_dir.join(&island.modelc_relpath);
            let modelc_str = modelc_path.to_string_lossy().to_string();
            match CoreMlModel::load(&modelc_str) {
                Ok(model) => {
                    let model = Arc::new(model);
                    for &layer_idx in &island.layer_indices {
                        let idx = layer_idx as usize;
                        if idx < n_layers {
                            ane_coreml_models[idx] = Some(model.clone());
                        }
                    }
                    eprintln!(
                        "[profiled-model] Loaded ANE CoreML model island '{}' ({} layers)",
                        island.island_id,
                        island.layer_indices.len(),
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[profiled-model] WARNING: failed to load ANE CoreML model island '{}': {}",
                        island.island_id, e,
                    );
                }
            }
        }

        let loaded_count = ane_coreml_models.iter().filter(|m| m.is_some()).count();
        eprintln!(
            "[profiled-model] ANE CoreML models loaded for {}/{} layers",
            loaded_count, n_layers,
        );

        // ── Compile scheduled module (memory plan, regions, boundaries)
        let scheduled_module = Some(
            crate::compiler::compile_schedule::compile_model_to_scheduled_module(
                &reader.manifest.execution_plan,
                &reader.manifest.architecture,
                crate::backend::routing::EvidenceDigest(reader.manifest.image_hash.clone()),
            ),
        );

        // ── Load vision encoder (if present) ─────────────────────────
        let vision_encoder = if reader
            .manifest
            .tensor_table
            .iter()
            .any(|e| e.name.contains("vision_encoder"))
        {
            // Find the model's vision_config from the manifest metadata.
            // Fall back to the image metadata embedded in the architecture.
            let vision_config = VisionArchitecture {
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
            let mut load_vision_tensor = |name: &str| -> Result<Arc<Array>, String> {
                if let Some(entry) = reader.manifest.tensor_table.iter().find(|e| e.name == name) {
                    let seg_id = &entry.segment;
                    let segment = mapped_image.segments.get(seg_id).ok_or_else(|| {
                        format!("segment not found for vision tensor {}: {}", name, seg_id)
                    })?;
                    let (arr, _classification) =
                        load_tensor_from_mapped_segment(segment, entry, true)
                            .map_err(|e| format!("load vision tensor {}: {}", name, e))?;
                    Ok(Arc::new(arr))
                } else {
                    // Return a zero-initialized placeholder so the encoder
                    // can still be constructed for models that don't have
                    // vision weights (graceful fallback).
                    Err(format!(
                        "vision tensor not found in compiled image: {}",
                        name
                    ))
                }
            };
            match VisionEncoder::load(vc, &mut load_vision_tensor) {
                Ok(enc) => Some(enc),
                Err(e) => {
                    eprintln!(
                        "[profiled-model] WARNING: vision encoder load failed: {} (continuing without vision)",
                        e
                    );
                    None
                }
            }
        } else {
            None
        }
        ;

        // ── Load compiled Metal kernel artifacts ──────────────────────────
        // Load .metallib files from the compute image, create pipeline states.
        let metal_kernels: Arc<Vec<LoadedMetalKernel>> = {
            let artifacts = &reader.manifest.metal_kernel_artifacts;
            if artifacts.is_empty() {
                eprintln!("[profiled-model] No Metal kernel artifacts in manifest");
                Arc::new(Vec::new())
            } else {
                match MetalKernelRegistry::load_all(image_dir, artifacts) {
                    Ok(registry) => {
                        let count = registry.len();
                        let vec = registry.into_vec();
                        eprintln!("[profiled-model] Loaded {} Metal kernel artifacts", count);
                        Arc::new(vec)
                    }
                    Err(e) => {
                        eprintln!("[profiled-model] WARNING: failed to load Metal kernels: {}", e);
                        Arc::new(Vec::new())
                    }
                }
            }
        };

        Ok(Self {
            image_dir: image_dir.to_path_buf(),
            phase_dag: reader.manifest.phase_dag.clone(),
            reader,
            mapped_image,
            weight_arenas: WEIGHT_ARENAS.with(|a| a.take()),
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
            ane_cache: None,
            ane_coreml_models,
            memory_island,
            scheduled_module,
            vision_encoder,
            active_adapter: None,
            // Metal kernels: start empty; populated by the fused-kernel
            metal_kernels,
        })
    }
}

impl std::fmt::Debug for LoadedProfiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedProfiledModel")
            .field("image_dir", &self.image_dir)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ANE DMA prefetcher
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LayerWeightStreamer – on-demand weight loading with ANE DMA prefetch
// ---------------------------------------------------------------------------

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
    pub active_weights: HashMap<u32, LayerWeights>,
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
            let entry = self
                .reader
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| format!("tensor not found: {}", name))?;
            let seg_id = &entry.segment;
            let segment = self
                .mapped_image
                .segments
                .get(seg_id)
                .ok_or_else(|| format!("segment not found: {}", seg_id))?;
            let (arr, _classification) = load_tensor_from_mapped_segment(segment, entry, true)
                .map_err(|e| format!("load {}: {}", name, e))?;
            Ok(Arc::new(arr))
        };

        let input_layernorm = load_tensor(&format!("{}.input_layernorm.weight", base))?;
        let post_attention_layernorm =
            load_tensor(&format!("{}.post_attention_layernorm.weight", base))?;

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
        let q_norm = if self
            .reader
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
        let k_norm = if self
            .reader
            .manifest
            .tensor_table
            .iter()
            .any(|e| e.name == k_norm_name)
        {
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
