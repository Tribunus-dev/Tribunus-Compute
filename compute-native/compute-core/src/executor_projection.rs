//! Projection dispatch: quantized matmul, epilogue (LM head), and sampling.
//!
//! All projection dispatch logic extracted from `executor.rs`. Contains the
//! fallback-chain `qmatmul()` / `run_epilogue()` and their helpers.

use crate::ane::hot_row_predictor::HotRowPredictor;
use crate::ane::weight_row_cache::WeightRowCache;
use crate::backend::MlxBackend;
use crate::config::EpiloguePlan;
use crate::log_warn;
use crate::primitives;
use crate::projection_executor::{
    MaterializationClass, ProjectionExecutor, QuantizedProjectionDescriptor, RuntimeMode,
    StorageDtype,
};
use crate::projection_identity::{dtype_to_storage, ProjectionContext, ProjectionFamily};
use crate::session::SamplerConfig;
use mlx_rs::error::Result as MlxResult;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Fallback tracking ──────────────────────────────────────────────────────

/// Global counter of backend fallback events during quantized matmul dispatch.
/// Incremented each time a primary backend fails and a secondary/tertiary
/// backend is used as fallback.  Read by compute-engine for observability.
static FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Return the current fallback count.
pub fn fallback_count() -> u64 {
    FALLBACK_COUNT.load(Ordering::Relaxed)
}

/// Reset the fallback counter to zero.
pub fn reset_fallback_count() {
    FALLBACK_COUNT.store(0, Ordering::Relaxed);
}

// ── Fallback helpers ───────────────────────────────────────────────────────

static T2_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Try one quantized matmul with a fresh MlxBackend in the given mode.
fn try_qmatmul_mlx(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    desc: &QuantizedProjectionDescriptor,
    mode: RuntimeMode,
) -> MlxResult<Array> {
    let mut backend = MlxBackend::new();
    let x_h = backend.alloc(x.clone());
    let w_h = backend.alloc_weight(w.clone());
    let s_h = backend.alloc(s.clone());
    let b_h = backend.alloc(b.clone());
    let result_h = {
        let mut executor = ProjectionExecutor {
            backend: &mut backend,
            mode,
        };
        executor
            .run_projection(x_h, w_h, s_h, b_h, desc)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?
    };
    let result = backend
        .get(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(e))?
        .clone();
    Ok(result)
}

/// Fallback to CandleCpuBackend (CPU dequantize + f32 matmul).
#[cfg(feature = "candle-cpu")]
fn try_qmatmul_candle(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    desc: &QuantizedProjectionDescriptor,
) -> MlxResult<Array> {
    use crate::backend::TensorBackend;
    use crate::candle_cpu_backend::CandleCpuBackend;
    use candle_core::{Device, Tensor};

    x.eval()?;
    w.eval()?;
    s.eval()?;
    b.eval()?;

    let x_shape: Vec<i32> = x.shape().iter().map(|&d| d as i32).collect();
    let w_shape: Vec<i32> = w.shape().iter().map(|&d| d as i32).collect();
    let s_shape: Vec<i32> = s.shape().iter().map(|&d| d as i32).collect();
    let b_shape: Vec<i32> = b.shape().iter().map(|&d| d as i32).collect();

    let x_data: Vec<f32> = x
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback read x: {e}")))?
        .to_vec();
    let w_data: Vec<u32> = w
        .try_as_slice::<u32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback read w: {e}")))?
        .to_vec();
    let s_data: Vec<f32> = s
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback read s: {e}")))?
        .to_vec();
    let b_data: Vec<f32> = b
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback read b: {e}")))?
        .to_vec();

    let mut cb = CandleCpuBackend::new();
    let cb_x = cb
        .create_f32(&x_data, &x_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback create x: {e}")))?;
    let cb_s = cb
        .create_f32(&s_data, &s_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback create s: {e}")))?;
    let cb_b = cb
        .create_f32(&b_data, &b_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback create b: {e}")))?;
    let w_tensor = Tensor::from_slice(
        &w_data,
        w_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        &Device::Cpu,
    )
    .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback create w tensor: {e}")))?;
    let cb_w = cb.alloc_weight(w_tensor);

    let op = crate::backend::QuantizedMatmulOp {
        m: desc.logical_in_features,
        n: desc.logical_out_features,
        k: desc.logical_in_features,
        input_dtype: crate::backend::DType::F32,
        weight_dtype: crate::backend::DType::U32,
        scale_dtype: crate::backend::DType::F32,
        bias_dtype: crate::backend::DType::F32,
        output_dtype: crate::backend::DType::F32,
        group_size: desc.group_size,
        bits: desc.bits,
        transpose: true,
    };

    let result_h = cb
        .quantized_matmul(&op, cb_x, cb_w, cb_s, cb_b)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback quantized_matmul: {e}")))?;
    let result_shape = cb
        .shape(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback shape: {e}")))?;
    let result_data = cb
        .read_f32(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("fallback read_f32: {e}")))?;
    Ok(Array::from_slice(&result_data.data, &result_shape))
}

fn qmatmul(x: &Array, w: &Array, s: &Array, b: &Array) -> MlxResult<Array> {
    // Correct quantization parameters derived from logical feature dimensions.
    let in_features = x.shape().last().copied().unwrap_or(1) as i32;
    let n_groups = s.shape().last().copied().unwrap_or(1) as i32;
    let group_size = if n_groups > 0 {
        in_features / n_groups
    } else {
        64
    };

    // Derive bits from weight packing: U32 contains N logical values.
    // NF4: 8 values per u32 → bits = 4.
    // AF8: 4 values per u32 → bits = 8.
    let physical_in = w.shape()[1] as i32; // last dim of packed weight
    let logical_per_word = if physical_in > 0 {
        in_features / physical_in
    } else {
        1
    };
    let bits = if logical_per_word > 0 {
        32 / logical_per_word
    } else {
        8
    };

    // Build descriptor and executor for this projection.
    let logical_in_features = in_features as u32;
    let logical_out_features = w.shape()[0] as u32;
    let mut backend = MlxBackend::new();
    let x_h = backend.alloc(x.clone());
    let w_h = backend.alloc_weight(w.clone());
    let s_h = backend.alloc(s.clone());
    let b_h = backend.alloc(b.clone());
    let qmatmul_desc = QuantizedProjectionDescriptor {
        family: ProjectionFamily::OProj,
        logical_in_features,
        logical_out_features,
        bits: bits as u8,
        group_size: group_size as u32,
        storage_dtype: StorageDtype::U32,
        physical_weight_shape: vec![w.shape()[0] as u32, w.shape()[1] as u32],
        layer_index: 0,
        weight_materialization: MaterializationClass::MlxOwned,
    };

    // OPT-0006-T2 diagnostic: first-call stride/contiguity probe.
    // Answers: do external (mmap-backed) arrays trigger hidden copies?
    use std::sync::atomic::Ordering;
    if T2_PROBE.swap(false, Ordering::Relaxed) {
        // Force authority path for both probe calls (external and copied arrays)
        // so the timing comparison is apples-to-apples.
        let mut probe_desc = qmatmul_desc.clone();
        probe_desc.weight_materialization = MaterializationClass::MappedReadOnly;

        let ws = w.shape();
        w.eval()?;
        let t_ext_start = std::time::Instant::now();
        let r1_h = {
            let mut executor = ProjectionExecutor {
                backend: &mut backend,
                mode: RuntimeMode::Safe,
            };
            executor
                .run_projection(x_h, w_h, s_h, b_h, &probe_desc)
                .map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?
        };
        let r1 = backend
            .get(r1_h)
            .map_err(|e| mlx_rs::error::Exception::custom(e))?
            .clone();
        let _ = r1.eval()?;
        let t_ext = t_ext_start.elapsed();
        let ws_str: Vec<String> = ws.iter().map(|d| d.to_string()).collect();
        // Try contiguous copy comparison — may fail for external arrays
        // with dtype mismatch; fall back to external-only timing.
        let w_read: Option<Vec<u8>> = w.try_as_slice::<u8>().ok().map(|s| s.to_vec());
        if let Some(w_vec) = w_read {
            let wc = Array::from_slice(&w_vec, &ws);
            let s_vec: Vec<f32> = s
                .try_as_slice::<f32>()
                .map_err(|e| mlx_rs::error::Exception::custom(format!("s as_slice: {:?}", e)))?
                .to_vec();
            let sc = Array::from_slice(&s_vec, &s.shape());
            let b_vec: Vec<f32> = b
                .try_as_slice::<f32>()
                .map_err(|e| mlx_rs::error::Exception::custom(format!("b as_slice: {:?}", e)))?
                .to_vec();
            let bc = Array::from_slice(&b_vec, &b.shape());
            let wc_h = backend.alloc_weight(wc);
            let sc_h = backend.alloc(sc);
            let bc_h = backend.alloc(bc);
            let t_copy_start = std::time::Instant::now();
            let r2_h = {
                let mut executor = ProjectionExecutor {
                    backend: &mut backend,
                    mode: RuntimeMode::Safe,
                };
                executor
                    .run_projection(x_h, wc_h, sc_h, bc_h, &probe_desc)
                    .map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?
            };
            let r2 = backend
                .get(r2_h)
                .map_err(|e| mlx_rs::error::Exception::custom(e))?
                .clone();
            let _ = r2.eval()?;
            let t_copy = t_copy_start.elapsed();
            let wss: Vec<String> = ws.iter().map(|d| d.to_string()).collect();
            eprintln!(
                "[OPT-0006-T2] first-qmatmul shape=[{}] strides=[{}] ext={:.1}ms copy={:.1}ms ratio={:.2}x",
                ws_str.join(","), wss.join(","),
                t_ext.as_secs_f64() * 1000.0,
                t_copy.as_secs_f64() * 1000.0,
                t_copy.as_secs_f64() / t_ext.as_secs_f64(),
            );
        } else {
            eprintln!(
                "[OPT-0006-T2] first-qmatmul shape=[{}] ext={:.1}ms (copy comparison unavailable — external array dtype mismatch)",
                ws_str.join(","),
                t_ext.as_secs_f64() * 1000.0,
            );
        }
        return Ok(r1);
    }

    // Route through ProjectionExecutor which applies safe-mode dispatch.
    // ── Fallback chain ────────────────────────────────────────────────
    // Primary:   MlxBackend in Experimental mode (native fused quantized matmul)
    // Secondary: MlxBackend in Safe mode (dequantize + f32 matmul authority path)
    // Tertiary:  CandleCpuBackend (CPU dequantize + f32 matmul)

    // ── Fallback chain ──
    // Primary: MLX Experimental, Secondary: MLX Safe, Tertiary: Candle CPU

    let primary = try_qmatmul_mlx(x, w, s, b, &qmatmul_desc, RuntimeMode::Experimental);
    if let Ok(r) = primary {
        return Ok(r);
    }
    let primary_msg = primary.unwrap_err().to_string();
    eprintln!("[qmatmul] Primary MLX error: {}", primary_msg);
    FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);

    let secondary = try_qmatmul_mlx(x, w, s, b, &qmatmul_desc, RuntimeMode::Safe);
    if let Ok(r) = secondary {
        return Ok(r);
    }
    FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);

    // Tertiary
    #[cfg(feature = "candle-cpu")]
    if let Ok(r) = try_qmatmul_candle(x, w, s, b, &qmatmul_desc) {
        return Ok(r);
    }
    #[cfg(feature = "candle-cpu")]
    FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);

    Err(mlx_rs::error::Exception::custom(format!(
        "[fallback] All backends failed for quantized projection: {}",
        primary_msg
    )))
}

/// Wrapper around qmatmul that conditionally emits a projection attribution
/// event when `TRIBUNUS_PROJECTION_ATTRIBUTION=1`.
///
/// When the env var is unset (the common case), this function degenerates
/// to a direct call to `qmatmul` — zero overhead beyond an AtomicBool load.
/// No eval, synchronization, or host readback occurs inside the timer.
pub fn qmatmul_attributed(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    _transpose: bool,
    _group_size: i32,
    _bits: i32,
    ctx: &ProjectionContext,
    family: ProjectionFamily,
    invocation: usize,
) -> MlxResult<Array> {
    // One-time gate check: cache in a static AtomicBool so the fast path
    // is a single relaxed load (no env var string allocation).
    static GATE_INIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static GATE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    if !GATE_INIT.load(std::sync::atomic::Ordering::Relaxed) {
        let enabled = std::env::var("TRIBUNUS_PROJECTION_ATTRIBUTION")
            .ok()
            .as_deref()
            == Some("1");
        GATE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
        GATE_INIT.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    if !GATE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return qmatmul(x, w, s, b);
    }

    // ---- Instrumented path ----
    let start = std::time::Instant::now();
    let result = qmatmul(x, w, s, b)?;
    let delta_ns = start.elapsed().as_nanos() as u64;

    let input_shape = x.shape();
    let w_shape = w.shape();

    // Compute group_size from scales shape (same logic as qmatmul).
    let gs = if s.shape().len() >= 1 {
        (w.shape()[1] as i32 * 4) / s.shape()[s.shape().len() - 1]
    } else {
        64
    };

    let runtime_dtype_str = format!("{:?}", w.dtype());
    let storage_dtype_str = dtype_to_storage(&w.dtype());

    // Build weight_physical = weight_logical for now; the physical shape
    // is identical because we operate on the canonical MLX representation.
    let w_d0 = w_shape.first().copied().unwrap_or(0);
    let w_d1 = w_shape.get(1).copied().unwrap_or(0);

    let token_step_str = match ctx.token_step {
        Some(ts) => format!("{}", ts),
        None => String::new(),
    };

    eprintln!(
        "[proj] run_id={} phase={} forward_pass={} token_step={} layer={} kind={} family={} invocation={} graph_build_ns={} input=[{},{}] weight_logical=[{},{}] weight_physical=[{},{}] storage_dtype={} runtime_dtype={} group_size={} bits={} transpose={}",
        ctx.run_id,
        ctx.phase.as_str(),
        ctx.forward_pass_index,
        token_step_str,
        ctx.layer_index,
        ctx.attention_kind.as_str(),
        family.as_str(),
        invocation,
        delta_ns,
        input_shape.first().copied().unwrap_or(0),
        input_shape.get(1).copied().unwrap_or(0),
        w_d0,
        w_d1,
        w_d0,
        w_d1,
        storage_dtype_str,
        runtime_dtype_str,
        gs,
        8,   // bits — always 8 for our quantized matmul
        true, // transpose — always true for qmatmul
    );

    Ok(result)
}

// ── Epilogue ───────────────────────────────────────────────────────────────

/// Result of an epilogue execution.
///
/// The caller MUST `eval()` the `selected_token` before reading the scalar
/// value. The `logits` field (when `Some`) holds the full logits tensor
/// (shape `[1, seq_len, vocab_size]`) for optional inspection.
pub struct EpilogueResult {
    /// Scalar token array — caller MUST eval() before reading.
    pub selected_token: Array,
    /// Full logits tensor [1, seq_len, vocab_size] before last-token slicing.
    pub logits: Option<Array>,
}

// ── Epilogue fallback helpers ──────────────────────────────────────────────

/// Try one epilogue LM head projection with a fresh MlxBackend in the given mode.
fn try_epilogue_mlx(
    normed: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    desc: &QuantizedProjectionDescriptor,
    mode: RuntimeMode,
) -> MlxResult<Array> {
    let mut backend = MlxBackend::new();
    let normed_h = backend.alloc(normed.clone());
    let w_h = backend.alloc_weight(output_weight.clone());
    let s_h = backend.alloc(output_scales.clone());
    let b_h = backend.alloc(output_biases.clone());
    let result_h = {
        let mut executor = ProjectionExecutor {
            backend: &mut backend,
            mode,
        };
        executor
            .run_projection(normed_h, w_h, s_h, b_h, desc)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?
    };
    let result = backend
        .get(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(e))?
        .clone();
    Ok(result)
}

/// Fallback epilogue LM head projection using CandleCpuBackend (CPU dequantize + f32 matmul).
#[cfg(feature = "candle-cpu")]
fn try_epilogue_candle(
    normed: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    desc: &QuantizedProjectionDescriptor,
) -> MlxResult<Array> {
    use crate::backend::TensorBackend;
    use crate::candle_cpu_backend::CandleCpuBackend;
    use candle_core::{Device, Tensor};

    normed.eval()?;
    output_weight.eval()?;
    output_scales.eval()?;
    output_biases.eval()?;

    let x_shape: Vec<i32> = normed.shape().iter().map(|&d| d as i32).collect();
    let w_shape: Vec<i32> = output_weight.shape().iter().map(|&d| d as i32).collect();
    let s_shape: Vec<i32> = output_scales.shape().iter().map(|&d| d as i32).collect();
    let b_shape: Vec<i32> = output_biases.shape().iter().map(|&d| d as i32).collect();

    let x_data: Vec<f32> = normed
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback read x: {e}")))?
        .to_vec();
    let w_data: Vec<u32> = output_weight
        .try_as_slice::<u32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback read w: {e}")))?
        .to_vec();
    let s_data: Vec<f32> = output_scales
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback read s: {e}")))?
        .to_vec();
    let b_data: Vec<f32> = output_biases
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback read b: {e}")))?
        .to_vec();

    let mut cb = CandleCpuBackend::new();
    let cb_x = cb
        .create_f32(&x_data, &x_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback create x: {e}")))?;
    let cb_s = cb
        .create_f32(&s_data, &s_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback create s: {e}")))?;
    let cb_b = cb
        .create_f32(&b_data, &b_shape)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback create b: {e}")))?;
    let w_tensor = Tensor::from_slice(
        &w_data,
        w_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        &Device::Cpu,
    )
    .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback create w tensor: {e}")))?;
    let cb_w = cb.alloc_weight(w_tensor);

    let op = crate::backend::QuantizedMatmulOp {
        m: desc.logical_in_features,
        n: desc.logical_out_features,
        k: desc.logical_in_features,
        input_dtype: crate::backend::DType::F32,
        weight_dtype: crate::backend::DType::U32,
        scale_dtype: crate::backend::DType::F32,
        bias_dtype: crate::backend::DType::F32,
        output_dtype: crate::backend::DType::F32,
        group_size: desc.group_size,
        bits: desc.bits,
        transpose: true,
    };

    let result_h = cb
        .quantized_matmul(&op, cb_x, cb_w, cb_s, cb_b)
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!("ep fallback quantized_matmul: {e}"))
        })?;
    let result_shape = cb
        .shape(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback shape: {e}")))?;
    let result_data = cb
        .read_f32(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("ep fallback read_f32: {e}")))?;
    Ok(Array::from_slice(&result_data.data, &result_shape))
}

/// Final normalization, tied output projection, softcapping, and token selection.
///
/// Returns an `EpilogueResult` so the caller can explicitly `eval()` the
/// selected token before reading it. Logits are returned as an `Option` for
/// optional inspection — the caller can `eval()` and inspect them as needed.
///
/// This function does NOT force `eval()` on the returned arrays.
pub fn run_epilogue(
    hidden: &Array,
    final_norm: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    plan: &EpiloguePlan,
    rms_norm_eps: f32,
    _tie_word_embeddings: bool,
    sampler: &SamplerConfig,
) -> MlxResult<EpilogueResult> {
    // Shape contract: hidden state is batchless [tokens, hidden_size].
    debug_assert_eq!(
        hidden.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        hidden.ndim()
    );

    // Final RMSNorm
    let normed = primitives::rms_norm(hidden, final_norm, rms_norm_eps)?;

    // Tied output projection: quantized matmul with embedding weights
    let logical_in_features = normed.shape()[1] as u32;
    let logical_out_features = output_weight.shape()[0] as u32;
    let n_groups = if output_scales.shape().len() >= 1 {
        output_scales.shape()[output_scales.shape().len() - 1] as u32
    } else {
        1
    };
    let gs = if n_groups > 0 {
        logical_in_features / n_groups
    } else {
        64
    };
    // Derive bits from weight packing: U32 holds 32/bits logical values.
    let physical_in = output_weight.shape()[1] as u32;
    let logical_per_word = if physical_in > 0 {
        logical_in_features / physical_in
    } else {
        1
    };
    let bits: u8 = if logical_per_word > 0 {
        (32 / logical_per_word) as u8
    } else {
        8u8
    };
    let lm_head_desc = QuantizedProjectionDescriptor {
        family: ProjectionFamily::LmHead,
        logical_in_features,
        logical_out_features,
        bits,
        group_size: gs,
        storage_dtype: StorageDtype::U32,
        physical_weight_shape: vec![
            output_weight.shape()[0] as u32,
            output_weight.shape()[1] as u32,
        ],
        layer_index: 0,
        weight_materialization: MaterializationClass::MlxOwned,
    };

    // ── Fallback chain for LM head projection ─────────────────────────
    // Primary:   MlxBackend Experimental (native fused)
    // Secondary: MlxBackend Safe (dequantize + f32)
    // Tertiary:  CandleCpuBackend (CPU dequantize + f32)
    let logits = match try_epilogue_mlx(
        &normed,
        output_weight,
        output_scales,
        output_biases,
        &lm_head_desc,
        RuntimeMode::Experimental,
    ) {
        Ok(l) => l,
        Err(e1) => {
            log_warn!("[fallback] Primary epilogue MLX projection failed: {}", e1);
            FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
            match try_epilogue_mlx(
                &normed,
                output_weight,
                output_scales,
                output_biases,
                &lm_head_desc,
                RuntimeMode::Safe,
            ) {
                Ok(l) => l,
                Err(e2) => {
                    log_warn!(
                        "[fallback] Secondary epilogue MLX projection (safe) failed: {}",
                        e2
                    );
                    FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "candle-cpu")]
                    {
                        match try_epilogue_candle(
                            &normed,
                            output_weight,
                            output_scales,
                            output_biases,
                            &lm_head_desc,
                        ) {
                            Ok(l) => l,
                            Err(e3) => {
                                log_warn!(
                                    "[fallback] Tertiary epilogue Candle CPU projection failed: {}",
                                    e3
                                );
                                FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                                return Err(mlx_rs::error::Exception::custom(
                                    "[fallback] All epilogue backends failed".to_string(),
                                ));
                            }
                        }
                    }
                    #[cfg(not(feature = "candle-cpu"))]
                    {
                        let epi_msg = format!("{}", e1);
                        return Err(mlx_rs::error::Exception::custom(format!(
                            "[fallback] All epilogue backends failed: {}",
                            epi_msg
                        )));
                    }
                }
            }
        }
    };

    // Final logit softcapping
    let logits = if let Some(cap) = plan.final_logit_softcapping {
        let cap_f32 = cap as f32;
        let scaled = logits.divide(&Array::from_f32(cap_f32))?;
        let tanh = mlx_rs::ops::tanh(&scaled)?;
        tanh.multiply(&Array::from_f32(cap_f32))?
    } else {
        logits
    };

    // logits is rank 2 [tokens, vocab]. Extract last token row, then restore
    // batch+seq dims for argmax compat: [1, 1, vocab_size].
    let last_row = logits.index(((logits.shape()[0] - 1)..logits.shape()[0], ..));
    let last_logits = last_row.reshape(&[1, 1, -1])?;

    // Check if grammar masking is required.
    let has_grammar = sampler.grammar.is_some() && sampler.grammar_tokenizer.is_some();

    // Greedy path without grammar: fast argmax (no f32 extraction overhead)
    if sampler.is_greedy() && !has_grammar {
        let token_arr = ops::indexing::argmax_axis(&last_logits, -1, false)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("argmax: {:?}", e)))?;
        return Ok(EpilogueResult {
            selected_token: token_arr,
            logits: Some(logits),
        });
    }

    // Must extract f32 logits for grammar masking and/or non-greedy sampling.
    last_logits.eval()?;

    // Flatten to 1D for contiguous extraction
    let flat = last_logits.reshape(&[-1])?;
    let vocab_size = flat.shape()[0] as usize;
    let mut logits_vec: Vec<f32> = flat
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("read logits: {:?}", e)))?
        .to_vec();

    // 0. Grammar mask: set invalid token logits to -inf
    //    Applied before temperature/top-k/top-p so invalid tokens are
    //    never candidates regardless of other parameters.
    if has_grammar {
        sampler.apply_grammar_mask(&mut logits_vec);
    }

    // Greedy path with grammar: argmax on the masked logits
    if sampler.is_greedy() {
        let token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(EpilogueResult {
            selected_token: Array::from_slice(&[token], &[1]),
            logits: Some(logits),
        });
    }

    // 1. Temperature scaling
    if let Some(temp) = sampler.temperature {
        if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
            let scale = 1.0 / temp;
            for v in &mut logits_vec {
                *v *= scale;
            }
        }
    }

    // 2. Top-k filtering
    if let Some(k) = sampler.top_k {
        let k = (k as usize).min(vocab_size);
        if k > 0 && k < vocab_size {
            let mut sorted = logits_vec.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];
            for v in &mut logits_vec {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 3. Top-p (nucleus) filtering
    if let Some(p) = sampler.top_p {
        if p > 0.0 && p < 1.0 {
            // Compute softmax probabilities for sorting
            let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; vocab_size];
            let mut prob_sum = 0.0f32;
            for (i, &v) in logits_vec.iter().enumerate() {
                let e = (v - max_l).exp();
                probs[i] = e;
                prob_sum += e;
            }
            if prob_sum > 0.0 {
                for v in &mut probs {
                    *v /= prob_sum;
                }
            }

            // Sort indices by probability descending
            let mut indices: Vec<usize> = (0..vocab_size).collect();
            indices.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find cumulative cutoff; zero out logits beyond it
            let mut cumsum = 0.0f32;
            for (rank, &idx) in indices.iter().enumerate() {
                cumsum += probs[idx];
                if cumsum > p {
                    for &i in &indices[rank..] {
                        logits_vec[i] = f32::NEG_INFINITY;
                    }
                    break;
                }
            }
        }
    }

    // 4. Check if everything was filtered — fall back to argmax
    let all_inf = logits_vec.iter().all(|v| !v.is_finite() || v.is_nan());
    if all_inf {
        let token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(EpilogueResult {
            selected_token: Array::from_slice(&[token], &[1]),
            logits: Some(logits),
        });
    }

    // 5. Categorical sample via MLX
    let shape = [1i32, 1, vocab_size as i32];
    let filtered_arr = Array::from_slice(&logits_vec, &shape);
    let key = match sampler.seed {
        Some(s) => Some(mlx_rs::random::key(s)?),
        None => None,
    };
    let token_arr = mlx_rs::random::categorical(&filtered_arr, None, None, key.as_ref())?;
    Ok(EpilogueResult {
        selected_token: token_arr,
        logits: Some(logits),
    })
}

/// Accelerated epilogue path with ANE weight prefetch.
///
/// Same interface as [`run_epilogue`] but additionally predicts the next
/// token candidate(s) via the ANE predictor, pre-fetches their LM head weight
/// rows into ANE SRAM, and uses the hybrid LM head path that reads cached
/// rows from IOSurface-backed memory for zero-latency access.
///
/// When the predictor is unavailable (`None`), falls back to the standard
/// `run_epilogue` path unchanged.
pub fn run_epilogue_prefetch(
    hidden: &Array,
    final_norm: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    plan: &EpiloguePlan,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    sampler: &SamplerConfig,
    predictor: Option<&mut HotRowPredictor>,
    row_cache: Option<&mut WeightRowCache>,
) -> MlxResult<EpilogueResult> {
    // Fall back to standard epilogue if prefetch is not configured.
    let (predictor, row_cache) = match (predictor, row_cache) {
        (Some(p), Some(c)) => (p, c),
        _ => {
            return run_epilogue(
                hidden,
                final_norm,
                output_weight,
                output_scales,
                output_biases,
                plan,
                rms_norm_eps,
                tie_word_embeddings,
                sampler,
            );
        }
    };

    // Shape contract: hidden state is [1, hidden_size] for decode.
    debug_assert_eq!(
        hidden.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        hidden.ndim()
    );

    // Final RMSNorm (same as run_epilogue)
    let normed = primitives::rms_norm(hidden, final_norm, rms_norm_eps)?;

    // 1. Read hidden state as f32 slice for the predictor.
    let hidden_slice: Vec<f32> = normed
        .try_as_slice::<f32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!(
                "run_epilogue_prefetch: read normed hidden: {:?}",
                e
            ))
        })?
        .to_vec();

    // 2. Predict next token candidates on ANE.
    let candidates = predictor.predict(&hidden_slice).map_err(|e| {
        mlx_rs::error::Exception::custom(format!("run_epilogue_prefetch: predictor error: {}", e))
    })?;

    // 3. Pre-fetch those rows into ANE SRAM.
    row_cache
        .prefetch_rows(&candidates, output_weight)
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!(
                "run_epilogue_prefetch: prefetch error: {}",
                e
            ))
        })?;

    // 4. Run hybrid LM head (cached rows are fast, rest are normal).
    let logits = row_cache
        .hybrid_lm_head(&normed, output_weight)
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!(
                "run_epilogue_prefetch: hybrid_lm_head error: {}",
                e
            ))
        })?;

    // 5. Final logit softcapping (same as run_epilogue)
    let logits = if let Some(cap) = plan.final_logit_softcapping {
        let cap_f32 = cap as f32;
        let scaled = logits.divide(&Array::from_f32(cap_f32))?;
        let tanh = mlx_rs::ops::tanh(&scaled)?;
        tanh.multiply(&Array::from_f32(cap_f32))?
    } else {
        logits
    };

    // 6. Restore batch+seq dims: [1, 1, vocab_size] for argmax compat.
    let last_logits = logits.reshape(&[1, 1, -1])?;

    // Check if grammar masking is required.
    let has_grammar = sampler.grammar.is_some() && sampler.grammar_tokenizer.is_some();

    // 7. Sample.
    let selected_token = if sampler.is_greedy() && !has_grammar {
        // Fast path: argmax directly (no f32 extraction)
        let token_arr = ops::indexing::argmax_axis(&last_logits, -1, false)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("argmax: {:?}", e)))?;
        token_arr
    } else {
        // Must extract f32 for grammar masking and/or non-greedy sampling.
        last_logits
            .eval()
            .map_err(|e| mlx_rs::error::Exception::custom(format!("last_logits eval: {:?}", e)))?;

        let flat = last_logits.reshape(&[-1])?;
        let vocab_size = flat.shape()[0] as usize;
        let mut logits_vec: Vec<f32> = flat
            .try_as_slice::<f32>()
            .map_err(|e| mlx_rs::error::Exception::custom(format!("read logits: {:?}", e)))?
            .to_vec();

        // 0. Grammar mask: set invalid token logits to -inf
        if has_grammar {
            sampler.apply_grammar_mask(&mut logits_vec);
        }

        // Greedy path with grammar: argmax on the masked logits
        if sampler.is_greedy() {
            // With grammar mask applied, pick highest-scoring valid token
            let token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            Array::from_slice(&[token], &[1])
        } else {
            // Non-greedy sampling: temperature, top-k, top-p
            // Temperature
            if let Some(temp) = sampler.temperature {
                if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
                    let scale = 1.0 / temp;
                    for v in &mut logits_vec {
                        *v *= scale;
                    }
                }
            }

            // Top-k
            if let Some(k) = sampler.top_k {
                let k = (k as usize).min(vocab_size);
                if k > 0 && k < vocab_size {
                    let mut sorted = logits_vec.clone();
                    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let threshold = sorted[k - 1];
                    for v in &mut logits_vec {
                        if *v < threshold {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Top-p
            if let Some(p) = sampler.top_p {
                if p > 0.0 && p < 1.0 {
                    let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs = vec![0.0f32; vocab_size];
                    let mut prob_sum = 0.0f32;
                    for (i, &v) in logits_vec.iter().enumerate() {
                        let e = (v - max_l).exp();
                        probs[i] = e;
                        prob_sum += e;
                    }
                    if prob_sum > 0.0 {
                        for v in &mut probs {
                            *v /= prob_sum;
                        }
                    }
                    let mut indices: Vec<usize> = (0..vocab_size).collect();
                    indices.sort_by(|&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut cumsum = 0.0f32;
                    for (rank, &idx) in indices.iter().enumerate() {
                        cumsum += probs[idx];
                        if cumsum > p {
                            for &i in &indices[rank..] {
                                logits_vec[i] = f32::NEG_INFINITY;
                            }
                            break;
                        }
                    }
                }
            }

            let all_inf = logits_vec.iter().all(|v| !v.is_finite() || v.is_nan());
            let token = if all_inf {
                logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                let shape = [1i32, 1, vocab_size as i32];
                let filtered_arr = Array::from_slice(&logits_vec, &shape);
                let key = match sampler.seed {
                    Some(s) => Some(mlx_rs::random::key(s)?),
                    None => None,
                };
                let token_arr =
                    mlx_rs::random::categorical(&filtered_arr, None, None, key.as_ref())?;
                let t: Vec<u32> = token_arr
                    .try_as_slice::<u32>()
                    .map_err(|e| mlx_rs::error::Exception::custom(format!("categorical: {:?}", e)))?
                    .to_vec();
                t.first().copied().unwrap_or(0)
            };
            Array::from_slice(&[token], &[1])
        }
    };

    // 8. Update prediction statistics.
    let token_slice: Vec<u32> = selected_token
        .try_as_slice::<u32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!("run_epilogue_prefetch: read token: {:?}", e))
        })?
        .to_vec();
    let selected_token_id = token_slice.first().copied().unwrap_or(0);
    predictor.record_outcome(selected_token_id);

    Ok(EpilogueResult {
        selected_token,
        logits: Some(logits),
    })
}
