//! Central projection dispatch for quantized matmuls.
//!
//! All quantized matmul calls must go through [`ProjectionExecutor`], which
//! enforces a typed [`QuantizedProjectionDescriptor`], chooses the backend
//! (MLX fused, MLX authority/dequantize path, or hard error), and applies
//! the current [`RuntimeMode`] policy.

use mlx_rs::error::Result as MlxResult;
use mlx_rs::Array;

use crate::crash_breadcrumb;
use crate::log_debug;
use serde::Serialize;
use std::cell::RefCell;
use crate::projection_identity::ProjectionFamily;

// ── Types ──────────────────────────────────────────────────────────────────

/// Describes a quantized projection operation with full contract.
#[derive(Debug, Clone)]
pub struct QuantizedProjectionDescriptor {
    /// Which projection in the transformer layer.
    pub family: ProjectionFamily,
    /// Logical input feature dimension (hidden_size).
    pub logical_in_features: u32,
    /// Logical output feature dimension.
    pub logical_out_features: u32,
    /// Quantization bit width (4 for int4, 8 for int8).
    pub bits: u8,
    /// Number of elements per quantization group.
    pub group_size: u32,
    /// Physical storage dtype of the packed weight array.
    pub storage_dtype: StorageDtype,
    /// Physical shape of the packed weight array.
    pub physical_weight_shape: Vec<u32>,
    /// Layer index (0-based).
    pub layer_index: u32,
    /// Materialization class of the weight arrays.
    pub weight_materialization: MaterializationClass,
}

/// How the weight arrays are stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationClass {
    /// MLX-owned array (copied from mmap at load time).
    MlxOwned,
    /// Mmap-backed external array (no-copy, unsafe for fused kernels).
    MappedReadOnly,
    /// Copied into MLX-owned buffer for safety.
    CopiedSafe,
}

/// Physical storage dtype of the packed weight array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype {
    U32,
    U8,
    I8,
}

/// Current runtime mode affecting dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Safe mode: no fused MLX int4 for FFN projections, no mapped no-copy
    /// for quantized weights — use the authority (dequantize + matmul) path.
    Safe,
    /// Qualified mode: operations permitted only after a crash-free parity
    /// probe has passed for this exact shape class.
    Qualified,
    /// Experimental mode: all paths enabled.
    Experimental,
}

/// A record of which backend was used for a single quantized projection.
#[derive(Debug, Clone, Serialize)]
pub struct RouteRecord {
    /// Transformer layer index (0-based).
    pub layer: u32,
    /// Projection family name (e.g. "GateProj", "UpProj", "DownProj", "QProj", "KProj", "VProj", "OProj").
    pub projection: String,
    /// Backend that executed this projection ("mlx_fused", "mlx_authority", or "accelerate").
    pub backend: String,
    /// Quantization bit width.
    pub bits: u8,
    /// Number of elements per quantization group.
    pub group_size: u32,
}

thread_local! {
    static ROUTE_RECEIPTS: RefCell<Vec<RouteRecord>> = const { RefCell::new(Vec::new()) };
}

/// Record the backend route for one quantized projection.
pub fn record_route(desc: &QuantizedProjectionDescriptor, backend: &str) {
    ROUTE_RECEIPTS.with_borrow_mut(|r| {
        r.push(RouteRecord {
            layer: desc.layer_index,
            projection: format!("{:?}", desc.family),
            backend: backend.to_string(),
            bits: desc.bits,
            group_size: desc.group_size,
        });
    });
}

/// Drain all collected route receipts (typically called at the end of a
/// generation to include the profile in the HTTP response).
pub fn drain_route_receipts() -> Vec<RouteRecord> {
    ROUTE_RECEIPTS.with_borrow_mut(|r| std::mem::take(r))
}

// ── ProjectionExecutor ─────────────────────────────────────────────────────

/// Central dispatcher for all quantized projections.
pub struct ProjectionExecutor {
    /// Current runtime mode (affects dispatch decisions).
    pub mode: RuntimeMode,
}

impl ProjectionExecutor {
    /// Execute one quantized projection. Returns the result Array (not eval'd).
    pub fn run_projection(
        &self,
        x: &Array,
        w: &Array,
        s: &Array,
        b: &Array,
        desc: &QuantizedProjectionDescriptor,
    ) -> MlxResult<Array> {
        // Write crash breadcrumb before entering native code.
        let pid = std::process::id();
        let x_shape: Vec<i32> = x.shape().iter().map(|&d| d as i32).collect();
        let w_shape: Vec<i32> = w.shape().iter().map(|&d| d as i32).collect();
        crash_breadcrumb::before_native(
            pid,
            desc.layer_index,
            "decode",
            desc.family.as_str(),
            "mlx",
            match desc.weight_materialization {
                MaterializationClass::MlxOwned => "mlx-owned",
                MaterializationClass::MappedReadOnly => "mapped-readonly",
                MaterializationClass::CopiedSafe => "copied-safe",
            },
            &x_shape,
            &w_shape,
            desc.bits,
            desc.group_size,
        );

        let start = std::time::Instant::now();
        let result = match self.mode {
            RuntimeMode::Safe => self.run_safe(x, w, s, b, desc),
            RuntimeMode::Qualified => self.run_qualified(x, w, s, b, desc),
            RuntimeMode::Experimental => self.run_experimental(x, w, s, b, desc),
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        crash_breadcrumb::after_native(pid, elapsed_us);
        result
    }

    /// Safe-mode dispatch: avoids fused MLX kernels that are known to
    /// segfault for certain shapes, and refuses mapped no-copy weights.
    fn run_safe(
        &self,
        x: &Array,
        w: &Array,
        s: &Array,
        b: &Array,
        desc: &QuantizedProjectionDescriptor,
    ) -> MlxResult<Array> {
        // Safe mode rules:
        // 1. No fused MLX quantized_matmul for FFN projections (gate/up/down)
        //    with logical_in_features > 2048 — use authority path.
        // 2. No mapped no-copy weights for quantized projections — must be
        //    copied-safe or MLX-owned.
        let needs_authority = match desc.family {
            ProjectionFamily::GateProj
            | ProjectionFamily::UpProj
            | ProjectionFamily::DownProj => desc.logical_in_features > 2048,
            _ => false,
        };

        if needs_authority || desc.weight_materialization == MaterializationClass::MappedReadOnly
        {
            let result = quantized_matmul_authority(x, w, s, b, desc.group_size as i32);
            record_route(desc, "mlx_authority");
            result
        } else {
            let result = mlx_rs::ops::quantized_matmul(
                x,
                w,
                s,
                b,
                true,
                desc.group_size as i32,
                desc.bits as i32,
            );
            record_route(desc, "mlx_fused");
            result
        }
    }

    /// Qualified-mode dispatch: same as safe for now; extended later with
    /// parity-gated paths.
    fn run_qualified(
        &self,
        x: &Array,
        w: &Array,
        s: &Array,
        b: &Array,
        desc: &QuantizedProjectionDescriptor,
    ) -> MlxResult<Array> {
        self.run_safe(x, w, s, b, desc)
    }

    /// Experimental-mode dispatch: use fused MLX quantized_matmul
    /// unconditionally.
    fn run_experimental(
        &self,
        x: &Array,
        w: &Array,
        s: &Array,
        b: &Array,
        desc: &QuantizedProjectionDescriptor,
    ) -> MlxResult<Array> {
        let result = mlx_rs::ops::quantized_matmul(
            x,
            w,
            s,
            b,
            true,
            desc.group_size as i32,
            desc.bits as i32,
        );
        record_route(desc, "mlx_fused");
        result
    }
}

// ── Authority quantized matmul ────────────────────────────────────────

/// Authority path for quantized matmul: dequantize packed U32 int4 weights
/// to f32 using correct nibble extraction, then call regular MLX matmul.
/// Avoids MLX fused quantized_matmul which segfaults for certain shapes.
fn quantized_matmul_authority(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    group_size: i32,
) -> MlxResult<Array> {
    log_debug!("[infer] op=authority x_shape={:?}", x.shape());
    let x_shape = x.shape();
    let w_shape = w.shape();
    let s_shape = s.shape();

    let m = x_shape[0] as usize;
    let k = x_shape[1] as usize;
    let n_out = w_shape[0] as usize;
    let n_groups = s_shape[s_shape.len() - 1] as usize;
    let packed_cols = w_shape[1] as usize;

    // Force evaluation of external/mmap-backed tensors before reading.
    // try_as_slice requires the tensor to be evaluated on the current device.
    w.eval()?;
    s.eval()?;
    b.eval()?;

    // Read packed weight (U32)
    let w_u32: Vec<u32> = w
        .try_as_slice::<u32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!("authority: w slice: {:?}", e))
        })?
        .to_vec();

    let scales: Vec<f32> = s
        .try_as_slice::<f32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!("authority: s slice: {:?}", e))
        })?
        .to_vec();
    let biases: Vec<f32> = b
        .try_as_slice::<f32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!("authority: b slice: {:?}", e))
        })?
        .to_vec();

    // Dequantize: packed U32 int4 (8 nibbles per u32) -> f32 [n_out, k]
    let gs = group_size as usize;
    let mut w_f32 = vec![0.0f32; n_out * k];
    for row in 0..n_out {
        for g in 0..n_groups {
            let scale = scales[row * n_groups + g];
            let bias = biases[row * n_groups + g];
            let start = g * gs;
            let end = (start + gs).min(k);
            for elem_idx in start..end {
                let word_idx = row * packed_cols + elem_idx / 8;
                let lane = elem_idx % 8;
                let qval = (w_u32[word_idx] >> (lane * 4)) & 0xF;
                w_f32[row * k + elem_idx] = (qval as f32) * scale + bias;
            }
        }
    }

    // Dequantized weight: [n_out, k], transpose to [k, n_out] for matmul
    let w_arr = Array::from_slice(&w_f32, &[n_out as i32, k as i32]);
    let wt = mlx_rs::ops::transpose_axes(&w_arr, &[1, 0])?;
    let result = mlx_rs::ops::matmul(x, &wt)?;
    result.eval()?;
    Ok(result)
}
