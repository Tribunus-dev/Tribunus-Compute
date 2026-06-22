//! Central projection dispatch for quantized matmuls.
//!
//! All quantized matmul calls must go through [`ProjectionExecutor`], which
//! enforces a typed [`QuantizedProjectionDescriptor`] and dispatches to the
//! configured [`TensorBackend`] implementation.
//!
//! The executor is backend-agnostic — it never calls MLX or candle directly.
//! It translates the descriptor into a [`QuantizedMatmulOp`] and delegates to
//! the backend's [`TensorBackend::quantized_matmul`].

use crate::backend::{
    DType as BackendDType, QuantizedMatmulOp, QuantizedWeightHandle, TensorBackend, TensorHandle,
};
use crate::crash_breadcrumb;
use crate::projection_identity::ProjectionFamily;
use serde::Serialize;
use std::cell::RefCell;

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

/// A record of which backend was used for a single quantized projection.
#[derive(Debug, Clone, Serialize)]
pub struct RouteRecord {
    /// Transformer layer index (0-based).
    pub layer: u32,
    /// Projection family name (e.g. "GateProj", "UpProj", "DownProj", "QProj", "KProj", "VProj", "OProj").
    pub projection: String,
    /// Backend that executed this projection.
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

// ── Re-exports ─────────────────────────────────────────────────────────────

/// Runtime mode affecting dispatch decisions.
///
/// Re-exported from `projection_identity` for convenience.
pub use crate::projection_identity::RuntimeMode;

// ── ProjectionExecutor ─────────────────────────────────────────────────────

/// Central dispatcher for all quantized projections.
///
/// Holds a mutable reference to a [`TensorBackend`] implementation and
/// delegates all quantized matmul work through the trait.
///
/// The executor should be scoped to a single call via a block so the
/// mutable borrow on the backend is released after the projection runs:
///
/// ```ignore
/// let result_h = {
///     let mut executor = ProjectionExecutor { backend: &mut backend, mode: RuntimeMode::Safe };
///     executor.run_projection(x_h, w_h, s_h, b_h, &desc)?
/// };
/// ```
pub struct ProjectionExecutor<'a> {
    /// Backend to dispatch operations through.
    pub backend: &'a mut dyn TensorBackend,
    /// Current runtime mode (affects dispatch decisions).
    pub mode: RuntimeMode,
}

impl ProjectionExecutor<'_> {
    /// Execute one quantized projection.
    ///
    /// Translates the descriptor into a [`QuantizedMatmulOp`] and delegates
    /// to [`TensorBackend::quantized_matmul`].
    pub fn run_projection(
        &mut self,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        s: TensorHandle,
        b: TensorHandle,
        desc: &QuantizedProjectionDescriptor,
    ) -> Result<TensorHandle, String> {
        // Write crash breadcrumb before entering native code.
        let pid = std::process::id();
        let x_shape = self.backend.shape(x)?;
        let w_shape = self.backend.shape(s)?;
        let caps = self.backend.backend_capabilities();
        crash_breadcrumb::before_native(
            pid,
            desc.layer_index,
            "decode",
            desc.family.as_str(),
            &caps.backend_name,
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
        let result = self.dispatch(x, w, s, b, desc);

        let elapsed_us = start.elapsed().as_micros() as u64;
        crash_breadcrumb::after_native(pid, elapsed_us);
        result
    }

    /// Build a [`QuantizedMatmulOp`] from the descriptor and delegate to the backend.
    fn dispatch(
        &mut self,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        s: TensorHandle,
        b: TensorHandle,
        desc: &QuantizedProjectionDescriptor,
    ) -> Result<TensorHandle, String> {
        // M = batch/seq dimension from input tensor, not from descriptor.
        // K = last dimension of input (logical hidden size).
        let x_shape = self.backend.shape(x)?;
        let n = desc.logical_out_features;

        // M = batch/seq dimension, K = feature dimension.
        // MlxBackend validates x_shape[-2] == op.m and x_shape[-1] == op.k.
        eprintln!("[dispatch] x_shape={:?}", x_shape);
        let (m, k): (u32, u32) = match x_shape.len() {
            1 => (1u32, x_shape[0] as u32),
            2 => (x_shape[0] as u32, x_shape[1] as u32),
            3 => (x_shape[1] as u32, x_shape[2] as u32),
            _ => (
                x_shape[x_shape.len() - 2] as u32,
                x_shape[x_shape.len() - 1] as u32,
            ),
        };
        eprintln!("[dispatch] m={} k={}", m, k);

        let input_dtype = BackendDType::F32;

        let weight_dtype = match desc.storage_dtype {
            StorageDtype::U32 => BackendDType::U32,
            StorageDtype::U8 => BackendDType::U8,
            StorageDtype::I8 => BackendDType::I8,
        };

        let op = QuantizedMatmulOp {
            m,
            n,
            k,
            input_dtype,
            weight_dtype,
            scale_dtype: BackendDType::F32,
            bias_dtype: BackendDType::F32,
            output_dtype: BackendDType::F32,
            group_size: desc.group_size,
            bits: desc.bits,
            transpose: true,
        };

        let result = self.backend.quantized_matmul(&op, x, w, s, b)?;
        record_route(desc, &self.backend.backend_capabilities().backend_name);
        Ok(result)
    }
}
