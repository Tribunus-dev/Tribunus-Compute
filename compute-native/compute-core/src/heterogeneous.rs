//! Heterogeneous backend dispatch — routes individual operations to
//! Accelerate, CoreML/ANE, or MLX per OperationRoute, operating on a
//! shared IOSurface memory island for zero-copy across backends.

use mlx_rs::error::Result as MlxResult;
use mlx_rs::Array;

use crate::arena::Arena;
use crate::config::operation_route::OperationRoute;
use crate::log_debug;
use crate::memory::allocator::IosurfaceAllocator;

use crate::compute_lane::{spawn_lane, ComputeCommand, ComputeLaneId, DeviceIdentity, LaneHandle};
use crate::coreml_bridge::CoreMlModel;
use parking_lot::Mutex;
use std::sync::Arc;

// ── Backend Identifiers ────────────────────────────────────────────────

pub const MLX: u32 = 0;
pub const ACCELERATE: u32 = 1;
pub const COREML: u32 = 2;
pub const ANE: u32 = 3;

// ── Shared Memory Island ───────────────────────────────────────────────

/// A pre-allocated pool of IOSurface-backed memory shared between MLX,
/// Accelerate, and CoreML backends. Output allocations come from this pool
/// so downstream MLX ops read zero-copy from the same physical pages.
pub struct SharedMemoryIsland {
    pub allocator: Arc<Mutex<IosurfaceAllocator>>,
}

impl Clone for SharedMemoryIsland {
    fn clone(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
        }
    }
}

impl SharedMemoryIsland {
    pub fn new() -> Self {
        Self {
            allocator: Arc::new(Mutex::new(IosurfaceAllocator::new(0))),
        }
    }

    /// Create a new island with a specific pool limit (in bytes).
    /// Pass `0` for no limit (unbounded — allocates up to physical RAM).
    pub fn with_limit(max_pool_bytes: u64) -> Self {
        Self {
            allocator: Arc::new(Mutex::new(IosurfaceAllocator::new(max_pool_bytes))),
        }
    }

    /// Allocate an IOSurface Arena and wrap it as an MLX Array.
    /// Returns `(Arc<Arena>, Array)` — keep the Arena alive until MLX finishes.
    /// MLX's deleter callback will drop the Arc when the array is released.
    pub fn alloc_mlx_array(
        &self,
        shape: &[i32],
        dtype: mlx_rs::Dtype,
    ) -> MlxResult<(Arc<Arena>, Array)> {
        let n = shape.iter().product::<i32>() as u32;
        let alloc = self.allocator.lock();
        let arena_id = alloc
            .allocate(1, n, dtype)
            .map_err(|e| mlx_rs::error::Exception::from(e.as_str()))?;
        let arena = alloc
            .get_arena(arena_id)
            .ok_or_else(|| mlx_rs::error::Exception::from("arena not found"))?;
        drop(alloc);
        let arena_arc = Arc::new(arena);
        let arr =
            crate::memory::iosurface_storage::arena_to_mlx_array(arena_arc.clone(), shape, dtype)
                .map_err(|e| mlx_rs::error::Exception::from(e.as_str()))?;
        Ok((arena_arc, arr))
    }
}

// ── Pre-allocated Layer Slots ───────────────────────────────────────────

/// Four pre-allocated IOSurface arenas for a single layer's intermediate
/// tensors.  Using persistent IOSurface memory avoids MLX temporary handle
/// churn between layers.
pub struct PreallocatedSlots {
    pub hidden_a: Arc<Arena>,
    pub hidden_b: Arc<Arena>,
    pub attn_out: Arc<Arena>,
    pub ffn_out: Arc<Arena>,
}

impl PreallocatedSlots {
    #[inline]
    pub fn get_hidden_a(&self) -> &Arena {
        &self.hidden_a
    }

    #[inline]
    pub fn get_hidden_b(&self) -> &Arena {
        &self.hidden_b
    }

    #[inline]
    pub fn get_attn_out(&self) -> &Arena {
        &self.attn_out
    }

    #[inline]
    pub fn get_ffn_out(&self) -> &Arena {
        &self.ffn_out
    }
}

impl SharedMemoryIsland {
    /// Pre-allocate four [n_tokens, hidden_dim] f32 arenas for one layer pass.
    pub fn preallocate_layer_slots(&self, n_tokens: i32, hidden_dim: i32) -> PreallocatedSlots {
        let shape = &[n_tokens, hidden_dim];
        let dtype = mlx_rs::Dtype::Float32;
        let (a, _) = self.alloc_mlx_array(shape, dtype).expect("hidden_a");
        let (b, _) = self.alloc_mlx_array(shape, dtype).expect("hidden_b");
        let (o, _) = self.alloc_mlx_array(shape, dtype).expect("attn_out");
        let (f, _) = self.alloc_mlx_array(shape, dtype).expect("ffn_out");
        PreallocatedSlots {
            hidden_a: a,
            hidden_b: b,
            attn_out: o,
            ffn_out: f,
        }
    }
}

/// Evaluate an MLX array such that its result materializes into `arena`'s
/// IOSurface backing store.  This is a safe wrapper around
/// [`Arena::evaluate_into`].
pub fn evaluate_into_island(arena: &Arena, array: &Array) -> MlxResult<()> {
    unsafe { arena.evaluate_into(array) }.map_err(|e| {
        let msg = e.to_string();
        mlx_rs::error::Exception::from(msg.as_str())
    })
}

// ── Accelerate Dispatch (IOSurface-backed) ─────────────────────────────

/// Evaluate an MLX array and extract a float32 slice for Accelerate.
/// The eval is necessary because MLX intermediate tensors are in GPU memory.
fn eval_and_extract(x: &Array) -> MlxResult<(Vec<i32>, Vec<f32>)> {
    x.eval()?;
    let shape = x.shape().to_vec();
    let data = match x.try_as_slice::<f32>() {
        Ok(s) => s.to_vec(),
        Err(_) => return Err(mlx_rs::error::Exception::from("extract: as_slice failed")),
    };
    Ok((shape, data))
}

/// Dispatch RMS norm to Accelerate, allocating the output from the shared
/// IOSurface island when available.
pub fn dispatch_rms_norm(
    x: &Array,
    weight: &Array,
    eps: f32,
    route: &OperationRoute,
    island: Option<&SharedMemoryIsland>,
) -> MlxResult<Array> {
    if route.rms_norm != ACCELERATE {
        log_debug!(
            "[infer] backend=mlx op=rms_norm shape={:?} elems={}",
            x.shape(),
            x.shape().iter().product::<i32>()
        );
        return crate::primitives::rms_norm(x, weight, eps);
    }
    log_debug!(
        "[infer] backend=accelerate op=rms_norm shape={:?} elems={}",
        x.shape(),
        x.shape().iter().product::<i32>()
    );
    let (shape, x_data) = eval_and_extract(x)?;
    let weight_data = match weight.try_as_slice::<f32>() {
        Ok(s) => s.to_vec(),
        Err(_) => return crate::primitives::rms_norm(x, weight, eps),
    };
    let dim = shape.last().copied().unwrap_or(1) as usize;
    let n = shape.iter().product::<i32>() as usize;
    let batch = n / dim;

    // Try to allocate output from the IOSurface island
    if let Some(island) = island {
        let (arena, out_arr) = match island.alloc_mlx_array(&shape, mlx_rs::Dtype::Float32) {
            Ok(v) => v,
            Err(_) => return crate::primitives::rms_norm(x, weight, eps),
        };
        // Get raw pointer to the IOSurface memory
        let ptr = unsafe { arena.base_ptr() as *mut f32 };
        for b in 0..batch {
            let row_start = b * dim;
            let x_row = &x_data[row_start..row_start + dim];
            let out_row = unsafe { std::slice::from_raw_parts_mut(ptr.add(row_start), dim) };
            // vDSP_vmul: x_sq = x * x
            let dim_i32 = dim as i32;
            let mut x_sq = vec![0.0f32; dim];
            unsafe {
                crate::backend::accelerate_ffi::vDSP_vmul(
                    x_row.as_ptr(),
                    1,
                    x_row.as_ptr(),
                    1,
                    x_sq.as_mut_ptr(),
                    1,
                    dim_i32,
                );
            }
            let mut sum = 0.0f32;
            unsafe {
                crate::backend::accelerate_ffi::vDSP_sve(x_sq.as_ptr(), 1, &mut sum, dim_i32);
            }
            let inv_rms = 1.0 / ((sum / dim as f32) + eps).sqrt();
            let scalar = [inv_rms];
            unsafe {
                crate::backend::accelerate_ffi::vDSP_vsmul(
                    x_row.as_ptr(),
                    1,
                    scalar.as_ptr(),
                    out_row.as_mut_ptr(),
                    1,
                    dim_i32,
                );
                crate::backend::accelerate_ffi::vDSP_vmul(
                    out_row.as_mut_ptr(),
                    1,
                    weight_data.as_ptr(),
                    1,
                    out_row.as_mut_ptr(),
                    1,
                    dim_i32,
                );
            }
        }
        // Eval the result array so MLX knows the data is ready
        out_arr.eval()?;
        return Ok(out_arr);
    }

    // Fallback: heap-allocated output
    let mut out = vec![0.0f32; n];
    for b in 0..batch {
        let row_start = b * dim;
        let x_row = &x_data[row_start..row_start + dim];
        let out_row = &mut out[row_start..row_start + dim];
        let dim_i32 = dim as i32;
        let mut x_sq = vec![0.0f32; dim];
        unsafe {
            crate::backend::accelerate_ffi::vDSP_vmul(
                x_row.as_ptr(),
                1,
                x_row.as_ptr(),
                1,
                x_sq.as_mut_ptr(),
                1,
                dim_i32,
            );
        }
        let mut sum = 0.0f32;
        unsafe {
            crate::backend::accelerate_ffi::vDSP_sve(x_sq.as_ptr(), 1, &mut sum, dim_i32);
        }
        let inv_rms = 1.0 / ((sum / dim as f32) + eps).sqrt();
        let scalar = [inv_rms];
        unsafe {
            crate::backend::accelerate_ffi::vDSP_vsmul(
                x_row.as_ptr(),
                1,
                scalar.as_ptr(),
                out_row.as_mut_ptr(),
                1,
                dim_i32,
            );
            crate::backend::accelerate_ffi::vDSP_vmul(
                out_row.as_mut_ptr(),
                1,
                weight_data.as_ptr(),
                1,
                out_row.as_mut_ptr(),
                1,
                dim_i32,
            );
        }
    }
    Ok(Array::from_slice(&out, &shape))
}

/// Dispatch element-wise add to Accelerate.
pub fn dispatch_add(
    a: &Array,
    b: &Array,
    route: &OperationRoute,
    island: Option<&SharedMemoryIsland>,
) -> MlxResult<Array> {
    if route.add != ACCELERATE {
        log_debug!(
            "[infer] backend=mlx op=add shape={:?} elems={}",
            a.shape(),
            a.shape().iter().product::<i32>()
        );
        return a.add(b);
    }
    log_debug!(
        "[infer] backend=accelerate op=add shape={:?} elems={}",
        a.shape(),
        a.shape().iter().product::<i32>()
    );
    let (shape, a_data) = eval_and_extract(a)?;
    let b_data = match b.try_as_slice::<f32>() {
        Ok(s) => s.to_vec(),
        Err(_) => return a.add(b),
    };
    let n = a_data.len().min(b_data.len());
    let n_i32 = n as i32;

    if let Some(island) = island {
        let (arena, out_arr) = match island.alloc_mlx_array(&shape, mlx_rs::Dtype::Float32) {
            Ok(v) => v,
            Err(_) => return a.add(b),
        };
        let ptr = unsafe { arena.base_ptr() as *mut f32 };
        unsafe {
            crate::backend::accelerate_ffi::vDSP_vadd(
                a_data.as_ptr(),
                1,
                b_data.as_ptr(),
                1,
                ptr,
                1,
                n_i32,
            );
        }
        out_arr.eval()?;
        return Ok(out_arr);
    }

    let mut out = vec![0.0f32; n];
    unsafe {
        crate::backend::accelerate_ffi::vDSP_vadd(
            a_data.as_ptr(),
            1,
            b_data.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n_i32,
        );
    }
    Ok(Array::from_slice(&out, &shape))
}

/// Dispatch element-wise multiply to Accelerate.
pub fn dispatch_multiply(
    a: &Array,
    b: &Array,
    route: &OperationRoute,
    island: Option<&SharedMemoryIsland>,
) -> MlxResult<Array> {
    if route.multiply != ACCELERATE {
        log_debug!(
            "[infer] backend=mlx op=multiply shape={:?} elems={}",
            a.shape(),
            a.shape().iter().product::<i32>()
        );
        return a.multiply(b);
    }
    log_debug!(
        "[infer] backend=accelerate op=multiply shape={:?} elems={}",
        a.shape(),
        a.shape().iter().product::<i32>()
    );
    log_debug!(
        "[infer] op=multiply_extract a_shape={:?} b_shape={:?}",
        a.shape(),
        b.shape()
    );
    let (shape, a_data) = eval_and_extract(a)?;
    log_debug!(
        "[infer] op=multiply_extract_b_before_eval shape={:?}",
        shape
    );
    let b_data = match b.try_as_slice::<f32>() {
        Ok(s) => {
            log_debug!("[infer] op=multiply_extract_b_ok len={}", s.len());
            s.to_vec()
        }
        Err(e) => {
            log_debug!("[infer] op=multiply_extract_b_err {:?}", e);
            return a.multiply(b);
        }
    };
    let n = a_data.len().min(b_data.len());
    let n_i32 = n as i32;

    if let Some(island) = island {
        let (arena, out_arr) = match island.alloc_mlx_array(&shape, mlx_rs::Dtype::Float32) {
            Ok(v) => v,
            Err(_) => return a.multiply(b),
        };
        let ptr = unsafe { arena.base_ptr() as *mut f32 };
        unsafe {
            crate::backend::accelerate_ffi::vDSP_vmul(
                a_data.as_ptr(),
                1,
                b_data.as_ptr(),
                1,
                ptr,
                1,
                n_i32,
            );
        }
        out_arr.eval()?;
        return Ok(out_arr);
    }

    let mut out = vec![0.0f32; n];
    unsafe {
        crate::backend::accelerate_ffi::vDSP_vmul(
            a_data.as_ptr(),
            1,
            b_data.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            n_i32,
        );
    }
    Ok(Array::from_slice(&out, &shape))
}

/// Reshape — no-op in Accelerate (view change).
pub fn dispatch_reshape(x: &Array, shape: &[i32], route: &OperationRoute) -> MlxResult<Array> {
    if route.reshape != ACCELERATE {
        log_debug!(
            "[infer] backend=mlx op=reshape from={:?} to={:?}",
            x.shape(),
            shape
        );
        return x.reshape(shape);
    }
    log_debug!(
        "[infer] backend=accelerate op=reshape from={:?} to={:?}",
        x.shape(),
        shape
    );
    x.reshape(shape)
}

// ── CoreML / ANE Dispatch (stub) ──────────────────────────────────────

// ── CoreML / ANE Dispatch ────────────────────────────────────────────

/// Dispatch attention to the ANE via a pre-loaded CoreML model.
///
/// Extracts Q, K, V data from MLX arrays into an IOSurface-backed input
/// arena, runs the CoreML model via `predict`, and reads the result back
/// as an MLX Array from the output arena.
pub fn dispatch_attention_ane(
    query: &Array,
    key: &Array,
    value: &Array,
    model: &CoreMlModel,
    island: &SharedMemoryIsland,
) -> MlxResult<Array> {
    // 1. Eval and extract Q, K, V as f32 slices
    query.eval()?;
    key.eval()?;
    value.eval()?;

    let q_slice = query
        .try_as_slice::<f32>()
        .map_err(|_| mlx_rs::error::Exception::from("ANE: query try_as_slice failed"))?;
    let k_slice = key
        .try_as_slice::<f32>()
        .map_err(|_| mlx_rs::error::Exception::from("ANE: key try_as_slice failed"))?;
    let v_slice = value
        .try_as_slice::<f32>()
        .map_err(|_| mlx_rs::error::Exception::from("ANE: value try_as_slice failed"))?;

    let q_len = q_slice.len();
    let k_len = k_slice.len();
    let v_len = v_slice.len();
    let total = q_len + k_len + v_len;

    // 2. Concatenate Q, K, V into a single flat buffer
    let mut qkv = Vec::with_capacity(total);
    qkv.extend_from_slice(q_slice);
    qkv.extend_from_slice(k_slice);
    qkv.extend_from_slice(v_slice);

    // 3. Allocate input IOSurface arena and write QKV data
    let (input_arena, _input_arr) = island
        .alloc_mlx_array(&[1, total as i32], mlx_rs::Dtype::Float32)
        .map_err(|_| mlx_rs::error::Exception::from("ANE alloc input arena"))?;
    unsafe {
        let ptr = input_arena.base_ptr() as *mut f32;
        std::ptr::copy_nonoverlapping(qkv.as_ptr(), ptr, total);
    }

    // 4. Allocate output IOSurface arena (same shape as query)
    let out_shape = query.shape();
    let (output_arena, output_arr) = island
        .alloc_mlx_array(out_shape, mlx_rs::Dtype::Float32)
        .map_err(|_| mlx_rs::error::Exception::from("ANE alloc output arena"))?;

    // 5. Run CoreML prediction
    model
        .predict("query", &input_arena.info, "output", &output_arena.info)
        .map_err(|e| -> mlx_rs::error::Exception {
            let msg = format!("ANE predict: {}", e);
            mlx_rs::error::Exception::from(msg.as_str())
        })?;

    // 6. Return the output MLX Array backed by the IOSurface
    Ok(output_arr)
}

/// Dispatch attention to the ANE via a per-layer cache of CoreML models.
///
/// Looks up the pre-loaded model for `layer_idx`, calls
/// [`dispatch_attention_ane`], and falls back to an error if no model is
/// available for this layer.
pub fn dispatch_attention_coreml(
    query: &Array,
    key: &Array,
    value: &Array,
    ane_models: &[Option<std::sync::Arc<CoreMlModel>>],
    layer_idx: usize,
    island: &SharedMemoryIsland,
) -> MlxResult<Array> {
    if let Some(Some(model)) = ane_models.get(layer_idx) {
        dispatch_attention_ane(query, key, value, model, island)
    } else {
        Err(mlx_rs::error::Exception::from(
            "ANE model not available for this layer",
        ))
    }
}

// ── Backend execution lanes ──────────────────────────────────────────

/// A set of one lane per backend for heterogeneous execution.
pub struct BackendLanes {
    pub mlx: LaneHandle,
    pub accelerate: LaneHandle,
    pub coreml: LaneHandle,
}

/// Drain commands from a stub backend lane — accepts commands but does
/// not process them.  The channel is kept open so senders do not fail.
fn stub_runner(mut rx: tokio::sync::mpsc::Receiver<ComputeCommand>) {
    while let Some(_cmd) = rx.blocking_recv() {
        // Stub: accept and discard all commands.
    }
}

/// Create one stub lane per backend (MLX, Accelerate, CoreML).
pub fn create_backend_lanes() -> BackendLanes {
    let mlx = spawn_lane(
        ComputeLaneId(0),
        DeviceIdentity {
            lane_id: ComputeLaneId(0),
            backend_name: "mlx".into(),
            substrate: "stub".into(),
        },
        64,
        move |rx| stub_runner(rx),
    );

    let accelerate = spawn_lane(
        ComputeLaneId(1),
        DeviceIdentity {
            lane_id: ComputeLaneId(1),
            backend_name: "accelerate".into(),
            substrate: "stub".into(),
        },
        64,
        move |rx| stub_runner(rx),
    );

    let coreml = spawn_lane(
        ComputeLaneId(2),
        DeviceIdentity {
            lane_id: ComputeLaneId(2),
            backend_name: "coreml".into(),
            substrate: "stub".into(),
        },
        64,
        move |rx| stub_runner(rx),
    );

    BackendLanes {
        mlx,
        accelerate,
        coreml,
    }
}

/// Combined compute runtime — shared memory island and per-backend lanes.
pub struct ComputeRuntime {
    pub island: SharedMemoryIsland,
    pub lanes: BackendLanes,
}

// ── Heterogeneous Layer Runner ─────────────────────────────────────────

pub fn run_layer_heterogeneous(
    hidden: &Array,
    plan: &crate::config::LayerPlan,
    route: &OperationRoute,
    island: Option<&SharedMemoryIsland>,
    attn_norm: &Array,
    ffn_norm: &Array,
    qw: &Array,
    qs: &Array,
    qb: &Array,
    kw: &Array,
    ks: &Array,
    kb: &Array,
    vw: &Array,
    vs: &Array,
    vb: &Array,
    ow: &Array,
    os: &Array,
    ob: &Array,
    q_norm_weight: Option<&Array>,
    k_norm_weight: Option<&Array>,
    gw: &Array,
    gs: &Array,
    gb: &Array,
    uw: &Array,
    us: &Array,
    ub: &Array,
    dw: &Array,
    ds: &Array,
    db: &Array,
    rope_cos: &Array,
    rope_sin: &Array,
    cache: &mut crate::kv_cache::KvCache,
    kv_offset: u32,
    rms_norm_eps: f32,
    ctx: &crate::projection_identity::ProjectionContext,
) -> MlxResult<Array> {
    crate::executor::run_layer(
        hidden,
        plan,
        route,
        island,
        &[],
        attn_norm,
        ffn_norm,
        qw,
        qs,
        qb,
        kw,
        ks,
        kb,
        vw,
        vs,
        vb,
        ow,
        os,
        ob,
        q_norm_weight,
        k_norm_weight,
        gw,
        gs,
        gb,
        uw,
        us,
        ub,
        dw,
        ds,
        db,
        rope_cos,
        rope_sin,
        cache,
        kv_offset,
        rms_norm_eps,
        ctx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_rms_norm_falls_through() {
        let route = OperationRoute {
            rms_norm: 1,
            ..Default::default()
        };
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let w = Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[4]);
        let result = dispatch_rms_norm(&x, &w, 1e-6, &route, None).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.try_as_slice::<f32>().unwrap().to_vec();
        assert_eq!(data.len(), 4);
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dispatch_rms_norm_with_island() {
        let island = SharedMemoryIsland::new();
        let route = OperationRoute {
            rms_norm: 1,
            ..Default::default()
        };
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let w = Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[4]);
        let result = dispatch_rms_norm(&x, &w, 1e-6, &route, Some(&island)).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.try_as_slice::<f32>().unwrap().to_vec();
        assert_eq!(data.len(), 4);
        for &v in &data {
            assert!(v.is_finite());
        }
    }
}
