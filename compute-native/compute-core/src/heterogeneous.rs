//! Heterogeneous backend dispatch — routes individual operations to
//! Accelerate, CoreML/ANE, or MLX per OperationRoute, operating on a
//! shared IOSurface memory island for zero-copy across backends.

use mlx_rs::Array;
use mlx_rs::error::Result as MlxResult;

use crate::config::operation_route::OperationRoute;
use crate::memory::allocator::IosurfaceAllocator;
use crate::arena::Arena;

use std::sync::Arc;
use parking_lot::Mutex;

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

impl SharedMemoryIsland {
    pub fn new() -> Self {
        Self {
            allocator: Arc::new(Mutex::new(IosurfaceAllocator::new(1024 * 1024))),
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
        let arr = crate::memory::iosurface_storage::arena_to_mlx_array(
            arena_arc.clone(), shape, dtype,
        )
        .map_err(|e| mlx_rs::error::Exception::from(e.as_str()))?;
        Ok((arena_arc, arr))
    }
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
        return crate::primitives::rms_norm(x, weight, eps);
    }
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
                    x_row.as_ptr(), 1, x_row.as_ptr(), 1,
                    x_sq.as_mut_ptr(), 1, dim_i32,
                );
            }
            let mut sum = 0.0f32;
            unsafe {
                crate::backend::accelerate_ffi::vDSP_sve(
                    x_sq.as_ptr(), 1, &mut sum, dim_i32,
                );
            }
            let inv_rms = 1.0 / ((sum / dim as f32) + eps).sqrt();
            let scalar = [inv_rms];
            unsafe {
                crate::backend::accelerate_ffi::vDSP_vsmul(
                    x_row.as_ptr(), 1, scalar.as_ptr(),
                    out_row.as_mut_ptr(), 1, dim_i32,
                );
                crate::backend::accelerate_ffi::vDSP_vmul(
                    out_row.as_mut_ptr(), 1, weight_data.as_ptr(), 1,
                    out_row.as_mut_ptr(), 1, dim_i32,
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
                x_row.as_ptr(), 1, x_row.as_ptr(), 1,
                x_sq.as_mut_ptr(), 1, dim_i32,
            );
        }
        let mut sum = 0.0f32;
        unsafe {
            crate::backend::accelerate_ffi::vDSP_sve(
                x_sq.as_ptr(), 1, &mut sum, dim_i32,
            );
        }
        let inv_rms = 1.0 / ((sum / dim as f32) + eps).sqrt();
        let scalar = [inv_rms];
        unsafe {
            crate::backend::accelerate_ffi::vDSP_vsmul(
                x_row.as_ptr(), 1, scalar.as_ptr(),
                out_row.as_mut_ptr(), 1, dim_i32,
            );
            crate::backend::accelerate_ffi::vDSP_vmul(
                out_row.as_mut_ptr(), 1, weight_data.as_ptr(), 1,
                out_row.as_mut_ptr(), 1, dim_i32,
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
        return a.add(b);
    }
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
                a_data.as_ptr(), 1, b_data.as_ptr(), 1,
                ptr, 1, n_i32,
            );
        }
        out_arr.eval()?;
        return Ok(out_arr);
    }

    let mut out = vec![0.0f32; n];
    unsafe {
        crate::backend::accelerate_ffi::vDSP_vadd(
            a_data.as_ptr(), 1, b_data.as_ptr(), 1,
            out.as_mut_ptr(), 1, n_i32,
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
        return a.multiply(b);
    }
    let (shape, a_data) = eval_and_extract(a)?;
    let b_data = match b.try_as_slice::<f32>() {
        Ok(s) => s.to_vec(),
        Err(_) => return a.multiply(b),
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
                a_data.as_ptr(), 1, b_data.as_ptr(), 1,
                ptr, 1, n_i32,
            );
        }
        out_arr.eval()?;
        return Ok(out_arr);
    }

    let mut out = vec![0.0f32; n];
    unsafe {
        crate::backend::accelerate_ffi::vDSP_vmul(
            a_data.as_ptr(), 1, b_data.as_ptr(), 1,
            out.as_mut_ptr(), 1, n_i32,
        );
    }
    Ok(Array::from_slice(&out, &shape))
}

/// Reshape — no-op in Accelerate (view change).
pub fn dispatch_reshape(x: &Array, shape: &[i32], route: &OperationRoute) -> MlxResult<Array> {
    if route.reshape != ACCELERATE {
        return x.reshape(shape);
    }
    x.reshape(shape)
}

// ── CoreML / ANE Dispatch (stub) ──────────────────────────────────────

pub fn dispatch_attention_ane(
    _query: &Array, _key: &Array, _value: &Array,
    _cache: &mut crate::kv_cache::KvCache, _layer_name: &str,
) -> MlxResult<Array> {
    Array::zeros::<f32>(&_query.shape())
}

// ── Heterogeneous Layer Runner ─────────────────────────────────────────

pub fn run_layer_heterogeneous(
    hidden: &Array, plan: &crate::config::LayerPlan, route: &OperationRoute,
    island: Option<&SharedMemoryIsland>,
    attn_norm: &Array, ffn_norm: &Array,
    qw: &Array, qs: &Array, qb: &Array,
    kw: &Array, ks: &Array, kb: &Array,
    vw: &Array, vs: &Array, vb: &Array,
    ow: &Array, os: &Array, ob: &Array,
    q_norm_weight: Option<&Array>, k_norm_weight: Option<&Array>,
    gw: &Array, gs: &Array, gb: &Array,
    uw: &Array, us: &Array, ub: &Array,
    dw: &Array, ds: &Array, db: &Array,
    rope_cos: &Array, rope_sin: &Array,
    cache: &mut crate::kv_cache::KvCache,
    kv_offset: u32, rms_norm_eps: f32,
    ctx: &crate::projection_identity::ProjectionContext,
) -> MlxResult<Array> {
    crate::executor::run_layer(
        hidden, plan, route, island,
        attn_norm, ffn_norm,
        qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob,
        q_norm_weight, k_norm_weight,
        gw, gs, gb, uw, us, ub, dw, ds, db,
        rope_cos, rope_sin, cache, kv_offset, rms_norm_eps, ctx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_rms_norm_falls_through() {
        let route = OperationRoute { rms_norm: 1, ..Default::default() };
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let w = Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[4]);
        let result = dispatch_rms_norm(&x, &w, 1e-6, &route, None).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.try_as_slice::<f32>().unwrap().to_vec();
        assert_eq!(data.len(), 4);
        for &v in &data { assert!(v.is_finite()); }
    }

    #[test]
    fn test_dispatch_rms_norm_with_island() {
        let island = SharedMemoryIsland::new();
        let route = OperationRoute { rms_norm: 1, ..Default::default() };
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let w = Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[4]);
        let result = dispatch_rms_norm(&x, &w, 1e-6, &route, Some(&island)).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.try_as_slice::<f32>().unwrap().to_vec();
        assert_eq!(data.len(), 4);
        for &v in &data { assert!(v.is_finite()); }
    }
}
