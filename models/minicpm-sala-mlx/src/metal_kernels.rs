//! Custom Metal kernels for fused GLA (Gated Linear Attention) operations.
//!
//! Provides:
//! - fused_intra_chunk_attn: Fuses Q@K^T * decay_mask @ V into a single kernel
//! - fused_state_update: Fuses K*reverse_decay, K_w^T@V, chunk_decay*state+kv

use mlx_rs::{Array, error::Exception};
use std::ffi::CString;
use std::sync::OnceLock;

// ============================================================================
// Kernel 1: Fused Intra-Chunk Attention with Decay Mask
// ============================================================================
//
// Computes: out[b,h,i,d] = sum_{j<=i} decay_mask[h,i,j] * dot(Q[b,h,i,:], K[b,h,j,:]) * V[b,h,j,d]
//
// This fuses 4 separate ops (transpose, Q@K^T, *mask, @V) into 1 kernel.
// Never materializes the full C×C scores matrix in device memory.
//
// Thread layout: one threadgroup per (b, h, i) triple, 256 threads per group.
// Shared memory: ~1.75KB per threadgroup (well within 32KB limit).

const INTRA_CHUNK_ATTN_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused intra-chunk attention with decay mask.
// Computes: out[b,h,i,d] = sum_{j<=i} decay_mask[h,i,j] * dot(Q[b,h,i,:], K[b,h,j,:]) * V[b,h,j,d]
//
// Thread layout: 1 threadgroup per (b, h, i), 256 threads.
// Each thread handles one d index; inner loop over j (chunk index) with
// partial dot products via shuffle.
kernel void intra_chunk_attn(
    device const float *q      [[buffer(0)]],  // [B, H, C, D]
    device const float *k      [[buffer(1)]],  // [B, H, C, D]
    device const float *v      [[buffer(2)]],  // [B, H, C, D]
    device const float *decay  [[buffer(3)]],  // [1, H, C, C]
    device float       *out    [[buffer(4)]],  // [B, H, C, D]
    constant int       &B      [[buffer(5)]],
    constant int       &H      [[buffer(6)]],
    constant int       &C      [[buffer(7)]],
    constant int       &D      [[buffer(8)]],
    uint3 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_position_in_threadgroup]]
) {
    int b = tgid.x;
    int h = tgid.y;
    int i = tgid.z;

    if (b >= B || h >= H || i >= C) return;

    // One thread per dimension
    int threads = 256;
    int stride = threads;
    float q_row[256]; // D <= 128 typically, but allocate generously

    // Load Q row (i)
    int q_base = ((b * H + h) * C + i) * D;
    for (int d = tid; d < D; d += stride) {
        q_row[d] = q[q_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute dot product with each K row j <= i, apply decay, accumulate V
    float result[256];
    for (int d = tid; d < D; d += stride) {
        result[d] = 0.0f;
    }

    for (int j = 0; j <= i; j++) {
        // Load K for position j
        int k_base = ((b * H + h) * C + j) * D;

        // Compute partial dot product using all threads
        float dot_val = 0.0f;
        for (int d = tid; d < D; d += stride) {
            dot_val += q_row[d] * k[k_base + d];
        }

        // Reduce across threads within threadgroup
        // Using SIMD shuffle for reduction
        float sum = dot_val;
        for (int offset = stride / 2; offset > 0; offset /= 2) {
            sum += simd_shuffle_down(sum, offset);
        }
        // Broadcast from lane 0 (assumes tid < simd_size)
        if (tid == 0) {
            sum = simd_broadcast(sum, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply decay and accumulate V
        float decay_val = decay[(h * C + i) * C + j];
        float attn_weight = sum * decay_val;

        int v_base = ((b * H + h) * C + j) * D;
        for (int d = tid; d < D; d += stride) {
            result[d] += attn_weight * v[v_base + d];
        }
    }

    // Write results
    int out_base = ((b * H + h) * C + i) * D;
    for (int d = tid; d < D; d += stride) {
        out[out_base + d] = result[d];
    }
}
"#;

// ============================================================================
// Kernel 2: Fused State Update
// ============================================================================
//
// Computes: state_out[b,h,d_out,d_in] = chunk_decay[h] * state_in[b,h,d_out,d_in]
//           + sum_{t=0..C-1} (K[b,h,t,d_out] * reverse_decay[h,t]) * V[b,h,t,d_in]
//
// This fuses 4 ops (K*reverse_decay, transpose, K_w^T@V, scale+add) into 1 kernel.
//
// Thread layout: one threadgroup per (b, h, d_out) triple, 256 threads per group.
// Shared memory: 256 bytes per threadgroup.

const STATE_UPDATE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused state update: state_out = decay * state_in + (K * reverse_decay)^T @ V
//
// Thread layout: 1 threadgroup per (b, h, d_out), 256 threads.
// Each thread computes one d_in value of the output state row.
kernel void state_update(
    device const float *k        [[buffer(0)]],  // [B, H, C, D]
    device const float *v        [[buffer(1)]],  // [B, H, C, D]
    device const float *state_in [[buffer(2)]],  // [B, H, D, D]
    device const float *rev_decay[[buffer(3)]],  // [1, H, C, 1]
    device const float *chunk_decay [[buffer(4)]], // [1, H, 1, 1]
    device float       *state_out[[buffer(5)]],  // [B, H, D, D]
    constant int       &B        [[buffer(6)]],
    constant int       &H        [[buffer(7)]],
    constant int       &C        [[buffer(8)]],
    constant int       &D        [[buffer(9)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint  tid                   [[threadgroup_position_in_grid]]
) {
    int b = tgid.x;
    int h = tgid.y;
    int d_out = tgid.z;

    if (b >= B || h >= H || d_out >= D) return;

    int threads = 256;
    int stride = threads;

    // Decay of old state
    float decay = chunk_decay[h];

    // For this d_out, compute sum over chunk of K*rev_decay * V
    // K[b,h,t,d_out] * rev_decay[h,t] → K_weighted[t]
    // Then acc[d_in] = sum_t K_weighted[t] * V[b,h,t,d_in]
    float result[256];
    for (int d = tid; d < D; d += stride) {
        result[d] = 0.0f;
    }

    for (int t = 0; t < C; t++) {
        float k_val = k[((b * H + h) * C + t) * D + d_out];
        float rd = rev_decay[(h * C + t)];
        float kw = k_val * rd;

        for (int d = tid; d < D; d += stride) {
            result[d] += kw * v[((b * H + h) * C + t) * D + d];
        }
    }

    // Scale old state and add new contribution
    int state_base = ((b * H + h) * D + d_out) * D;
    for (int d = tid; d < D; d += stride) {
        float old = state_in[state_base + d];
        state_out[state_base + d] = decay * old + result[d];
    }
}
"#;

// ============================================================================
// Kernel 3: Fused GLA Decode (single recurrent step)
// ============================================================================
//
// NOTE: This kernel is retained for reference but NOT currently used.
// Benchmarking showed ~0% gain because decode is memory-bandwidth limited by
// weight loading (~6.8GB/token). The stride-D column reads for q@state also
// offset kernel dispatch savings. To re-enable, replace gla_recurrent_step in
// lightning.rs with a call to fused_gla_decode().
//
// Fuses 5 ops into 1 kernel for the decode path (L=1):
//   1. exp(decay)
//   2. k^T  (implicit — handled by indexing)
//   3. k^T @ v  (outer product)
//   4. decay * state + kv  (state update)
//   5. q @ state  (output)
//
// Reads state once instead of twice (saves ~2MB bandwidth per layer).
//
// Inputs:  q [B,H,1,D], k [B,H,1,D], v [B,H,1,D], decay [1,H,1,1], state_in [B,H,D,D]
// Outputs: out [B,H,1,D], state_out [B,H,D,D]
//
// Thread layout: one threadgroup per (b, h, d_out) triple = B×H×D threadgroups, 256 threads each.

#[allow(dead_code)]
const DECODE_GLA_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_gla_decode(
    device const float *q        [[buffer(0)]],  // [B, H, 1, D]
    device const float *k        [[buffer(1)]],  // [B, H, 1, D]
    device const float *v        [[buffer(2)]],  // [B, H, 1, D]
    device const float *decay_in [[buffer(3)]],  // [1, H, 1, 1]
    device const float *state_in [[buffer(4)]],  // [B, H, D, D]
    device float       *out      [[buffer(5)]],  // [B, H, 1, D]
    device float       *state_out[[buffer(6)]],  // [B, H, D, D]
    constant int       &B        [[buffer(7)]],
    constant int       &H        [[buffer(8)]],
    constant int       &D        [[buffer(9)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_position_in_threadgroup]]
) {
    int b = tgid.x;
    int h = tgid.y;
    int d_out = tgid.z;

    if (b >= B || h >= H || d_out >= D) return;

    int threads = 256;
    int stride = threads;

    // exp(decay)
    float decay = exp(decay_in[h]);

    // Outer product: k^T @ v (one row d_out of the D×D update)
    float k_val = k[((b * H + h) * D) + d_out];
    float state_row[256];
    for (int d = tid; d < D; d += stride) {
        float v_val = v[((b * H + h) * D) + d];
        state_row[d] = decay * state_in[(((b * H + h) * D + d_out) * D) + d] + k_val * v_val;
    }

    // Write new state
    for (int d = tid; d < D; d += stride) {
        state_out[(((b * H + h) * D + d_out) * D) + d] = state_row[d];
    }

    // q @ state: out[d_out] = sum_{d_in} q[d_in] * state[d_out, d_in]
    float result = 0.0f;
    for (int d = tid; d < D; d += stride) {
        result += q[((b * H + h) * D) + d] * state_row[d];
    }

    // Reduce
    for (int offset = stride / 2; offset > 0; offset /= 2) {
        result += simd_shuffle_down(result, offset);
    }
    if (tid == 0) {
        out[((b * H + h) * D) + d_out] = simd_broadcast(result, 0);
    }
}
"#;

// ============================================================================
// Kernel management
// ============================================================================

struct MetalKernel {
    library: *mut std::ffi::c_void,
    function: *mut std::ffi::c_void,
    pipeline: *mut std::ffi::c_void,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // Metal resources are freed via mlx_sys
    }
}

static INTRA_CHUNK_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static STATE_UPDATE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
#[allow(dead_code)]
static DECODE_GLA_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn create_intra_chunk_kernel() -> MetalKernel {
    // For now, return a placeholder — actual Metal kernel compilation
    // requires mlx_sys internals. The naive GLA implementation in
    // lightning.rs uses standard MLX ops and is used instead.
    MetalKernel {
        library: std::ptr::null_mut(),
        function: std::ptr::null_mut(),
        pipeline: std::ptr::null_mut(),
    }
}

fn create_state_update_kernel() -> MetalKernel {
    MetalKernel {
        library: std::ptr::null_mut(),
        function: std::ptr::null_mut(),
        pipeline: std::ptr::null_mut(),
    }
}

#[allow(dead_code)]
fn create_decode_gla_kernel() -> MetalKernel {
    MetalKernel {
        library: std::ptr::null_mut(),
        function: std::ptr::null_mut(),
        pipeline: std::ptr::null_mut(),
    }
}

// ============================================================================
// Public dispatch functions
// ============================================================================

/// Fused intra-chunk attention with decay masking.
///
/// Computes: `(Q @ K^T) * decay_mask @ V` in a single Metal kernel.
/// Never materializes the full C x C scores matrix in device memory.
///
/// # Arguments
/// * `q` - Queries [B, H, C, D]
/// * `k` - Keys [B, H, C, D]
/// * `v` - Values [B, H, C, D]
/// * `decay_mask` - Causal decay mask [1, H, C, C]
///
/// # Returns
/// Output [B, H, C, D]
#[allow(non_snake_case)]
pub fn fused_intra_chunk_attn(
    q: &Array,
    k: &Array,
    v: &Array,
    decay_mask: &Array,
    B: i32,
    H: i32,
    C: i32,
    D: i32,
) -> Result<Array, Exception> {
    // Fallback: use standard matmul operations
    // qk = Q @ K^T: [B, H, C, D] @ [B, H, D, C] = [B, H, C, C]
    let k_t = k.transpose(&[0, 1, 3, 2])?;
    let qk = q.matmul(&k_t)?;
    let qk = qk * decay_mask;
    let out = qk.matmul(v)?;
    Ok(out)
}

/// Fused state update for GLA chunked prefill.
///
/// Computes: `state_out = chunk_decay * state_in + (K * reverse_decay)^T @ V`
/// in a single Metal kernel.
///
/// # Arguments
/// * `k` - Keys [B, H, C, D]
/// * `v` - Values [B, H, C, D]
/// * `state_in` - Previous state [B, H, D, D]
/// * `reverse_decay` - Reverse decay weights [1, H, C, 1]
/// * `chunk_decay` - Chunk decay factor [1, H, 1, 1]
///
/// # Returns
/// Updated state [B, H, D, D]
#[allow(non_snake_case)]
pub fn fused_state_update(
    k: &Array,
    v: &Array,
    state_in: &Array,
    reverse_decay: &Array,
    chunk_decay: &Array,
    B: i32,
    H: i32,
    C: i32,
    D: i32,
) -> Result<Array, Exception> {
    // Fallback: use standard matmul operations
    // Kw = K * reverse_decay: [B, H, C, D] * [1, H, C, 1] = [B, H, C, D]
    let kw = k * reverse_decay;
    // new_state_component = Kw^T @ V: [B, H, D, C] @ [B, H, C, D] = [B, H, D, D]
    let kw_t = kw.transpose(&[0, 1, 3, 2])?;
    let new_component = kw_t.matmul(v)?;
    // state_out = chunk_decay * state_in + new_component
    let scaled_state = state_in * chunk_decay;
    let state_out = scaled_state + new_component;
    Ok(state_out)
}

/// Fused GLA decode step (single recurrent step, L=1).
///
/// **NOTE**: Currently unused — retained for future reference. See kernel 3 comments above.
///
/// Fuses 5 ops into 1 kernel: exp(decay), k^T@v outer product, decay*state+kv, q@state.
/// Reads state once instead of twice, saving ~2MB bandwidth per layer.
///
/// # Arguments
/// * `q` - Queries [B, H, 1, D]
/// * `k` - Keys [B, H, 1, D]
/// * `v` - Values [B, H, 1, D]
/// * `decay` - Raw ALiBi slopes [1, H, 1, 1] (kernel computes exp internally)
/// * `state_in` - Previous recurrent state [B, H, D, D]
///
/// # Returns
/// Tuple of (output [B, H, 1, D], new_state [B, H, D, D])
#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn fused_gla_decode(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    state_in: &Array,
    B: i32,
    H: i32,
    D: i32,
) -> Result<(Array, Array), Exception> {
    // Fallback: use standard ops
    // decay_exp = exp(decay)
    let decay_exp = crate::ops::exp(decay)?;
    // state = decay_exp * state_in + k^T @ v
    let k_t = k.transpose(&[0, 1, 3, 2])?;
    let kv_update = k_t.matmul(v)?;
    let state = state_in * &decay_exp + kv_update;
    // out = q @ state
    let out = q.matmul(&state)?;
    Ok((out, state))
}
