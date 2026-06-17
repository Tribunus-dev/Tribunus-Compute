use std::cmp::min;

use mlx_rs::fast;
use mlx_rs::module::{ModuleParamMut, ModuleParamRef, ModuleParameters};
use mlx_rs::nn::{
    Linear, Module, ModuleParameters as ModuleParametersTrait, Quantizable, RmsNorm,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{self, arange, astype, expand, full, split, stack, transpose, triu, zeros, ArrayOps};
use mlx_rs::{Array, DType, Exception, StreamOrDevice};

use crate::config::ModelArgs;

/// Default chunk size for chunked GLA prefill.
const DEFAULT_CHUNK_SIZE: i32 = 64;

/// Recurrent state cache for lightning (GLA) attention layers.
#[derive(Debug, Clone)]
pub struct LightningCache {
    /// Recurrent state: [B, n_heads, head_dim, head_dim]
    pub state: Option<Array>,
    /// Current sequence offset (total tokens processed)
    pub offset: i32,
    /// Number of heads
    pub n_heads: i32,
    /// Head dimension
    pub head_dim: i32,
    /// Batch size (preallocated)
    pub batch: i32,
    /// ALiBi slopes for this layer's heads
    pub alibi_slopes: Option<Vec<f32>>,
    /// Cached decay value for the last chunk: pow(chunk_decay, offset)
    pub cum_decay: Option<Array>,
}

impl LightningCache {
    pub fn new(args: &ModelArgs) -> Self {
        let n_heads = args.lightning_num_heads();
        let head_dim = args.lightning_head_dim();
        Self {
            state: None,
            offset: 0,
            n_heads,
            head_dim,
            batch: 1,
            alibi_slopes: None,
            cum_decay: None,
        }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }
}

// ============================================================================
// ALiBi Slopes
// ============================================================================

/// Build ALiBi slopes (negated) for GLA decay.
/// These are NOT learnable — they are derived from the number of heads.
fn build_alibi_slopes(n_heads: i32) -> Vec<f32> {
    let n = n_heads as usize;
    let mut slopes = Vec::with_capacity(n);
    // Powers of 2: 2^(-8/n * i) for i = 1..n
    // Negated for exponential decay: decay = exp(slope * position_diff)
    for i in 0..n {
        let base = 2.0_f32.powf(-8.0 / n as f32);
        let slope = -base.powi(i as i32 + 1);
        slopes.push(slope);
    }
    slopes
}

/// Public wrapper for `build_alibi_slopes`, used by quantized loading.
pub fn build_alibi_slopes_pub(n_heads: i32) -> Vec<f32> {
    build_alibi_slopes(n_heads)
}

// ============================================================================
// Decay Tensor Builders (computed on CPU, cached for reuse)
// ============================================================================

/// Build intra-chunk causal decay mask: [1, H, C, C]
/// mask[h, i, j] = exp(slope_h * (i - j)) for j <= i, 0 otherwise
fn build_intra_decay_mask(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len() as i32;
    let mut data = vec![0.0f32; (c * c * h) as usize];
    for hi in 0..h {
        let slope = slopes[hi as usize];
        for i in 0..c {
            for j in 0..=i {
                let idx = (hi * c + i) * c + j;
                data[idx as usize] = (slope * (i - j) as f32).exp();
            }
        }
    }
    Array::from_slice(&data, &[1, h, c, c])
}

/// Build query decay for inter-chunk state lookup: [1, H, C, 1]
/// query_decay[h, t] = exp(slope_h * (t + 1)) for t = 0..C-1
fn build_query_decay(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len() as i32;
    let mut data = vec![0.0f32; (c * h) as usize];
    for hi in 0..h {
        let slope = slopes[hi as usize];
        for t in 0..c {
            data[(hi * c + t) as usize] = (slope * (t + 1) as f32).exp();
        }
    }
    Array::from_slice(&data, &[1, h, c, 1])
}

/// Build reverse decay for key weighting in state update: [1, H, C, 1]
/// reverse_decay[h, t] = exp(slope_h * (C - 1 - t)) for t = 0..C-1
fn build_reverse_decay(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len() as i32;
    let mut data = vec![0.0f32; (c * h) as usize];
    for hi in 0..h {
        let slope = slopes[hi as usize];
        for t in 0..c {
            data[(hi * c + t) as usize] = (slope * (c - 1 - t) as f32).exp();
        }
    }
    Array::from_slice(&data, &[1, h, c, 1])
}

/// Build chunk decay factor for state propagation: [1, H, 1, 1]
/// chunk_decay[h] = exp(slope_h * C)
fn build_chunk_decay(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len() as i32;
    let mut data = vec![0.0f32; h as usize];
    for hi in 0..h {
        let slope = slopes[hi as usize];
        data[hi as usize] = (slope * c as f32).exp();
    }
    Array::from_slice(&data, &[1, h, 1, 1])
}

/// Build all four decay tensors for a given chunk size.
fn build_decay_tensors(c: i32, slopes: &[f32]) -> (Array, Array, Array, Array) {
    (
        build_intra_decay_mask(c, slopes),
        build_query_decay(c, slopes),
        build_reverse_decay(c, slopes),
        build_chunk_decay(c, slopes),
    )
}

/// Zero-pad a 4D tensor along axis 2 (sequence dimension).
/// Input: [B, H, L, D] -> Output: [B, H, L + pad, D]
#[allow(non_snake_case)]
fn pad_seq_dim(x: &Array, pad: i32) -> Result<Array, Exception> {
    if pad <= 0 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let b = shape[0] as i32;
    let h = shape[1] as i32;
    let d = shape[3] as i32;

    // Create zero padding
    let pad_tensor: Array = zeros(&[b, h, pad, d], x.dtype(), x.stream_or_device())?;
    // Concatenate along seq dimension
    let result = ops::concatenate(&[x.view(), pad_tensor.view()], 2)?;
    Ok(result)
}

// ============================================================================
// Lightning Attention
// ============================================================================

/// Lightning attention using Gated Linear Attention (GLA) with recurrent state.
///
/// Uses chunked prefill for L > 1 (batched matmul within chunks of size C)
/// and single-step recurrence for decode (L = 1).
#[derive(Debug, ModuleParameters, Quantizable)]
#[module(root = mlx_rs)]
pub struct LightningAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // Gating projection
    z_proj: Linear,
    // QK normalization
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // Output normalization/gating
    o_norm: RmsNorm,
    pub args: ModelArgs,
}

impl LightningAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let hidden_size = args.hidden_size;
        let num_heads = args.lightning_num_heads();
        let num_kv_heads = args.lightning_num_kv_heads();
        let head_dim = args.lightning_head_dim();
        let rms_norm_eps = args.rms_norm_eps;

        Ok(Self {
            q_proj: Linear::new(hidden_size, num_heads * head_dim)?,
            k_proj: Linear::new(hidden_size, num_kv_heads * head_dim)?,
            v_proj: Linear::new(hidden_size, num_kv_heads * head_dim)?,
            o_proj: Linear::new(num_heads * head_dim, hidden_size)?,
            z_proj: Linear::new(hidden_size, num_heads * head_dim)?,
            q_norm: RmsNorm::new(head_dim, rms_norm_eps)?,
            k_norm: RmsNorm::new(head_dim, rms_norm_eps)?,
            o_norm: RmsNorm::new(hidden_size, rms_norm_eps)?,
            args: args.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0] as i32;
        let seq_len = x.shape()[1] as i32;
        let hidden_size = self.args.hidden_size;
        let num_heads = self.args.lightning_num_heads();
        let num_kv_heads = self.args.lightning_num_kv_heads();
        let head_dim = self.args.lightning_head_dim();
        let use_rope = self.args.lightning_use_rope;

        // Project to Q, K, V, Z
        let q_proj = self.q_proj.forward(x)?;
        let k_proj = self.k_proj.forward(x)?;
        let v_proj = self.v_proj.forward(x)?;
        let z = self.z_proj.forward(x)?;

        // Reshape to multi-head: [B, L, n_heads * head_dim] -> [B, L, n_heads, head_dim]
        // then transpose to [B, n_heads, L, head_dim]
        let mut q = q_proj.reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let mut k = k_proj.reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let mut v = v_proj.reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let z = z.reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;

        // QK normalization
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // RoPE (if enabled)
        if use_rope {
            let theta = self.args.rope_theta;
            // Apply RoPE in-place
            // For B > 1, L = 1 (decode), merge batch into head dim to avoid MLX RoPE bug
            if batch > 1 && seq_len == 1 {
                // Merge B into H: [B, H, 1, D] -> [1, B*H, 1, D]
                let q_merged = q.reshape(&[1, batch * num_heads, 1, head_dim])?;
                let k_merged = k.reshape(&[1, batch * num_kv_heads, 1, head_dim])?;
                q = fast::rope(&q_merged, &[], 1, theta, true, None, None)?;
                k = fast::rope(&k_merged, &[], 1, theta, true, None, None)?;
                // Reshape back
                q = q.reshape(&[batch, num_heads, 1, head_dim])?;
                k = k.reshape(&[batch, num_kv_heads, 1, head_dim])?;
            } else {
                q = fast::rope(&q, None, seq_len, theta, true, None, None)?;
                k = fast::rope(&k, None, seq_len, theta, true, None, None)?;
            }
        }

        // Compute attention
        let attn_out = if seq_len == 1 {
            // Decode step: single-token recurrent GLA
            self.decode_step(&q, &k, &v, cache)?
        } else {
            // Prefill: chunked GLA
            self.chunked_prefill(&q, &k, &v, cache)?
        };

        // Gate with Z (sigmoid)
        let z_gate = ops::sigmoid(&z)?;
        let gated = attn_out * z_gate;

        // Transpose back: [B, n_heads, L, head_dim] -> [B, L, n_heads * head_dim]
        let gated = gated.transpose(&[0, 2, 1, 3])?;
        let combined = gated.reshape(&[batch, seq_len, num_heads * head_dim])?;

        // Output projection
        let out = self.o_proj.forward(&combined)?;

        // Output norm and gate
        let out = self.o_norm.forward(&out)?;

        Ok(out)
    }

    /// Single-token decode step: recurrent GLA with state update.
    fn decode_step(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let batch = q.shape()[0] as i32;
        let num_heads = self.args.lightning_num_heads();
        let num_kv_heads = self.args.lightning_num_kv_heads();
        let head_dim = self.args.lightning_head_dim();
        let scale = self.args.lightning_scale_value();

        // Expand KV heads if GQA (should be same for lightning)
        let k = if num_kv_heads != num_heads {
            expand_kv_heads(k, num_heads)?
        } else {
            k.clone()
        };
        let v = if num_kv_heads != num_heads {
            expand_kv_heads(v, num_heads)?
        } else {
            v.clone()
        };

        // Initialize or load ALiBi slopes
        if cache.alibi_slopes.is_none() {
            cache.alibi_slopes = Some(build_alibi_slopes(num_heads));
        }
        let slopes = cache.alibi_slopes.as_ref().unwrap();

        // Initialize state if needed (first token)
        if cache.state.is_none() {
            let state = zeros(&[batch, num_heads, head_dim, head_dim], DType::Float32, None)?;
            cache.state = Some(state);
        }

        let state = cache.state.as_ref().unwrap();

        // Decay = exp(ALiBi slope) — same for all positions
        // For each head, decay_h = exp(slope_h)
        let decay_arr = Array::from_slice(
            &slopes.iter().map(|s| s.exp()).collect::<Vec<f32>>(),
            &[1, num_heads, 1, 1],
        )?;

        // GLA recurrent step:
        // 1. state = decay * state + k^T @ v   (outer product)
        // 2. out = q @ state * scale           (query state with scaling)

        // k^T @ v: [B, H, 1, D] -> k_t [B, H, D, 1], v [B, H, 1, D] -> kv [B, H, D, D]
        let k_t = k.transpose(&[0, 1, 3, 2])?;
        let kv_update = k_t.matmul(&v)?;

        let new_state = state * &decay_arr + kv_update;
        cache.state = Some(new_state.clone());

        // q @ state: [B, H, 1, D] @ [B, H, D, D] = [B, H, 1, D]
        let out = q.matmul(&new_state)?;
        let out = out * scale;

        cache.offset += 1;
        Ok(out)
    }

    /// Chunked GLA prefill for L > 1.
    fn chunked_prefill(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let batch = q.shape()[0] as i32;
        let seq_len = q.shape()[2] as i32;
        let num_heads = self.args.lightning_num_heads();
        let num_kv_heads = self.args.lightning_num_kv_heads();
        let head_dim = self.args.lightning_head_dim();
        let scale = self.args.lightning_scale_value();

        // Expand KV heads if GQA
        let k = if num_kv_heads != num_heads {
            expand_kv_heads(k, num_heads)?
        } else {
            k.clone()
        };
        let v = if num_kv_heads != num_heads {
            expand_kv_heads(v, num_heads)?
        } else {
            v.clone()
        };

        // Initialize or load ALiBi slopes
        if cache.alibi_slopes.is_none() {
            cache.alibi_slopes = Some(build_alibi_slopes(num_heads));
        }
        let slopes = cache.alibi_slopes.as_ref().unwrap();

        // Compute chunk size
        let chunk_size = DEFAULT_CHUNK_SIZE.min(seq_len);

        // Pre-allocate output
        let mut output_parts: Vec<Array> = Vec::new();

        // Process in chunks
        for start in (0..seq_len).step_by(chunk_size as usize) {
            let end = min(start + chunk_size, seq_len);
            let c = end - start;

            // Slice chunk
            let q_chunk = q.slice(&[0, 0, start, 0], &[batch, num_heads, c, head_dim])?;
            let k_chunk = k.slice(&[0, 0, start, 0], &[batch, num_heads, c, head_dim])?;
            let v_chunk = v.slice(&[0, 0, start, 0], &[batch, num_heads, c, head_dim])?;

            // Build decay tensors for this chunk size
            let (intra_mask, query_decay, reverse_decay, chunk_decay) =
                build_decay_tensors(c, slopes);

            // Intra-chunk attention: (Q @ K^T) * decay_mask @ V (fused in custom kernel)
            // Using fallback matmul path
            let k_chunk_t = k_chunk.transpose(&[0, 1, 3, 2])?;
            let scores = q_chunk.matmul(&k_chunk_t)?;
            let scores = scores * scale;
            let scores = scores * &intra_mask;
            let intra_out = scores.matmul(&v_chunk)?;

            // Inter-chunk: query against accumulated state
            let inter_out = match &cache.state {
                Some(state) => {
                    // query_decay[h, t] scales each query position
                    let q_decayed = q_chunk * &query_decay;
                    let out = q_decayed.matmul(state)?;
                    // Adjust: inter-chunk output needs scale
                    out * scale
                }
                None => zeros(&[batch, num_heads, c, head_dim], DType::Float32, None)?,
            };

            let chunk_out = intra_out + inter_out;
            output_parts.push(chunk_out);

            // Update state: new_state = chunk_decay * state + (K * reverse_decay)^T @ V
            let kw = k_chunk * &reverse_decay;
            let kw_t = kw.transpose(&[0, 1, 3, 2])?;
            let kv_new = kw_t.matmul(&v_chunk)?;

            let new_state = match &cache.state {
                Some(state) => state * &chunk_decay + kv_new,
                None => kv_new,
            };
            cache.state = Some(new_state);
        }

        cache.offset += seq_len;

        // Concatenate along sequence dimension
        let output = ops::concatenate(
            &output_parts.iter().map(|a| a.view()).collect::<Vec<_>>(),
            2,
        )?;

        Ok(output)
    }
}

/// Expand KV heads from num_kv_heads to num_heads by repeating.
#[allow(non_snake_case)]
fn expand_kv_heads(x: &Array, num_heads: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let n_kv = shape[1] as i32;
    if n_kv == num_heads {
        return Ok(x.clone());
    }
    let repeats = num_heads / n_kv;
    // Repeat along the head dimension
    let mut parts = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        parts.push(x.view());
    }
    ops::concatenate(&parts, 1)
}
