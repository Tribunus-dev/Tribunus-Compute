use mlx_rs::fast;
use mlx_rs::module::{ModuleParamMut, ModuleParamRef, ModuleParameters};
use mlx_rs::nn::{
    Linear, Module, ModuleParameters as ModuleParametersTrait, Quantizable, RmsNorm,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    self, arange, astype, expand, full, split, stack, transpose, triu, zeros, ArrayOps,
};
use mlx_rs::{
    Array, DType, Error, Exception, ScaledDotProductAttentionMask, ScaledDotProductAttentionOptions,
    StreamOrDevice,
};

use crate::config::{ModelArgs, SparseConfig};

// ============================================================================
// SparseKVCache — stores full KV history for InfLLMv2 sparse attention
// ============================================================================

/// KV cache for sparse (InfLLMv2) attention layers.
///
/// Stores the full key/value history and provides methods for:
/// - Dense SDPA when total_len <= dense_len
/// - InfLLMv2 sparse attention when total_len > dense_len
#[derive(Debug, Clone)]
pub struct SparseKVCache {
    /// Cached keys [B, n_kv_heads, total_len, head_dim]
    pub keys: Option<Array>,
    /// Cached values [B, n_kv_heads, total_len, head_dim]
    pub values: Option<Array>,
    /// Current total length
    pub offset: i32,
}

impl SparseKVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }
}

// ============================================================================
// InfLLMv2 Helper Functions
// ============================================================================

/// Compress keys by mean-pooling with non-overlapping windows.
/// Input: [B, H, L, D] -> Output: [B, H, num_blocks, D]
/// where num_blocks = L / kernel_size (truncated).
#[allow(non_snake_case)]
fn compress_keys(keys: &Array, kernel_size: i32) -> Result<Array, Exception> {
    let shape = keys.shape();
    let b = shape[0] as i32;
    let h = shape[1] as i32;
    let l = shape[2] as i32;
    let d = shape[3] as i32;

    let num_blocks = l / kernel_size;
    if num_blocks <= 0 {
        return Err(Exception::from("Sequence too short for compression"));
    }

    let truncated = l - (l % kernel_size);
    let valid = keys.slice(&[0, 0, 0, 0], &[b, h, truncated, d])?;

    // Reshape to [B, H, num_blocks, kernel_size, D] and mean over kernel_size dim
    let reshaped = valid.reshape(&[b, h, num_blocks, kernel_size, d])?;
    let compressed = reshaped.mean(&[3], true, None::<f32>)?;
    // Remove the meaned dimension: [B, H, num_blocks, 1, D] -> [B, H, num_blocks, D]
    let compressed = compressed.squeeze(&[3])?;

    Ok(compressed)
}

/// InfLLMv2 sparse attention for long contexts.
///
/// Two-stage algorithm:
/// 1. CompressK: Mean-pool keys from the "middle" region into block representatives
/// 2. Score queries against compressed keys, select top-K blocks
/// 3. Gather K,V from: init blocks + selected blocks + sliding window
/// 4. Run SDPA on gathered subset
#[allow(non_snake_case)]
fn infllmv2_attention(
    queries: &Array,
    cache: &SparseKVCache,
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
) -> Result<Array, Exception> {
    // Placeholder: fall back to dense SDPA for now.
    // Full InfLLMv2 would require custom Metal kernels for
    // block selection and sparse gathering.
    let keys = cache.keys.as_ref().ok_or_else(|| Exception::from("No cached keys"))?;
    let values = cache.values.as_ref().ok_or_else(|| Exception::from("No cached values"))?;

    let total_len = keys.shape()[2] as i32;

    // Use standard SDPA as fallback (dense attention over entire cache)
    let q = queries;
    let k = keys;
    let v = values;

    let mask = ScaledDotProductAttentionMask::Causal;
    let opts = ScaledDotProductAttentionOptions::new()
        .scale(scale)
        .mask(mask);

    let out = fast::scaled_dot_product_attention(q, k, v, None::<&Array>, &opts)?;
    Ok(out)
}

// ============================================================================
// SparseAttention
// ============================================================================

/// Sparse attention layer using standard SDPA for short contexts,
/// InfLLMv2 two-stage sparse attention for long contexts.
///
/// Sparse layers do NOT have q_norm/k_norm (those are lightning-only).
/// Sparse layers DO have o_gate for output gating.
#[derive(Debug, ModuleParameters, Quantizable)]
#[module(root = mlx_rs)]
pub struct SparseAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    o_gate: Linear,
    pub args: ModelArgs,
}

impl SparseAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let hidden_size = args.hidden_size;
        let num_heads = args.num_attention_heads;
        let num_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;

        Ok(Self {
            q_proj: Linear::new(hidden_size, num_heads * head_dim)?,
            k_proj: Linear::new(hidden_size, num_kv_heads * head_dim)?,
            v_proj: Linear::new(hidden_size, num_kv_heads * head_dim)?,
            o_proj: Linear::new(num_heads * head_dim, hidden_size)?,
            o_gate: Linear::new(num_heads * head_dim, hidden_size)?,
            args: args.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut SparseKVCache,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0] as i32;
        let seq_len = x.shape()[1] as i32;
        let hidden_size = self.args.hidden_size;
        let num_heads = self.args.num_attention_heads;
        let num_kv_heads = self.args.num_key_value_heads;
        let head_dim = self.args.head_dim;
        let use_rope = self.args.attn_use_rope;
        let scale = (head_dim as f32).sqrt().recip();

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head
        let mut q = q.reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let mut k = k.reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;

        // RoPE (if enabled for sparse layers)
        if use_rope {
            let theta = self.args.rope_theta;
            q = fast::rope(&q, None, seq_len, theta, true, None, None)?;
            k = fast::rope(&k, None, seq_len, theta, true, None, None)?;
        }

        // Update KV cache
        let total_len = cache.offset + seq_len;
        let new_keys = match &cache.keys {
            Some(existing) => ops::concatenate(&[existing.view(), k.view()], 2)?,
            None => k.clone(),
        };
        let new_values = match &cache.values {
            Some(existing) => ops::concatenate(&[existing.view(), v.view()], 2)?,
            None => v.clone(),
        };

        // Decide: dense SDPA vs sparse attention
        let attn_out = if cache.offset > 0 && total_len > 8192 {
            // Long context: use InfLLMv2 sparse attention
            // Temporilly cache before calling sparse
            cache.keys = Some(new_keys);
            cache.values = Some(new_values);
            cache.offset = total_len;

            infllmv2_attention(&q, cache, num_heads, num_kv_heads, scale)?
        } else {
            // Short context: standard SDPA with causal mask
            let k = &new_keys;
            let v = &new_values;

            // Use causal mask (always for autoregressive)
            let mask = ScaledDotProductAttentionMask::Causal;
            let opts = ScaledDotProductAttentionOptions::new()
                .scale(scale)
                .mask(mask);

            let out = fast::scaled_dot_product_attention(&q, k, v, None::<&Array>, &opts)?;

            // Update cache
            cache.keys = Some(new_keys);
            cache.values = Some(new_values);
            cache.offset = total_len;

            out
        };

        // Transpose back: [B, n_heads, L, head_dim] -> [B, L, n_heads * head_dim]
        let attn_out = attn_out.transpose(&[0, 2, 1, 3])?;
        let combined = attn_out.reshape(&[batch, seq_len, num_heads * head_dim])?;

        // Output gating: o_gate(x) = sigmoid(o_proj(x))   (simplified)
        let gate = ops::sigmoid(&self.o_gate.forward(x)?)?;
        let out = self.o_proj.forward(&combined)?;
        let out = out * gate;

        Ok(out)
    }
}
