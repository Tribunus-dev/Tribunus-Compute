//! Cross-attention between vision features and text tokens.
//!
//! In multimodal Gemma models, vision features are injected into the text
//! model's hidden state via cross-attention layers placed at specific
//! positions.  This module provides the cross-attention computation and
//! a helper to splice vision features into the embedding sequence.

use crate::quantized::QuantizedLinearBinding;
use mlx_rs::Array;

/// A cross-attention layer between vision features and text tokens.
///
/// Vision features (keys/values) attend into the text hidden state (queries).
/// This is typically the same shape as a standard attention layer, but the
/// keys and values come from the vision encoder output rather than the text
/// hidden state.
#[derive(Debug, Clone)]
pub struct CrossAttentionLayer {
    /// Q projection for the text hidden state.
    pub q_proj: QuantizedLinearBinding,
    /// K projection (input from vision features).
    pub k_proj: QuantizedLinearBinding,
    /// V projection (input from vision features).
    pub v_proj: QuantizedLinearBinding,
    /// Output projection.
    pub o_proj: QuantizedLinearBinding,
    /// Number of attention heads for cross-attention.
    pub num_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
}

/// Inject vision features into the text hidden state.
///
/// This inserts the projected vision feature embeddings at the positions
/// of the `<image>` placeholder tokens in the text sequence, then runs
/// cross-attention so the text tokens can attend to the image patches.
///
/// # Arguments
/// * `hidden` — text hidden state `[1, seq_len, hidden_size]`
/// * `vision_features` — vision features `[num_patches, projection_dim]`
/// * `image_token_positions` — positions in the sequence where `<image>` tokens sit
/// * `cross_attn` — the cross-attention layer weights and config
/// * `rms_norm_eps` — epsilon for RMS normalization
///
/// # Returns
/// Updated hidden state with vision information injected.
pub fn inject_vision_features(
    hidden: &Array,
    vision_features: &Array,
    image_token_positions: &[usize],
    cross_attn: &CrossAttentionLayer,
    rms_norm_eps: f32,
) -> Result<Array, String> {
    if image_token_positions.is_empty() {
        return Ok(hidden.clone());
    }

    let seq_len = hidden.shape().get(1).copied().unwrap_or(1) as usize;

    // Check bounds for all positions.
    for &pos in image_token_positions {
        if pos >= seq_len {
            return Err(format!(
                "image token position {} out of range (seq_len={})",
                pos, seq_len
            ));
        }
    }

    // 1. Replace the hidden state at image token positions with vision features.
    //    For each image token position, insert the corresponding vision feature.
    let mut h_data: Vec<f32> = hidden
        .try_as_slice::<f32>()
        .map_err(|e| format!("hidden as_slice: {:?}", e))?
        .to_vec();
    let hidden_size = hidden.shape().get(2).copied().unwrap_or(1) as usize;

    let vf_data: Vec<f32> = vision_features
        .try_as_slice::<f32>()
        .map_err(|e| format!("vision_features as_slice: {:?}", e))?
        .to_vec();
    let vf_dim = vision_features.shape().get(1).copied().unwrap_or(1) as usize;

    if vf_dim != hidden_size {
        return Err(format!(
            "vision feature dim {} != hidden size {}",
            vf_dim, hidden_size
        ));
    }

    let num_vision_tokens = vision_features.shape().get(0).copied().unwrap_or(1) as usize;

    // Distribute vision tokens across image token positions.
    let tokens_per_pos = if image_token_positions.len() > 0 {
        num_vision_tokens / image_token_positions.len()
    } else {
        0
    };

    for (i, &pos) in image_token_positions.iter().enumerate() {
        let vf_start = i * tokens_per_pos;
        for j in 0..tokens_per_pos {
            let vf_idx = (vf_start + j).min(num_vision_tokens - 1);
            let start = pos * hidden_size + j * (hidden_size / tokens_per_pos.max(1));
            let end = (start + hidden_size / tokens_per_pos.max(1)).min((pos + 1) * hidden_size);
            for k in start..end {
                let vf_src = vf_idx * vf_dim + (k - pos * hidden_size);
                h_data[k] = if vf_src < vf_data.len() {
                    vf_data[vf_src]
                } else {
                    0.0
                };
            }
        }
    }

    let hidden_dims: Vec<i32> = vec![1, seq_len as i32, hidden_size as i32];
    let hidden_with_vision = Array::from_slice(&h_data, &hidden_dims);

    // 2. Apply cross-attention: use text hidden state as Q, vision_features as KV.
    let cross_out = cross_attention(
        &hidden_with_vision,
        vision_features,
        cross_attn,
        rms_norm_eps,
    )?;

    // 3. Residual connection.
    let updated = mlx_rs::ops::add(&hidden_with_vision, &cross_out)
        .map_err(|e| format!("cross-attn residual: {:?}", e))?;

    Ok(updated)
}

/// Run a single cross-attention step.
///
/// Query comes from the text hidden state (pre-normed).
/// Key/Value come from the vision features (pre-normed).
fn cross_attention(
    hidden: &Array,
    vision_features: &Array,
    cross_attn: &CrossAttentionLayer,
    rms_norm_eps: f32,
) -> Result<Array, String> {
    let num_heads = cross_attn.num_heads as i32;
    let head_dim = cross_attn.head_dim as i32;

    // Pre-norm for Q (from text features).
    let q_normed = rms_norm_simple(hidden, rms_norm_eps)?;
    let q = cross_attn
        .q_proj
        .forward(&q_normed)
        .map_err(|e| format!("cross q_proj: {:?}", e))?;

    // Pre-norm for K/V (from vision features).
    let kv_normed = rms_norm_simple(vision_features, rms_norm_eps)?;
    let k = cross_attn
        .k_proj
        .forward(&kv_normed)
        .map_err(|e| format!("cross k_proj: {:?}", e))?;
    let v = cross_attn
        .v_proj
        .forward(&kv_normed)
        .map_err(|e| format!("cross v_proj: {:?}", e))?;

    // Reshape to multi-head.
    let q_seq = q.shape().get(0).copied().unwrap_or(1) as i32;
    let kv_seq = k.shape().get(0).copied().unwrap_or(1) as i32;
    let q_heads = q.shape().get(1).copied().unwrap_or(1) as i32 / head_dim;

    let q_mh = q
        .reshape(&[q_seq, q_heads, head_dim])
        .map_err(|e| format!("q mh reshape: {:?}", e))?;
    let k_mh = k
        .reshape(&[kv_seq, q_heads, head_dim])
        .map_err(|e| format!("k mh reshape: {:?}", e))?;
    let v_mh = v
        .reshape(&[kv_seq, q_heads, head_dim])
        .map_err(|e| format!("v mh reshape: {:?}", e))?;

    // Transpose K for matmul: [kv_seq, q_heads, head_dim] -> [q_heads, head_dim, kv_seq]
    let k_t = mlx_rs::ops::transpose_axes(&k_mh, &[1, 2, 0])
        .map_err(|e| format!("k_t transpose: {:?}", e))?;

    // Q @ K^T: [q_seq, q_heads, head_dim] x [q_heads, head_dim, kv_seq]
    // We need to transpose Q too for batch matmul.
    let q_t = mlx_rs::ops::transpose_axes(&q_mh, &[1, 0, 2])
        .map_err(|e| format!("q_t transpose: {:?}", e))?;

    let scale = (head_dim as f32).sqrt().recip();
    let scores = mlx_rs::ops::matmul(&q_t, &k_t).map_err(|e| format!("cross scores: {:?}", e))?;
    let scores_scaled = mlx_rs::ops::multiply(&scores, &Array::from_slice(&[scale], &[1]))
        .map_err(|e| format!("cross scale: {:?}", e))?;

    let attn = mlx_rs::ops::softmax_axis(&scores_scaled, -1, false)
        .map_err(|e| format!("cross softmax: {:?}", e))?;

    // V transpose for matmul: [kv_seq, q_heads, head_dim] -> [q_heads, kv_seq, head_dim]
    let v_t = mlx_rs::ops::transpose_axes(&v_mh, &[1, 0, 2])
        .map_err(|e| format!("v_t transpose: {:?}", e))?;

    let out = mlx_rs::ops::matmul(&attn, &v_t).map_err(|e| format!("cross output: {:?}", e))?;

    // Collapse heads: [q_heads, q_seq, head_dim] -> [q_seq, q_heads * head_dim]
    let out_collapsed = mlx_rs::ops::transpose_axes(&out, &[1, 0, 2])
        .map_err(|e| format!("out transpose: {:?}", e))?
        .reshape(&[q_seq, num_heads * head_dim])
        .map_err(|e| format!("out reshape: {:?}", e))?;

    // Output projection.
    let result = cross_attn
        .o_proj
        .forward(&out_collapsed)
        .map_err(|e| format!("cross o_proj: {:?}", e))?;

    Ok(result)
}

/// Simple RMS normalization.
fn rms_norm_simple(x: &Array, eps: f32) -> Result<Array, String> {
    let x_sq = mlx_rs::ops::multiply(x, x).map_err(|e| format!("rms x_sq: {:?}", e))?;
    let mean =
        mlx_rs::ops::mean_axis(&x_sq, -1, false).map_err(|e| format!("rms mean: {:?}", e))?;
    let rms = mlx_rs::ops::rsqrt(
        &mlx_rs::ops::add(&mean, &Array::from_slice(&[eps], &[1]))
            .map_err(|e| format!("rms add eps: {:?}", e))?,
    )
    .map_err(|e| format!("rms rsqrt: {:?}", e))?;
    mlx_rs::ops::multiply(x, &rms).map_err(|e| format!("rms multiply: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_simple_identity() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let result = rms_norm_simple(&x, 1e-6).unwrap();
        // After eval, result should be a valid tensor.
        assert!(result.shape().len() > 0);
    }

    #[test]
    fn test_empty_positions() {
        let hidden = Array::from_slice(&[1.0f32; 6], &[1, 2, 3]);
        let vf = Array::from_slice(&[0.1f32; 6], &[2, 3]);
        let cross_attn = CrossAttentionLayer {
            q_proj: dummy_binding(),
            k_proj: dummy_binding(),
            v_proj: dummy_binding(),
            o_proj: dummy_binding(),
            num_heads: 1,
            head_dim: 3,
        };
        let result = inject_vision_features(&hidden, &vf, &[], &cross_attn, 1e-6).unwrap();
        assert_eq!(result.shape(), &[1, 2, 3]);
    }

    #[cfg(test)]
    fn dummy_binding() -> QuantizedLinearBinding {
        QuantizedLinearBinding::new(0xDEAD, 0xBEEF, 0xCAFE, 64, 64, 64, 8, false)
    }
}
