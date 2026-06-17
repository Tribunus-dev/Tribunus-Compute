//! Mimi Speech Encoder: encodes reference audio to 16-codebook discrete codes.
//!
//! Used for ICL voice cloning in Base model mode. Encodes 24kHz audio
//! into 12Hz codec frames with 16 codebooks (1 semantic + 15 acoustic).
//!
//! Architecture: SEANet Conv Encoder → Transformer → Downsample → RVQ
//!
//! Weight keys from `speech_tokenizer/model.safetensors`:
//!   `encoder.encoder.layers.*`          — SEANet convolutional encoder
//!   `encoder.encoder_transformer.layers.*` — 8-layer transformer
//!   `encoder.downsample.conv.*`         — 25Hz → 12.5Hz
//!   `encoder.quantizer.*`               — Semantic + Acoustic RVQ

use std::collections::HashMap;

use mlx_rs::{array, Array};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

use crate::error::{Error, Result};

// ============================================================================
// Conv helpers (reusing patterns from speech_tokenizer decoder)
// ============================================================================

/// Transpose Conv1d weight from PyTorch [out, in, kernel] to MLX [out, kernel, in].
fn transpose_conv_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 1])?)
}

/// A causal Conv1d with configurable padding mode (constant/replicate).
/// Matches HF MimiConv1d: causal left padding + dynamic extra right padding
/// to ensure output length = ceil(input_length / stride).
struct CausalConv1d {
    conv: nn::Conv1d,
    left_pad: i32,
    kernel_size: i32,
    stride: i32,
    replicate: bool, // true = replicate padding (edge values), false = constant/zero padding
}

impl CausalConv1d {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C] in MLX (channel-last)
        let length = x.dim(1) as i32;

        // Compute extra right padding (matches Python _get_extra_padding_for_conv1d)
        let n_frames_num = length - self.kernel_size + self.left_pad;
        let n_frames_ceil = (n_frames_num + self.stride - 1) / self.stride + 1;
        let ideal_length = (n_frames_ceil - 1) * self.stride + self.kernel_size - self.left_pad;
        let extra_pad = (ideal_length - length).max(0);

        let x = if self.left_pad > 0 || extra_pad > 0 {
            let b = x.dim(0) as i32;
            let c = x.dim(2) as i32;
            let mut parts: Vec<Array> = Vec::new();
            if self.left_pad > 0 {
                if self.replicate {
                    let first = x.index((.., ..1i32, ..));
                    let left = ops::broadcast_to(&first, &[b, self.left_pad, c])?;
                    parts.push(left);
                } else {
                    parts.push(ops::zeros::<f32>(&[b, self.left_pad, c])?);
                }
            }
            parts.push(x.clone());
            if extra_pad > 0 {
                if self.replicate {
                    let last = x.index((.., -1i32.., ..));
                    let right = ops::broadcast_to(&last, &[b, extra_pad, c])?;
                    parts.push(right);
                } else {
                    parts.push(ops::zeros::<f32>(&[b, extra_pad, c])?);
                }
            }
            let refs: Vec<&Array> = parts.iter().collect();
            ops::concatenate_axis(&refs[..], 1)?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

// ============================================================================
// Residual Block (for SEANet encoder)
// ============================================================================

/// Residual block in the SEANet encoder.
/// Two convolutions with optional dimension change + skip connection.
struct EncoderResBlock {
    conv1: CausalConv1d, // bottleneck: C → C/2, k=3
    conv2: CausalConv1d, // expand: C/2 → C, k=1
    shortcut: Option<CausalConv1d>, // if input/output dims differ
}

impl EncoderResBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // ELU activation before each conv (pre-activation residual block)
        let h = elu(x)?;
        let h = self.conv1.forward(&h)?;
        let h = elu(&h)?;
        let h = self.conv2.forward(&h)?;

        let skip = if let Some(ref mut sc) = self.shortcut {
            sc.forward(x)?
        } else {
            x.clone()
        };

        Ok(h.add(&skip)?)
    }
}

/// ELU activation: x if x > 0, else alpha * (exp(x) - 1) where alpha=1.0
fn elu(x: &Array) -> Result<Array> {
    let zero = array!(0.0f32);
    let one = array!(1.0f32);
    let positive = x.gt(&zero)?;
    let exp_x = ops::exp(x)?;
    let neg_part = exp_x.subtract(&one)?;
    Ok(ops::r#where(&positive, x, &neg_part)?)
}

// ============================================================================
// Affine LayerNorm (encoder transformer uses LN with bias)
// ============================================================================

struct AffineLayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl AffineLayerNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        // x: [B, T, D]
        let mean = ops::mean_axis(x, -1, true)?;
        let diff = x.subtract(&mean)?;
        let var = ops::mean_axis(&diff.multiply(&diff)?, -1, true)?;
        let inv_std = ops::rsqrt(&var.add(&array!(self.eps))?)?;
        let normed = diff.multiply(&inv_std)?;
        let scaled = normed.multiply(&self.weight)?;
        Ok(scaled.add(&self.bias)?)
    }
}

// ============================================================================
// Encoder Transformer Layer
// ============================================================================

struct EncoderTransformerLayer {
    input_layernorm: AffineLayerNorm,
    q_proj: Array, // [D, D]
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    self_attn_layer_scale: Array, // [D]
    post_attention_layernorm: AffineLayerNorm,
    fc1: Array, // [4D, D]
    fc2: Array, // [D, 4D]
    mlp_layer_scale: Array, // [D]
    num_heads: i32,
    head_dim: i32,
    /// Precomputed inverse frequency vector for RoPE [half_dim]
    inv_freqs: Vec<f32>,
}

impl EncoderTransformerLayer {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // Self-attention with pre-norm
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attention(&normed)?;
        let attn_scaled = attn_out.multiply(&self.self_attn_layer_scale)?;
        let x = x.add(&attn_scaled)?;

        // MLP with pre-norm
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp(&normed)?;
        let mlp_scaled = mlp_out.multiply(&self.mlp_layer_scale)?;
        Ok(x.add(&mlp_scaled)?)
    }

    fn self_attention(&self, x: &Array) -> Result<Array> {
        // x: [B, T, D]
        let b = x.dim(0) as i32;
        let t = x.dim(1) as i32;
        let t_usize = t as usize;

        // Q, K, V projections
        let mut q = ops::matmul(x, &self.q_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut k = ops::matmul(x, &self.k_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = ops::matmul(x, &self.v_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

        // RoPE using precomputed inverse frequencies
        let half = self.inv_freqs.len();
        let mut cos_data = vec![0.0f32; t_usize * half];
        let mut sin_data = vec![0.0f32; t_usize * half];
        for pos in 0..t_usize {
            for i in 0..half {
                let angle = pos as f32 * self.inv_freqs[i];
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }
        let cos_arr = Array::from_slice(&cos_data, &[1, 1, t, half as i32]);
        let sin_arr = Array::from_slice(&sin_data, &[1, 1, t, half as i32]);

        // Apply RoPE: split first/second half
        let q1 = q.index((.., .., .., ..half as i32));
        let q2 = q.index((.., .., .., half as i32..));
        q = mlx_rs::ops::concatenate_axis(
            &[&q1.multiply(&cos_arr)?.subtract(&q2.multiply(&sin_arr)?)?,
              &q2.multiply(&cos_arr)?.add(&q1.multiply(&sin_arr)?)?],
            -1,
        )?;
        let k1 = k.index((.., .., .., ..half as i32));
        let k2 = k.index((.., .., .., half as i32..));
        k = mlx_rs::ops::concatenate_axis(
            &[&k1.multiply(&cos_arr)?.subtract(&k2.multiply(&sin_arr)?)?,
              &k2.multiply(&cos_arr)?.add(&k1.multiply(&sin_arr)?)?],
            -1,
        )?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(array!(1.0 / scale))?;

        // Causal mask with sliding window (window=250)
        let window = 250usize;
        let mut mask_data = vec![0.0f32; t_usize * t_usize];
        for row in 0..t_usize {
            for col in 0..t_usize {
                if col > row || row - col >= window {
                    mask_data[row * t_usize + col] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Array::from_slice(&mask_data, &[1, 1, t, t]);
        let scores = scores.add(&mask)?;

        let attn_weights = ops::softmax_axis(&scores, -1, None::<bool>)?;
        let attn_out = ops::matmul(&attn_weights, &v)?;

        // Reshape and output projection
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;
        Ok(ops::matmul(&attn_out, &self.o_proj.t())?)
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        // Standard MLP: fc1 → GELU → fc2
        let h = ops::matmul(x, &self.fc1.t())?;
        let h = nn::gelu(&h)?;
        Ok(ops::matmul(&h, &self.fc2.t())?)
    }
}

// ============================================================================
// RVQ Codebook
// ============================================================================

struct RvqCodebook {
    embedding: Array, // [codebook_size, codebook_dim] (normalized)
}

impl RvqCodebook {
    /// Find nearest codebook entry for each vector.
    /// Input: [B, T, D] (already projected to codebook_dim)
    /// Output: [B, T] codes (u32)
    fn quantize(&self, x: &Array) -> Result<(Array, Array)> {
        // L2 distance: ||x - e||^2 = ||x||^2 - 2*x*e^T + ||e||^2
        let x_sq = ops::sum_axis(&x.multiply(x)?, -1, true)?; // [B, T, 1]
        let e_sq = ops::sum_axis(&self.embedding.multiply(&self.embedding)?, -1, true)?; // [1, codebook_size]
        let x_e = ops::matmul(x, &self.embedding.t())?; // [B, T, codebook_size]

        let dists = x_sq.subtract(&x_e.multiply(array!(2.0f32))?)?.add(&e_sq.t())?;

        // Argmin
        let codes = ops::indexing::argmin_axis(&dists, -1, None)?; // [B, T]

        // Lookup embeddings
        let flat_codes = codes.reshape(&[-1])?;
        let quantized = self.embedding.index(flat_codes);
        let quantized = quantized.reshape(&[x.dim(0) as i32, x.dim(1) as i32, -1])?;

        Ok((codes, quantized))
    }
}

/// Normalize codebook: embedding = embed_sum / cluster_usage.clamp(min=epsilon)
/// Matches Python MimiEuclideanCodebook: epsilon=1e-5
fn normalize_codebook(embed_sum: &Array, cluster_usage: &Array) -> Result<Array> {
    let usage = cluster_usage.reshape(&[-1, 1])?;
    let usage = ops::maximum(&usage, &array!(1e-5f32))?;
    Ok(embed_sum.divide(&usage)?)
}


// ... (remaining RVQ, downsample, and full encoder follow)
