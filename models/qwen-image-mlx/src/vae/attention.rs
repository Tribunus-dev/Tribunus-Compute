//! 3D Attention Block for VAE
//!
//! Reference: diffusers QwenImageAttentionBlock
//! "causal self-attention with a single head"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv2d, Conv2dBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::QwenImageRMSNorm;

/// 3D Attention Block for VAE
/// Applies self-attention on flattened H*W per time frame
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageAttentionBlock3D {
    pub channels: i32,
    #[param]
    pub norm: QwenImageRMSNorm,
    #[param]
    pub to_qkv: Conv2d, // Projects to Q, K, V combined
    #[param]
    pub proj: Conv2d, // Output projection
}

impl QwenImageAttentionBlock3D {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            channels,
            norm: QwenImageRMSNorm::new(channels, 1e-6, true)?,
            to_qkv: Conv2dBuilder::new(channels, channels * 3, 1).build()?, // 1x1 conv
            proj: Conv2dBuilder::new(channels, channels, 1).build()?,
        })
    }
}

impl Module<&Array> for QwenImageAttentionBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.to_qkv.training_mode(mode);
        self.proj.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [N, C, T, H, W] for 3D or [N, C, H, W] for 2D
        let is_3d = x.ndim() == 5;

        if is_3d {
            let n = x.dim(0);
            let c = x.dim(1);
            let t = x.dim(2);
            let h = x.dim(3);
            let w = x.dim(4);

            // Process each time frame independently
            let mut outputs = Vec::with_capacity(t as usize);
            for frame in 0..t {
                let frame_x = x.index(&[.., .., frame..=frame, .., ..])?; // [N, C, 1, H, W]
                let frame_x = frame_x.reshape(&[n, c, h, w])?; // [N, C, H, W]
                let frame_out = self.forward_2d(&frame_x)?; // [N, C, H, W]
                let frame_out = frame_out.reshape(&[n, c, 1, h, w])?; // [N, C, 1, H, W]
                outputs.push(frame_out);
            }

            ops::concatenate_axis(&outputs.iter().collect::<Vec<_>>(), 2)
        } else {
            self.forward_2d(x)
        }
    }
}

impl QwenImageAttentionBlock3D {
    fn forward_2d(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [N, C, H, W]
        let residual = x.clone();
        let n = x.dim(0);
        let c = x.dim(1);
        let h = x.dim(2);
        let w = x.dim(3);

        // Normalize
        let x = self.norm.forward(x)?;

        // Project to Q, K, V (1x1 conv)
        // to_qkv: [N, 3*C, H, W]
        let x = x.transpose_axes(&[0, 2, 3, 1])?; // [N, H, W, C] for Conv2d
        let qkv = self.to_qkv.forward(&x)?;
        let qkv = qkv.transpose_axes(&[0, 3, 1, 2])?; // [N, 3*C, H, W]

        let q = qkv.index(&[.., ..c, .., ..])?;
        let k = qkv.index(&[.., c..2*c, .., ..])?;
        let v = qkv.index(&[.., 2*c.., .., ..])?;

        // Flatten spatial dimensions for attention
        let q = q.reshape(&[n, c, h * w])?; // [N, C, H*W]
        let k = k.reshape(&[n, c, h * w])?;
        let v = v.reshape(&[n, c, h * w])?;

        // Transpose for attention: q, k: [N, H*W, C], v: [N, H*W, C]
        let q = q.transpose(&[0, 2, 1])?; // [N, H*W, C]
        let k = k.transpose(&[0, 2, 1])?;
        let v = v.transpose(&[0, 2, 1])?;

        // Scaled dot-product attention
        let scale = 1.0 / (c as f32).sqrt();
        let attn = fast_sdp(&q, &k, &v, scale)?;

        // Reshape back to spatial
        let attn = attn.transpose(&[0, 2, 1])?; // [N, C, H*W]
        let attn = attn.reshape(&[n, c, h, w])?; // [N, C, H, W]

        // Output projection
        let attn = attn.transpose_axes(&[0, 2, 3, 1])?; // [N, H, W, C] for Conv2d
        let out = self.proj.forward(&attn)?;
        let out = out.transpose_axes(&[0, 3, 1, 2])?; // [N, C, H, W]

        // Residual connection
        ops::add(&residual, &out)
    }
}

// Simple SDPA for VAE attention (uses matmul directly since head_dim = channels)
fn fast_sdp(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array, Exception> {
    // q, k: [N, H*W, C], v: [N, H*W, C]
    let scores = ops::matmul(q, &k.transpose(&[0, 2, 1])?)?;
    let scores = ops::multiply(&scores, &Array::from_f32(scale))?;
    let weights = ops::softmax(&scores, -1)?;
    ops::matmul(&weights, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_block() {
        let mut attn = QwenImageAttentionBlock3D::new(64).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 8, 8]).unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64, 8, 8]);
    }

    #[test]
    fn test_attention_block_3d() {
        let mut attn = QwenImageAttentionBlock3D::new(64).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 8, 8]).unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64, 2, 8, 8]);
    }
}
