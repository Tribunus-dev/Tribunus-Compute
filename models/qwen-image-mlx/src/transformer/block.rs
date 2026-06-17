//! Transformer block for Qwen-Image
//!
//! Reference: diffusers QwenImageTransformerBlock
//! "Processes paired image and text hidden states with shared attention"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::attention::QwenTransformerAttention;
use super::feedforward::QwenFeedForward;
use super::norm::QwenLayerNorm;

/// Transformer block with dual-stream processing
/// Reference: diffusers QwenImageTransformerBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerBlock {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image stream
    #[param]
    pub norm1: QwenLayerNorm,
    #[param]
    pub ff: QwenFeedForward,

    // Text stream
    #[param]
    pub norm1_context: QwenLayerNorm,
    #[param]
    pub ff_context: QwenFeedForward,

    // Shared attention
    #[param]
    pub attn: QwenTransformerAttention,
}

impl QwenTransformerBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            norm1: QwenLayerNorm::new(dim)?,
            ff: QwenFeedForward::new(dim)?,
            norm1_context: QwenLayerNorm::new(dim)?,
            ff_context: QwenFeedForward::new(dim)?,
            attn: QwenTransformerAttention::new(dim, num_heads, head_dim)?,
        })
    }

    /// Forward pass with dual streams
    /// Returns (image_output, text_output)
    pub fn forward(
        &mut self,
        image_hidden: &Array,
        text_hidden: &Array,
        temb: &Array,           // Time embedding [batch, dim]
        image_rotary: &(Array, Array),
        text_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        // Image stream: modulate, attention, FFN
        let (img_modulated, img_gate, img_mod2) = self.norm1.forward(image_hidden, temb)?;
        let (txt_modulated, txt_gate, txt_mod2) = self.norm1_context.forward(text_hidden, temb)?;

        // Joint attention
        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            image_rotary,
            text_rotary,
            mask,
        )?;

        // Gate + residual for attention outputs
        let image_hidden = ops::add(image_hidden, &ops::multiply(&img_gate, &img_attn)?)?;
        let text_hidden = ops::add(text_hidden, &ops::multiply(&txt_gate, &txt_attn)?)?;

        // Apply modulation for image FFN
        let image_hidden = self.apply_ffn(&image_hidden, &img_mod2, &mut *self.ff)?;

        // Apply modulation for text FFN
        let text_hidden = self.apply_ffn(&text_hidden, &txt_mod2, &mut *self.ff_context)?;

        Ok((image_hidden, text_hidden))
    }

    fn apply_ffn(
        &self,
        hidden: &Array,
        mod2: &Array,
        ff: &mut QwenFeedForward,
    ) -> Result<Array, Exception> {
        let shift2 = mod2.index(&[.., ..self.dim])?;
        let scale2 = mod2.index(&[.., self.dim..2*self.dim])?;
        let gate2 = mod2.index(&[.., 2*self.dim..])?;

        // Modulate: (1 + scale) * norm(x) + shift
        let one = Array::from_f32(1.0);
        let normed = self.norm1.norm.forward(hidden)?;
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale2)?)?,
            &shift2,
        )?;

        let ffn_out = ff.forward(&modulated)?;

        // Gate + residual
        ops::add(hidden, &ops::multiply(&gate2, &ffn_out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = QwenTransformerBlock::new(3072, 24, 128).unwrap();
        assert_eq!(block.dim, 3072);
    }

    #[test]
    fn test_block_forward() {
        let mut block = QwenTransformerBlock::new(64, 4, 16).unwrap();
        let img = Array::zeros::<f32>(&[1, 16, 64]).unwrap();
        let txt = Array::zeros::<f32>(&[1, 8, 64]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 64]).unwrap();
        let img_rope = (Array::zeros::<f32>(&[1, 16, 1, 16]).unwrap(), Array::zeros::<f32>(&[1, 16, 1, 16]).unwrap());
        let txt_rope = (Array::zeros::<f32>(&[1, 8, 1, 16]).unwrap(), Array::zeros::<f32>(&[1, 8, 1, 16]).unwrap());

        let (img_out, txt_out) = block.forward(&img, &txt, &temb, &img_rope, &txt_rope, None).unwrap();
        assert_eq!(img_out.shape(), &[1, 16, 64]);
        assert_eq!(txt_out.shape(), &[1, 8, 64]);
    }
}
