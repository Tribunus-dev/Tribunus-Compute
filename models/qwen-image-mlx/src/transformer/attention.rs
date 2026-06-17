//! Joint text-image attention
//!
//! Reference: diffusers QwenDoubleStreamAttnProcessor2_0

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::fast;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder, RmsNorm, RmsNormBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::rope::apply_rope;

/// Joint text-image attention
/// Reference: diffusers QwenDoubleStreamAttnProcessor2_0
/// "Computing separate QKV projections for each stream"
/// "Concatenating streams for unified attention computation"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerAttention {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image projections
    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,

    // Text projections
    #[param]
    pub add_q_proj: Linear,
    #[param]
    pub add_k_proj: Linear,
    #[param]
    pub add_v_proj: Linear,

    // QK normalization
    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
    #[param]
    pub norm_added_q: RmsNorm,
    #[param]
    pub norm_added_k: RmsNorm,

    // Output projections
    #[param]
    pub attn_to_out: Linear,
    #[param]
    pub to_add_out: Linear,
}

impl QwenTransformerAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        let total_dim = num_heads * head_dim;
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            to_q: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            to_k: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            to_v: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_q_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_k_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_v_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            norm_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            attn_to_out: LinearBuilder::new(total_dim, dim).bias(true).build()?,
            to_add_out: LinearBuilder::new(total_dim, dim).bias(true).build()?,
        })
    }

    pub fn forward(
        &mut self,
        img_hidden: &Array,
        txt_hidden: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img_hidden.dim(0);
        let img_seq = img_hidden.dim(1);
        let txt_seq = txt_hidden.dim(1);

        // Project Q, K, V for image stream
        let img_q = self.to_q.forward(img_hidden)?;
        let img_k = self.to_k.forward(img_hidden)?;
        let img_v = self.to_v.forward(img_hidden)?;

        // Project Q, K, V for text stream
        let txt_q = self.add_q_proj.forward(txt_hidden)?;
        let txt_k = self.add_k_proj.forward(txt_hidden)?;
        let txt_v = self.add_v_proj.forward(txt_hidden)?;

        // Reshape for multi-head attention
        let img_q = img_q.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_k = img_k.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_v = img_v.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;

        let txt_q = txt_q.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_k = txt_k.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_v = txt_v.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;

        // Apply RoPE
        let img_q = apply_rope(&img_q, img_rotary)?;
        let img_k = apply_rope(&img_k, img_rotary)?;
        let txt_q = apply_rope(&txt_q, txt_rotary)?;
        let txt_k = apply_rope(&txt_k, txt_rotary)?;

        // QK Norm
        let img_q = self.norm_q.forward(&img_q)?;
        let img_k = self.norm_k.forward(&img_k)?;
        let txt_q = self.norm_added_q.forward(&txt_q)?;
        let txt_k = self.norm_added_k.forward(&txt_k)?;

        // Transpose to [batch, heads, seq, head_dim] for attention
        let img_q = img_q.transpose(&[0, 2, 1, 3])?;
        let img_k = img_k.transpose(&[0, 2, 1, 3])?;
        let img_v = img_v.transpose(&[0, 2, 1, 3])?;
        let txt_q = txt_q.transpose(&[0, 2, 1, 3])?;
        let txt_k = txt_k.transpose(&[0, 2, 1, 3])?;
        let txt_v = txt_v.transpose(&[0, 2, 1, 3])?;

        // Concatenate image and text for joint attention
        let q = ops::concatenate_axis(&[&img_q, &txt_q], 2)?;
        let k = ops::concatenate_axis(&[&img_k, &txt_k], 2)?;
        let v = ops::concatenate_axis(&[&img_v, &txt_v], 2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = fast::scaled_dot_product_attention(
            &q, &k, &v,
            Some(scale),
            mask,
        )?;

        // Split back into image and text outputs
        let img_out = attn_out.index(&[.., .., ..img_seq, ..])?;
        let txt_out = attn_out.index(&[.., .., img_seq.., ..])?;

        // Transpose back and reshape for output projection
        let img_out = img_out.transpose(&[0, 2, 1, 3])?;
        let img_out = img_out.reshape(&[batch, img_seq, self.num_heads * self.head_dim])?;
        let img_out = self.attn_to_out.forward(&img_out)?;

        let txt_out = txt_out.transpose(&[0, 2, 1, 3])?;
        let txt_out = txt_out.reshape(&[batch, txt_seq, self.num_heads * self.head_dim])?;
        let txt_out = self.to_add_out.forward(&txt_out)?;

        Ok((img_out, txt_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attn = QwenTransformerAttention::new(3072, 24, 128).unwrap();
        assert_eq!(attn.num_heads, 24);
        assert_eq!(attn.head_dim, 128);
    }
}
