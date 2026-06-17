//! Adaptive Layer Normalization for DiT blocks
//!
//! Reference: diffusers "Apply modulation to input tensor"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, LayerNorm, LayerNormBuilder, Linear, LinearBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

/// Adaptive Layer Normalization for DiT blocks
/// Projects conditioning to shift/scale/gate parameters
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenLayerNorm {
    pub dim: i32,
    #[param]
    pub mod_linear: Linear, // Projects to 6 * dim
    #[param]
    pub norm: LayerNorm, // Non-affine LayerNorm
}

impl QwenLayerNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            mod_linear: LinearBuilder::new(dim, 6 * dim).bias(true).build()?,
            norm: LayerNormBuilder::new(dim).elementwise_affine(false).eps(1e-6).build()?,
        })
    }

    /// Returns (modulated_hidden, gate, mod2_params)
    /// - modulated_hidden: normalized and modulated by shift1/scale1
    /// - gate: gate1 for attention gating
    /// - mod2_params: [shift2, scale2, gate2] for FFN modulation
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        text_embeddings: &Array,
    ) -> Result<(Array, Array, Array), Exception> {
        // Project conditioning to modulation params
        let cond = self.mod_linear.forward(text_embeddings)?; // [batch, 6*dim]
        let cond = nn::silu(&cond)?;

        // Split into 6 sets of parameters
        let shift1 = cond.index(&[.., ..self.dim])?;
        let scale1 = cond.index(&[.., self.dim..2*self.dim])?;
        let gate1 = cond.index(&[.., 2*self.dim..3*self.dim])?;
        let shift2 = cond.index(&[.., 3*self.dim..4*self.dim])?;
        let scale2 = cond.index(&[.., 4*self.dim..5*self.dim])?;
        let gate2 = cond.index(&[.., 5*self.dim..])?;

        // Modulate: (1 + scale) * LayerNorm(x) + shift
        let normed = self.norm.forward(hidden_states)?;
        let one = Array::from_f32(1.0);
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale1)?)?,
            &shift1,
        )?;

        // Combine shift2, scale2, gate2 for later use
        let mod2 = ops::concatenate_axis(&[&shift2, &scale2, &gate2], -1)?;

        Ok((modulated, gate1, mod2))
    }
}

/// Continuous AdaLN for final output normalization
/// Reference: diffusers QwenImageTransformer2DModel "Apply final normalization and output projection"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAdaLayerNormContinuous {
    pub dim: i32,
    #[param]
    pub linear: Linear, // Projects to 2*dim (shift + scale)
}

impl QwenAdaLayerNormContinuous {
    pub fn new(dim: i32, conditioning_embedding_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            linear: LinearBuilder::new(conditioning_embedding_dim, 2 * dim).bias(true).build()?,
        })
    }

    /// Returns modulated hidden states
    pub fn forward(&mut self, hidden_states: &Array, conditioning: &Array) -> Result<Array, Exception> {
        let cond = self.linear.forward(conditioning)?;
        let cond = nn::silu(&cond)?;

        let shift = cond.index(&[.., ..self.dim])?;
        let scale = cond.index(&[.., self.dim..])?;

        let one = Array::from_f32(1.0);
        ops::add(
            &ops::multiply(hidden_states, &ops::add(&one, &scale)?)?,
            &shift,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let ln = QwenLayerNorm::new(64).unwrap();
        assert_eq!(ln.dim, 64);
    }

    #[test]
    fn test_layer_norm_forward() {
        let mut ln = QwenLayerNorm::new(64).unwrap();
        let x = Array::zeros::<f32>(&[2, 10, 64]).unwrap();
        let temb = Array::zeros::<f32>(&[2, 64]).unwrap();
        let (modulated, gate, mod2) = ln.forward(&x, &temb).unwrap();
        assert_eq!(modulated.shape(), &[2, 10, 64]);
        assert_eq!(gate.shape(), &[2, 64]);
        assert_eq!(mod2.shape(), &[2, 192]);
    }
}
