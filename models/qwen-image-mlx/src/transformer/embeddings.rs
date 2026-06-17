//! Timestep embeddings for diffusion
//!
//! Reference: diffusers QwenImageTransformer2DModel "Generate timestep embeddings"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Linear, LinearBuilder};
use mlx_rs::ops;
use mlx_rs::Dtype;
use mlx_rs::Array;

/// Timestep projection using sinusoidal embeddings
#[derive(Debug, Clone)]
pub struct QwenTimesteps {
    pub projection_dim: i32,
    pub scale: f32,
}

impl QwenTimesteps {
    pub fn new(projection_dim: i32, scale: f32) -> Self {
        Self {
            projection_dim,
            scale,
        }
    }

    pub fn forward(&self, timesteps: &Array) -> Result<Array, Exception> {
        let half_dim = self.projection_dim / 2;
        let exponent: Vec<f32> = (0..half_dim)
            .map(|i| -(i as f32) * (10000.0f32.ln()) / half_dim as f32)
            .collect();
        let exponent = Array::from_slice(&exponent, &[half_dim]);

        // timesteps: [batch] or scalar
        let timesteps = ops::multiply(timesteps, &Array::from_f32(self.scale))?;
        let timesteps = timesteps.as_dtype(Dtype::Float32)?;
        let timesteps = timesteps.reshape(&[-1, 1])?;

        let angles = ops::matmul(&timesteps, &exponent.reshape(&[half_dim, 1])?)?;

        let cos = ops::cos(&angles)?;
        let sin = ops::sin(&angles)?;

        ops::concatenate_axis(&[&cos, &sin], -1)
    }
}

/// Timestep embedding MLP
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimestepEmbedding {
    #[param]
    pub linear_1: Linear,
    #[param]
    pub linear_2: Linear,
}

impl QwenTimestepEmbedding {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: LinearBuilder::new(dim, dim).bias(true).build()?,
            linear_2: LinearBuilder::new(dim, dim).bias(true).build()?,
        })
    }
}

impl Module<&Array> for QwenTimestepEmbedding {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = nn::silu(&self.linear_1.forward(x)?)?;
        self.linear_2.forward(&h)
    }
}

/// Combined time-text embedding
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimeTextEmbed {
    #[param]
    pub timestep_embedder: QwenTimestepEmbedding,
    #[param]
    pub text_embedder: Linear,
}

impl QwenTimeTextEmbed {
    pub fn new(hidden_size: i32, caption_projection_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            timestep_embedder: QwenTimestepEmbedding::new(hidden_size)?,
            text_embedder: LinearBuilder::new(caption_projection_dim, hidden_size).bias(true).build()?,
        })
    }
}

impl Module<(&Array, &Array)> for QwenTimeTextEmbed {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.timestep_embedder.training_mode(mode);
        self.text_embedder.training_mode(mode);
    }

    fn forward(&mut self, (timestep, text_emb): (&Array, &Array)) -> Result<Array, Exception> {
        let t_emb = self.timestep_embedder.forward(timestep)?;
        let txt_emb = self.text_embedder.forward(text_emb)?;
        ops::add(&t_emb, &txt_emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timesteps() {
        let ts = QwenTimesteps::new(256, 1000.0);
        let t = Array::from_slice::<f32>(&[0.5], &[1]);
        let out = ts.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 256]);
    }

    #[test]
    fn test_timestep_embedding() {
        let mut emb = QwenTimestepEmbedding::new(256).unwrap();
        let x = Array::zeros::<f32>(&[1, 256]).unwrap();
        let out = emb.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 256]);
    }
}
