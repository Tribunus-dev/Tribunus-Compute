//! Qwen-Image VAE with 3D encoder and decoder
//!
//! Reference: diffusers QwenImageEncoder3d, QwenImageDecoder3d

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{self, Conv2d, Conv2dBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::{
    QwenImageCausalConv3D, QwenImageDownBlock3D, QwenImageMidBlock3D,
    QwenImageResBlock3D, QwenImageUpBlock3D,
};

/// Latent normalization constants (16 channels)
/// Reference: diffusers pipeline - "denormalizes using stored VAE statistics"
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075,
    -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.8907, -0.2202,
    -0.2508, -0.7155, -0.4311, 0.5396,
];

pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558,
    2.1779, 2.6922, 2.3901, 5.8310,
    7.6994, 6.6214, 5.0793, 5.1179,
    5.2592, 5.4057, 5.7659, 7.0673,
];

// Channel configuration
const BASE_CHANNELS: i32 = 96;
const STAGE_MULTIPLIERS: [i32; 5] = [1, 1, 2, 4, 4]; // [96, 96, 192, 384, 384]

/// 3D Encoder for Qwen-Image VAE
/// Reference: diffusers QwenImageEncoder3d
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageEncoder3D {
    #[param]
    pub conv_in: QwenImageCausalConv3D,
    #[param]
    pub down_blocks: Vec<QwenImageDownBlock3D>,
    #[param]
    pub mid_block: QwenImageMidBlock3D,
    #[param]
    pub conv_norm_out: QwenImageCausalConv3D, // Final norm conv
    #[param]
    pub conv_out: QwenImageCausalConv3D,
}

impl QwenImageEncoder3D {
    pub fn new() -> Result<Self, Exception> {
        let emb_channels = 256; // Time embedding channels (shared)
        let dropout = 0.0;

        // Input conv: 3 channels (RGB) -> 96 channels, 3x3x3 kernel
        let conv_in = QwenImageCausalConv3D::new(
            3, BASE_CHANNELS, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;

        // Down blocks (5 stages)
        let mut down_blocks = Vec::new();
        for stage in 0..5 {
            let in_ch = BASE_CHANNELS * STAGE_MULTIPLIERS[if stage > 0 { stage - 1 } else { 0 }];
            let out_ch = BASE_CHANNELS * STAGE_MULTIPLIERS[stage];
            let has_downsample = stage < 4; // Last stage has no downsample
            // Each stage has 2 res blocks
            down_blocks.push(QwenImageDownBlock3D::new(
                in_ch, out_ch, emb_channels, dropout, 2, has_downsample,
            )?);
        }

        // Mid block (1 res block + 1 attention block)
        let mid_channels = BASE_CHANNELS * STAGE_MULTIPLIERS[4]; // 384
        let mid_block = QwenImageMidBlock3D::new(mid_channels, emb_channels, dropout, 1)?;

        // Output convs
        let conv_norm_out = QwenImageCausalConv3D::new(
            mid_channels, mid_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;
        let conv_out = QwenImageCausalConv3D::new(
            mid_channels, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;

        Ok(Self {
            conv_in: Param::new(conv_in),
            down_blocks: Param::new(down_blocks),
            mid_block: Param::new(mid_block),
            conv_norm_out: Param::new(conv_norm_out),
            conv_out: Param::new(conv_out),
        })
    }
}

impl Module<&Array> for QwenImageEncoder3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.conv_in.training_mode(mode);
        for block in &mut self.down_blocks {
            block.training_mode(mode);
        }
        self.mid_block.training_mode(mode);
        self.conv_norm_out.training_mode(mode);
        self.conv_out.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [N, 3, H, W] -> [N, 3, 1, H, W] (add temporal dim)
        let x = if x.ndim() == 4 {
            let n = x.dim(0);
            let h = x.dim(2);
            let w = x.dim(3);
            x.reshape(&[n, 3, 1, h, w])?
        } else {
            x.clone()
        };

        // Create empty time embedding (VAE doesn't use timestep conditioning)
        let batch = x.dim(0);
        let temb = Array::zeros::<f32>(&[batch, 256])?;

        let mut h = self.conv_in.forward(&x)?;

        for block in &mut self.down_blocks {
            h = block.forward((&h, &temb))?;
        }

        h = self.mid_block.forward((&h, &temb))?;

        // Final norm and output
        h = nn::silu(&self.conv_norm_out.forward(&h)?)?;
        h = self.conv_out.forward(&h)?;

        Ok(h)
    }
}

/// 3D Decoder for Qwen-Image VAE
/// Reference: diffusers QwenImageDecoder3d
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageDecoder3D {
    #[param]
    pub conv_in: QwenImageCausalConv3D,
    #[param]
    pub mid_block: QwenImageMidBlock3D,
    #[param]
    pub up_blocks: Vec<QwenImageUpBlock3D>,
    #[param]
    pub conv_norm_out: QwenImageCausalConv3D,
    #[param]
    pub conv_out: QwenImageCausalConv3D,
}

impl QwenImageDecoder3D {
    pub fn new() -> Result<Self, Exception> {
        let emb_channels = 256;
        let dropout = 0.0;

        // Input conv: 16 -> 384
        let mid_channels = BASE_CHANNELS * STAGE_MULTIPLIERS[4]; // 384
        let conv_in = QwenImageCausalConv3D::new(
            16, mid_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;

        // Mid block
        let mid_block = QwenImageMidBlock3D::new(mid_channels, emb_channels, dropout, 1)?;

        // Up blocks (4 stages, reverse of encoder)
        let mut up_blocks = Vec::new();
        for stage in (0..4).rev() {
            let ch = BASE_CHANNELS * STAGE_MULTIPLIERS[stage + 1];
            let has_upsample = stage > 0;
            up_blocks.push(QwenImageUpBlock3D::new(
                ch, emb_channels, dropout, 2, has_upsample,
            )?);
        }

        // Output convs
        let conv_norm_out = QwenImageCausalConv3D::new(
            BASE_CHANNELS, BASE_CHANNELS, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;
        let conv_out = QwenImageCausalConv3D::new(
            BASE_CHANNELS, 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        )?;

        Ok(Self {
            conv_in: Param::new(conv_in),
            mid_block: Param::new(mid_block),
            up_blocks: Param::new(up_blocks),
            conv_norm_out: Param::new(conv_norm_out),
            conv_out: Param::new(conv_out),
        })
    }
}

impl Module<&Array> for QwenImageDecoder3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.conv_in.training_mode(mode);
        for block in &mut self.up_blocks {
            block.training_mode(mode);
        }
        self.mid_block.training_mode(mode);
        self.conv_norm_out.training_mode(mode);
        self.conv_out.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [N, 16, T, H, W]
        let batch = x.dim(0);
        let temb = Array::zeros::<f32>(&[batch, 256])?;

        let mut h = self.conv_in.forward(x)?;

        h = self.mid_block.forward((&h, &temb))?;

        for block in &mut self.up_blocks {
            h = block.forward((&h, &temb))?;
        }

        // Final norm and output
        h = nn::silu(&self.conv_norm_out.forward(&h)?)?;
        h = self.conv_out.forward(&h)?;

        Ok(h)
    }
}

/// Complete VAE with encoder and decoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenVAE {
    #[param]
    pub encoder: QwenImageEncoder3D,
    #[param]
    pub decoder: QwenImageDecoder3D,
}

impl QwenVAE {
    pub fn new() -> Result<Self, Exception> {
        Ok(Self {
            encoder: QwenImageEncoder3D::new()?,
            decoder: QwenImageDecoder3D::new()?,
        })
    }

    /// Encode image to latent
    pub fn encode(&mut self, x: &Array) -> Result<Array, Exception> {
        self.encoder.forward(x)
    }

    /// Decode latent to image
    pub fn decode(&mut self, z: &Array) -> Result<Array, Exception> {
        self.decoder.forward(z)
    }

    /// Normalize latent: (z - mean) / std
    pub fn normalize_latent(&self, z: &Array) -> Result<Array, Exception> {
        // z: [N, 16, T, H, W]
        let mean = Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1]);
        let std = Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1, 1]);
        ops::divide(&ops::subtract(z, &mean)?, &std)
    }

    /// Denormalize latent: z * std + mean
    pub fn denormalize_latent(&self, z: &Array) -> Result<Array, Exception> {
        let mean = Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1]);
        let std = Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1, 1]);
        ops::add(&ops::multiply(z, &std)?, &mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_creation() {
        let vae = QwenVAE::new().unwrap();
        assert_eq!(vae.encoder.down_blocks.len(), 5);
        assert_eq!(vae.decoder.up_blocks.len(), 4);
    }

    #[test]
    fn test_latent_normalization() {
        let vae = QwenVAE::new().unwrap();
        let z = Array::zeros::<f32>(&[1, 16, 1, 8, 8]).unwrap();
        let normalized = vae.normalize_latent(&z).unwrap();
        assert_eq!(normalized.shape(), &[1, 16, 1, 8, 8]);
        let denormalized = vae.denormalize_latent(&normalized).unwrap();
        assert_eq!(denormalized.shape(), &[1, 16, 1, 8, 8]);
    }
}
