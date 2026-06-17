//! VAE building blocks: ResBlock3D, MidBlock3D, UpBlock3D, DownBlock3D
//!
//! Reference: diffusers QwenImageResidualBlock, QwenImageMidBlock, QwenImageUpBlock

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::fast;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Conv2d, Conv2dBuilder, GroupNorm, GroupNormBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::{QwenImageAttentionBlock3D, QwenImageCausalConv3D, QwenImageRMSNorm, QwenImageResample3D, ResampleMode};

/// 3D Residual Block for VAE
/// Reference: diffusers QwenImageResidualBlock
/// "RMS normalization, causal 3D convolutions, and optional dropout"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageResBlock3D {
    pub channels: i32,
    pub emb_channels: i32,
    pub dropout: f32,
    pub out_channels: i32,
    pub use_conv_shortcut: bool,

    // RMS normalization
    #[param]
    pub norm1: QwenImageRMSNorm,
    #[param]
    pub norm2: QwenImageRMSNorm,
    #[param]
    pub norm3: Option<QwenImageRMSNorm>, // Only when channels != out_channels

    // Convolutions
    #[param]
    pub conv1: QwenImageCausalConv3D,
    #[param]
    pub conv2: QwenImageCausalConv3D,
    #[param]
    pub conv_shortcut: Option<QwenImageCausalConv3D>, // Only when channels != out_channels

    // Time embedding projection
    #[param]
    pub time_emb_proj: nn::Linear,
}

impl QwenImageResBlock3D {
    pub fn new(
        channels: i32,
        emb_channels: i32,
        dropout: f32,
        out_channels: i32,
        use_conv_shortcut: bool,
    ) -> Result<Self, Exception> {
        let use_3d = true; // Qwen-Image VAE uses 3D convs

        let (norm1, norm2) = if use_3d {
            (QwenImageRMSNorm::new(channels, 1e-6, false)?,
             QwenImageRMSNorm::new(out_channels, 1e-6, false)?)
        } else {
            (QwenImageRMSNorm::new(channels, 1e-6, true)?,
             QwenImageRMSNorm::new(out_channels, 1e-6, true)?)
        };

        Ok(Self {
            channels,
            emb_channels,
            dropout,
            out_channels,
            use_conv_shortcut,

            norm1,
            norm2,
            norm3: if channels != out_channels {
                Some(if use_3d {
                    QwenImageRMSNorm::new(channels, 1e-6, false)?
                } else {
                    QwenImageRMSNorm::new(channels, 1e-6, true)?
                })
            } else {
                None
            },

            conv1: QwenImageCausalConv3D::new(
                channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
            )?,
            conv2: QwenImageCausalConv3D::new(
                out_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
            )?,
            conv_shortcut: if channels != out_channels {
                Some(QwenImageCausalConv3D::new(
                    channels, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), true,
                )?)
            } else {
                None
            },

            time_emb_proj: nn::LinearBuilder::new(emb_channels, out_channels).bias(true).build()?,
        })
    }
}

impl Module<(&Array, &Array)> for QwenImageResBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, (x, temb): (&Array, &Array)) -> Result<Array, Exception> {
        // x: [N, C, T, H, W] for 3D
        let residual = x.clone();

        // First norm + conv
        let h = self.norm1.forward(x)?;
        let h = nn::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        // Add time embedding
        // temb: [N, emb_channels] -> [N, out_channels, 1, 1, 1]
        let temb = nn::silu(&temb)?;
        let temb = self.time_emb_proj.forward(&temb)?;
        let temb = temb.reshape(&[-1, self.out_channels, 1, 1, 1])?;
        let h = ops::add(&h, &temb)?;

        // Second norm + conv
        let h = self.norm2.forward(&h)?;
        let h = nn::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        // Shortcut connection
        let residual = if self.channels != self.out_channels {
            if let Some(ref mut conv) = self.conv_shortcut {
                conv.forward(&residual)?
            } else {
                residual
            }
        } else {
            residual
        };

        ops::add(&residual, &h)
    }
}

/// Mid block with alternating residual and attention blocks
/// Reference: diffusers QwenImageMidBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageMidBlock3D {
    #[param]
    pub res_blocks: Vec<QwenImageResBlock3D>,
    #[param]
    pub attn_blocks: Vec<QwenImageAttentionBlock3D>,
}

impl QwenImageMidBlock3D {
    pub fn new(
        in_channels: i32,
        emb_channels: i32,
        dropout: f32,
        num_layers: i32,
    ) -> Result<Self, Exception> {
        let mut res_blocks = Vec::new();
        let mut attn_blocks = Vec::new();

        for i in 0..num_layers {
            let channels = if i == 0 { in_channels } else { in_channels };
            res_blocks.push(QwenImageResBlock3D::new(
                channels, emb_channels, dropout, in_channels, false,
            )?);
            attn_blocks.push(QwenImageAttentionBlock3D::new(in_channels)?);
        }

        Ok(Self {
            res_blocks,
            attn_blocks,
        })
    }
}

impl Module<(&Array, &Array)> for QwenImageMidBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for block in &mut self.res_blocks {
            block.training_mode(mode);
        }
        for block in &mut self.attn_blocks {
            block.training_mode(mode);
        }
    }

    fn forward(&mut self, (x, temb): (&Array, &Array)) -> Result<Array, Exception> {
        let mut h = x.clone();
        for (res, attn) in self.res_blocks.iter_mut().zip(self.attn_blocks.iter_mut()) {
            h = res.forward((&h, temb))?;
            h = attn.forward(&h)?;
        }
        Ok(h)
    }
}

/// Up block for decoder
/// Reference: diffusers QwenImageUpBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageUpBlock3D {
    #[param]
    pub res_blocks: Vec<QwenImageResBlock3D>,
    #[param]
    pub upsamplers: Vec<QwenImageResample3D>,
}

impl QwenImageUpBlock3D {
    pub fn new(
        channels: i32,
        emb_channels: i32,
        dropout: f32,
        num_layers: i32,
        has_upsample: bool,
    ) -> Result<Self, Exception> {
        let mut res_blocks = Vec::new();
        let mut upsamplers = Vec::new();

        for _ in 0..num_layers {
            res_blocks.push(QwenImageResBlock3D::new(
                channels, emb_channels, dropout, channels, false,
            )?);
        }

        if has_upsample {
            upsamplers.push(QwenImageResample3D::new(
                channels,
                ResampleMode::Upsample3D,
            )?);
        }

        Ok(Self {
            res_blocks,
            upsamplers,
        })
    }
}

impl Module<(&Array, &Array)> for QwenImageUpBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for block in &mut self.res_blocks {
            block.training_mode(mode);
        }
        for up in &mut self.upsamplers {
            up.training_mode(mode);
        }
    }

    fn forward(&mut self, (x, temb): (&Array, &Array)) -> Result<Array, Exception> {
        let mut h = x.clone();
        for res in &mut self.res_blocks {
            h = res.forward((&h, temb))?;
        }
        for up in &mut self.upsamplers {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Down block for encoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageDownBlock3D {
    #[param]
    pub res_blocks: Vec<QwenImageResBlock3D>,
    #[param]
    pub downsamplers: Vec<QwenImageResample3D>,
}

impl QwenImageDownBlock3D {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        emb_channels: i32,
        dropout: f32,
        num_layers: i32,
        has_downsample: bool,
    ) -> Result<Self, Exception> {
        let mut res_blocks = Vec::new();
        let mut downsamplers = Vec::new();

        for i in 0..num_layers {
            let ch = if i == 0 { in_channels } else { out_channels };
            res_blocks.push(QwenImageResBlock3D::new(
                ch, emb_channels, dropout, out_channels, false,
            )?);
        }

        if has_downsample {
            downsamplers.push(QwenImageResample3D::new(
                out_channels,
                ResampleMode::Downsample3D,
            )?);
        }

        Ok(Self {
            res_blocks,
            downsamplers,
        })
    }
}

impl Module<(&Array, &Array)> for QwenImageDownBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for block in &mut self.res_blocks {
            block.training_mode(mode);
        }
        for down in &mut self.downsamplers {
            down.training_mode(mode);
        }
    }

    fn forward(&mut self, (x, temb): (&Array, &Array)) -> Result<Array, Exception> {
        let mut h = x.clone();
        for res in &mut self.res_blocks {
            h = res.forward((&h, temb))?;
        }
        for down in &mut self.downsamplers {
            h = down.forward(&h)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_res_block_3d() {
        let mut block = QwenImageResBlock3D::new(64, 256, 0.0, 64, false).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 8, 8]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 256]).unwrap();
        let out = block.forward((&x, &temb)).unwrap();
        assert_eq!(out.shape(), &[1, 64, 2, 8, 8]);
    }

    #[test]
    fn test_down_block() {
        let mut block = QwenImageDownBlock3D::new(64, 128, 256, 0.0, 2, true).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 16, 16]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 256]).unwrap();
        let out = block.forward((&x, &temb)).unwrap();
        assert_eq!(out.shape(), &[1, 128, 1, 8, 8]); // After stride-2 downsample
    }

    #[test]
    fn test_up_block() {
        let mut block = QwenImageUpBlock3D::new(128, 256, 0.0, 2, true).unwrap();
        let x = Array::zeros::<f32>(&[1, 128, 1, 8, 8]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 256]).unwrap();
        let out = block.forward((&x, &temb)).unwrap();
        // After upsample: spatial dims double, temporal stays
        assert_eq!(out.shape(), &[1, 128, 2, 16, 16]);
    }

    #[test]
    fn test_mid_block() {
        let mut block = QwenImageMidBlock3D::new(64, 256, 0.0, 1).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 8, 8]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 256]).unwrap();
        let out = block.forward((&x, &temb)).unwrap();
        assert_eq!(out.shape(), &[1, 64, 2, 8, 8]);
    }
}
