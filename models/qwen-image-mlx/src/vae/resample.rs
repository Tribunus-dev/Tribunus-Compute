//! Resampling module for 2D/3D upsampling and downsampling
//!
//! Reference: diffusers QwenImageResample

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv2d, Conv2dBuilder};
use mlx_rs::ops;
use mlx_rs::Array;

use super::QwenImageCausalConv3D;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResampleMode {
    Upsample3D,
    Upsample2D,
    Downsample3D,
    Downsample2D,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageResample3D {
    pub mode: ResampleMode,
    #[param]
    pub resample_conv: Conv2d, // 2D conv for spatial resampling
    #[param]
    pub time_conv: Option<QwenImageCausalConv3D>, // Only for 3D modes
}

impl QwenImageResample3D {
    pub fn new(channels: i32, mode: ResampleMode) -> Result<Self, Exception> {
        let conv2d_channels = match mode {
            ResampleMode::Upsample3D | ResampleMode::Upsample2D => channels,
            ResampleMode::Downsample3D | ResampleMode::Downsample2D => channels,
        };

        let resample_conv = Conv2dBuilder::new(conv2d_channels, channels, 3)
            .padding(1)
            .build()?;

        let time_conv = match mode {
            ResampleMode::Upsample3D | ResampleMode::Downsample3D => {
                Some(QwenImageCausalConv3D::new(
                    channels, channels,
                    (3, 1, 1), // T, H, W - only temporal
                    (1, 1, 1), // stride
                    (1, 0, 0), // padding
                    true,
                )?)
            }
            ResampleMode::Upsample2D | ResampleMode::Downsample2D => None,
        };

        Ok(Self {
            mode,
            resample_conv: Param::new(resample_conv),
            time_conv: Param::new(time_conv),
        })
    }

    /// Nearest neighbor 2D upsampling using repeat (matches mflux implementation)
    fn nearest_upsample_2d(x: &Array, scale: i32) -> Result<Array, Exception> {
        // x: [N, C, H, W]
        // Repeat H and W by scale factor
        let repeats = &[1, 1, scale, scale];
        ops::repeat(x, repeats)
    }

    /// Nearest neighbor 3D upsampling
    fn nearest_upsample_3d(x: &Array, scale: i32) -> Result<Array, Exception> {
        // x: [N, C, T, H, W] - repeat T, H, W by scale factor
        let repeats = &[1, 1, scale, scale, scale];
        ops::repeat(x, repeats)
    }

    /// 2D downsampling using strided conv
    fn downsample_2d(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [N, C, H, W]
        let x = x.transpose_axes(&[0, 2, 3, 1])?; // [N, H, W, C] for MLX conv
        let out = self.resample_conv.forward(&x)?;
        out.transpose_axes(&[0, 3, 1, 2])? // [N, C, H, W]
    }
}

impl Module<&Array> for QwenImageResample3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.resample_conv.training_mode(mode);
        if let Some(ref mut conv) = self.time_conv {
            conv.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self.mode {
            ResampleMode::Upsample3D => {
                // x: [N, C, T, H, W]
                let scale = 2i32;
                let up = Self::nearest_upsample_3d(x, scale)?; // [N, C, 2T, 2H, 2W]

                // Apply temporal conv
                let up = if let Some(ref mut tconv) = self.time_conv {
                    tconv.forward(&up)?
                } else {
                    up
                };

                // Apply spatial conv on each time frame
                let n = up.dim(0);
                let c = up.dim(1);
                let t = up.dim(2);
                let h = up.dim(3);
                let w = up.dim(4);

                // Reshape to 4D: [N*T, C, H, W]
                let up_4d = up.reshape(&[n * t, c, h, w])?;
                let up_4d = up_4d.transpose_axes(&[0, 2, 3, 1])?; // [N*T, H, W, C] for Conv2d
                let out = self.resample_conv.forward(&up_4d)?;
                let out = out.transpose_axes(&[0, 3, 1, 2])?; // [N*T, C, H, W]

                // Reshape back to 5D
                out.reshape(&[n, c, t, h, w])
            }
            ResampleMode::Upsample2D => {
                // x: [N, C, H, W]
                let scale = 2i32;
                let up = Self::nearest_upsample_2d(x, scale)?; // [N, C, 2H, 2W]

                // Apply spatial conv
                let up = up.transpose_axes(&[0, 2, 3, 1])?; // [N, H, W, C] for Conv2d
                let out = self.resample_conv.forward(&up)?;
                out.transpose_axes(&[0, 3, 1, 2]) // [N, C, H, W]
            }
            ResampleMode::Downsample3D => {
                // x: [N, C, T, H, W]

                // Apply temporal conv first
                let x = if let Some(ref mut tconv) = self.time_conv {
                    tconv.forward(x)?
                } else {
                    x.clone()
                };

                // Apply spatial downsampling on each time frame
                let n = x.dim(0);
                let c = x.dim(1);
                let t = x.dim(2);
                let h = x.dim(3);
                let w = x.dim(4);

                let x_4d = x.reshape(&[n * t, c, h, w])?;
                let x_4d = x_4d.transpose_axes(&[0, 2, 3, 1])?; // [N*T, H, W, C] for Conv2d
                let out = self.resample_conv.forward(&x_4d)?;
                let out = out.transpose_axes(&[0, 3, 1, 2])?; // [N*T, C, H, W]

                // Strided conv reduces H and W, reshape back
                let out_h = out.dim(2);
                let out_w = out.dim(3);
                out.reshape(&[n, c, t, out_h, out_w])
            }
            ResampleMode::Downsample2D => {
                // x: [N, C, H, W]
                // Use spatial conv with stride 2
                let x = x.transpose_axes(&[0, 2, 3, 1])?; // [N, H, W, C] for Conv2d
                let out = self.resample_conv.forward(&x)?;
                out.transpose_axes(&[0, 3, 1, 2]) // [N, C, H, W]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsample_2d() {
        let mut resample = QwenImageResample3D::new(64, ResampleMode::Upsample2D).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 8, 8]).unwrap();
        let out = resample.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64, 16, 16]);
    }

    #[test]
    fn test_upsample_3d() {
        let mut resample = QwenImageResample3D::new(64, ResampleMode::Upsample3D).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 8, 8]).unwrap();
        let out = resample.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64, 4, 16, 16]);
    }

    #[test]
    fn test_nearest_upsample_2d() {
        let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let up = QwenImageResample3D::nearest_upsample_2d(&x, 2).unwrap();
        assert_eq!(up.shape(), &[1, 1, 4, 4]);
    }
}
