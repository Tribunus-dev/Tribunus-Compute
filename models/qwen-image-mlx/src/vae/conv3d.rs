//! 3D Causal Convolution
//!
//! Reference: diffusers QwenImageCausalConv3d
//! "causal padding in the time dimension and feature caching for efficient inference"

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::ops;
use mlx_rs::Array;

/// 3D Causal Convolution
/// Pads temporally only in the "past" direction for causal generation
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageCausalConv3D {
    pub in_channels: i32,
    pub out_channels: i32,
    pub kernel_size: (i32, i32, i32), // (T, H, W)
    pub stride: (i32, i32, i32),
    pub padding: (i32, i32, i32),

    #[param]
    pub weight: Param<Array>, // [out_ch, in_ch, kT, kH, kW]
    #[param]
    pub bias: Param<Option<Array>>, // [out_ch]
}

impl QwenImageCausalConv3D {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: (i32, i32, i32),
        stride: (i32, i32, i32),
        padding: (i32, i32, i32),
        use_bias: bool,
    ) -> Result<Self, Exception> {
        let (k_t, k_h, k_w) = kernel_size;

        // Kaiming uniform initialization for 3D conv weights
        let fan_in = in_channels * k_t * k_h * k_w;
        let scale = (1.0 / fan_in as f32).sqrt();
        let weight = Array::uniform::<f32>(
            -scale,
            scale,
            &[out_channels, in_channels, k_t, k_h, k_w],
        )?;

        let bias = if use_bias {
            Some(Param::new(Array::zeros::<f32>(&[out_channels])?))
        } else {
            None
        };

        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Apply causal padding: only pad "past" in temporal dimension
    /// Standard causal padding: pad_before = kernel_size - 1, pad_after = 0
    fn apply_causal_padding(&self, x: &Array) -> Result<Array, Exception> {
        // x: [N, C, T, H, W]
        let (kT, kH, kW) = self.kernel_size;
        let (sT, sH, sW) = self.stride;
        let (pT, pH, pW) = self.padding;

        // Causal temporal padding: pad (kT-1, 0) in temporal dim (dim 2)
        // Spatial padding: symmetric (pH, pH), (pW, pW)
        let pad_width = [
            (0, 0), // N
            (0, 0), // C
            (kT - 1, 0), // T: causal (past only)
            (pH, pH), // H: symmetric
            (pW, pW), // W: symmetric
        ];

        ops::pad(x, &pad_width, ops::PadMode::Constant(0.0))
    }
}

impl Module<&Array> for QwenImageCausalConv3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Input: [N, C, T, H, W]
        let padded = self.apply_causal_padding(x)?;

        // MLX expects conv3d in format:
        // input: [N, T, H, W, C] (NTHWC)
        // weight: [out_C, kT, kH, kW, in_C]

        // Transpose from NCTHW to NTHWC
        let input = padded.transpose_axes(&[0, 2, 3, 4, 1])?; // [N, T, H, W, C]

        // Transpose weight from [out, in, kT, kH, kW] to [out, kT, kH, kW, in]
        let weight = self.weight.transpose_axes(&[0, 2, 3, 4, 1])?; // [out, kT, kH, kW, in]

        let strides = (self.stride.1, self.stride.2, self.stride.0); // (sH, sW, sT) for NTHWC
        let padding: [(i32, i32); 3] = [(0, 0), (0, 0), (0, 0)]; // Already padded

        let result = ops::conv_general::<3>(
            &input,
            &weight,
            &strides,
            &padding,
            None, // dilation
        )?;

        // Add bias
        let result = if let Some(ref bias) = *self.bias {
            // bias: [out_C] -> broadcast over N, T, H, W
            ops::add(&result, &bias.reshape(&[1, 1, 1, 1, self.out_channels])?)?
        } else {
            result
        };

        // Transpose back from NTHWC to NCTHW
        result.transpose_axes(&[0, 4, 1, 2, 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv3d_forward() {
        let mut conv = QwenImageCausalConv3D::new(
            3, 64, (2, 3, 3), (1, 1, 1), (0, 1, 1), true
        ).unwrap();
        let x = Array::zeros::<f32>(&[1, 3, 2, 16, 16]).unwrap();
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64, 2, 16, 16]);
    }

    #[test]
    fn test_causal_padding() {
        let conv = QwenImageCausalConv3D::new(
            3, 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), false
        ).unwrap();
        let x = Array::zeros::<f32>(&[1, 3, 2, 16, 16]).unwrap();
        let padded = conv.apply_causal_padding(&x).unwrap();
        // Temporal dim: 2 + (3-1) = 4 (past-only padding)
        assert_eq!(padded.shape(), &[1, 3, 4, 18, 18]);
    }
}
