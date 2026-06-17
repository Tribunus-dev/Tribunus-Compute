//! 3-axis Rotary Position Embedding for Qwen-Image
//!
//! Reference: diffusers QwenEmbedRope
//! "Implements rotary embeddings for video/image sequences with frame, height, and width dimensions"

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Dtype;
use mlx_rs::Array;

/// 3-axis Rotary Position Embedding for Qwen-Image
/// Different from Z-Image: theta=10000, axes=[16, 56, 56], scaled=true
#[derive(Debug, Clone)]
pub struct QwenEmbedRope {
    pub theta: i32,
    pub axes_dimensions: [i32; 3], // [16, 56, 56] for (frame, height, width)
    pub scale_rope: bool,

    // Pre-computed frequencies
    positive_cos: Vec<Array>,
    positive_sin: Vec<Array>,
    negative_cos: Vec<Array>,
    negative_sin: Vec<Array>,
}

impl QwenEmbedRope {
    const MAX_INDEX: i32 = 4096;

    pub fn new(theta: i32, axes_dimensions: [i32; 3], scale_rope: bool) -> Result<Self, Exception> {
        Ok(Self {
            theta,
            axes_dimensions,
            scale_rope,
            positive_cos: Vec::new(),
            positive_sin: Vec::new(),
            negative_cos: Vec::new(),
            negative_sin: Vec::new(),
        })
    }

    fn compute_rope_params(
        indices: &[i32],
        dim: i32,
        theta: i32,
    ) -> Result<(Array, Array), Exception> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let theta_f = theta as f32;
                1.0 / theta_f.powf(2.0 * i as f32 / dim as f32)
            })
            .collect();
        let inv_freq = Array::from_slice(&inv_freq, &[half_dim]);

        // Convert indices to f32 and compute sin/cos
        let indices_f32: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
        let indices = Array::from_slice(&indices_f32, &[indices.len() as i32]);

        // [seq_len, half_dim] = [seq_len, 1] * [1, half_dim]
        let args = ops::matmul(
            &indices.reshape(&[indices.len() as i32, 1])?,
            &inv_freq.reshape(&[1, half_dim])?,
        )?;

        let cos = ops::cos(&args)?;
        let sin = ops::sin(&args)?;

        Ok((cos, sin))
    }

    /// Build rotary embeddings for image and text
    /// Returns ((image_cos, image_sin), (text_cos, text_sin))
    pub fn forward(
        &self,
        video_segments: &[(i32, i32, i32)], // [(frame, height, width), ...]
        text_sequence_lengths: &[i32],
    ) -> Result<((Array, Array), (Array, Array)), Exception> {
        // Build image frequencies
        let total_frames: i32 = video_segments.iter().map(|(f, _, _)| f).sum();
        let max_frames = video_segments.iter().map(|(f, _, _)| f).max().unwrap_or(1);

        let mut image_cos_parts = Vec::new();
        let mut image_sin_parts = Vec::new();
        let mut frame_offset = 0;

        for (_frame, height, width) in video_segments {
            let (cos, sin) = self.build_video_frequencies(
                *max_frames, *height, *width, frame_offset,
            )?;
            image_cos_parts.push(cos);
            image_sin_parts.push(sin);
            frame_offset += *max_frames;
        }

        let image_cos = ops::concatenate_axis(&image_cos_parts.iter().collect::<Vec<_>>(), 0)?;
        let image_sin = ops::concatenate_axis(&image_sin_parts.iter().collect::<Vec<_>>(), 0)?;

        // Build text frequencies
        let total_text_len: i32 = text_sequence_lengths.iter().sum();
        let mut text_cos_parts = Vec::new();
        let mut text_sin_parts = Vec::new();

        for &len in text_sequence_lengths {
            let text_indices: Vec<i32> = (0..len).collect();
            let total_dim: i32 = self.axes_dimensions.iter().sum();
            let (cos, sin) = Self::compute_rope_params(&text_indices, total_dim, self.theta)?;
            text_cos_parts.push(cos);
            text_sin_parts.push(sin);
        }

        let text_cos = ops::concatenate_axis(&text_cos_parts.iter().collect::<Vec<_>>(), 0)?;
        let text_sin = ops::concatenate_axis(&text_sin_parts.iter().collect::<Vec<_>>(), 0)?;

        // Reshape to [1, seq, 1, dim]
        let total_img_seq = image_cos.dim(0);
        let total_dim: i32 = self.axes_dimensions.iter().sum();

        let image_cos = image_cos.reshape(&[1, total_img_seq, 1, total_dim])?;
        let image_sin = image_sin.reshape(&[1, total_img_seq, 1, total_dim])?;
        let text_cos = text_cos.reshape(&[1, total_text_len, 1, total_dim])?;
        let text_sin = text_sin.reshape(&[1, total_text_len, 1, total_dim])?;

        Ok(((image_cos, image_sin), (text_cos, text_sin)))
    }

    fn build_video_frequencies(
        &self,
        frame: i32,
        height: i32,
        width: i32,
        frame_offset: i32,
    ) -> Result<(Array, Array), Exception> {
        // Build position indices for each axis
        let h_positions: Vec<i32> = (0..height).collect();
        let w_positions: Vec<i32> = (0..width).collect();

        // Total patches = height * width
        let total_patches = height * width;

        // For each patch, compute the 3D position and concatenate axis frequencies
        let mut cos_list = Vec::new();
        let mut sin_list = Vec::new();

        for h in 0..height {
            for w in 0..width {
                // Frame frequency
                let (f_cos, f_sin) =
                    Self::compute_rope_params(&[0], self.axes_dimensions[0], self.theta)?;
                // Height frequency
                let (h_cos, h_sin) =
                    Self::compute_rope_params(&[h_positions[h as usize]], self.axes_dimensions[1], self.theta)?;
                // Width frequency
                let (w_cos, w_sin) =
                    Self::compute_rope_params(&[w_positions[w as usize]], self.axes_dimensions[2], self.theta)?;

                let cos = ops::concatenate_axis(&[&f_cos, &h_cos, &w_cos], -1)?;
                let sin = ops::concatenate_axis(&[&f_sin, &h_sin, &w_sin], -1)?;

                cos_list.push(cos);
                sin_list.push(sin);
            }
        }

        let cos = ops::concatenate_axis(&cos_list.iter().collect::<Vec<_>>(), 0)?;
        let sin = ops::concatenate_axis(&sin_list.iter().collect::<Vec<_>>(), 0)?;

        Ok((cos, sin))
    }
}

/// Apply RoPE to query and key tensors
/// Uses even/odd split method (mathematically equivalent to rotation matrix)
pub fn apply_rope(
    x: &Array,
    rotary: &(Array, Array),
) -> Result<Array, Exception> {
    let (cos, sin) = rotary;
    // x: [batch, seq, heads, head_dim]
    // cos/sin: [1, seq, 1, dim]
    let half_dim = x.dim(-1) / 2;

    // Split into two halves
    let x1 = x.index(&[.., .., .., ..half_dim])?;
    let x2 = x.index(&[.., .., .., half_dim..])?;

    // [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    // Cos and sin might have different dim than half -> broadcast
    let cos_part = ops::multiply(&x1, cos)?;
    let sin_part = ops::multiply(&x2, sin)?;
    let rotated1 = ops::subtract(&cos_part, &sin_part)?;

    let cos_part = ops::multiply(&x2, cos)?;
    let sin_part = ops::multiply(&x1, sin)?;
    let rotated2 = ops::add(&cos_part, &sin_part)?;

    ops::concatenate_axis(&[&rotated1, &rotated2], -1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let rope = QwenEmbedRope::new(10000, [16, 56, 56], true).unwrap();
        assert_eq!(rope.theta, 10000);
    }

    #[test]
    fn test_apply_rope() {
        let x = Array::zeros::<f32>(&[1, 4, 2, 128]).unwrap();
        let cos = Array::ones::<f32>(&[1, 4, 1, 128]).unwrap();
        let sin = Array::zeros::<f32>(&[1, 4, 1, 128]).unwrap();
        let rotated = apply_rope(&x, &(cos, sin)).unwrap();
        assert_eq!(rotated.shape(), &[1, 4, 2, 128]);
    }

    #[test]
    fn test_rope_params() {
        let (cos, sin) = QwenEmbedRope::compute_rope_params(&[0, 1, 2, 3], 16, 10000).unwrap();
        assert_eq!(cos.shape(), &[4, 8]); // seq_len=4, half_dim=8
        assert_eq!(sin.shape(), &[4, 8]);
    }
}
