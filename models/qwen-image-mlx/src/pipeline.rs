//! Qwen-Image generation pipeline
//!
//! Reference: diffusers QwenImagePipeline
//! "End-to-end text-to-image generation pipeline"

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::Array;

use crate::vae::QwenVAE;

/// Flow-matching Euler scheduler
/// Reference: diffusers FlowMatchEulerDiscreteScheduler
#[derive(Debug, Clone)]
pub struct FlowMatchEulerScheduler {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl FlowMatchEulerScheduler {
    /// Create new scheduler for Qwen-Image
    ///
    /// The scheduler uses:
    /// - image_seq_len: number of patches (e.g. 32*32 = 1024 for 512x512)
    /// - num_steps: number of denoising steps
    pub fn new(num_steps: i32, image_seq_len: i32) -> Self {
        let num_steps = num_steps as usize;

        // Qwen-Image scheduler parameters
        let base_shift = 0.5f32;
        let max_shift = 0.9f32; // Qwen-Image uses 0.9 for 512x512
        let base_image_seq_len = 256.0f32;
        let max_image_seq_len = 8192.0f32;
        let shift_terminal = 0.02f32;

        // Step 1: Compute mu (linear interpolation)
        let m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len);
        let b = base_shift - m * base_image_seq_len;
        let mu = m * image_seq_len as f32 + b;

        // Step 2: Generate input sigmas (from 1.0 to 1/num_steps)
        let input_sigmas: Vec<f32> = (0..num_steps)
            .map(|i| {
                1.0 - (i as f32 / (num_steps - 1) as f32) * (1.0 - 1.0 / num_steps as f32)
            })
            .collect();

        // Step 3: Apply exponential time shift
        let exp_mu = mu.exp();
        let shifted_sigmas: Vec<f32> = input_sigmas
            .iter()
            .map(|&t| {
                if t >= 1.0 {
                    1.0
                } else if t <= 0.0 {
                    0.0
                } else {
                    let inv_t = 1.0 / t - 1.0;
                    exp_mu / (exp_mu + inv_t.powf(1.0))
                }
            })
            .collect();

        // Step 4: Apply stretch_shift_to_terminal
        let last_sigma = shifted_sigmas[shifted_sigmas.len() - 1];
        let scale_factor = (1.0 - last_sigma) / (1.0 - shift_terminal);

        let sigmas: Vec<f32> = shifted_sigmas
            .iter()
            .map(|&t| 1.0 - (1.0 - t) / scale_factor)
            .collect();

        // Timesteps: sigma * 1000
        let timesteps: Vec<f32> = sigmas.iter().map(|&s| s * 1000.0).collect();

        // Last timestep should be 20.0 (shift_terminal * 1000)
        // Ensure last sigma equals shift_terminal
        assert!(
            (sigmas[sigmas.len() - 1] - shift_terminal).abs() < 0.001,
            "Last sigma should be shift_terminal ({}), got {}",
            shift_terminal,
            sigmas[sigmas.len() - 1]
        );

        Self { sigmas, timesteps }
    }

    /// Euler step: sample_{t-1} = sample_t + dt * model_output
    ///
    /// Where dt = sigma_{t-1} - sigma_t (note: sigmas increase in our implementation
    /// so dt is negative, matching the reverse diffusion direction)
    pub fn step(
        &self,
        model_output: &Array,
        timestep_idx: usize,
        sample: &Array,
    ) -> Result<Array, Exception> {
        // dt = sigma_{t+1} - sigma_t (move to next timestep)
        let dt = self.sigmas[timestep_idx + 1] - self.sigmas[timestep_idx];
        let dt_arr = Array::from_f32(dt);
        ops::add(sample, &ops::multiply(model_output, &dt_arr)?)
    }
}

/// Qwen-Image generation pipeline
pub struct QwenImagePipeline {
    pub transformer: Option<crate::qwen_quantized::QwenQuantizedTransformer>,
    pub text_encoder: crate::text_encoder::QwenTextEncoder,
    pub vae: crate::vae::QwenVAE,
    pub scheduler: FlowMatchEulerScheduler,
    pub config: QwenImageConfig,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct QwenImageConfig {
    pub height: i32,
    pub width: i32,
    pub num_steps: i32,
    pub guidance_scale: f32,
}

impl Default for QwenImageConfig {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            num_steps: 20,
            guidance_scale: 4.0,
        }
    }
}

impl QwenImagePipeline {
    pub fn new(
        vae: QwenVAE,
        text_encoder: crate::text_encoder::QwenTextEncoder,
        config: QwenImageConfig,
    ) -> Self {
        let image_seq_len = (config.height / 16) * (config.width / 16);
        let scheduler = FlowMatchEulerScheduler::new(config.num_steps, image_seq_len);

        Self {
            transformer: None,
            text_encoder,
            vae,
            scheduler,
            config,
        }
    }

    /// Generate image from prompt text
    pub fn generate(
        &mut self,
        prompt: &str,
    ) -> Result<Array, Box<dyn std::error::Error>> {
        // TODO: Implement end-to-end generation
        // This requires: tokenizer, noise generation, CFG guidance loop
        //
        // Steps:
        // 1. Tokenize prompt
        // 2. Encode text with text_encoder
        // 3. Generate random latent noise
        // 4. Diffusion loop with CFG
        // 5. Decode with VAE
        // 6. Return image tensor
        todo!("End-to-end generation not yet implemented")
    }
}

/// Attention mask builder for variable-length sequences
pub fn build_attention_mask(
    image_seq_len: i32,
    text_seq_len: i32,
    batch_size: i32,
) -> Result<Array, Exception> {
    let total_seq = image_seq_len + text_seq_len;

    // Create mask: image tokens attend to all, text tokens attend to text only
    let mut mask_data: Vec<f32> = Vec::with_capacity((total_seq * total_seq) as usize);
    for i in 0..total_seq {
        for j in 0..total_seq {
            if i < image_seq_len {
                // Image tokens attend to all
                mask_data.push(0.0);
            } else {
                // Text tokens attend to text only
                if j >= image_seq_len {
                    mask_data.push(0.0);
                } else {
                    mask_data.push(f32::NEG_INFINITY);
                }
            }
        }
    }

    let mask = Array::from_slice(&mask_data, &[1, 1, total_seq, total_seq]);
    ops::broadcast(&mask, &[batch_size, 1, total_seq, total_seq])
}

// ─── Latent packing/unpacking (patchify for DiT) ────────────────────────────

/// Pack latents: [B, C, 1, H, W] -> [B, (H/2)*(W/2), C*4]
/// Rearranges spatial dims into patch tokens for the DiT transformer.
pub fn pack_latents(latents: &Array) -> Result<Array, Exception> {
    let batch = latents.dim(0);
    let channels = latents.dim(1);
    let _frame = latents.dim(2);
    let height = latents.dim(3);
    let width = latents.dim(4);

    // Patch: 2x2 spatial merge -> (C*4) per patch
    let patch_size = 2i32;
    let patch_h = height / patch_size;
    let patch_w = width / patch_size;

    // Reshape: [B, C, 1, H, W] -> [B, C, 1, patch_h, patch_size, patch_w, patch_size]
    // Then permute to [B, 1, patch_h, patch_w, patch_size, patch_size, C]
    // Then reshape to [B, patch_h*patch_w, C*4]
    let reshaped = latents.reshape(&[batch, channels, 1, patch_h, patch_size, patch_w, patch_size])?;
    let permuted = reshaped.transpose(&[0, 2, 3, 5, 4, 6, 1])?;
    permuted.reshape(&[batch, 1 * patch_h * patch_w, channels * patch_size * patch_size])
}

/// Unpack latents: [B, seq, C*4] -> [B, C, 1, H, W]
/// height/width are original image dimensions (in pixels).
pub fn unpack_latents(latents: &Array, height: i32, width: i32) -> Result<Array, Exception> {
    let batch = latents.dim(0);
    let patch_channels = latents.dim(2); // C*4
    let patch_size = 2i32;
    let channels = patch_channels / (patch_size * patch_size);
    let latent_h = height / 16;
    let latent_w = width / 16;
    let patch_h = latent_h / patch_size;
    let patch_w = latent_w / patch_size;

    // Reshape: [B, 1, patch_h*patch_w, C*4] -> [B, 1, patch_h, patch_w, patch_size, patch_size, C]
    let reshaped = latents.reshape(&[batch, 1, patch_h, patch_w, patch_size, patch_size, channels])?;
    // Transpose to [B, C, 1, patch_h*patch_size, patch_w*patch_size]
    let permuted = reshaped.transpose(&[0, 6, 1, 2, 4, 3, 5])?;
    permuted.reshape(&[batch, channels, 1, latent_h, latent_w])
}

/// Encode a reference image through the VAE and pack into DiT patches.
/// image: [B, 4, H, W] (RGBA, values in [-1, 1])
/// Returns packed patches [B, (H/16)*(W/16), 64] ready for forward_edit.
pub fn encode_reference_latent(vae: &mut QwenVAE, image: &Array) -> Result<Array, Exception> {
    // VAE encode: [B, 4, H, W] -> [B, 16, 1, H/8, W/8]
    let latent = vae.encode(image)?;
    // Pack: [B, 16, 1, H/8, W/8] -> [B, (H/16)*(W/16), 64]
    pack_latents(&latent)
}

/// Compute the ref_shape in patchified space for a reference image.
/// latent_h, latent_w: VAE latent dimensions (image_dim / 8)
/// Returns (frame=1, patch_h, patch_w)
pub fn ref_shape_from_latent(latent_h: i32, latent_w: i32) -> (i32, i32, i32) {
    (1, latent_h / 2, latent_w / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = FlowMatchEulerScheduler::new(20, 1024);
        assert_eq!(scheduler.sigmas.len(), 20);
        assert_eq!(scheduler.timesteps.len(), 20);
        // First sigma should be 1.0
        assert!((scheduler.sigmas[0] - 1.0).abs() < 0.001);
        // Last sigma should be 0.02 (shift_terminal)
        assert!((scheduler.sigmas[19] - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let batch = 1;
        let channels = 16;
        let height = 8;
        let width = 8;
        let latent = Array::zeros::<f32>(&[batch, channels, 1, height, width]).unwrap();

        let packed = pack_latents(&latent).unwrap();
        let packed_h = (height as f32 / 2.0) as i32; // patch_size=2
        let packed_w = (width as f32 / 2.0) as i32;
        assert_eq!(packed.shape(), &[batch, packed_h * packed_w, channels * 4]);

        let unpacked = unpack_latents(&packed, height * 16, width * 16).unwrap();
        assert_eq!(unpacked.shape(), &[batch, channels, 1, height, width]);
    }

    #[test]
    fn test_attention_mask() {
        let mask = build_attention_mask(4, 2, 1).unwrap();
        assert_eq!(mask.shape(), &[1, 1, 6, 6]);
    }

    #[test]
    fn test_ref_shape() {
        let shape = ref_shape_from_latent(32, 32);
        assert_eq!(shape, (1, 16, 16));
    }
}
