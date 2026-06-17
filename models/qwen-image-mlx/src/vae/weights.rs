//! VAE weight loading
//!
//! Loads weights from safetensors into the VAE decoder.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::Array;
use safetensors::SafeTensors;

use super::{QwenVAE, QwenImageDecoder3D};

/// Load VAE weights from safetensors file
pub fn load_vae_weights(
    vae: &mut QwenVAE,
    weights: &HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (name, weight) in weights {
        let mut mapped = name.to_string();

        // Map weight names to match our struct fields
        // The decoder has specific weight names from HuggingFace format
        mapped = mapped.replace("decoder.", "");

        // Map conv layers
        mapped = mapped.replace("conv_in.weight", "decoder.conv_in.weight");
        mapped = mapped.replace("conv_in.bias", "decoder.conv_in.bias");
        mapped = mapped.replace("conv_out.weight", "decoder.conv_out.weight");
        mapped = mapped.replace("conv_out.bias", "decoder.conv_out.bias");

        // Map block layers
        mapped = mapped.replace("mid_block.res_blocks.", "decoder.mid_block.res_blocks.");
        mapped = mapped.replace("mid_block.attn_blocks.", "decoder.mid_block.attn_blocks.");
        mapped = mapped.replace("up_blocks.", "decoder.up_blocks.");

        // Map norm outputs
        mapped = mapped.replace("conv_norm_out.weight", "decoder.conv_norm_out.weight");
        mapped = mapped.replace("conv_norm_out.bias", "decoder.conv_norm_out.bias");

        // Push into the VAE decoder
        vae.update_with_flattened(mapped, weight.clone())?;
    }

    Ok(())
}

fn load_conv3d_weights(
    conv: &mut super::QwenImageCausalConv3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        conv.weight = super::QwenImageCausalConv3D::extract_param(w.clone())?;
    }
    if let Some(w) = weights.get(&format!("{}.bias", prefix)) {
        conv.bias = super::QwenImageCausalConv3D::extract_param(Some(w.clone()))?;
    }
    Ok(())
}

fn load_rms_norm_weights(
    norm: &mut super::QwenImageRMSNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        norm.weight = super::QwenImageRMSNorm::extract_param(w.clone())?;
    }
    Ok(())
}

fn load_resblock_weights(
    block: &mut super::QwenImageResBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    load_conv3d_weights(&mut block.conv1, weights, &format!("{}.conv1", prefix))?;
    load_conv3d_weights(&mut block.conv2, weights, &format!("{}.conv2", prefix))?;
    if let Some(ref mut conv_shortcut) = block.conv_shortcut {
        load_conv3d_weights(conv_shortcut, weights, &format!("{}.conv_shortcut", prefix))?;
    }
    Ok(())
}

fn load_attention_weights(
    attn: &mut super::QwenImageAttentionBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    load_rms_norm_weights(&mut attn.norm, weights, &format!("{}.norm", prefix))?;
    Ok(())
}

fn load_midblock_weights(
    mid: &mut super::QwenImageMidBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, block) in mid.res_blocks.iter_mut().enumerate() {
        load_resblock_weights(block, weights, &format!("{}.res_blocks.{}", prefix, i))?;
    }
    for (i, block) in mid.attn_blocks.iter_mut().enumerate() {
        load_attention_weights(block, weights, &format!("{}.attn_blocks.{}", prefix, i))?;
    }
    Ok(())
}

fn load_resample_weights(
    resample: &mut super::QwenImageResample3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn load_upblock_weights(
    block: &mut super::QwenImageUpBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, res_block) in block.res_blocks.iter_mut().enumerate() {
        load_resblock_weights(res_block, weights, &format!("{}.res_blocks.{}", prefix, i))?;
    }
    Ok(())
}

fn load_decoder_weights(
    decoder: &mut QwenImageDecoder3D,
    weights: &HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    load_conv3d_weights(&mut decoder.conv_in, weights, "decoder.conv_in")?;
    load_midblock_weights(&mut decoder.mid_block, weights, "decoder.mid_block")?;
    for (i, block) in decoder.up_blocks.iter_mut().enumerate() {
        load_upblock_weights(block, weights, &format!("decoder.up_blocks.{}", i))?;
    }
    load_conv3d_weights(&mut decoder.conv_norm_out, weights, "decoder.conv_norm_out")?;
    load_conv3d_weights(&mut decoder.conv_out, weights, "decoder.conv_out")?;
    Ok(())
}

/// Load VAE from model directory.
/// Checks for `qwen_image_vae.safetensors` (edit model layout)
/// then falls back to `vae/0.safetensors` (text-to-image layout).
pub fn load_vae_from_dir(model_dir: impl AsRef<Path>) -> Result<QwenVAE, Box<dyn std::error::Error>> {
    let model_dir = model_dir.as_ref();

    // Check for single file in root (edit model layout)
    let root_vae = model_dir.join("qwen_image_vae.safetensors");
    if root_vae.exists() {
        let data = std::fs::read(&root_vae)?;
        let tensors = SafeTensors::deserialize(&data)?;
        let mut weights = HashMap::new();

        for (name, view) in tensors.tensors() {
            let data = view.into_data();
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => mlx_rs::Dtype::Float32,
                safetensors::Dtype::F16 => mlx_rs::Dtype::Float16,
                safetensors::Dtype::BF16 => mlx_rs::Dtype::BFloat16,
                _ => mlx_rs::Dtype::Float32,
            };
            let shape: Vec<i32> = view.shape().iter().map(|&s| s as i32).collect();
            let array = Array::from_slice_into_dtype(
                bytemuck::cast_slice(&data),
                &shape,
                dtype,
            )?;
            weights.insert(name, array);
        }

        let mut vae = QwenVAE::new()?;
        for (name, weight) in weights {
            vae.update_with_flattened(name, weight)?;
        }
        return Ok(vae);
    }

    // Check for vae/ directory
    let vae_dir = model_dir.join("vae");
    if vae_dir.exists() {
        let mut weights = HashMap::new();
        let vae_file = vae_dir.join("0.safetensors");
        if vae_file.exists() {
            let data = std::fs::read(&vae_file)?;
            let tensors = SafeTensors::deserialize(&data)?;
            for (name, view) in tensors.tensors() {
                let data = view.into_data();
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => mlx_rs::Dtype::Float32,
                    safetensors::Dtype::F16 => mlx_rs::Dtype::Float16,
                    safetensors::Dtype::BF16 => mlx_rs::Dtype::BFloat16,
                    _ => mlx_rs::Dtype::Float32,
                };
                let shape: Vec<i32> = view.shape().iter().map(|&s| s as i32).collect();
                let array = Array::from_slice_into_dtype(
                    bytemuck::cast_slice(&data),
                    &shape,
                    dtype,
                )?;
                weights.insert(name, array);
            }

            let mut vae = QwenVAE::new()?;
            for (name, weight) in weights {
                vae.update_with_flattened(name, weight)?;
            }
            return Ok(vae);
        }
    }

    Err(format!("VAE not found at {}", model_dir.display()).into())
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
}
