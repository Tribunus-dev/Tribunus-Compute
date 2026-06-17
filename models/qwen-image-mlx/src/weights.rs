//! Weight loading utilities for Qwen-Image
//!
//! Handles loading SafeTensors weights and mapping them to model structure

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;
use safetensors::SafeTensors;

use crate::error::QwenImageError;
use crate::transformer::{QwenTransformer, QwenTransformerConfig};
use crate::vae::QwenVAE;

/// Load SafeTensors file into a HashMap
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, Array>, QwenImageError> {
    let data = std::fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&data)?;
    let mut weights = HashMap::new();

    for (name, view) in tensors.tensors() {
        let data = view.into_data();
        let dtype = dtype_from_safetensors(&view);
        let shape: Vec<i32> = view.shape().iter().map(|&s| s as i32).collect();
        let array = Array::from_slice_into_dtype(
            bytemuck::cast_slice(&data),
            &shape,
            dtype,
        )?;
        weights.insert(name, array);
    }

    Ok(weights)
}

/// Load multiple SafeTensors shards
pub fn load_safetensors_shards<P: AsRef<Path>>(
    paths: &[P],
) -> Result<HashMap<String, Array>, QwenImageError> {
    let mut all_weights = HashMap::new();
    for path in paths {
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }
    Ok(all_weights)
}

fn dtype_from_safetensors(tensor: &safetensors::tensor::TensorView) -> mlx_rs::Dtype {
    use safetensors::Dtype;
    match tensor.dtype() {
        Dtype::F32 => mlx_rs::Dtype::Float32,
        Dtype::F16 => mlx_rs::Dtype::Float16,
        Dtype::BF16 => mlx_rs::Dtype::BFloat16,
        Dtype::I64 => mlx_rs::Dtype::Int64,
        Dtype::I32 => mlx_rs::Dtype::Int32,
        Dtype::I8 => mlx_rs::Dtype::Int8,
        Dtype::U8 => mlx_rs::Dtype::Uint8,
        _ => mlx_rs::Dtype::Float32,
    }
}

/// Weight name mapping for transformer
pub struct TransformerWeightMapper;

impl TransformerWeightMapper {
    /// Map HuggingFace weight name to mlx-rs weight name
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Image patch embedding
        name = name.replace("pos_embed.proj.weight", "img_in.weight");
        name = name.replace("pos_embed.proj.bias", "img_in.bias");

        // Position embedding
        name = name.replace("pos_embed.pos_embed", "pos_embedding");

        // Attention
        name = name.replace("attn.to_q.weight", "to_q.weight");
        name = name.replace("attn.to_k.weight", "to_k.weight");
        name = name.replace("attn.to_v.weight", "to_v.weight");
        name = name.replace("attn.to_out.0.weight", "attn_to_out.weight");
        name = name.replace("attn.add_q_proj.weight", "add_q_proj.weight");
        name = name.replace("attn.add_k_proj.weight", "add_k_proj.weight");
        name = name.replace("attn.add_v_proj.weight", "add_v_proj.weight");
        name = name.replace("attn.add_out.weight", "to_add_out.weight");
        name = name.replace("attn.norm_q.weight", "norm_q.weight");
        name = name.replace("attn.norm_k.weight", "norm_k.weight");
        name = name.replace("attn.norm_added_q.weight", "norm_added_q.weight");
        name = name.replace("attn.norm_added_k.weight", "norm_added_k.weight");

        // FeedForward
        name = name.replace("ff.net.0.proj.weight", "mlp_in.weight");
        name = name.replace("ff.net.0.proj.bias", "mlp_in.bias");
        name = name.replace("ff.net.2.weight", "mlp_out.weight");
        name = name.replace("ff.net.2.bias", "mlp_out.bias");

        // LayerNorm
        name = name.replace("norm1.norm.weight", "norm1.norm.weight");
        name = name.replace("norm1.norm.bias", "norm1.norm.bias");
        name = name.replace("norm1_context.norm.weight", "norm1_context.norm.weight");
        name = name.replace("norm1_context.norm.bias", "norm1_context.norm.bias");

        // TimeText embed
        name = name.replace("time_text_embed.timestep_embedder.linear_1.", "time_text_embed.timestep_embedder.linear_1.");
        name = name.replace("time_text_embed.timestep_embedder.linear_2.", "time_text_embed.timestep_embedder.linear_2.");
        name = name.replace("time_text_embed.text_embedder.", "time_text_embed.text_embedder.");

        return name;
    }
}

/// Weight name mapping for VAE
pub struct VAEWeightMapper;

impl VAEWeightMapper {
    /// Map HuggingFace VAE weight name to mlx-rs weight name
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Conv3d
        name = name.replace("conv_in.weight", "conv_in.weight");
        name = name.replace("conv_in.bias", "conv_in.bias");
        name = name.replace("conv_out.weight", "conv_out.weight");
        name = name.replace("conv_out.bias", "conv_out.bias");

        // Decoder conv
        name = name.replace("decoder.conv_in.weight", "decoder_conv_in.weight");

        return name;
    }
}

/// Convert HashMap<String, Array> to HashMap<Rc<str>, Array> for update_flattened
fn to_rc_keys(weights: HashMap<String, Array>) -> HashMap<std::rc::Rc<str>, Array> {
    weights
        .into_iter()
        .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
        .collect()
}

/// Load transformer from SafeTensors
pub fn load_transformer<P: AsRef<Path>>(
    path: P,
    config: QwenTransformerConfig,
) -> Result<QwenTransformer, QwenImageError> {
    let weights = load_safetensors(path)?;

    // Map weight names
    let mapped: HashMap<String, Array> = weights
        .into_iter()
        .map(|(k, v)| (TransformerWeightMapper::map_name(&k), v))
        .collect();

    let mut transformer = QwenTransformer::new(config)?;
    transformer.update(to_rc_keys(mapped))?;
    Ok(transformer)
}

/// Load VAE from SafeTensors
pub fn load_vae<P: AsRef<Path>>(path: P) -> Result<QwenVAE, QwenImageError> {
    let weights = load_safetensors(path)?;

    // Map weight names
    let mapped: HashMap<String, Array> = weights
        .into_iter()
        .map(|(k, v)| (VAEWeightMapper::map_name(&k), v))
        .collect();

    let mut vae = QwenVAE::new()?;
    vae.update(to_rc_keys(mapped))?;
    Ok(vae)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_weight_mapping() {
        let hf_name = "transformer_blocks.0.attn.to_q.weight";
        let mapped = TransformerWeightMapper::map_name(hf_name);
        assert_eq!(mapped, "transformer_blocks.0.attn.to_q.weight");

        let hf_name = "pos_embed.proj.weight";
        let mapped = TransformerWeightMapper::map_name(hf_name);
        assert_eq!(mapped, "img_in.weight");
    }

    #[test]
    fn test_vae_weight_mapping() {
        let hf_name = "decoder.conv_in.weight";
        let mapped = VAEWeightMapper::map_name(hf_name);
        assert_eq!(mapped, "decoder_conv_in.weight");
    }
}
