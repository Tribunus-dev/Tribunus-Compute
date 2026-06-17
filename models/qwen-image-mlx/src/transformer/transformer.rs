//! Top-level Qwen-Image Transformer
//!
//! Reference: diffusers QwenImageTransformer2DModel
//! "Main transformer model for Qwen-Image generation"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder};
use mlx_rs::ops;
use mlx_rs::Array;

use super::block::QwenTransformerBlock;
use super::embeddings::QwenTimeTextEmbed;
use super::norm::QwenAdaLayerNormContinuous;
use super::rope::QwenEmbedRope;

/// Configuration for Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenTransformerConfig {
    pub patch_size: i32,
    pub in_channels: i32,
    pub num_layers: i32,
    pub attention_head_dim: i32,
    pub num_attention_heads: i32,
    pub caption_projection_dim: i32,
    pub pooled_projection_dim: i32,
    pub out_channels: i32,
    pub pos_embed_max_size: i32,
    pub axes_dimensions: [i32; 3],
    pub theta: i32,
}

impl Default for QwenTransformerConfig {
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            caption_projection_dim: 3584,
            pooled_projection_dim: 3584,
            out_channels: 64,
            pos_embed_max_size: 96,
            axes_dimensions: [16, 56, 56],
            theta: 10000,
        }
    }
}

impl QwenTransformerConfig {
    pub fn from_json(json: &serde_json::Value) -> Self {
        Self {
            patch_size: json["patch_size"].as_i64().unwrap_or(2) as i32,
            in_channels: json["in_channels"].as_i64().unwrap_or(64) as i32,
            num_layers: json["num_layers"].as_i64().unwrap_or(60) as i32,
            attention_head_dim: json["attention_head_dim"].as_i64().unwrap_or(128) as i32,
            num_attention_heads: json["num_attention_heads"].as_i64().unwrap_or(24) as i32,
            caption_projection_dim: json["caption_projection_dim"].as_i64().unwrap_or(3584) as i32,
            pooled_projection_dim: json["pooled_projection_dim"].as_i64().unwrap_or(3584) as i32,
            out_channels: json["out_channels"].as_i64().unwrap_or(64) as i32,
            pos_embed_max_size: json["pos_embed_max_size"].as_i64().unwrap_or(96) as i32,
            axes_dimensions: {
                let arr = json["axes_dim"].as_array()
                    .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(16) as i32).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![16, 56, 56]);
                [arr[0], arr[1], arr[2]]
            },
            theta: json["theta"].as_i64().unwrap_or(10000) as i32,
        }
    }
}

/// Qwen-Image Transformer (DiT-style)
/// Reference: diffusers QwenImageTransformer2DModel
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformer {
    pub config: QwenTransformerConfig,

    #[param]
    pub img_in: Linear,
    #[param]
    pub txt_in: Linear,
    #[param]
    pub time_text_embed: QwenTimeTextEmbed,
    #[param]
    pub pos_embedding: Array,
    #[param]
    pub blocks: Vec<QwenTransformerBlock>,
    #[param]
    pub norm_out: QwenAdaLayerNormContinuous,
    #[param]
    pub proj_out: Linear,
}

impl QwenTransformer {
    pub fn new(config: QwenTransformerConfig) -> Result<Self, Exception> {
        let in_channels = config.in_channels * config.patch_size * config.patch_size;
        let out_channels = config.out_channels * config.patch_size * config.patch_size;

        // Position embedding (learned positional encodings)
        let max_patches = config.pos_embed_max_size * config.pos_embed_max_size;
        let pos_embedding = Array::zeros::<f32>(&[1, max_patches, in_channels])?;

        // Input projections
        let img_in = LinearBuilder::new(in_channels, config.num_attention_heads * config.attention_head_dim)
            .bias(true).build()?;
        let txt_in = LinearBuilder::new(config.caption_projection_dim, config.num_attention_heads * config.attention_head_dim)
            .bias(true).build()?;

        // Time-text embedding
        let time_text_embed = QwenTimeTextEmbed::new(
            config.num_attention_heads * config.attention_head_dim,
            config.caption_projection_dim,
        )?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            blocks.push(QwenTransformerBlock::new(
                config.num_attention_heads * config.attention_head_dim,
                config.num_attention_heads,
                config.attention_head_dim,
            )?);
        }

        // Output
        let norm_out = QwenAdaLayerNormContinuous::new(
            config.num_attention_heads * config.attention_head_dim,
            config.num_attention_heads * config.attention_head_dim,
        )?;
        let proj_out = LinearBuilder::new(
            config.num_attention_heads * config.attention_head_dim,
            out_channels,
        ).bias(true).build()?;

        Ok(Self {
            config,
            img_in: Param::new(img_in),
            txt_in: Param::new(txt_in),
            time_text_embed: Param::new(time_text_embed),
            pos_embedding: Param::new(pos_embedding),
            blocks: Param::new(blocks),
            norm_out: Param::new(norm_out),
            proj_out: Param::new(proj_out),
        })
    }

    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        timestep: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
    ) -> Result<(Array, Array), Exception> {
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);

        // Image input projection + position embedding
        let img = self.img_in.forward(img)?;
        let pos_embed = self.pos_embedding.index(&[.., ..img_seq, ..])?;
        let img = ops::add(&img, &pos_embed)?;

        // Text input projection
        let txt = self.txt_in.forward(txt)?;

        // Time-text embedding
        let temb = self.time_text_embed.forward((timestep, txt))?;

        // Pass through transformer blocks
        let mut img_hidden = img;
        let mut txt_hidden = temb;

        for block in &mut self.blocks {
            let (img_out, txt_out) = block.forward(
                &img_hidden,
                &txt_hidden,
                &temb,
                img_rotary,
                txt_rotary,
                None,
            )?;
            img_hidden = img_out;
            txt_hidden = txt_out;
        }

        // Final norm and output projection
        let img_out = self.norm_out.forward(&img_hidden, &temb)?;
        let img_out = self.proj_out.forward(&img_out)?;

        Ok((img_out, txt_hidden))
    }
}

fn ops::add<'a>(a: &'a Array, b: &'a Array) -> Result<Array, Exception> {
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let config = QwenTransformerConfig::default();
        let transformer = QwenTransformer::new(config).unwrap();
        assert_eq!(transformer.blocks.len(), 60);
    }
}
