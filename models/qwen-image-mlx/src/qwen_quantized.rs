//! Quantized Qwen-Image Transformer
//!
//! Matches the weight structure of mlx-community/Qwen-Image-2512-4bit

use std::collections::HashMap;
use std::rc::Rc;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::fast;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{self, QuantizedLinear, QuantizedLinearBuilder, RmsNorm, RmsNormBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Dtype;
use mlx_rs::Array;

/// Configuration for Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub head_dim: i32,
    pub caption_projection_dim: i32,
    pub patch_size: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub pos_embed_max_size: i32,
    pub axes_dimensions: [i32; 3],
    pub theta: i32,
    pub quantized_bits: i32,
    pub quantized_group_size: i32,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3072,
            intermediate_size: 12288,
            num_attention_heads: 24,
            num_hidden_layers: 60,
            head_dim: 128,
            caption_projection_dim: 3584,
            patch_size: 2,
            in_channels: 64,
            out_channels: 64,
            pos_embed_max_size: 96,
            axes_dimensions: [16, 56, 56],
            theta: 10000,
            quantized_bits: 4,
            quantized_group_size: 64,
        }
    }
}

impl QwenConfig {
    pub fn from_hf_json(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_hf_json_str(&content)
    }

    pub fn from_hf_json_str(json_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json: serde_json::Value = serde_json::from_str(json_str)?;

        let hidden_size = json["hidden_size"].as_i64().unwrap_or(3072) as i32;
        let num_attention_heads = json["num_attention_heads"].as_i64().unwrap_or(24) as i32;
        let head_dim = json["head_dim"].as_i64().or_else(|| json["attention_head_dim"].as_i64()).unwrap_or(128) as i32;

        Ok(Self {
            hidden_size,
            intermediate_size: json["intermediate_size"].as_i64().unwrap_or(12288) as i32,
            num_attention_heads,
            num_hidden_layers: json["num_layers"].as_i64().or_else(|| json["num_hidden_layers"].as_i64()).unwrap_or(60) as i32,
            head_dim,
            caption_projection_dim: json["caption_projection_dim"].as_i64().unwrap_or(3584) as i32,
            patch_size: json["patch_size"].as_i64().unwrap_or(2) as i32,
            in_channels: json["in_channels"].as_i64().unwrap_or(64) as i32,
            out_channels: json["out_channels"].as_i64().unwrap_or(64) as i32,
            pos_embed_max_size: json["pos_embed_max_size"].as_i64().unwrap_or(96) as i32,
            axes_dimensions: {
                let arr = json["axes_dim"].as_array()
                    .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(16) as i32).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![16, 56, 56]);
                [arr[0], arr[1], arr[2]]
            },
            theta: json["theta"].as_i64().unwrap_or(10000) as i32,
            quantized_bits: json["quantization_bits"].as_i64().unwrap_or(4) as i32,
            quantized_group_size: json["quantization_group_size"].as_i64().unwrap_or(64) as i32,
        })
    }
}

/// Quantized Feed Forward network
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFeedForward {
    #[param]
    pub mlp_in: QuantizedLinear,
    #[param]
    pub mlp_out: QuantizedLinear,
}

impl QwenFeedForward {
    pub fn new(dim: i32, intermediate_dim: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        let mlp_in = QuantizedLinearBuilder::new(dim, intermediate_dim)
            .bits(bits)
            .group_size(group_size)
            .bias(true)
            .build()?;
        let mlp_out = QuantizedLinearBuilder::new(intermediate_dim, dim)
            .bits(bits)
            .group_size(group_size)
            .bias(true)
            .build()?;
        Ok(Self { mlp_in, mlp_out })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.mlp_in.forward(x)?;
        let hidden = nn::silu(&hidden)?;
        self.mlp_out.forward(&hidden)
    }
}

/// Quantized Attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAttention {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    #[param]
    pub to_q: QuantizedLinear,
    #[param]
    pub to_k: QuantizedLinear,
    #[param]
    pub to_v: QuantizedLinear,

    #[param]
    pub add_q_proj: QuantizedLinear,
    #[param]
    pub add_k_proj: QuantizedLinear,
    #[param]
    pub add_v_proj: QuantizedLinear,

    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
    #[param]
    pub norm_added_q: RmsNorm,
    #[param]
    pub norm_added_k: RmsNorm,

    #[param]
    pub attn_to_out: QuantizedLinear,
    #[param]
    pub to_add_out: QuantizedLinear,
}

impl QwenAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        let total_dim = num_heads * head_dim;
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            to_q: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            to_k: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            to_v: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            add_q_proj: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            add_k_proj: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            add_v_proj: QuantizedLinearBuilder::new(dim, total_dim).bits(bits).group_size(group_size).bias(true).build()?,
            norm_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            attn_to_out: QuantizedLinearBuilder::new(total_dim, dim).bits(bits).group_size(group_size).bias(true).build()?,
            to_add_out: QuantizedLinearBuilder::new(total_dim, dim).bits(bits).group_size(group_size).bias(true).build()?,
        })
    }

    pub fn forward(
        &mut self,
        img_hidden: &Array,
        txt_hidden: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
    ) -> Result<(Array, Array), Exception> {
        let batch = img_hidden.dim(0);
        let img_seq = img_hidden.dim(1);
        let txt_seq = txt_hidden.dim(1);

        // Project Q, K, V for image
        let img_q = self.to_q.forward(img_hidden)?;
        let img_k = self.to_k.forward(img_hidden)?;
        let img_v = self.to_v.forward(img_hidden)?;

        // Project Q, K, V for text
        let txt_q = self.add_q_proj.forward(txt_hidden)?;
        let txt_k = self.add_k_proj.forward(txt_hidden)?;
        let txt_v = self.add_v_proj.forward(txt_hidden)?;

        // Reshape for multi-head
        let img_q = img_q.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_k = img_k.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_v = img_v.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;

        let txt_q = txt_q.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_k = txt_k.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_v = txt_v.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;

        // Apply RoPE
        let (img_cos, img_sin) = img_rotary;
        let (txt_cos, txt_sin) = txt_rotary;

        let img_q = apply_rope_qwen(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope_qwen(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope_qwen(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope_qwen(&txt_k, txt_cos, txt_sin)?;

        // QK Norm
        let img_q = self.norm_q.forward(&img_q)?;
        let img_k = self.norm_k.forward(&img_k)?;
        let txt_q = self.norm_added_q.forward(&txt_q)?;
        let txt_k = self.norm_added_k.forward(&txt_k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let img_q = img_q.transpose(&[0, 2, 1, 3])?;
        let img_k = img_k.transpose(&[0, 2, 1, 3])?;
        let img_v = img_v.transpose(&[0, 2, 1, 3])?;
        let txt_q = txt_q.transpose(&[0, 2, 1, 3])?;
        let txt_k = txt_k.transpose(&[0, 2, 1, 3])?;
        let txt_v = txt_v.transpose(&[0, 2, 1, 3])?;

        // Concatenate image and text for joint attention
        let q = ops::concatenate_axis(&[&img_q, &txt_q], 2)?;
        let k = ops::concatenate_axis(&[&img_k, &txt_k], 2)?;
        let v = ops::concatenate_axis(&[&img_v, &txt_v], 2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = fast::scaled_dot_product_attention(&q, &k, &v, Some(scale), None)?;

        // Split back: first img_seq tokens are image, rest are text
        let img_out = attn_out.index(&[.., .., ..img_seq, ..])?;
        let txt_out = attn_out.index(&[.., .., img_seq.., ..])?;

        // Transpose back and project
        let img_out = img_out.transpose(&[0, 2, 1, 3])?;
        let img_out = img_out.reshape(&[batch, img_seq, self.num_heads * self.head_dim])?;
        let img_out = self.attn_to_out.forward(&img_out)?;

        let txt_out = txt_out.transpose(&[0, 2, 1, 3])?;
        let txt_out = txt_out.reshape(&[batch, txt_seq, self.num_heads * self.head_dim])?;
        let txt_out = self.to_add_out.forward(&txt_out)?;

        Ok((img_out, txt_out))
    }
}

/// Quantized Transformer Block
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerBlock {
    pub dim: i32,

    #[param]
    pub norm1: QwenAdaLayerNorm,
    #[param]
    pub norm1_context: QwenAdaLayerNorm,
    #[param]
    pub attn: QwenAttention,
    #[param]
    pub ff: QwenFeedForward,
    #[param]
    pub ff_context: QwenFeedForward,
}

impl QwenTransformerBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32, intermediate_size: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            norm1: QwenAdaLayerNorm::new(dim, bits, group_size)?,
            norm1_context: QwenAdaLayerNorm::new(dim, bits, group_size)?,
            attn: QwenAttention::new(dim, num_heads, head_dim, bits, group_size)?,
            ff: QwenFeedForward::new(dim, intermediate_size, bits, group_size)?,
            ff_context: QwenFeedForward::new(dim, intermediate_size, bits, group_size)?,
        })
    }

    pub fn forward(
        &mut self,
        img_hidden: &Array,
        txt_hidden: &Array,
        temb: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
    ) -> Result<(Array, Array), Exception> {
        // Image modulation
        let (img_modulated, img_gate, img_mod2) = self.norm1.forward(img_hidden, temb)?;

        // Text modulation
        let (txt_modulated, txt_gate, txt_mod2) = self.norm1_context.forward(txt_hidden, temb)?;

        // Joint attention
        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_rotary,
            txt_rotary,
        )?;

        // Gate + residual
        let img_hidden = ops::add(img_hidden, &ops::multiply(&img_gate, &img_attn)?)?;
        let txt_hidden = ops::add(txt_hidden, &ops::multiply(&txt_gate, &txt_attn)?)?;

        // Apply modulation for MLP
        let mod2_params = &img_mod2;
        let shift2 = mod2_params.index(&[.., ..self.dim])?;
        let scale2 = mod2_params.index(&[.., self.dim..2*self.dim])?;
        let gate2 = mod2_params.index(&[.., 2*self.dim..])?;

        let img_modulated_ff = modulate(&img_hidden, &shift2, &scale2)?;
        let img_ffn = self.ff.forward(&img_modulated_ff)?;
        let img_hidden = ops::add(&img_hidden, &ops::multiply(&gate2, &img_ffn)?)?;

        // Text FFN
        let mod2_ctx = &txt_mod2;
        let shift2_ctx = mod2_ctx.index(&[.., ..self.dim])?;
        let scale2_ctx = mod2_ctx.index(&[.., self.dim..2*self.dim])?;
        let gate2_ctx = mod2_ctx.index(&[.., 2*self.dim..])?;

        let txt_modulated_ff = modulate(&txt_hidden, &shift2_ctx, &scale2_ctx)?;
        let txt_ffn = self.ff_context.forward(&txt_modulated_ff)?;
        let txt_hidden = ops::add(&txt_hidden, &ops::multiply(&gate2_ctx, &txt_ffn)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

/// Timestep Embedder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimestepEmbedder {
    #[param]
    pub linear_1: QuantizedLinear,
    #[param]
    pub linear_2: QuantizedLinear,
}

impl QwenTimestepEmbedder {
    pub fn new(dim: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: QuantizedLinearBuilder::new(256, dim).bits(bits).group_size(group_size).bias(true).build()?,
            linear_2: QuantizedLinearBuilder::new(dim, dim).bits(bits).group_size(group_size).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        let t = ops::multiply(t, &Array::from_f32(1000.0))?;
        let half_dim = 128i32;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
            .collect();
        let freqs = Array::from_slice(&freqs, &[1, half_dim]);

        let t_expanded = t.reshape(&[-1, 1])?;
        let args = ops::multiply(&t_expanded, &freqs)?;

        let cos = ops::cos(&args)?;
        let sin = ops::sin(&args)?;
        let emb = ops::concatenate_axis(&[&cos, &sin], -1)?;

        let h = nn::silu(&self.linear_1.forward(&emb)?)?;
        self.linear_2.forward(&h)
    }
}

/// Time-Text Embed
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimeTextEmbed {
    #[param]
    pub timestep_embedder: QwenTimestepEmbedder,
    #[param]
    pub text_embedder: QuantizedLinear,
}

impl QwenTimeTextEmbed {
    pub fn new(hidden_size: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            timestep_embedder: QwenTimestepEmbedder::new(hidden_size, bits, group_size)?,
            text_embedder: QuantizedLinearBuilder::new(hidden_size, hidden_size).bits(bits).group_size(group_size).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array, txt_emb: &Array) -> Result<Array, Exception> {
        let t_emb = self.timestep_embedder.forward(t)?;
        let txt_emb = self.text_embedder.forward(txt_emb)?;
        ops::add(&t_emb, &txt_emb)
    }
}

/// AdaLayerNorm for output
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAdaLayerNormOut {
    pub dim: i32,
    #[param]
    pub linear: QuantizedLinear,
}

impl QwenAdaLayerNormOut {
    pub fn new(dim: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            linear: QuantizedLinearBuilder::new(dim, 2 * dim).bits(bits).group_size(group_size).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<(Array, Array), Exception> {
        // Upcast to FP32 for numerical stability before final LayerNorm
        let input_dtype = x.dtype();
        let x = x.as_dtype(Dtype::Float32)?;
        let temb = temb.as_dtype(Dtype::Float32)?;

        let cond = self.linear.forward(&temb)?;
        let shift = cond.index(&[.., ..self.dim])?;
        let scale = cond.index(&[.., self.dim..])?;

        let gate = ops::mean_axis(&scale, -1, true)?;

        let normed = layer_norm(&x, 1e-6)?;
        let one = Array::from_f32(1.0);
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale)?)?,
            &shift,
        )?;

        let modulated = modulated.as_dtype(input_dtype)?;
        Ok((modulated, gate))
    }
}

/// RMS Norm for text input
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerRMSNorm {
    #[param]
    pub weight: Array,
    pub eps: f32,
}

impl QwenTransformerRMSNorm {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Array::ones::<f32>(&[dim])?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        fast::rms_norm(x, &self.weight, self.eps)
    }
}

/// Main Quantized Transformer
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenQuantizedTransformer {
    pub config: QwenConfig,

    #[param]
    pub img_in: QuantizedLinear,
    #[param]
    pub txt_in: QuantizedLinear,
    #[param]
    pub time_text_embed: QwenTimestepEmbedder,
    #[param]
    pub pos_embedding: Array,
    #[param]
    pub txt_norm: Option<RmsNorm>,
    #[param]
    pub blocks: Vec<QwenTransformerBlock>,
    #[param]
    pub norm_out: QwenAdaLayerNormOut,
    #[param]
    pub proj_out: QuantizedLinear,

    // RoPE pre-computation (optional — computed once or per-forward)
    pub img_rotary: Option<(Array, Array)>,
    pub txt_rotary: Option<(Array, Array)>,
}

impl QwenQuantizedTransformer {
    pub fn new(config: QwenConfig) -> Result<Self, Exception> {
        let in_channels = config.in_channels * config.patch_size * config.patch_size;
        let out_channels = config.out_channels * config.patch_size * config.patch_size;
        let max_patches = config.pos_embed_max_size * config.pos_embed_max_size;
        let bits = config.quantized_bits;
        let group_size = config.quantized_group_size;

        let img_in = QuantizedLinearBuilder::new(in_channels, config.hidden_size)
            .bits(bits).group_size(group_size).bias(true).build()?;
        let txt_in = QuantizedLinearBuilder::new(config.caption_projection_dim, config.hidden_size)
            .bits(bits).group_size(group_size).bias(true).build()?;
        let time_text_embed = QwenTimestepEmbedder::new(config.hidden_size, bits, group_size)?;

        // Position embedding (learnt)
        let pos_embedding = Array::zeros::<f32>(&[1, max_patches, in_channels])?;

        // Text RMS norm (optional - some models have it)
        let txt_norm = None;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            blocks.push(QwenTransformerBlock::new(
                config.hidden_size,
                config.num_attention_heads,
                config.head_dim,
                config.intermediate_size,
                bits,
                group_size,
            )?);
        }

        // Output
        let norm_out = QwenAdaLayerNormOut::new(config.hidden_size, bits, group_size)?;
        let proj_out = QuantizedLinearBuilder::new(config.hidden_size, out_channels)
            .bits(bits).group_size(group_size).bias(true).build()?;

        Ok(Self {
            config,
            img_in: Param::new(img_in),
            txt_in: Param::new(txt_in),
            time_text_embed: Param::new(time_text_embed),
            pos_embedding: Param::new(pos_embedding),
            txt_norm: Param::new(txt_norm),
            blocks: Param::new(blocks),
            norm_out: Param::new(norm_out),
            proj_out: Param::new(proj_out),
            img_rotary: None,
            txt_rotary: None,
        })
    }

    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        timestep: &Array,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.dim(0);
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);

        // Image input projection + position embedding
        let img = self.img_in.forward(img)?;
        let pos_embed = self.pos_embedding.index(&[.., ..img_seq, ..])?;
        let img = ops::add(&img, &pos_embed)?;

        // Text input projection
        let txt = self.txt_in.forward(txt)?;
        let txt = if let Some(ref mut txt_norm) = self.txt_norm {
            txt_norm.forward(&txt)?
        } else {
            txt
        };

        // Time-text embedding
        let temb = nn::silu(&self.time_text_embed.forward(timestep)?)?;

        // Get RoPE from pre-computed or compute here
        let img_rotary = self.img_rotary.as_ref()
            .map(|r| (r.0.clone(), r.1.clone()))
            .unwrap_or_else(|| (Array::zeros::<f32>(&[1, img_seq, 1, self.config.head_dim]).unwrap(),
                               Array::zeros::<f32>(&[1, img_seq, 1, self.config.head_dim]).unwrap()));

        let txt_rotary = self.txt_rotary.as_ref()
            .map(|r| (r.0.clone(), r.1.clone()))
            .unwrap_or_else(|| (Array::zeros::<f32>(&[1, txt_seq, 1, self.config.head_dim]).unwrap(),
                               Array::zeros::<f32>(&[1, txt_seq, 1, self.config.head_dim]).unwrap()));

        // Pass through transformer blocks
        let mut img_hidden = img;
        let mut txt_hidden = txt;

        for block in &mut self.blocks {
            let (img_out, txt_out) = block.forward(
                &img_hidden,
                &txt_hidden,
                &temb,
                &img_rotary,
                &txt_rotary,
            )?;
            img_hidden = img_out;
            txt_hidden = txt_out;
        }

        // Final norm and output projection
        let (img_out_norm, _img_gate_out) = self.norm_out.forward(&img_hidden, &temb)?;
        let img_out = self.proj_out.forward(&img_out_norm)?;

        Ok((img_out, txt_hidden))
    }

    /// Forward pass for edit mode with reference images
    /// Returns (img_out, txt_hidden)
    pub fn forward_edit(
        &mut self,
        img: &Array,
        txt: &Array,
        timestep: &Array,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.dim(0);
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);

        // Image input projection + position embedding
        let img = self.img_in.forward(img)?;
        let pos_embed = self.pos_embedding.index(&[.., ..img_seq, ..])?;
        let img = ops::add(&img, &pos_embed)?;

        // Text input projection
        let txt = self.txt_in.forward(txt)?;
        let txt = if let Some(ref mut txt_norm) = self.txt_norm {
            txt_norm.forward(&txt)?
        } else {
            txt
        };

        // Time-text embedding
        let temb = nn::silu(&self.time_text_embed.forward(timestep)?)?;

        // Get RoPE
        let img_rotary = self.img_rotary.as_ref()
            .map(|r| (r.0.clone(), r.1.clone()))
            .unwrap_or_else(|| (Array::zeros::<f32>(&[1, img_seq, 1, self.config.head_dim]).unwrap(),
                               Array::zeros::<f32>(&[1, img_seq, 1, self.config.head_dim]).unwrap()));

        let txt_rotary = self.txt_rotary.as_ref()
            .map(|r| (r.0.clone(), r.1.clone()))
            .unwrap_or_else(|| (Array::zeros::<f32>(&[1, txt_seq, 1, self.config.head_dim]).unwrap(),
                               Array::zeros::<f32>(&[1, txt_seq, 1, self.config.head_dim]).unwrap()));

        // Pass through transformer blocks
        let mut img_hidden = img;
        let mut txt_hidden = txt;

        for block in &mut self.blocks {
            let (img_out, txt_out) = block.forward(
                &img_hidden,
                &txt_hidden,
                &temb,
                &img_rotary,
                &txt_rotary,
            )?;
            img_hidden = img_out;
            txt_hidden = txt_out;
        }

        // Final norm and output projection
        let (img_out_norm, _img_gate_out) = self.norm_out.forward(&img_hidden, &temb)?;
        let img_out = self.proj_out.forward(&img_out_norm)?;

        Ok((img_out, txt_hidden))
    }
}

// Helper functions

fn split_half(x: &Array) -> Result<(Array, Array), Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.index(&[.., .., .., ..half])?;
    let x2 = x.index(&[.., .., .., half..])?;
    Ok((x1, x2))
}

/// LayerNorm without learnable weights (for pre-modulation normalization).
/// Computes in float32 for numerical stability, returns in input dtype.
fn layer_norm(x: &Array, eps: f32) -> Result<Array, Exception> {
    let input_dtype = x.dtype();
    let x = x.as_dtype(Dtype::Float32)?;
    let mean = ops::mean_axis(&x, -1, true)?;
    let x_centered = ops::subtract(&x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let eps_arr = Array::from_f32(eps);
    let result = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps_arr)?)?)?;
    result.as_dtype(input_dtype)
}

/// Clip values to prevent numerical explosion (like FLUX-klein's ±65504)
fn clip_values(x: &Array) -> Result<Array, Exception> {
    let max_val = Array::from_f32(65504.0);
    let min_val = Array::from_f32(-65504.0);
    ops::clip(x, &min_val, &max_val)
}

fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let eps = Array::from_f32(1e-6);
    let one = Array::from_f32(1.0);

    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps)?)?)?;

    let scaled = ops::multiply(&normalized, &ops::add(&one, scale)?)?;
    ops::add(&scaled, shift)
}

fn apply_rope_qwen(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let (x1, x2) = split_half(x)?;
    let rotated1 = ops::subtract(&ops::multiply(&x1, cos)?, &ops::multiply(&x2, sin)?)?;
    let rotated2 = ops::add(&ops::multiply(&x2, cos)?, &ops::multiply(&x1, sin)?)?;
    ops::concatenate_axis(&[&rotated1, &rotated2], -1)
}

// ─── Edit mode helpers ───────────────────────────────────────────────────────

/// Blend image modulation parameters per-token using modulate_index.
/// img_mod_params: [2, 6*dim] (row 0 = real, row 1 = zero)
/// modulate_index: [total_img_seq] (0.0 for main, 1.0 for ref)
/// Returns 6 arrays each [1, total_img_seq, dim]: shift1, scale1, gate1, shift2, scale2, gate2
fn prepare_img_mod_edit(
    img_mod_params: &Array,
    modulate_index: &Array,
    dim: i32,
) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
    // img_mod_params: [2, 6*dim]
    let main_mod = img_mod_params.index(&[0, ..])?; // [6*dim]
    let zero_mod = img_mod_params.index(&[1, ..])?; // [6*dim] (all zeros)

    // Reshape modulate_index for broadcasting: [total_img_seq] -> [1, total_img_seq, 1]
    let total_img_seq = modulate_index.dim(0);
    let idx = modulate_index.reshape(&[1, total_img_seq, 1])?;

    // For each param group, blend: (1 - idx) * main + idx * zero
    // Since zero is all zeros, this simplifies to: (1 - idx) * main
    let one = Array::from_f32(1.0);
    let blend_factor = ops::subtract(&one, &idx)?; // [1, total_img_seq, 1]

    let shift1 = main_mod.index(&[....dim])?;
    let scale1 = main_mod.index(&[dim..2*dim])?;
    let gate1 = main_mod.index(&[2*dim..3*dim])?;
    let shift2 = main_mod.index(&[3*dim..4*dim])?;
    let scale2 = main_mod.index(&[4*dim..5*dim])?;
    let gate2 = main_mod.index(&[5*dim..])?;

    // Reshape each to [1, 1, dim] for broadcasting
    let shift1 = shift1.reshape(&[1, 1, dim])?;
    let scale1 = scale1.reshape(&[1, 1, dim])?;
    let gate1 = gate1.reshape(&[1, 1, dim])?;
    let shift2 = shift2.reshape(&[1, 1, dim])?;
    let scale2 = scale2.reshape(&[1, 1, dim])?;
    let gate2 = gate2.reshape(&[1, 1, dim])?;

    // Apply blend factor: (1 - idx) * param
    let shift1 = ops::multiply(&blend_factor, &shift1)?;
    let scale1 = ops::multiply(&blend_factor, &scale1)?;
    let gate1 = ops::multiply(&blend_factor, &gate1)?;
    let shift2 = ops::multiply(&blend_factor, &shift2)?;
    let scale2 = ops::multiply(&blend_factor, &scale2)?;
    let gate2 = ops::multiply(&blend_factor, &gate2)?;

    Ok((shift1, scale1, gate1, shift2, scale2, gate2))
}

/// Apply modulation with 3D shift/scale (edit mode: per-token blending)
fn modulate_flex(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let eps = Array::from_f32(1e-6);
    let one = Array::from_f32(1.0);

    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps)?)?)?;

    let scaled = ops::multiply(&normalized, &ops::add(&one, scale)?)?;
    ops::add(&scaled, shift)
}

/// Apply gating with 3D gate (edit mode: per-token)
fn gate_flex(gate: &Array, y: &Array) -> Result<Array, Exception> {
    ops::multiply(gate, y)
}

/// Apply modulation with 2D shift/scale (standard: broadcast over seq)
fn modulate_2d(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let eps = Array::from_f32(1e-6);
    let one = Array::from_f32(1.0);

    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps)?)?)?;

    let scaled = ops::multiply(&normalized, &ops::add(&one, scale)?)?;
    ops::add(&scaled, shift)
}

/// Centered position indices: [-ceil(n/2), ..., -1, 0, 1, ..., floor(n/2)-1]
fn centered_positions_vec(length: i32) -> Vec<f32> {
    let half = length / 2;
    let is_odd = length % 2 == 1;
    let mut positions = Vec::with_capacity(length as usize);
    if is_odd {
        for i in -half..=half {
            positions.push(i as f32);
        }
    } else {
        for i in -half..half {
            positions.push(i as f32);
        }
    }
    positions
}

/// 1D RoPE frequencies: positions -> (cos, sin), each [seq, dim/2]
fn rope_frequencies_1d(positions: &Array, dim: i32, theta: f32) -> Result<(Array, Array), Exception> {
    let half_dim = dim / 2;
    let inv_freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powi(i * 2))
        .collect();
    let inv_freqs = Array::from_slice(&inv_freqs, &[half_dim]);

    let args = ops::matmul(positions, &inv_freqs.reshape(&[half_dim, 1])?)?;
    Ok((ops::cos(&args)?, ops::sin(&args)?))
}

/// Build 3-axis RoPE for image-edit: main image + reference images + text.
/// img_shape: (frame, patch_h, patch_w) in patchified space
/// ref_shapes: per-ref (frame, patch_h, patch_w)
/// Returns ((img_cos, img_sin), (txt_cos, txt_sin))
pub fn build_edit_rope(
    img_shape: (i32, i32, i32),
    ref_shapes: &[(i32, i32, i32)],
    txt_seq_len: i32,
    theta: f32,
    axes_dims: [i32; 3],
) -> Result<((Array, Array), (Array, Array)), Exception> {
    let (_frame, patch_h, patch_w) = img_shape;
    let total_img_patches = patch_h * patch_w;
    let total_ref_patches: i32 = ref_shapes.iter().map(|(_, h, w)| h * w).sum();
    let total_img_seq = total_img_patches + total_ref_patches;

    // Build position indices for each axis
    let h_positions = centered_positions_vec(patch_h);
    let w_positions = centered_positions_vec(patch_w);

    // For each patch, compute the 3D position
    let mut img_cos_partial = Vec::new();
    let mut img_sin_partial = Vec::new();

    for h in 0..patch_h {
        for w in 0..patch_w {
            // One position per axis per patch
            let f_pos = 0.0; // No frame dimension for images
            let h_pos = h_positions[h as usize];
            let w_pos = w_positions[w as usize];

            // Compute per-axis frequencies and accumulate
            let f_cos_sin = rope_frequencies_1d(
                &Array::from_slice(&[f_pos], &[1, 1]),
                axes_dims[0],
                theta,
            )?;
            let h_cos_sin = rope_frequencies_1d(
                &Array::from_slice(&[h_pos], &[1, 1]),
                axes_dims[1],
                theta,
            )?;
            let w_cos_sin = rope_frequencies_1d(
                &Array::from_slice(&[w_pos], &[1, 1]),
                axes_dims[2],
                theta,
            )?;

            let cos = ops::concatenate_axis(&[&f_cos_sin.0, &h_cos_sin.0, &w_cos_sin.0], -1)?;
            let sin = ops::concatenate_axis(&[&f_cos_sin.1, &h_cos_sin.1, &w_cos_sin.1], -1)?;
            img_cos_partial.push(cos);
            img_sin_partial.push(sin);
        }
    }

    // Concatenate all patch frequencies
    let img_cos = ops::concatenate_axis(&img_cos_partial.iter().collect::<Vec<_>>(), 0)?;
    let img_sin = ops::concatenate_axis(&img_sin_partial.iter().collect::<Vec<_>>(), 0)?;

    // Reshape to [1, total_img_seq, 1, total_dim]
    let total_dim: i32 = axes_dims.iter().sum();
    let img_cos = img_cos.reshape(&[1, total_img_seq, 1, total_dim])?;
    let img_sin = img_sin.reshape(&[1, total_img_seq, 1, total_dim])?;

    // Text RoPE
    let txt_positions: Vec<f32> = (0..txt_seq_len).map(|i| i as f32).collect();
    let txt_pos = Array::from_slice(&txt_positions, &[txt_seq_len, 1]);

    let (txt_cos, txt_sin) = rope_frequencies_1d(&txt_pos, total_dim, theta)?;
    let txt_cos = txt_cos.reshape(&[1, txt_seq_len, 1, total_dim])?;
    let txt_sin = txt_sin.reshape(&[1, txt_seq_len, 1, total_dim])?;

    Ok(((img_cos, img_sin), (txt_cos, txt_sin)))
}

fn get_timestep_embedding(t: &Array, dim: i32) -> Result<Array, Exception> {
    let half_dim = dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
        .collect();
    let freqs = Array::from_slice(&freqs, &[1, half_dim]);

    let t = ops::multiply(t, &Array::from_f32(1000.0))?;
    let t_expanded = t.reshape(&[-1, 1])?;
    let args = ops::multiply(&t_expanded, &freqs)?;

    let cos = ops::cos(&args)?;
    let sin = ops::sin(&args)?;
    ops::concatenate_axis(&[&cos, &sin], -1)
}

// AdaLayerNorm
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAdaLayerNorm {
    pub dim: i32,
    #[param]
    pub linear: QuantizedLinear,
}

impl QwenAdaLayerNorm {
    pub fn new(dim: i32, bits: i32, group_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            linear: QuantizedLinearBuilder::new(dim, 6 * dim).bits(bits).group_size(group_size).bias(true).build()?,
        })
    }

    pub fn forward(&self, hidden_states: &Array, temb: &Array) -> Result<(Array, Array, Array), Exception> {
        let cond = self.linear.forward(temb)?;

        let shift1 = cond.index(&[.., ..self.dim])?;
        let scale1 = cond.index(&[.., self.dim..2*self.dim])?;
        let gate1 = cond.index(&[.., 2*self.dim..3*self.dim])?;
        let shift2 = cond.index(&[.., 3*self.dim..4*self.dim])?;
        let scale2 = cond.index(&[.., 4*self.dim..5*self.dim])?;
        let gate2 = cond.index(&[.., 5*self.dim..])?;

        let modulated = modulate(hidden_states, &shift1, &scale1)?;

        let mod2 = ops::concatenate_axis(&[&shift2, &scale2, &gate2], -1)?;

        Ok((modulated, gate1, mod2))
    }
}

/// Load weights from HashMap into the model
///
/// Transforms weight keys to match mlx-rs QuantizedLinear structure:
/// - `xxx.weight` -> `xxx.inner.weight` (for quantized weights)
/// - `xxx.bias` -> `xxx.inner.bias` (for quantized linear output bias)
/// (but keeps `xxx.scales` and `xxx.biases` as-is since they match)
pub fn load_transformer_weights(
    model: &mut QwenQuantizedTransformer,
    weights: HashMap<String, Array>,
) -> Result<(), Exception> {
    for (name, weight) in weights {
        let mut mapped = name.to_string();

        // Map transformer block names
        if let Some(rest) = mapped.strip_prefix("transformer_blocks.") {
            mapped = format!("blocks.{}", rest);
        }

        // Map attention layers
        mapped = mapped.replace("attn.to_q", "attn.to_q.inner");
        mapped = mapped.replace("attn.to_k", "attn.to_k.inner");
        mapped = mapped.replace("attn.to_v", "attn.to_v.inner");
        mapped = mapped.replace("attn.add_q_proj", "attn.add_q_proj.inner");
        mapped = mapped.replace("attn.add_k_proj", "attn.add_k_proj.inner");
        mapped = mapped.replace("attn.add_v_proj", "attn.add_v_proj.inner");
        mapped = mapped.replace("attn.to_out.0", "attn.attn_to_out.inner");
        mapped = mapped.replace("attn.add_out", "attn.to_add_out.inner");

        mapped = mapped.replace("ff.net.0.proj", "ff.mlp_in.inner");
        mapped = mapped.replace("ff.net.2", "ff.mlp_out.inner");
        mapped = mapped.replace("ff_context.net.0.proj", "ff_context.mlp_in.inner");
        mapped = mapped.replace("ff_context.net.2", "ff_context.mlp_out.inner");

        // Map output norm
        mapped = mapped.replace("norm_out.linear", "norm_out.linear.inner");

        // Map input layers
        mapped = mapped.replace("pos_embed.proj", "img_in.inner");
        mapped = mapped.replace("caption_proj", "txt_in.inner");
        mapped = mapped.replace("proj_out", "proj_out.inner");

        // Map time_text_embed layers
        mapped = mapped.replace("time_text_embed.timestep_embedder.linear_1", "time_text_embed.linear_1.inner");
        mapped = mapped.replace("time_text_embed.timestep_embedder.linear_2", "time_text_embed.linear_2.inner");

        // Map AdaLayerNorm
        mapped = mapped.replace("norm1.linear", "norm1.linear.inner");
        mapped = mapped.replace("norm1_context.linear", "norm1_context.linear.inner");

        // Keep position embedding as-is
        mapped = mapped.replace("pos_embed.pos_embedding", "pos_embedding");

        model.update_with_flattened(mapped, weight)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let config = QwenConfig::default();
        let transformer = QwenQuantizedTransformer::new(config).unwrap();
        assert_eq!(transformer.blocks.len(), 60);
    }

    #[test]
    fn test_config_from_json() {
        let json = r#"{
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_layers": 60,
            "patch_size": 2,
            "in_channels": 64
        }"#;
        let config = QwenConfig::from_hf_json_str(json).unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 60);
    }

    #[test]
    fn test_split_half() {
        let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let (x1, x2) = split_half(&x).unwrap();
        assert_eq!(x1.shape(), &[1, 1, 1, 2]);
        assert_eq!(x2.shape(), &[1, 1, 1, 2]);
    }

    #[test]
    fn test_centered_positions() {
        let pos = centered_positions_vec(4);
        assert_eq!(pos, vec![-2.0, -1.0, 0.0, 1.0]);
        let pos = centered_positions_vec(5);
        assert_eq!(pos, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_build_edit_rope() {
        let ((img_cos, img_sin), (txt_cos, txt_sin)) = build_edit_rope(
            (1, 4, 4),
            &[(1, 2, 2)],
            10,
            10000.0,
            [16, 56, 56],
        ).unwrap();
        // img_seq = 16 + 4 = 20
        assert_eq!(img_cos.shape(), &[1, 20, 1, 128]);
        assert_eq!(img_sin.shape(), &[1, 20, 1, 128]);
        assert_eq!(txt_cos.shape(), &[1, 10, 1, 128]);
        assert_eq!(txt_sin.shape(), &[1, 10, 1, 128]);
    }
}
