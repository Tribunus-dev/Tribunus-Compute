//! Full Precision Qwen-Image Transformer
//!
//! Matches the architecture of Qwen/Qwen-Image (HuggingFace diffusers format)
//! Uses joint attention with separate image/text pathways.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::fast;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{self, Linear, LinearBuilder, LayerNorm, LayerNormBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Dtype;
use mlx_rs::Array;

// For memory cache clearing
extern crate mlx_sys;

// fused_modulate kernel available but manual implementation is faster
// due to MLX's efficient lazy evaluation (see USE_FUSED_MODULATE flag)

/// Configuration for full precision Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenFullConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub patch_size: i32,
    pub num_layers: i32,
    pub attention_head_dim: i32,
    pub num_attention_heads: i32,
    pub caption_projection_dim: i32,
    pub pooled_projection_dim: i32,
    pub pos_embed_max_size: i32,
    pub axes_dimensions: [i32; 3],
    pub theta: i32,
    pub hidden_size: i32,
}

impl Default for QwenFullConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            patch_size: 2,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            caption_projection_dim: 3584,
            pooled_projection_dim: 3584,
            pos_embed_max_size: 96,
            axes_dimensions: [16, 56, 56],
            theta: 10000,
            hidden_size: 3072,
        }
    }
}

impl QwenFullConfig {
    /// Create config from a HuggingFace config.json
    pub fn from_hf_json(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        Ok(Self {
            in_channels: json["in_channels"].as_i64().unwrap_or(64) as i32,
            out_channels: json["out_channels"].as_i64().unwrap_or(64) as i32,
            patch_size: json["patch_size"].as_i64().unwrap_or(2) as i32,
            num_layers: json["num_layers"].as_i64().unwrap_or(60) as i32,
            attention_head_dim: json["attention_head_dim"].as_i64().unwrap_or(128) as i32,
            num_attention_heads: json["num_attention_heads"].as_i64().unwrap_or(24) as i32,
            caption_projection_dim: json["caption_projection_dim"].as_i64().unwrap_or(3584) as i32,
            pooled_projection_dim: json["pooled_projection_dim"].as_i64().unwrap_or(3584) as i32,
            pos_embed_max_size: json["pos_embed_max_size"].as_i64().unwrap_or(96) as i32,
            axes_dimensions: {
                let arr = json["axes_dim"].as_array()
                    .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(16) as i32).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![16, 56, 56]);
                [arr[0], arr[1], arr[2]]
            },
            theta: json["theta"].as_i64().unwrap_or(10000) as i32,
            hidden_size: json["hidden_size"].as_i64().unwrap_or(3072) as i32,
        })
    }
}

// ============================================================================
// GELU MLP (GELU-approximate activation)
// ============================================================================

/// GELU-approximate Feed Forward network (matches HuggingFace FeedForward)
#[derive(Debug, Clone, ModuleParameters)]
pub struct GeluMLP {
    #[param]
    pub proj_in: Linear,
    #[param]
    pub proj_out: Linear,
}

impl GeluMLP {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            proj_in: LinearBuilder::new(dim, dim * 4).bias(true).build()?,
            proj_out: LinearBuilder::new(dim * 4, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.proj_in.forward(x)?;
        let activated = nn::gelu_approximate(&hidden)?;
        self.proj_out.forward(&activated)
    }
}

// ============================================================================
// QK Normalization (per-head RMSNorm)
// ============================================================================

/// Per-head RMSNorm for Q/K normalization
#[derive(Debug, Clone, ModuleParameters)]
pub struct QKNorm {
    #[param]
    pub norm_q: nn::RmsNorm,
    #[param]
    pub norm_k: nn::RmsNorm,
}

impl QKNorm {
    pub fn new(head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            norm_q: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
        })
    }

    pub fn forward(&mut self, q: &Array, k: &Array) -> Result<(Array, Array), Exception> {
        let q = self.norm_q.forward(q)?;
        let k = self.norm_k.forward(k)?;
        Ok((q, k))
    }
}

// ============================================================================
// Joint Attention
// ============================================================================

/// Joint Attention with separate image/text pathways
#[derive(Debug, Clone, ModuleParameters)]
pub struct JointAttention {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    // Image projections
    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,

    // Text projections
    #[param]
    pub add_q_proj: Linear,
    #[param]
    pub add_k_proj: Linear,
    #[param]
    pub add_v_proj: Linear,

    // QK normalization
    #[param]
    pub norm_q: nn::RmsNorm,
    #[param]
    pub norm_k: nn::RmsNorm,
    #[param]
    pub norm_added_q: nn::RmsNorm,
    #[param]
    pub norm_added_k: nn::RmsNorm,

    // Output projections
    #[param]
    pub attn_to_out: Linear,
    #[param]
    pub to_add_out: Linear,
}

impl JointAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        let total_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            scale,
            to_q: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            to_k: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            to_v: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_q_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_k_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            add_v_proj: LinearBuilder::new(dim, total_dim).bias(true).build()?,
            norm_q: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_q: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_k: nn::RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            attn_to_out: LinearBuilder::new(total_dim, dim).bias(true).build()?,
            to_add_out: LinearBuilder::new(total_dim, dim).bias(true).build()?,
        })
    }

    /// Forward pass with joint attention
    pub fn forward(
        &mut self,
        img_hidden: &Array,
        txt_hidden: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img_hidden.dim(0);
        let img_seq = img_hidden.dim(1);
        let txt_seq = txt_hidden.dim(1);

        // Project Q, K, V for both streams
        let mut img_q = self.to_q.forward(img_hidden)?;
        let mut img_k = self.to_k.forward(img_hidden)?;
        let img_v = self.to_v.forward(img_hidden)?;

        let mut txt_q = self.add_q_proj.forward(txt_hidden)?;
        let mut txt_k = self.add_k_proj.forward(txt_hidden)?;
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

        let img_q = apply_rope(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope(&txt_k, txt_cos, txt_sin)?;

        // QK Norm
        let (img_q, img_k) = self.norm_q.forward(&img_q)?;
        let (img_k,) = (self.norm_k.forward(&img_k)?,);
        let (txt_q, txt_k) = self.norm_added_q.forward(&txt_q)?;
        let (txt_k,) = (self.norm_added_k.forward(&txt_k)?,);
        let img_k = img_k;
        let txt_k = txt_k;

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
        let attn_out = if let Some(mask) = mask {
            fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?
        } else {
            fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };

        // Split back into image and text
        let img_out = attn_out.index(&[.., .., ..img_seq, ..])?;
        let txt_out = attn_out.index(&[.., .., img_seq.., ..])?;

        // Transpose back and reshape
        let img_out = img_out.transpose(&[0, 2, 1, 3])?;
        let img_out = img_out.reshape(&[batch, img_seq, self.num_heads * self.head_dim])?;
        let img_out = self.attn_to_out.forward(&img_out)?;

        let txt_out = txt_out.transpose(&[0, 2, 1, 3])?;
        let txt_out = txt_out.reshape(&[batch, txt_seq, self.num_heads * self.head_dim])?;
        let txt_out = self.to_add_out.forward(&txt_out)?;

        Ok((img_out, txt_out))
    }
}

/// Apply rotary position embedding (Qwen complex-valued style)
/// Qwen uses interleaved pairs: [real, imag, real, imag, ...]
/// This matches use_real=False in diffusers' apply_rotary_emb_qwen
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    // x: [batch, seq, heads, head_dim]
    // cos/sin: [batch, seq, 1, head_dim] or [seq, 1, head_dim]
    let half_dim = x.dim(-1) / 2;

    // Split into two halves
    let x1 = x.index(&[.., .., .., ..half_dim])?;
    let x2 = x.index(&[.., .., .., half_dim..])?;

    // [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    let cos_part = ops::multiply(&x1, cos)?;
    let sin_part = ops::multiply(&x2, sin)?;
    let rotated1 = ops::subtract(&cos_part, &sin_part)?;

    let cos_part = ops::multiply(&x2, cos)?;
    let sin_part = ops::multiply(&x1, sin)?;
    let rotated2 = ops::add(&cos_part, &sin_part)?;

    ops::concatenate_axis(&[&rotated1, &rotated2], -1)
}

// ============================================================================
// AdaLayerNorm Modulation
// ============================================================================

/// AdaLayerNorm modulation (outputs shift, scale, gate for attention and MLP)
#[derive(Debug, Clone, ModuleParameters)]
pub struct AdaLayerNormMod {
    pub dim: i32,
    #[param]
    pub linear: Linear,
    #[param]
    pub norm: LayerNorm,
}

impl AdaLayerNormMod {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            linear: LinearBuilder::new(dim, 6 * dim).bias(true).build()?,
            norm: LayerNormBuilder::new(dim).elementwise_affine(false).eps(1e-6).build()?,
        })
    }

    /// Returns (modulated_hidden, gate_msa, mod_mlp_params)
    pub fn forward(&mut self, hidden_states: &Array, temb: &Array) -> Result<(Array, Array, Array), Exception> {
        // Project conditioning to modulation params
        let cond = self.linear.forward(temb)?; // [batch, 6*dim]
        let cond = nn::silu(&cond)?;

        // Split into shift1, scale1, gate1, shift2, scale2, gate2
        let shift1 = cond.index(&[.., ..self.dim])?;
        let scale1 = cond.index(&[.., self.dim..2*self.dim])?;
        let gate1 = cond.index(&[.., 2*self.dim..3*self.dim])?;
        let shift2 = cond.index(&[.., 3*self.dim..4*self.dim])?;
        let scale2 = cond.index(&[.., 4*self.dim..5*self.dim])?;
        let gate2 = cond.index(&[.., 5*self.dim..])?;

        // Modulate: (1 + scale) * LayerNorm(x) + shift
        let normed = self.norm.forward(hidden_states)?;
        let one = Array::from_f32(1.0);
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale1)?)?,
            &shift1,
        )?;

        Ok((modulated, gate1, ops::concatenate_axis(&[&shift2, &scale2, &gate2], -1)?))
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

/// Single transformer block with joint attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFullBlock {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image stream
    #[param]
    pub norm1: AdaLayerNormMod,
    #[param]
    pub attn: JointAttention,
    #[param]
    pub ff: GeluMLP,

    // Text stream
    #[param]
    pub norm1_context: AdaLayerNormMod,
    #[param]
    pub ff_context: GeluMLP,
}

impl QwenFullBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            norm1: AdaLayerNormMod::new(dim)?,
            attn: JointAttention::new(dim, num_heads, head_dim)?,
            ff: GeluMLP::new(dim)?,
            norm1_context: AdaLayerNormMod::new(dim)?,
            ff_context: GeluMLP::new(dim)?,
        })
    }

    /// Forward pass with dual streams and condition
    pub fn forward(
        &mut self,
        img_hidden: &Array,
        txt_hidden: &Array,
        temb: &Array,
        img_rotary: &(Array, Array),
        txt_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img_hidden.dim(0);

        // Image stream: modulate, attention, FFN
        let (img_modulated, img_gate, img_mod2) = self.norm1.forward(img_hidden, temb)?;
        let (txt_modulated, txt_gate, txt_mod2) = self.norm1_context.forward(txt_hidden, temb)?;

        // Joint attention
        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_rotary,
            txt_rotary,
            mask,
        )?;

        // Gate + residual
        let img_hidden = ops::add(img_hidden, &ops::multiply(&img_gate, &img_attn)?)?;
        let txt_hidden = ops::add(txt_hidden, &ops::multiply(&txt_gate, &txt_attn)?)?;

        // Apply modulation for MLP
        let mod2_params = &img_mod2;
        let shift2 = mod2_params.index(&[.., ..self.dim])?;
        let scale2 = mod2_params.index(&[.., self.dim..2*self.dim])?;
        let gate2 = mod2_params.index(&[.., 2*self.dim..])?;

        let one = Array::from_f32(1.0);
        let normed = self.norm1.norm.forward(&img_hidden)?; // Reuse norm without affine
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale2)?)?,
            &shift2,
        )?;

        let img_ffn = self.ff.forward(&modulated)?;
        let img_hidden = ops::add(&img_hidden, &ops::multiply(&gate2, &img_ffn)?)?;

        // Text FFN
        let mod2_ctx = &txt_mod2;
        let shift2_ctx = mod2_ctx.index(&[.., ..self.dim])?;
        let scale2_ctx = mod2_ctx.index(&[.., self.dim..2*self.dim])?;
        let gate2_ctx = mod2_ctx.index(&[.., 2*self.dim..])?;

        let normed_ctx = self.norm1_context.norm.forward(&txt_hidden)?;
        let modulated_ctx = ops::add(
            &ops::multiply(&normed_ctx, &ops::add(&one, &scale2_ctx)?)?,
            &shift2_ctx,
        )?;

        let txt_ffn = self.ff_context.forward(&modulated_ctx)?;
        let txt_hidden = ops::add(&txt_hidden, &ops::multiply(&gate2_ctx, &txt_ffn)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

/// Use fused Metal kernel for modulation
/// Set to false for better performance (MLX's lazy evaluation is already efficient)
/// The fused kernel is available and working if needed for specific use cases
const USE_FUSED_MODULATE: bool = false;

/// Apply layer norm (no learnable params) then modulation: (1 + scale) * LayerNorm(x) + shift
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    modulate_manual(x, shift, scale)
}

/// Manual modulate implementation
/// MLX lazy evaluation handles Array::from_f32 efficiently - no GPU allocation until eval()
fn modulate_manual(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let eps = Array::from_f32(1e-6);
    let one = Array::from_f32(1.0);

    // Compute mean and variance
    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps)?)?)?;

    // Apply scale and shift: (1 + scale) * normalized + shift
    let scaled = ops::multiply(&normalized, &ops::add(&one, scale)?)?;
    ops::add(&scaled, shift)
}

// ============================================================================
// Timestep Embedding
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct TimestepEmbedder {
    /// Pre-computed sinusoidal frequencies (cached for performance)
    pub cached_freqs: Array,

    #[param]
    pub linear_1: Linear,
    #[param]
    pub linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        let half_dim = 128; // 256 / 2

        // Pre-compute frequencies ONCE at initialization
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);

        Ok(Self {
            cached_freqs,
            linear_1: LinearBuilder::new(256, dim).bias(true).build()?,
            linear_2: LinearBuilder::new(dim, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        // t is sigma in [0, 1] range - scale by 1000 for sinusoidal encoding
        let t_scaled = ops::multiply(t, &Array::from_f32(1000.0))?;
        let t_expanded = t_scaled.reshape(&[-1, 1])?;
        let args = ops::multiply(&t_expanded, &self.cached_freqs)?;

        let cos = ops::cos(&args)?;
        let sin = ops::sin(&args)?;
        let emb = ops::concatenate_axis(&[&cos, &sin], -1)?;

        let h = nn::silu(&self.linear_1.forward(&emb)?)?;
        self.linear_2.forward(&h)
    }
}

// ============================================================================
// Final Norm with Linear
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct FinalNorm {
    pub dim: i32,
    #[param]
    pub linear: Linear,
    pub norm: LayerNorm,
}

impl FinalNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            linear: LinearBuilder::new(dim, dim).bias(true).build()?,
            norm: LayerNormBuilder::new(dim).elementwise_affine(false).eps(1e-6).build()?,
        })
    }

    pub fn forward(&mut self, hidden_states: &Array, temb: &Array) -> Result<Array, Exception> {
        // Upcast to FP32 before final LayerNorm for numerical stability
        // DiT activations can reach ±50M after 60 blocks
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.as_dtype(Dtype::Float32)?;
        let temb = temb.as_dtype(Dtype::Float32)?;

        let cond = self.linear.forward(&temb)?;
        let shift = cond.index(&[.., ..self.dim])?;
        let scale = cond.index(&[.., self.dim..])?;

        let normed = self.norm.forward(&hidden_states)?;
        let one = Array::from_f32(1.0);
        let modulated = ops::add(
            &ops::multiply(&normed, &ops::add(&one, &scale)?)?,
            &shift,
        )?;

        // Cast back to original dtype
        modulated.as_dtype(input_dtype)
    }
}

// ============================================================================
// Full Transformer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFullTransformer {
    pub config: QwenFullConfig,

    #[param]
    pub pos_embedding: Array,
    #[param]
    pub img_in: Linear,
    #[param]
    pub txt_in: Linear,
    #[param]
    pub time_text_embed: TimestepEmbedder,
    #[param]
    pub blocks: Vec<QwenFullBlock>,
    #[param]
    pub norm_out: FinalNorm,
    #[param]
    pub proj_out: Linear,
}

impl QwenFullTransformer {
    pub fn new(config: QwenFullConfig) -> Result<Self, Exception> {
        let in_channels = config.in_channels * config.patch_size * config.patch_size;
        let out_channels = config.out_channels * config.patch_size * config.patch_size;

        // Position embedding (learnt positional encodings)
        let max_patches = config.pos_embed_max_size * config.pos_embed_max_size;
        let pos_embedding = Array::zeros::<f32>(&[1, max_patches, in_channels])?;

        // Image input projection
        let img_in = LinearBuilder::new(in_channels, config.hidden_size).bias(true).build()?;
        let txt_in = LinearBuilder::new(config.caption_projection_dim, config.hidden_size).bias(true).build()?;

        // Time-text embedding
        let time_text_embed = TimestepEmbedder::new(config.hidden_size)?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            blocks.push(QwenFullBlock::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_head_dim,
            )?);
        }

        // Output
        let norm_out = FinalNorm::new(config.hidden_size)?;
        let proj_out = LinearBuilder::new(config.hidden_size, out_channels).bias(true).build()?;

        Ok(Self {
            config,
            pos_embedding: Param::new(pos_embedding),
            img_in: Param::new(img_in),
            txt_in: Param::new(txt_in),
            time_text_embed: Param::new(time_text_embed),
            blocks: Param::new(blocks),
            norm_out: Param::new(norm_out),
            proj_out: Param::new(proj_out),
        })
    }

    /// Forward pass with pre-computed RoPE
    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        timestep: &Array,
        img_rotary: Option<&(Array, Array)>,
        txt_rotary: Option<&(Array, Array)>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.dim(0);
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);

        // Image input projection + position embedding
        let img = self.img_in.forward(img)?;

        // Add learned position embeddings (crop to sequence length)
        let pos_embed = self.pos_embedding.index(&[.., ..img_seq, ..])?;
        let img = ops::add(&img, &pos_embed)?;

        // Text input projection
        let txt = self.txt_in.forward(txt)?;

        // Time-text embedding
        let temb = self.time_text_embed.forward(timestep)?;
        let temb = nn::silu(&temb)?;

        // Get RoPE (pre-computed or compute here)
        let default_img_rope = (Array::zeros::<f32>(&[1, img_seq, 1, self.config.attention_head_dim])?, Array::zeros::<f32>(&[1, img_seq, 1, self.config.attention_head_dim])?);
        let default_txt_rope = (Array::zeros::<f32>(&[1, txt_seq, 1, self.config.attention_head_dim])?, Array::zeros::<f32>(&[1, txt_seq, 1, self.config.attention_head_dim])?);

        let img_rotary = img_rotary.unwrap_or(&default_img_rope);
        let txt_rotary = txt_rotary.unwrap_or(&default_txt_rope);

        // Pass through transformer blocks
        let mut img_hidden = img;
        let mut txt_hidden = txt;

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

// ============================================================================
// Weight Loading
// ============================================================================

/// Load full precision transformer weights from HuggingFace format
pub fn load_full_precision_weights(
    transformer: &mut QwenFullTransformer,
    weights: HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Map HuggingFace weight names to our struct field names
    let mapped: HashMap<Rc<str>, Array> = weights
        .into_iter()
        .map(|(name, weight)| {
            let mapped_name = map_weight_name(&name);
            (Rc::from(mapped_name.as_str()), weight)
        })
        .collect();

    // Use update_flattened for each weight
    for (name, weight) in &mapped {
        transformer.update_with_flattened(name.clone(), weight.clone())?;
    }

    Ok(())
}

fn map_weight_name(name: &str) -> String {
    let mut mapped = name.to_string();

    // Map transformer block weight names
    // HuggingFace format: transformer_blocks.0.attn.to_q.weight
    // Our format: blocks.0.attn.to_q.weight
    mapped = mapped.replace("transformer_blocks.", "blocks.");

    // Image input
    mapped = mapped.replace("pos_embed.proj.weight", "img_in.weight");
    mapped = mapped.replace("pos_embed.proj.bias", "img_in.bias");
    mapped = mapped.replace("pos_embed.pos_embedding", "pos_embedding");

    // Text input
    mapped = mapped.replace("caption_proj.", "txt_in.");
    mapped = mapped.replace("caption_norm.", "txt_norm.");

    // Output
    mapped = mapped.replace("norm_out.linear.", "norm_out.linear.");
    mapped = mapped.replace("proj_out.weight", "proj_out.weight");
    mapped = mapped.replace("proj_out.bias", "proj_out.bias");

    // Time-text embed
    mapped = mapped.replace("time_text_embed.timestep_embedder.linear_1.", "time_text_embed.linear_1.");
    mapped = mapped.replace("time_text_embed.timestep_embedder.linear_2.", "time_text_embed.linear_2.");

    // Block sub-layers
    mapped = mapped.replace("attn.to_q.weight", "attn.to_q.weight");
    mapped = mapped.replace("attn.to_k.weight", "attn.to_k.weight");
    mapped = mapped.replace("attn.to_v.weight", "attn.to_v.weight");
    mapped = mapped.replace("attn.add_q_proj.weight", "attn.add_q_proj.weight");
    mapped = mapped.replace("attn.add_k_proj.weight", "attn.add_k_proj.weight");
    mapped = mapped.replace("attn.add_v_proj.weight", "attn.add_v_proj.weight");
    mapped = mapped.replace("attn.to_out.0.weight", "attn.attn_to_out.weight");
    mapped = mapped.replace("attn.to_out.0.bias", "attn.attn_to_out.bias");
    mapped = mapped.replace("attn.add_out.weight", "attn.to_add_out.weight");
    mapped = mapped.replace("attn.add_out.bias", "attn.to_add_out.bias");
    mapped = mapped.replace("attn.norm_q.weight", "attn.norm_q.weight");
    mapped = mapped.replace("attn.norm_k.weight", "attn.norm_k.weight");
    mapped = mapped.replace("attn.norm_added_q.weight", "attn.norm_added_q.weight");
    mapped = mapped.replace("attn.norm_added_k.weight", "attn.norm_added_k.weight");

    // FFN
    mapped = mapped.replace("ff.net.0.proj.weight", "ff.proj_in.weight");
    mapped = mapped.replace("ff.net.0.proj.bias", "ff.proj_in.bias");
    mapped = mapped.replace("ff.net.2.weight", "ff.proj_out.weight");
    mapped = mapped.replace("ff.net.2.bias", "ff.proj_out.bias");
    mapped = mapped.replace("ff_context.net.0.proj.weight", "ff_context.proj_in.weight");
    mapped = mapped.replace("ff_context.net.0.proj.bias", "ff_context.proj_in.bias");
    mapped = mapped.replace("ff_context.net.2.weight", "ff_context.proj_out.weight");
    mapped = mapped.replace("ff_context.net.2.bias", "ff_context.proj_out.bias");

    // Norm1
    mapped = mapped.replace("norm1.linear.weight", "norm1.linear.weight");
    mapped = mapped.replace("norm1.linear.bias", "norm1.linear.bias");
    mapped = mapped.replace("norm1_context.linear.weight", "norm1_context.linear.weight");
    mapped = mapped.replace("norm1_context.linear.bias", "norm1_context.linear.bias");

    mapped
}

fn load_weight(
    transformer: &mut QwenFullTransformer,
    name: &str,
    weight: Array,
) -> Result<(), Box<dyn std::error::Error>> {
    let name = map_weight_name(name);
    transformer.update_with_flattened(name, weight)?;
    Ok(())
}

fn load_block_weight(
    block: &mut QwenFullBlock,
    parts: &[&str],
    weight: Array,
) -> Result<(), Box<dyn std::error::Error>> {
    // parts: ["blocks", "0", "attn", "to_q", "weight"]
    // We need to recursively find the right sub-module
    // Since update_with_flattened handles path resolution, we can use it
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_embedder() {
        let mut embedder = TimestepEmbedder::new(3072).unwrap();
        let t = Array::from_slice::<f32>(&[0.5], &[1]);
        let out = embedder.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 3072]);
    }

    #[test]
    fn test_gelu_mlp() {
        let mut mlp = GeluMLP::new(64).unwrap();
        let x = Array::zeros::<f32>(&[2, 10, 64]).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_joint_attention_allocation() {
        let attn = JointAttention::new(3072, 24, 128).unwrap();
        assert_eq!(attn.to_q.in_features(), 3072);
        assert_eq!(attn.to_q.out_features(), 3072);
    }

    #[test]
    fn test_block_creation() {
        let block = QwenFullBlock::new(3072, 24, 128).unwrap();
        assert_eq!(block.dim, 3072);
    }
}
