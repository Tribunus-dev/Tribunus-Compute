//! # qwen3-vl-mlx
//!
//! Qwen3-VL Vision-Language Model inference on Apple Silicon with MLX.
//!
//! ## Architecture
//!
//! ```text
//! Image (H x W x 3)
//!   |-> Conv3d PatchEmbed -> [N_patches, 1024]
//!   |-> + pos_embed
//!   |-> 24 VisionBlocks (LN + QKV attn + LN + GELU MLP)
//!   |     DeepStack mergers at blocks [5, 11, 17] -> intermediate [N/4, 2560]
//!   |-> PatchMerger (final) -> [N/4, 2560]
//!   |-> sum all deepstack + final -> [N_visual, 2560]
//!
//! Qwen3 LM decoder (36 layers, GQA 32/8, q_norm/k_norm, RoPE theta=5M)
//!   - embed_tokens (quantized 8-bit)
//!   - Replace image token positions with visual features
//!   - Generate text autoregressively
//! ```

pub mod error;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use error::{Error, Result};

pub use mlx_rs_core::cache::{ConcatKeyValueCache, KVCache, KeyValueCache};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_hidden")]
    pub hidden_size: i32,
    #[serde(default = "default_num_heads")]
    pub num_heads: i32,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: i32,
    #[serde(default = "default_num_pos_emb")]
    pub num_position_embeddings: i32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_deepstack_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_depth() -> usize { 24 }
fn default_vision_hidden() -> i32 { 1024 }
fn default_num_heads() -> i32 { 16 }
fn default_intermediate() -> i32 { 4096 }
fn default_patch_size() -> i32 { 16 }
fn default_temporal_patch_size() -> i32 { 2 }
fn default_in_channels() -> i32 { 3 }
fn default_out_hidden_size() -> i32 { 2560 }
fn default_num_pos_emb() -> i32 { 2304 }
fn default_spatial_merge_size() -> i32 { 2 }
fn default_deepstack_indexes() -> Vec<usize> { vec![5, 11, 17] }

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 24,
            hidden_size: 1024,
            num_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
            out_hidden_size: 2560,
            num_position_embeddings: 2304,
            spatial_merge_size: 2,
            deepstack_visual_indexes: vec![5, 11, 17],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attn_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_text_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_vocab_size() -> i32 { 151936 }
fn default_hidden_size() -> i32 { 2560 }
fn default_num_layers() -> i32 { 36 }
fn default_num_attn_heads() -> i32 { 32 }
fn default_kv_heads() -> i32 { 8 }
fn default_head_dim() -> i32 { 128 }
fn default_text_intermediate() -> i32 { 9728 }
fn default_rms_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f64 { 5_000_000.0 }

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Qwen3VLConfig {
    #[serde(default)]
    pub vision_config: VisionConfig,
    #[serde(default)]
    pub text_config: TextConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: i32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: i32,
}

fn default_image_token_id() -> i32 { 151655 }
fn default_vision_start_token_id() -> i32 { 151652 }
fn default_vision_end_token_id() -> i32 { 151653 }

impl Default for Qwen3VLConfig {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            image_token_id: 151655,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Linear>> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases_arr = get_weight(weights, &format!("{}.biases", prefix))?;
    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };
    Ok(MaybeQuantized::Quantized(nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases_arr),
        inner,
    }))
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Embedding>> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases_arr = get_weight(weights, &format!("{}.biases", prefix))?;
    let inner = nn::Embedding {
        weight: Param::new(weight),
    };
    Ok(MaybeQuantized::Quantized(nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases_arr),
        inner,
    }))
}

// ============================================================================
// Vision Encoder
// ============================================================================

fn apply_gelu(x: Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
    nn::gelu_approximate(x)
}

/// PatchEmbed: Conv3d to extract image patches.
/// Weight: [out_channels, temporal, H_patch, W_patch, in_channels] = [1024, 2, 16, 16, 3]
pub struct PatchEmbed {
    pub proj: nn::Conv3d,
}

impl PatchEmbed {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [1, T, H, W, C] (channels last)
        let out = self.proj.forward(x)?;
        // out: [1, T', H', W', 1024] where T'=1, H'=H/16, W'=W/16
        let shape = out.shape();
        let n = shape[2] * shape[3]; // H' * W'
        // Reshape to [n, 1024]
        Ok(out.reshape(&[n, shape[4]])?)
    }
}

/// Vision block LayerNorm (with bias).
pub struct VisionLayerNorm {
    pub weight: Array,
    pub bias: Array,
    pub eps: f32,
}

impl VisionLayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Use mlx_rs layernorm fast path
        let w = Some(&self.weight);
        let b = Some(&self.bias);
        Ok(mlx_rs::fast::layer_norm(x, w, b, self.eps)?)
    }
}

/// Vision self-attention (full attention, no windowing).
/// Uses combined QKV projection then split.
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    // QKV combined [3*hidden, hidden]
    pub qkv: nn::Linear,
    pub proj: nn::Linear,
}

impl VisionAttention {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let n = shape[0]; // sequence length
        let hidden = shape[1];

        // QKV projection: [N, 3*hidden]
        let qkv = self.qkv.forward(x)?;
        // Split: each [N, hidden]
        let parts = qkv.split(3, -1)?;
        let q = parts[0].reshape(&[n, self.num_heads as i32, self.head_dim as i32])?;
        let k = parts[1].reshape(&[n, self.num_heads as i32, self.head_dim as i32])?;
        let v = parts[2].reshape(&[n, self.num_heads as i32, self.head_dim as i32])?;

        // Manual QK^T with scale
        let q_scaled = &q * self.scale;
        // Transpose the last two dims on k: [n, num_heads, head_dim]
        let k_t = k.transpose(&[0, 2, 1])?; // [n, head_dim, num_heads]
        // Actually we need [1, num_heads, n, head_dim] format
        // Reshape to [1, num_heads, n, head_dim]
        let q2 = q.reshape(&[1, self.num_heads as i32, n, self.head_dim as i32])?;
        let k2 = k.reshape(&[1, self.num_heads as i32, n, self.head_dim as i32])?;
        let v2 = v.reshape(&[1, self.num_heads as i32, n, self.head_dim as i32])?;

        let attn_weights = mlx_rs::fast::scaled_dot_product_attention(
            &q2, &k2, &v2, self.scale, None, None,
        )?;

        // Merge heads: [1, num_heads, n, head_dim] -> [n, num_heads * head_dim]
        let merged = attn_weights.reshape(&[n, self.num_heads as i32 * self.head_dim as i32])?;

        // Output projection
        Ok(self.proj.forward(&merged)?)
    }
}

/// Single vision transformer block.
pub struct VisionBlock {
    pub ln1: VisionLayerNorm,
    pub attn: VisionAttention,
    pub ln2: VisionLayerNorm,
    pub mlp: VisionMLP,
}

impl VisionBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // Self-attention with residual
        let residual = x.clone();
        let ln_out = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&ln_out)?;
        let x = (&residual + &attn_out)?;

        // MLP with residual
        let residual = x.clone();
        let ln_out = self.ln2.forward(&x)?;
        let mlp_out = self.mlp.forward(&ln_out)?;
        Ok((&residual + &mlp_out)?)
    }
}

/// Vision MLP: GELU gating MLP with intermediate expansion.
/// Uses merged gate+up weight, similar to Qwen2's approach.
pub struct VisionMLP {
    pub gate_up: nn::Linear,
    pub down: nn::Linear,
}

impl VisionMLP {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate_up = self.gate_up.forward(x)?;
        // Split into gate and proj halves
        let halved = gate_up.split(2, -1)?;
        let gate = apply_gelu(halved[0].clone())?;
        let proj = halved[1].clone();
        // Element-wise multiply
        let gated = (&gate * &proj)?;
        Ok(self.down.forward(&gated)?)
    }
}

/// DeepStack merger: merges spatial tokens at specified indices.
/// Actually a simple Conv3d that reduces spatial dim by spatial_merge_size.
pub struct DeepStackMerger {
    pub merger: nn::Conv3d,
    pub spatial_merge_size: i32,
}

impl DeepStackMerger {
    pub fn forward(&mut self, x: &Array, spatial_shape: &[i32; 2]) -> Result<Array> {
        // x: [N, hidden] — N tokens (H' * W')
        // Reshape to [1, 1, H', W', hidden]
        let reshaped = x.reshape(&[
            1,
            1,
            spatial_shape[0],
            spatial_shape[1],
            x.shape()[1],
        ])?;
        let out = self.merger.forward(&reshaped)?;
        // out: [1, 1, H'/s, W'/s, out_hidden]
        let shape = out.shape();
        let out_hidden = shape[4];
        let n = shape[2] * shape[3];
        Ok(out.reshape(&[n, out_hidden])?)
    }
}

/// PatchMerger: final merger layer.
/// Uses linear projection to merge groups of spatial_merge_size^2 tokens.
pub struct PatchMerger {
    pub projection: nn::Linear,
    pub spatial_merge_size: i32,
}

impl PatchMerger {
    pub fn forward(&mut self, x: &Array, spatial_shape: &[i32; 2]) -> Result<Array> {
        // x: [N, hidden] where N = H' * W'
        // Group tokens spatial_merge_size^2 at a time -> [N//sms^2, sms^2 * hidden]
        let sms = self.spatial_merge_size;
        let n = x.shape()[0];
        let hidden = x.shape()[1];
        let tokens_per_group = sms * sms;
        let n_groups = n / tokens_per_group;

        let grouped = x.reshape(&[n_groups, tokens_per_group * hidden])?;
        Ok(self.projection.forward(&grouped)?)
    }
}

/// Full Vision Encoder with optional DeepStack.
pub struct VisionEncoder {
    pub patch_embed: PatchEmbed,
    pub pos_embed: Array,
    pub blocks: Vec<VisionBlock>,
    pub deepstack_mergers: Vec<DeepStackMerger>,
    pub patch_merger: PatchMerger,
    pub spatial_merge_size: i32,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionEncoder {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [1, T, H, W, C] where T is temporal (usually 1)
        let patches = self.patch_embed.forward(x)?;
        // patches: [N, hidden] where N = H' * W'

        // Add position embedding
        let pos_embed = &self.pos_embed;
        let n = patches.shape()[0];
        let pos = pos_embed.index(&[array!(0), array!(".."), array!("..")])?;
        let pos = pos.reshape(&[pos.shape()[1], pos.shape()[2]])?;
        let pos = pos.index(&[array!(".."), array!(":")?]);
        // Trim pos to n if needed
        let mut x = (&patches + &pos)?;

        // Spatial shape before DeepStack
        let spatial_h = (x.shape()[0] as f64).sqrt() as i32;
        let spatial_w = x.shape()[0] / spatial_h;
        let mut spatial_shape = [spatial_h, spatial_w];

        // DeepStack accumulator
        let mut deepstack_outputs = Vec::new();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(&x)?;

            if self.deepstack_visual_indexes.contains(&i) {
                let idx = self
                    .deepstack_visual_indexes
                    .iter()
                    .position(|v| *v == i)
                    .unwrap();
                let merger = &mut self.deepstack_mergers[idx];
                let merged = merger.forward(&x, &spatial_shape)?;
                deepstack_outputs.push(merged);

                // Update spatial shape after merger
                let sms = self.spatial_merge_size;
                spatial_shape[0] /= sms;
                spatial_shape[1] /= sms;
                x = x.reshape(&[
                    1,
                    1,
                    spatial_shape[0] * sms,
                    spatial_shape[1] * sms,
                    x.shape()[1],
                ])?;
                let idx_merger = self
                    .deepstack_visual_indexes
                    .iter()
                    .position(|v| *v == i)
                    .unwrap();
                let merger = &mut self.deepstack_mergers[idx_merger];
                let merged = merger.forward(&x, &spatial_shape)?;
                // x after merger is now the merged output for this level
            }
        }

        // Final PatchMerger
        let final_merged = self.patch_merger.forward(&x, &spatial_shape)?;

        // Sum all deepstack outputs + final
        let mut visual_features = final_merged;
        for d in &deepstack_outputs {
            visual_features = (&visual_features + d)?;
        }

        Ok(visual_features)
    }
}

// ============================================================================
// Text Decoder
// ============================================================================

/// Qwen3 LM decoder with quantized embeddings, GQA, q_norm/k_norm.
pub struct Qwen3VLModel {
    pub config: Qwen3VLConfig,
    // Embedding
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    // Decoder layers
    pub layers: Vec<DecoderLayer>,
    // Final norm
    pub norm: nn::RmsNorm,
    // LM head (tied with embed)
    pub lm_head: nn::Linear,
    // Layer config cache
    pub hidden_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub num_layers: i32,
    pub rope_theta: f64,
    pub max_seq_len: i32,
}

/// Single decoder layer.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

/// GQA attention with QK normalization.
pub struct Attention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub rope_theta: f64,
    pub max_seq_len: i32,
}

impl Attention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];

        // Projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, S, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        // QK normalization
        let q_norm = self.q_norm.forward(&q.reshape(&[
            batch * seq_len,
            self.num_heads * self.head_dim,
        ])?)?;
        let k_norm = self.k_norm.forward(&k.reshape(&[
            batch * seq_len,
            self.num_kv_heads * self.head_dim,
        ])?)?;
        let q = q_norm.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k_norm.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE
        let (q, k) = nn::rope(
            &q,
            &k,
            self.head_dim,
            self.max_seq_len,
            self.rope_theta,
        )?;

        // Handle KV cache
        let (k, v) = if let Some(cache) = cache {
            cache.update(k, v)?
        } else {
            (k, v)
        };

        // GQA: expand kv heads to match q heads
        let n_kv_heads = self.num_kv_heads;
        let n_heads = self.num_heads;
        let n_groups = n_heads / n_kv_heads;

        // Transpose to [B, n_heads, S, head_dim]
        let q = q.transpose(&[0, 2, 1, 3])?;
        let k = k.transpose(&[0, 2, 1, 3])?;
        let v = v.transpose(&[0, 2, 1, 3])?;

        // For GQA, repeat kv heads
        // This is a simplification; actual impl uses group repeat
        let k = if n_groups > 1 {
            // Expand k from [B, n_kv, S, hd] to [B, n_heads, S, hd]
            let k = k.reshape(&[batch, n_kv_heads, 1, seq_len, self.head_dim])?;
            let k = k.broadcast(&[batch, n_kv_heads, n_groups, seq_len, self.head_dim])?;
            k.reshape(&[batch, n_heads, seq_len, self.head_dim])?
        } else {
            k
        };
        let v = if n_groups > 1 {
            let v = v.reshape(&[batch, n_kv_heads, 1, seq_len, self.head_dim])?;
            let v = v.broadcast(&[batch, n_kv_heads, n_groups, seq_len, self.head_dim])?;
            v.reshape(&[batch, n_heads, seq_len, self.head_dim])?
        } else {
            v
        };

        // Scaled dot-product attention
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v, self.scale, mask.map(ScaledDotProductAttentionMask::Array), None,
        )?;

        // Merge heads: [B, n_heads, S, head_dim] -> [B, S, n_heads * head_dim]
        let attn_out = attn_out.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Output projection
        Ok(self.o_proj.forward(&attn_out)?)
    }
}

/// SwiGLU MLP.
pub struct MLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl MLP {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gated = (&nn::silu(&gate)? * &up)?;
        Ok(self.down_proj.forward(&gated)?)
    }
}

impl DecoderLayer {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        // Self-attention with pre-norm
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let x = (&residual + &attn_out)?;

        // MLP with pre-norm
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok((&residual + &mlp_out)?)
    }
}

impl Qwen3VLModel {
    /// Create model from safetensor weights and config JSON.
    pub fn from_checkpoint(
        config_path: &Path,
        weight_dir: &Path,
    ) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Qwen3VLConfig = serde_json::from_str(&config_str)?;

        // Load weights
        let weights = mlx_rs_core::safetensors::load_all(weight_dir)?;

        Self::from_weights(config, weights)
    }

    /// Build model from config and weight map.
    pub fn from_weights(config: Qwen3VLConfig, weights: HashMap<String, Array>) -> Result<Self> {
        let tc = &config.text_config;
        let hidden_size = tc.hidden_size;
        let num_heads = tc.num_attention_heads;
        let num_kv_heads = tc.num_key_value_heads;
        let head_dim = tc.head_dim;
        let num_layers = tc.num_hidden_layers;
        let rope_theta = tc.rope_theta;
        let max_seq_len = 8192;

        // Embedding (quantized 8-bit)
        let embed_tokens = make_quantized_embedding(
            &weights,
            "model.embed_tokens",
            64,
            8,
        )?;

        // Decoder layers
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            let prefix = format!("model.layers.{}", i);

            // QKV projections
            let q_proj = nn::Linear::new(
                "q_proj",
                &format!("{}.self_attn.q_proj", prefix),
                &weights,
            )?;
            let k_proj = nn::Linear::new(
                "k_proj",
                &format!("{}.self_attn.k_proj", prefix),
                &weights,
            )?;
            let v_proj = nn::Linear::new(
                "v_proj",
                &format!("{}.self_attn.v_proj", prefix),
                &weights,
            )?;
            let o_proj = nn::Linear::new(
                "o_proj",
                &format!("{}.self_attn.o_proj", prefix),
                &weights,
            )?;

            // QK norms
            let q_norm = nn::RmsNorm::new(
                &format!("{}.self_attn.q_norm", prefix),
                head_dim,
                tc.rms_norm_eps,
                &weights,
            )?;
            let k_norm = nn::RmsNorm::new(
                &format!("{}.self_attn.k_norm", prefix),
                head_dim,
                tc.rms_norm_eps,
                &weights,
            )?;

            let self_attn = Attention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                rope_theta,
                max_seq_len,
            };

            // MLP (SwiGLU)
            let gate_proj = nn::Linear::new(
                "gate_proj",
                &format!("{}.mlp.gate_proj", prefix),
                &weights,
            )?;
            let up_proj = nn::Linear::new(
                "up_proj",
                &format!("{}.mlp.up_proj", prefix),
                &weights,
            )?;
            let down_proj = nn::Linear::new(
                "down_proj",
                &format!("{}.mlp.down_proj", prefix),
                &weights,
            )?;

            let mlp = MLP {
                gate_proj,
                up_proj,
                down_proj,
            };

            let input_layernorm = nn::RmsNorm::new(
                "input_layernorm",
                &format!("{}.input_layernorm", prefix),
                hidden_size,
                tc.rms_norm_eps,
                &weights,
            )?;
            let post_attention_layernorm = nn::RmsNorm::new(
                "post_attention_layernorm",
                &format!("{}.post_attention_layernorm", prefix),
                hidden_size,
                tc.rms_norm_eps,
                &weights,
            )?;

            layers.push(DecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = nn::RmsNorm::new(
            "model.norm",
            "model.norm",
            hidden_size,
            tc.rms_norm_eps,
            &weights,
        )?;

        // LM head (tied weights)
        let lm_head = nn::Linear::new("lm_head", "lm_head", &weights)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            rope_theta,
            max_seq_len,
        })
    }

    /// Forward pass through the decoder.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        // [B, S] token ids
        visual_features: Option<&Array>,
        image_token_positions: Option<&HashSet<usize>>,
        cache: Option<&mut Vec<Option<KVCache>>>,
    ) -> Result<Array> {
        let batch = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Embed tokens
        let mut hidden_states = match &mut self.embed_tokens {
            MaybeQuantized::Unquantized(emb) => emb.forward(input_ids)?,
            MaybeQuantized::Quantized(emb) => emb.forward(input_ids)?,
        };

        // Place visual features at image token positions
        if let (Some(vis), Some(positions)) = (visual_features, image_token_positions) {
            let mut vis_idx = 0;
            for &pos in positions.iter() {
                if pos as i64 >= seq_len as i64 {
                    break;
                }
                let vis_tokens = vis.index(&[array!(vis_idx), array!("..")])?;
                hidden_states = hidden_states.index_put(
                    &[array!(0), array!(pos as i64), array!("..")],
                    &vis_tokens,
                )?;
                vis_idx += 1;
            }
        }

        // Create causal mask
        let mask = mlx_rs_core::ops::create_causal_mask(seq_len)?;

        // Run decoder layers
        let mut cache = cache.unwrap_or(&mut Vec::new());
        if cache.is_empty() {
            for _ in 0..self.num_layers as usize {
                cache.push(None);
            }
        }

        for i in 0..self.num_layers as usize {
            hidden_states = self.layers[i].forward(
                &hidden_states,
                Some(&mask),
                cache[i].as_mut(),
            )?;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        visual_features: Option<&Array>,
        image_token_positions: Option<&HashSet<usize>>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<Vec<i32>> {
        let mut tokens = prompt_ids.to_vec();
        let mut cache: Vec<Option<KVCache>> = Vec::new();

        // Prefill
        let input = Array::from_slice(prompt_ids, &[1, prompt_ids.len() as i32]);
        let _ = self.forward(
            &input,
            visual_features,
            image_token_positions,
            Some(&mut cache),
        )?;

        // Generate
        for _ in 0..max_new_tokens {
            let last_token = tokens.last().copied().unwrap_or(0);
            let input = Array::from_slice(&[last_token], &[1, 1]);

            let logits = self.forward(&input, None, None, Some(&mut cache))?;

            // Sample next token
            let next_token = mlx_rs_core::sampling::sample(
                &logits.index(&[array!(0), array!(-1), array!("..")])?,
                temperature,
                top_p,
            )?;

            tokens.push(next_token);

            // Check for EOS
            if next_token == 151643 || next_token == 151645 {
                // <|im_end|> or <|endoftext|>
                break;
            }
        }

        Ok(tokens)
    }
}

// ============================================================================
// Image Processing
// ============================================================================

/// Load and prepare an image for the vision encoder.
pub fn load_image(path: &str) -> Result<Array> {
    let img = image::open(path)?;
    let img = img.resize_exact(
        448,
        448,
        image::imageops::FilterType::CatmullRom,
    );
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            vec![
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    // [1, H, W, C] channels-last
    Ok(Array::from_slice(&pixels, &[1, h as i32, w as i32, 3]))
}

/// Preprocess image with normalization.
pub fn preprocess_image(img: &Array) -> Result<Array> {
    // ImageNet mean/std
    let mean = array![0.48145466f32, 0.4578275, 0.40821073];
    let std = array![0.26862954f32, 0.26130258, 0.27577711];

    // Normalize: (x - mean) / std
    let img_f32 = img.as_type(mlx_rs::DType::Float32)?;
    let reshaped = img_f32.reshape(&[1, 1, 1, 3])?;
    let normalized = ((&reshaped - &mean) / &std)?;
    // Pad temporal dim: [1, T=1, H, W, C]
    Ok(normalized)
}

/// Full forward through vision encoder.
pub fn encode_image(
    vision_encoder: &mut VisionEncoder,
    image: &Array,
) -> Result<Array> {
    vision_encoder.forward(image)
}

// ============================================================================
// Text Processing (tokenizer wrapper)
// ============================================================================

/// Container for tokenized output.
pub struct TokenizedInput {
    pub input_ids: Vec<i32>,
    pub image_token_positions: Vec<usize>,
    pub image_token_positions_set: HashSet<usize>,
}

/// Tokenize prompt with image markers.
///
/// Expects prompt with `<|image_pad|>` tokens which get replaced
/// by visual feature tokens during generation.
pub fn tokenize_with_images(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<TokenizedInput> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    let tokens = encoding.get_ids().to_vec();
    let image_token_id = 151655; // <|image_pad|

    let image_token_positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, &id)| id == image_token_id)
        .map(|(i, _)| i)
        .collect();

    let image_token_positions_set: HashSet<usize> =
        image_token_positions.iter().cloned().collect();

    Ok(TokenizedInput {
        input_ids: tokens,
        image_token_positions,
        image_token_positions_set,
    })
}

// ============================================================================
// Weight loading helpers
// ============================================================================

/// Load all safetensor weight files from a directory.
pub fn load_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
    Ok(mlx_rs_core::safetensors::load_all(dir)?)
}

/// Load model configuration from JSON.
pub fn load_config(path: &Path) -> Result<Qwen3VLConfig> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}
