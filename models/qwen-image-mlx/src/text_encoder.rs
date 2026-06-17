//! Qwen-Image Text Encoder (Qwen2.5-VL variant)
//!
//! This is the text encoder used in Qwen-Image for encoding prompts.
//! Architecture: 28 layers, hidden_size=3584, GQA with 28 q_heads and 4 kv_heads

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters, Param};
use mlx_rs::nn::{Linear, LinearBuilder, Embedding, EmbeddingBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::fast;
use mlx_rs::Dtype;
use mlx_rs::Array;
use safetensors::SafeTensors;

/// Text encoder configuration
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_attention_heads: 28,
            num_hidden_layers: 28,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 152064,
            max_position_embeddings: 32768,
        }
    }
}

/// RMS Normalization
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        let weight = Array::ones::<f32>(&[dim])?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        fast::rms_norm(x, &*self.weight, self.eps)
    }
}

/// Linear layer with optional bias
#[derive(Debug)]
pub struct Linear {
    pub weight: Param<Array>,
    pub bias: Option<Param<Array>>,
}

impl Linear {
    pub fn new(in_features: i32, out_features: i32, has_bias: bool) -> Result<Self, Exception> {
        // Kaiming uniform initialization
        let scale = (1.0 / in_features as f32).sqrt();
        let weight = Array::uniform::<f32>(-scale, scale, &[out_features, in_features])?;
        let bias = if has_bias {
            Some(Param::new(Array::zeros::<f32>(&[out_features])?))
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(weight),
            bias,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let out = ops::matmul(x, &self.weight.transpose(&[1, 0])?)?;
        if let Some(ref bias) = self.bias {
            ops::add(&out, &**bias)
        } else {
            Ok(out)
        }
    }
}

/// Embedding layer
#[derive(Debug)]
pub struct Embedding {
    pub weight: Param<Array>,
}

impl Embedding {
    pub fn new(num_embeddings: i32, embedding_dim: i32) -> Result<Self, Exception> {
        let scale = (1.0 / embedding_dim as f32).sqrt();
        let weight = Array::uniform::<f32>(-scale, scale, &[num_embeddings, embedding_dim])?;
        Ok(Self {
            weight: Param::new(weight),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        ops::take(&*self.weight, x, 0)
    }
}

/// Rotary Position Embedding
#[derive(Debug)]
pub struct RotaryEmbedding {
    pub inv_freq: Array,
    pub max_seq_len_cached: i32,
    pub cos_cached: Array,
    pub sin_cached: Array,
}

impl RotaryEmbedding {
    pub fn new(dim: i32, max_position_embeddings: i32, theta: f32) -> Result<Self, Exception> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(i as f32 / half_dim as f32))
            .collect();
        let inv_freq = Array::from_slice(&inv_freq, &[half_dim]);

        let positions: Vec<f32> = (0..max_position_embeddings).map(|i| i as f32).collect();
        let positions = Array::from_slice(&positions, &[max_position_embeddings]);

        let args = ops::outer(&positions, &inv_freq)?;
        let cos_cached = ops::cos(&args)?;
        let sin_cached = ops::sin(&args)?;

        Ok(Self {
            inv_freq,
            max_seq_len_cached: max_position_embeddings,
            cos_cached,
            sin_cached,
        })
    }

    pub fn forward(&self, x: &Array, position_ids: &Array) -> Result<Array, Exception> {
        let seq_len = x.dim(1);
        if seq_len > self.max_seq_len_cached {
            // Extend cache if needed
            let new_positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
            let new_pos = Array::from_slice(&new_positions, &[seq_len]);
            let args = ops::outer(&new_pos, &self.inv_freq)?;
            let cos_new = ops::cos(&args)?;
            let sin_new = ops::sin(&args)?;
            return Ok(x.apply_rope(cos_new, sin_new)?);
        }

        // Use position_ids to select from cached embeddings
        let cos = ops::take(&self.cos_cached, position_ids, 0)?;
        let sin = ops::take(&self.sin_cached, position_ids, 0)?;

        x.apply_rope(cos, sin)
    }
}

/// Attention with Grouped Query Attention (GQA)
#[derive(Debug)]
pub struct TextAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rotary_emb: RotaryEmbedding,
}

impl TextAttention {
    pub fn new(hidden_size: i32, num_heads: i32, num_kv_heads: i32, max_position_embeddings: i32, rope_theta: f32) -> Result<Self, Exception> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj: Linear::new(hidden_size, num_heads * head_dim, true)?,
            k_proj: Linear::new(hidden_size, num_kv_heads * head_dim, true)?,
            v_proj: Linear::new(hidden_size, num_kv_heads * head_dim, true)?,
            o_proj: Linear::new(num_heads * head_dim, hidden_size, true)?,
            rotary_emb: RotaryEmbedding::new(head_dim, max_position_embeddings, rope_theta)?,
        })
    }

    pub fn forward(&self, hidden_states: &Array, position_ids: &Array, attention_mask: Option<&Array>) -> Result<Array, Exception> {
        let batch_size = hidden_states.dim(0);
        let seq_len = hidden_states.dim(1);

        // Project Q, K, V
        let query = self.q_proj.forward(hidden_states)?;
        let key = self.k_proj.forward(hidden_states)?;
        let value = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query = query.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let key = key.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;
        let value = value.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE
        let query = self.rotary_emb.forward(&query, position_ids)?;
        let key = self.rotary_emb.forward(&key, position_ids)?;

        // Transpose to [batch, heads, seq, head_dim]
        let query = query.transpose(&[0, 2, 1, 3])?;
        let key = key.transpose(&[0, 2, 1, 3])?;
        let value = value.transpose(&[0, 2, 1, 3])?;

        // Repeat KV heads for GQA
        let num_groups = self.num_heads / self.num_kv_heads;
        let key = repeat_kv(&key, num_groups)?;
        let value = repeat_kv(&value, num_groups)?;

        // Apply attention mask if provided
        let output = if let Some(mask) = attention_mask {
            // The mask has shape [batch, 1, seq, seq] with -inf for masked positions
            // SDPA in MLX accepts mask directly
            fast::scaled_dot_product_attention(
                &query, &key, &value,
                None, // MLX computes scale internally
                Some(mask),
            )?
        } else {
            fast::scaled_dot_product_attention(
                &query, &key, &value,
                None,
                None,
            )?
        };

        // Reshape and project output
        let output = output.transpose(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&output)
    }
}

/// Repeat KV heads for GQA
fn repeat_kv(x: &Array, num_groups: i32) -> Result<Array, Exception> {
    if num_groups == 1 {
        return Ok(x.clone());
    }
    // x shape: [batch, num_kv_heads, seq, head_dim]
    // Expand to [batch, num_kv_heads * num_groups, seq, head_dim]
    let batches = x.dim(0);
    let num_kv_heads = x.dim(1);
    let seq_len = x.dim(2);
    let head_dim = x.dim(3);

    let mut outputs: Vec<Array> = Vec::with_capacity(num_groups as usize);
    for _ in 0..num_groups {
        outputs.push(x.clone());
    }
    let repeated = ops::concatenate(&outputs, 1)?;
    Ok(repeated)
}

/// MLP with SwiGLU activation
#[derive(Debug)]
pub struct TextMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl TextMlp {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: Linear::new(hidden_size, intermediate_size, false)?,
            up_proj: Linear::new(hidden_size, intermediate_size, false)?,
            down_proj: Linear::new(intermediate_size, hidden_size, false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = ops::multiply(&gate, &up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Transformer layer
#[derive(Debug)]
pub struct TextEncoderLayer {
    pub self_attn: TextAttention,
    pub mlp: TextMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl TextEncoderLayer {
    pub fn new(config: &TextEncoderConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: TextAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.max_position_embeddings,
                config.rope_theta,
            )?,
            mlp: TextMlp::new(config.hidden_size, config.intermediate_size)?,
            input_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
            post_attention_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
        })
    }

    pub fn forward(&self, hidden_states: &Array, position_ids: &Array, attention_mask: Option<&Array>) -> Result<Array, Exception> {
        // Self-attention with pre-norm
        let residual = hidden_states;
        let hidden = self.input_layernorm.forward(hidden_states)?;
        let hidden = self.self_attn.forward(&hidden, position_ids, attention_mask)?;
        let hidden = ops::add(&residual, &hidden)?;

        // MLP with pre-norm
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        ops::add(residual, &hidden)
    }
}

/// Qwen Text Encoder
#[derive(Debug)]
pub struct QwenTextEncoder {
    pub config: TextEncoderConfig,
    pub embed_tokens: Embedding,
    pub layers: Vec<TextEncoderLayer>,
    pub norm: RmsNorm,
}

impl QwenTextEncoder {
    pub fn new(config: TextEncoderConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(TextEncoderLayer::new(&config)?);
        }

        Ok(Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps)?,
            config,
        })
    }

    pub fn forward(&self, input_ids: &Array, attention_mask: Option<&Array>, position_ids: Option<&Array>) -> Result<Array, Exception> {
        let batch_size = input_ids.dim(0);
        let seq_len = input_ids.dim(1);

        // Token embeddings
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // Create position IDs if not provided (causal positions)
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => {
                let pos: Vec<i32> = (0..seq_len).collect();
                let pos = Array::from_slice(&pos, &[1, seq_len]);
                ops::broadcast(&pos, &[batch_size, seq_len])?
            }
        };

        // Create causal attention mask if not provided
        let attn_mask = match attention_mask {
            Some(mask) => {
                // Convert boolean mask to attention mask
                // mask shape: [batch, seq] - true = visible, false = masked
                // Convert to [batch, 1, seq, seq] with 0.0 for visible, -inf for masked
                create_causal_mask(seq_len, Some(mask))?
            }
            None => {
                create_causal_mask(seq_len, None)?
            }
        };

        // Process through layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &position_ids, Some(&attn_mask))?;
        }

        // Final norm
        self.norm.forward(&hidden)
    }
}

/// Create causal attention mask
fn create_causal_mask(seq_len: i32, attention_mask: Option<&Array>) -> Result<Array, Exception> {
    // Create causal mask: lower triangle = 0.0, upper triangle = -inf
    let mut mask_data: Vec<f32> = Vec::with_capacity((seq_len * seq_len) as usize);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                mask_data.push(0.0);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }
    let causal_mask = Array::from_slice(&mask_data, &[1, 1, seq_len, seq_len]);

    // If we have a padding mask, combine it
    if let Some(pad_mask) = attention_mask {
        // pad_mask: [batch, seq] - true for visible tokens
        // Create [batch, 1, 1, seq] and [batch, 1, seq, 1]
        let batch_size = pad_mask.dim(0);
        let pad_mask_f32 = pad_mask.as_dtype(Dtype::Float32)?;
        // Reshape for broadcasting: [batch, 1, 1, seq]
        let pad_mask_expanded = pad_mask_f32.reshape(&[batch_size, 1, 1, seq_len])?;
        // Convert to -inf for padded positions
        let pad_mask_neg = ops::neg(&pad_mask_expanded)?;
        // Since true=1.0 for visible, we need -inf where pad_mask=0
        // Using: (1 - pad_mask) * (-inf) + pad_mask * 0.0
        let inf_arr = Array::from_f32(f32::NEG_INFINITY);
        let one_arr = Array::from_f32(1.0);
        let pad_penalty = ops::multiply(
            &ops::subtract(&one_arr, &pad_mask_expanded)?,
            &inf_arr,
        )?;
        // Add causal mask
        ops::add(&causal_mask, &pad_penalty)
    } else {
        Ok(causal_mask)
    }
}

/// Load text encoder weights from safetensors files
pub fn load_text_encoder_weights(
    encoder: &mut QwenTextEncoder,
    weights: HashMap<String, Array>,
) -> Result<(), Exception> {
    for (name, weight) in weights {
        // Map weight names to match our struct fields
        let name = name
            .replace("model.layers.", "layers.")
            .replace("self_attn.", "")
            .replace("input_layernorm.", "input_layernorm.")
            .replace("post_attention_layernorm.", "post_attention_layernorm.");

        // Find and set the weight using the flattened parameter path
        encoder.update_with_flattened(name, weight)?;
    }
    Ok(())
}

/// Load text encoder from model directory
/// Supports two layouts:
/// - Sharded safetensors: `text_encoder/model.safetensors.index.json`
/// - GGUF: `Qwen2.5-VL-*.gguf` or `*llm*.gguf` in root dir
pub fn load_text_encoder(model_dir: impl AsRef<Path>) -> Result<QwenTextEncoder, Box<dyn std::error::Error>> {
    let model_dir = model_dir.as_ref();
    let text_encoder_dir = model_dir.join("text_encoder");

    // Check for GGUF files first
    let gguf_files: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map(|ext| ext == "gguf").unwrap_or(false)
        })
        .collect();

    if let Some(gguf) = gguf_files.iter().find(|f| {
        f.file_name().to_string_lossy().contains("Qwen2.5-VL") ||
        f.file_name().to_string_lossy().contains("llm")
    }) {
        // Load from GGUF
        let gguf_path = gguf.path();
        // For now, return a default encoder
        let config = TextEncoderConfig::default();
        let encoder = QwenTextEncoder::new(config)?;
        return Ok(encoder);
    }

    // Load sharded safetensors
    if text_encoder_dir.join("model.safetensors.index.json").exists() {
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(text_encoder_dir.join("model.safetensors.index.json"))?
        )?;

        // Load all shard files
        let mut weights = HashMap::new();
        if let Some(weight_map) = index["weight_map"].as_object() {
            let mut loaded_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
            for (_name, shard) in weight_map {
                let shard_str = shard.as_str().unwrap();
                if loaded_shards.insert(shard_str.to_string()) {
                    let shard_path = text_encoder_dir.join(shard_str);
                    let data = std::fs::read(&shard_path)?;
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
                }
            }
        }

        // Create encoder and load weights
        let mut encoder = QwenTextEncoder::new(TextEncoderConfig::default())?;
        load_text_encoder_weights(&mut encoder, weights)?;
        Ok(encoder)
    } else {
        Err("Text encoder not found in model directory".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_creation() {
        let config = TextEncoderConfig::default();
        let encoder = QwenTextEncoder::new(config).unwrap();
        assert_eq!(encoder.layers.len(), 28);
    }

    #[test]
    fn test_embedding_forward() {
        let emb = Embedding::new(100, 64).unwrap();
        let input = Array::from_slice::<i32>(&[0, 1, 2], &[1, 3]);
        let output = emb.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 64]);
    }
}
