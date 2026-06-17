//! GLM-4 Model Implementation
//!
//! Features:
//! - Partial RoPE (rotary position embedding on half of dimensions)
//! - Fused gate_up_proj in MLP
//! - Extra LayerNorms (post_self_attn, post_mlp)

use std::{collections::HashMap, fs, path::Path, sync::Arc};

use mlx_rs::{
    linalg::copy,
    ops::{
        indexing::{IndexOp, NewAxis},
        self,
    },
    quantized,
    random,
    transforms::eval,
    Array, Device, Dtype, Stream, StreamOrDevice,
};
use mlx_rs::{
    module::{ModuleParameters, Module, Quantizable},
    nn,
};
use mlx_sys::{
    metal,
    c_bridge::mlx_pow,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::KeyValueCache,
    error::{Error, Result},
    utils::{create_attention_mask, scaled_dot_product_attention, AttentionMask, SdpaMask},
};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }
fn default_max_position_embeddings() -> i32 { 32768 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_partial_rotary_factor() -> f32 { 0.5 }
fn default_attention_bias() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: Option<f32>,
    pub vocab_size: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    pub head_dim: Option<i32>,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 40,
            intermediate_size: 13696,
            num_key_value_heads: 2,
            rms_norm_eps: None,
            vocab_size: 151552,
            max_position_embeddings: 32768,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            attention_bias: true,
            quantization: None,
            head_dim: None,
        }
    }
}

// ============================================================================
// GLM4 Attention with Partial RoPE
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Attention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    pub attention_bias: bool,
    pub kv_cache: Option<nn::KVCache>,
}

impl Glm4Attention {
    pub fn new(args: &ModelArgs) -> Self {
        let hidden_size = args.hidden_size;
        let num_heads = args.num_attention_heads;
        let num_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim.unwrap_or(hidden_size / num_heads);

        Self {
            q_proj: nn::Linear::new(hidden_size, num_heads * head_dim, args.attention_bias),
            k_proj: nn::Linear::new(hidden_size, num_kv_heads * head_dim, args.attention_bias),
            v_proj: nn::Linear::new(hidden_size, num_kv_heads * head_dim, args.attention_bias),
            o_proj: nn::Linear::new(num_heads * head_dim, hidden_size, false),
            num_heads,
            num_kv_heads,
            head_dim,
            partial_rotary_factor: args.partial_rotary_factor,
            rope_theta: args.rope_theta,
            max_position_embeddings: args.max_position_embeddings,
            attention_bias: args.attention_bias,
            kv_cache: None,
        }
    }

    // Partial RoPE: only apply to first `partial_rotary_factor * head_dim` dimensions
    fn partial_rope(&self, x: &Array, offset: i32) -> Result<Array> {
        let dim = (self.head_dim as f32 * self.partial_rotary_factor) as i32;
        let half = dim / 2;

        // Compute frequencies
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.rope_theta.powf(i as f32 / half as f32))
            .collect();
        let inv_freq = Array::from_slice(&inv_freq, &[half]);

        // Compute sin/cos for position offset
        let positions: Vec<i32> = (offset..offset + x.shape().last().unwrap_or(&1))
            .map(|i| i as i32)
            .collect();
        let positions = Array::from_slice(&positions, &[positions.len() as i32]);

        let freqs = ops::matmul(&positions, &inv_freq.reshape(&[1, half]))?;
        let emb = ops::concatenate(&[&freqs, &freqs], -1)?;
        let cos = emb.cos();
        let sin = emb.sin();

        // Apply to first dim dimensions
        let x_part = x.slice_axis(-1, 0, dim, 1);
        let x_part_rotated = x_part.clone();

        // x_rotated = x * cos + rotate_half(x) * sin
        let x_half = x_part.slice_axis(-1, half, dim - half, 1);
        let x_half = ops::concatenate(&[&x_half.neg()?, &x_part.slice_axis(-1, 0, half, 1)], -1)?;

        let rotated = &x_part_rotated * &cos + &x_half * &sin;

        // Combine with un-rotated part
        let x_rest = x.slice_axis(-1, dim, x.shape().last().unwrap_or(&0) - dim, 1);
        Ok(ops::concatenate(&[&rotated, &x_rest], -1)?)
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Option<(C, usize)>,
}

impl<C> Module<AttentionInput<'_, C>> for Glm4Attention
where
    C: KeyValueCache,
{
    type Output = Result<Array>;

    fn module(&self, input: AttentionInput<'_, C>) -> Self::Output {
        let AttentionInput { x, mask, cache } = input;
        let (b, s, _) = x.shape3()?;

        let query = self.q_proj.module(x)?;
        let key = self.k_proj.module(x)?;
        let value = self.v_proj.module(x)?;

        // Reshape for multi-head attention
        let query = query.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let key = key.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let value = value.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Apply partial RoPE
        let offset = match cache {
            Some((c, _)) => c.offset(),
            None => 0,
        };
        let query = self.partial_rope(&query, offset)?;
        let key = self.partial_rope(&key, offset)?;

        // KV cache update
        let (key, value) = match cache {
            Some((cache, _)) => {
                let (k, v) = cache.update(&key, &value)?;
                (k, v)
            }
            None => (key, value),
        };

        // SDPA
        let output = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            mask.copied(),
            SdpaMask::Causal,
        )?;

        // Reshape and project
        let output = output.reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_proj.module(&output)
    }
}

// ============================================================================
// GLM4 MLP with Fused gate_up_proj
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Mlp {
    pub gate_up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl Glm4Mlp {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            gate_up_proj: nn::Linear::new(args.hidden_size, 2 * args.intermediate_size, false),
            down_proj: nn::Linear::new(args.intermediate_size, args.hidden_size, false),
        }
    }
}

impl Module<&Array> for Glm4Mlp {
    type Output = Result<Array>;

    fn module(&self, x: &Array) -> Self::Output {
        let x = self.gate_up_proj.module(x)?;

        // Split into gate and up (each intermediate_size)
        let split = x.split_axis(-1, 1, 2)?;
        let gate = &split[0];
        let up = &split[1];

        // silu(gate) * up
        let gate_act = ops::silu(gate)?;
        let hidden = &gate_act * up;

        self.down_proj.module(&hidden)
    }
}

// ============================================================================
// GLM4 Decoder Layer with Extra LayerNorms
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4DecoderLayer {
    pub self_attn: Glm4Attention,
    pub mlp: Glm4Mlp,
    pub input_layernorm: nn::LayerNorm,
    pub post_self_attn_layernorm: nn::LayerNorm,
    pub post_attention_layernorm: nn::LayerNorm,
    pub post_mlp_layernorm: nn::LayerNorm,
}

impl Glm4DecoderLayer {
    pub fn new(args: &ModelArgs) -> Self {
        let eps = args.rms_norm_eps.unwrap_or(1e-5);
        Self {
            self_attn: Glm4Attention::new(args),
            mlp: Glm4Mlp::new(args),
            input_layernorm: nn::LayerNorm::new(args.hidden_size, eps),
            post_self_attn_layernorm: nn::LayerNorm::new(args.hidden_size, eps),
            post_attention_layernorm: nn::LayerNorm::new(args.hidden_size, eps),
            post_mlp_layernorm: nn::LayerNorm::new(args.hidden_size, eps),
        }
    }
}

impl<C> Module<AttentionInput<'_, C>> for Glm4DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Result<Array>;

    fn module(&self, input: AttentionInput<'_, C>) -> Self::Output {
        // Pre-attention norm
        let residual = input.x;
        let x = self.input_layernorm.module(input.x)?;

        // Attention with post-attention norm
        let attn_input = AttentionInput {
            x: &x,
            mask: input.mask,
            cache: input.cache,
        };
        let x = self.self_attn.module(attn_input)?;
        let x = self.post_self_attn_layernorm.module(&x)?;
        let x = &x + residual;

        // Pre-MLP norm
        let residual = x.clone();
        let x = self.post_attention_layernorm.module(&x)?;

        // MLP with post-MLP norm
        let x = self.mlp.module(&x)?;
        let x = self.post_mlp_layernorm.module(&x)?;
        Ok(&x + residual)
    }
}

// ============================================================================
// GLM4 Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Model {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Glm4DecoderLayer>,
    pub final_layernorm: nn::LayerNorm,
}

impl Glm4Model {
    pub fn new(args: &ModelArgs) -> Self {
        let eps = args.rms_norm_eps.unwrap_or(1e-5);
        Self {
            embed_tokens: nn::Embedding::new(args.vocab_size, args.hidden_size),
            layers: (0..args.num_hidden_layers)
                .map(|_| Glm4DecoderLayer::new(args))
                .collect(),
            final_layernorm: nn::LayerNorm::new(args.hidden_size, eps),
        }
    }
}

pub struct ModelInput<'a, C> {
    pub input_ids: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<(C, usize)>>,
}

impl<C> Module<ModelInput<'_, C>> for Glm4Model
where
    C: KeyValueCache + Default,
{
    type Output = Result<Array>;

    fn module(&self, input: ModelInput<'_, C>) -> Self::Output {
        let ModelInput { input_ids, mask, cache } = input;

        let x = self.embed_tokens.module(input_ids)?;

        // Process through layers
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = &mut cache[i];
            let attn_input = AttentionInput {
                x: &x,
                mask,
                cache,
            };
            x = layer.module(attn_input)?;
        }

        self.final_layernorm.module(&x)
    }
}

// ============================================================================
// Full Model with LM Head
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub model: Glm4Model,
    pub lm_head: nn::Linear,
}

impl Model {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            model: Glm4Model::new(args),
            lm_head: nn::Linear::new(args.hidden_size, args.vocab_size, false),
        }
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Result<Array>;

    fn module(&self, input: ModelInput<'_, C>) -> Self::Output {
        let x = self.model.module(input)?;
        let logits = self.lm_head.module(&x)?;

        // Only return last token's logits
        let last_idx = logits.shape().last().unwrap_or(&1) - 1;
        let logits = logits.index(IndexOp::new(NewAxis, last_idx as i64))?;

        Ok(logits)
    }
}

// ============================================================================
// Model Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::Tokenizer(tokenizer_path.display().to_string(), e.to_string()))
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| Error::Config(config_path.display().to_string(), e.to_string()))?;
    serde_json::from_str(&config_str)
        .map_err(|e| Error::Config(config_path.display().to_string(), e.to_string()))
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    #[serde(flatten)]
    weights: HashMap<String, String>,
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    // Check if quantized
    if args.quantization.is_some() {
        return load_model_quantized(model_dir, &args);
    }

    let weights = load_all_weights(model_dir)?;
    let mut model = Model::new(&args);

    // Load each parameter from the weight map
    let params = model.parameters();
    for (name, param) in params {
        if let Some(weight) = weights.get(&name) {
            let array = weight.clone();
            param.copy_from(&array)?;
        }
    }

    Ok(model)
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let mut weights = HashMap::new();

    // Read weight map
    let weight_map_path = model_dir.join("weight_map.json");
    if weight_map_path.exists() {
        let weight_map_str = fs::read_to_string(&weight_map_path)
            .map_err(|e| Error::Config(weight_map_path.display().to_string(), e.to_string()))?;
        let weight_map: WeightMap = serde_json::from_str(&weight_map_str)
            .map_err(|e| Error::Config(weight_map_path.display().to_string(), e.to_string()))?;

        // Load each weight file
        let mut loaded_files: HashMap<String, Vec<Array>> = HashMap::new();
        for (name, filename) in &weight_map.weights {
            let file_path = model_dir.join(filename);
            if !loaded_files.contains_key(filename) {
                let arrays = unsafe { mlx_rs::io::load_safetensors(&file_path)? };
                loaded_files.insert(filename.clone(), arrays);
            }
            if let Some(arrays) = loaded_files.get(filename) {
                if let Some(array) = arrays.iter().find(|a| a.name() == name.as_str()) {
                    weights.insert(name.clone(), array.clone());
                }
            }
        }
    } else {
        // Fallback: load all .safetensors files
        for entry in fs::read_dir(model_dir)
            .map_err(|e| Error::IO(e.to_string()))?
        {
            let entry = entry.map_err(|e| Error::IO(e.to_string()))?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                let arrays = unsafe { mlx_rs::io::load_safetensors(&path)? };
                for array in arrays {
                    weights.insert(array.name().to_string(), array);
                }
            }
        }
    }

    Ok(weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key).cloned().ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let scales = get_weight(weights, &format!("{prefix}.scales"))?;
    let biases = get_weight_optional(weights, &format!("{prefix}.bias"));

    Ok(nn::QuantizedLinear::new(w, scales, group_size, bits, biases))
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let scales = get_weight(weights, &format!("{prefix}.scales"))?;

    Ok(nn::QuantizedEmbedding::new(w, scales, group_size, bits))
}

fn load_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let config = args.quantization.as_ref().unwrap();
    let group_size = config.group_size;
    let bits = config.bits;

    let weights = load_all_weights(model_dir)?;
    let mut model = Model::new(args);

    // For quantized models, we need to handle the quantization loading
    // This is a simplified version - actual loading would iterate all parameters
    // and call make_quantized_linear for each
    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, Exception> {
    if temp == 0.0 {
        // Greedy
        let logits = ops::argmax(logits, -1, true)?;
        return Ok(logits);
    }

    let logits = &(&*logits / temp);
    let probs = ops::softmax(logits, -1)?;
    let sample = random::categorical(&probs, 1)?;
    Ok(sample)
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<(C, usize)>>,
    temp: f32,
    tokens: Vec<Array>,
    done: bool,
    eos_token: Option<i32>,
}

pub enum GenerateState<'a> {
    Ready(&'a mut [Array]),
    Done,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<(C, usize)>>,
        temp: f32,
        prompt: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            tokens: prompt.iter().map(|t| Array::from_slice(&[*t], &[1])).collect(),
            done: false,
            eos_token: None,
        }
    }

    pub fn generate(&mut self) -> Result<Option<Array>> {
        if self.done {
            return Ok(None);
        }

        let input_ids = if self.tokens.len() == 1 {
            self.tokens[0].clone()
        } else {
            ops::concatenate(
                &self.tokens.iter().map(|t| t.as_ref()).collect::<Vec<_>>(),
                0,
            )?
        };

        let input = ModelInput {
            input_ids: &input_ids,
            mask: None,
            cache: self.cache,
        };

        let logits = self.model.module(input)?;
        let next_token = sample(&logits, self.temp)?;

        // Check for EOS
        if let Some(eos) = self.eos_token {
            if next_token.item::<i32>() == eos {
                self.done = true;
                return Ok(None);
            }
        }

        // Update tokens for next iteration (only last token matters in step mode)
        self.tokens = vec![next_token.clone()];

        Ok(Some(next_token))
    }
}

#[macro_export]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        self.generate().transpose()
    }
}
