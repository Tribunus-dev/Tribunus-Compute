//! GLM-4.5 MoE (Mixture of Experts) model implementation
//!
//! This module implements the GLM-4.5 MoE architecture with:
//! - Partial RoPE (rotary position embedding on partial dimensions)
//! - Mixture of Experts with top-k routing
//! - Shared experts + routed experts
//! - 3-bit quantization support

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::ops::{
    addmm, argmin_all, arange, concatenate, expand_dims, full, gather,
    matmul, multiply, quantize, reshape, scatter_add, slice,
    softmax, split, sqrt, sum_all, take_along_axis, transpose, unsqueeze,
    where_op,
};
use mlx_rs::quantized;
use mlx_rs::transforms::{async_eval, eval, compile};
use mlx_rs::{array, Array, Dtype, Exception, Stream};
use mlx_rs::module::Module;
use mlx_rs::nn::{
    self, linear, LinearParameters, ModuleParameters, Quantizable, quantized_linear,
};

use mlx_rs_core::cache::{ConcatKeyValueCache, KVCache, KeyValueCache, QuantizedKVCache};
use mlx_rs_core::error::{Error, Result};
use mlx_rs_core::fused_swiglu;
use mlx_rs_core::utils::{
    create_attention_mask, scaled_dot_product_attention, AttentionMask, SdpaMask,
};

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quantize: Option<String>,
    pub group_size: Option<i32>,
    pub bits: Option<i32>,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: Option<i32>,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: Option<i32>,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    pub partial_rotary_factor: Option<f32>,
    pub attention_bias: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub head_dim: Option<i32>,
    // MoE-specific
    pub num_experts: Option<i32>,
    pub num_experts_per_tok: Option<i32>,
    pub first_k_dense_replace: Option<i32>,
    pub routed_scaling_factor: Option<f32>,
    pub n_group: Option<i32>,
    pub topk_group: Option<i32>,
    // Quantization
    pub quantization: Option<QuantizationConfig>,
}

fn default_max_position_embeddings() -> i32 { 131072 }
fn default_rope_theta() -> f32 { 1000000.0 }
fn default_partial_rotary_factor() -> f32 { 0.5 }
fn default_attention_bias() -> bool { true }
fn default_num_experts_per_tok() -> i32 { 8 }
fn default_first_k_dense_replace() -> i32 { 1 }
fn default_routed_scaling_factor() -> f32 { 1.0 }
fn default_n_group() -> i32 { 1 }
fn default_topk_group() -> i32 { 1 }

/// GLM4 MoE Attention with partial RoPE
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub wqkv: nn::QuantizedLinear,
    pub wo: nn::QuantizedLinear,
    pub n_kv_heads: i32,
    pub n_head: i32,
    pub n_rep: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    pub attention_bias: bool,
    pub kv_cache: bool,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
}

impl Attention {
    pub fn new(
        cfg: &ModelArgs,
        layer_idx: i32,
        group_size: i32,
        bits: i32,
        weights: &HashMap<String, Array>,
    ) -> Result<Self> {
        let n_head = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads.unwrap_or(n_head);
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / n_head);
        let rope_dim = (cfg.partial_rotary_factor.unwrap_or(default_partial_rotary_factor()) * head_dim as f32) as i32;
        let rope_theta = cfg.rope_theta.unwrap_or(default_rope_theta());
        let max_position_embeddings = cfg.max_position_embeddings.unwrap_or(default_max_position_embeddings());
        let attention_bias = cfg.attention_bias.unwrap_or(default_attention_bias());

        let wqkv = make_quantized_linear(weights, &format!("transformer.layers.{}.attention.wqkv", layer_idx), group_size, bits)?;
        let wo = make_quantized_linear(weights, &format!("transformer.layers.{}.attention.wo", layer_idx), group_size, bits)?;

        Ok(Self {
            wqkv,
            wo,
            n_kv_heads,
            n_head,
            n_rep: n_head / n_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            max_position_embeddings,
            attention_bias,
            kv_cache: true,
            rope_scaling: cfg.rope_scaling.clone(),
        })
    }
}

impl Attention {
    fn create_partial_rope_cache(&self, position_ids: &Array) -> Result<(Array, Array), Exception> {
        // Partial RoPE: only apply to first rope_dim dimensions
        let half = self.rope_dim / 2;
        let inv_freq: Array = arange(0, half, Dtype::Float32)?
            .map(|x: f32| 1.0 / self.rope_theta.powf(x as f32 / half as f32))?;

        let inv_freq = expand_dims(&inv_freq, &[0])?;
        let inv_freq = expand_dims(&inv_freq, &[1])?;

        let freqs = matmul(&position_ids?.astype(Dtype::Float32)?, &inv_freq)?;
        let emb = concatenate(&[&freqs, &freqs], -1)?;

        let cos_emb = emb.cos()?;
        let sin_emb = emb.sin()?;

        Ok((cos_emb, sin_emb))
    }

    fn partial_rotate_half(&self, x: &Array) -> Result<Array, Exception> {
        // Split x into [x1, x2] along last dim
        let d = x.shape().last().copied().unwrap_or(0);
        let half = d / 2;
        let x1 = slice(x, &[0, 0], &[-1, half], &[1, 1])?;
        let x2 = slice(x, &[0, half], &[-1, -1], &[1, 1])?;
        // Concatenate [-x2, x1]
        let neg_x2 = x2.neg()?;
        concatenate(&[&neg_x2, &x1], -1)
    }

    fn apply_partial_rope(
        &self,
        xq: &Array,
        xk: &Array,
        cos_cache: &Array,
        sin_cache: &Array,
    ) -> Result<(Array, Array), Exception> {
        // Partial RoPE: only apply to first rope_dim dimensions
        let xq_rotated = slice(xq, &[0, 0, 0, 0], &[-1, -1, -1, self.rope_dim], &[1, 1, 1, 1])?;
        let xq_pass = slice(xq, &[0, 0, 0, self.rope_dim], &[-1, -1, -1, -1], &[1, 1, 1, 1])?;

        let xk_rotated = slice(xk, &[0, 0, 0, 0], &[-1, -1, -1, self.rope_dim], &[1, 1, 1, 1])?;
        let xk_pass = slice(xk, &[0, 0, 0, self.rope_dim], &[-1, -1, -1, -1], &[1, 1, 1, 1])?;

        // Apply RoPE
        let xq_rot = (&xq_rotated * cos_cache - &self.partial_rotate_half(&xq_rotated)? * sin_cache)?;
        let xk_rot = (&xk_rotated * cos_cache - &self.partial_rotate_half(&xk_rotated)? * sin_cache)?;

        let xq_new = concatenate(&[&xq_rot, &xq_pass], -1)?;
        let xk_new = concatenate(&[&xk_rot, &xk_pass], -1)?;

        Ok((xq_new, xk_new))
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Result<Array, Exception>;

    fn forward(&self, input: AttentionInput<'_, C>) -> Self::Output {
        let AttentionInput { x, mask, cache } = input;
        let (B, T, C) = x.shape_vec().try_into().unwrap();

        let qkv = self.wqkv.forward(x)?;
        let qkv = qkv.reshape(&[B, T, -1])?;
        let n_kv_heads = self.n_kv_heads;
        let n_head = self.n_head;
        let head_dim = self.head_dim;
        let _n_rep = n_head / n_kv_heads;

        let splits = qkv.split(3, -1)?;
        let q = splits[0].reshape(&[B, T, n_head, head_dim])?;
        let k = splits[1].reshape(&[B, T, n_kv_heads, head_dim])?;
        let v = splits[2].reshape(&[B, T, n_kv_heads, head_dim])?;

        // Partial RoPE
        let position_ids = arange(0, T as i32, Dtype::Int32)?;
        let (cos_cache_, sin_cache_) = self.create_partial_rope_cache(&position_ids)?;
        let cos_cache = expand_dims(&cos_cache_, &[0, 2])?;
        let sin_cache = expand_dims(&sin_cache_, &[0, 2])?;

        let (q, k) = self.apply_partial_rope(&q, &k, &cos_cache, &sin_cache)?;

        // KVCache update
        let (k, v) = if let Some(cache) = cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };

        let attn_output = scaled_dot_product_attention(
            &q, &k, &v, mask, self.rope_theta,
        )?;

        let attn_output = attn_output.reshape(&[B, T, -1])?;
        let output = self.wo.forward(&attn_output)?;

        Ok(output)
    }
}

// Standard MLP
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MLP {
    pub gate_proj: nn::QuantizedLinear,
    pub up_proj: nn::QuantizedLinear,
    pub down_proj: nn::QuantizedLinear,
}

impl MLP {
    pub fn new(gate_proj: nn::QuantizedLinear, up_proj: nn::QuantizedLinear, down_proj: nn::QuantizedLinear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }
}

impl Module<&Array> for MLP {
    type Output = Result<Array, Exception>;
    fn forward(&self, x: &Array) -> Self::Output {
        let x = fused_swiglu::fused_swiglu_forward(x, &self.gate_proj, &self.up_proj)?;
        self.down_proj.forward(&x)
    }
}

// MoE Gate
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoEGate {
    pub n_group: i32,
    pub topk_group: i32,
    pub n_experts: i32,
    pub routed_scaling_factor: f32,
    pub gate: nn::Linear,
}

impl MoEGate {
    pub fn new(cfg: &ModelArgs, weights: &HashMap<String, Array>) -> Result<Self> {
        let n_experts = cfg.num_experts.unwrap_or(0);
        let n_group = cfg.n_group.unwrap_or(default_n_group());
        let topk_group = cfg.topk_group.unwrap_or(default_topk_group());
        let routed_scaling_factor = cfg.routed_scaling_factor.unwrap_or(default_routed_scaling_factor());

        // gate uses float weights (not quantized) for expert routing
        let weight = get_weight(weights, "transformer.gate.weight")?;
        let gate = nn::Linear::new(weight, None)?;

        Ok(Self { n_group, topk_group, n_experts, routed_scaling_factor, gate })
    }
}

impl MoEGate {
    pub fn forward(&self, x: &Array) -> Result<(Array, Array, Array, Array, f32), Exception> {
        let x = x.mean(&[-1], true, Dtype::Float32)?;
        let score = self.gate.forward(&x)?;
        let score = softmax(&score, -1)?;

        // Group-based expert selection (GLM4 style)
        let n_group = self.n_group;
        let topk_group = self.topk_group;
        let n_experts = self.n_experts;

        let scores = score.squeeze(None::<&[i32]>)?;
        let num_tokens = scores.shape()[0];
        let group_scores = scores
            .reshape(&[num_tokens, n_group, n_experts / n_group])?
            .max(&[-1])?;

        let group_idx = group_scores
            .topk(topk_group, -1)?
            .indices()?;

        let group_mask = full(&[num_tokens, n_experts as i32], 0.0_f32, Dtype::Float32)?;
        let mut final_mask = group_mask;

        for i in 0..n_group {
            let mask_val = group_idx.eq(i)?.astype(Dtype::Float32)?;
            let start = i * (n_experts / n_group);
            let end = (i + 1) * (n_experts / n_group);
            let slice_range: Vec<i32> = (start..end).collect();
            let slice_mask = mask_val.expand_dims(&[-1])?.broadcast(&[num_tokens, n_experts / n_group])?;
            let mut new_mask = final_mask;
            for (j, &pos) in slice_range.iter().enumerate() {
                let col = new_mask.index((.., pos))?;
                new_mask = new_mask.index_put(&[slice![.., pos]], &(col + slice_mask.index((.., j))?))?;
            }
            final_mask = new_mask;
        }

        // Apply group mask
        let score = score * final_mask;
        let num_experts_per_tok = 2;  // GLM4 uses top-2 for routed experts
        let (top_weights, top_indices) = score.topk(num_experts_per_tok, -1)?;
        let top_weights = softmax(&top_weights, -1)?;

        let routed_scaling_factor = self.routed_scaling_factor;
        let shared_weight = 1.0_f32;  // Shared expert always gets one route

        Ok((top_weights, top_indices, score, final_mask, routed_scaling_factor))
    }
}

// Quantized Switch Linear for MoE experts
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub weight: Array,
    pub scales: Array,
    pub biases: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedSwitchLinear {
    pub fn new(weight: Array, scales: Array, biases: Option<Array>, group_size: i32, bits: i32) -> Self {
        Self { weight, scales, biases, group_size, bits }
    }

    pub fn forward(&self, x: &Array, expert_indices: Option<&Array>) -> Result<Array, Exception> {
        let group_size = self.group_size;
        let bits = self.bits;

        match expert_indices {
            Some(indices) => {
                // Gather specific expert weights
                let w = gather(&self.weight, indices, 0)?;
                let s = gather(&self.scales, indices, 0)?;
                let b = self.biases.as_ref().map(|bias| gather(bias, indices, 0));

                // Quantized matmul with gathered weights
                let q_bits = bits;
                let q_group_size = group_size;
                quantized::quantized_matmul(x, &w, &s, b.as_ref(), q_bits, q_group_size)
            }
            None => {
                quantized::quantized_matmul(x, &self.weight, &self.scales, self.biases.as_ref(), bits, group_size)
            }
        }
    }
}

fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let num_tokens = x.shape()[0];
    let d_model = x.shape()[1];

    // Flatten indices and get sorted order
    let flat_indices = indices.reshape(&[-1])?;
    let sorted_indices = flat_indices.argsort(0, false)?;
    let sorted_indices_2d = sorted_indices.reshape(&[sorted_indices.shape()[0], 1])?;

    // Gather sorted x
    let sorted_x = gather(x, &sorted_indices_2d, 0)?;
    let sorted_x = sorted_x.reshape(&[num_tokens, d_model])?;

    // Inverse permutation
    let inverse_order = sorted_indices.argsort(0, false)?;

    // Gather sorted expert indices
    let sorted_expert_indices = gather(&flat_indices, &sorted_indices, 0)?;

    Ok((sorted_x, sorted_expert_indices, inverse_order))
}

fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> Result<Array, Exception> {
    let num_tokens = x.shape()[0];
    let d_model = x.shape()[1];

    let inv_order_2d = inv_order.reshape(&[num_tokens, 1])?;
    let result = gather(x, &inv_order_2d, 0)?;
    result.reshape(original_shape)
}

// SwitchGLU MLP for routed experts
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    pub gate_proj: QuantizedSwitchLinear,
    pub up_proj: QuantizedSwitchLinear,
    pub down_proj: QuantizedSwitchLinear,
}

impl SwitchGLU {
    pub fn new(
        gate_proj: QuantizedSwitchLinear,
        up_proj: QuantizedSwitchLinear,
        down_proj: QuantizedSwitchLinear,
    ) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Array, expert_indices: Option<&Array>) -> Result<Array, Exception> {
        // SwitchGLU with expert-specific weights
        let gate = self.gate_proj.forward(x, expert_indices)?;
        let up = self.up_proj.forward(x, expert_indices)?;
        let up = (&gate * &up.silu()?)?;  // SwiGLU activation
        let down = self.down_proj.forward(&up, expert_indices)?;
        Ok(down)
    }
}

// MoE block
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoE {
    pub shared_expert: MLP,
    pub gate: MoEGate,
    pub routed_experts: SwitchGLU,
    pub num_experts_per_tok: i32,
    pub n_experts: i32,
    pub first_k_dense_replace: i32,
}

impl Module<&Array> for MoE {
    type Output = Result<Array, Exception>;

    fn forward(&self, x: &Array) -> Self::Output {
        let orig_shape = x.shape().to_vec();
        let orig_shape_i32: Vec<i32> = orig_shape.iter().map(|&s| s as i32).collect();

        // Shared expert always runs
        let shared_output = self.shared_expert.forward(x)?;

        // Gate to determine routed expert selection
        let (top_weights, top_indices, _score, _final_mask, routed_scaling_factor) = self.gate.forward(x)?;

        // Gather inputs for each expert
        let num_tokens = x.shape()[0];
        let d_model = x.shape()[1];

        // Sort tokens by expert index for coalesced access
        let (sorted_x, sorted_expert_indices, inverse_order) = match top_indices.shape() {
            shape if shape.len() >= 2 => {
                let flat_indices = top_indices.reshape(&[-1])?;
                gather_sort(x, &flat_indices)?
            }
            _ => {
                gather_sort(x, top_indices)?
            }
        };

        // Run through routed experts
        let routed_output = self.routed_experts.forward(&sorted_x, Some(&sorted_expert_indices))?;

        // Restore original order
        let routed_output = scatter_unsort(&routed_output, &inverse_order, &orig_shape_i32)?;

        // Weight and combine
        let top_weights_3d = top_weights.reshape(&[-1, 1])?.broadcast(&[num_tokens, d_model])?;
        let routed_output = (&routed_output * &top_weights_3d * routed_scaling_factor)?;

        // Combine shared + routed
        let output = (shared_output + routed_output)?;

        Ok(output)
    }
}

// Decoder layer
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub attention: Attention,
    pub mlp: MoE,
    pub rms_norm1: nn::RMSNorm,
    pub rms_norm2: nn::RMSNorm,
    pub layer_idx: i32,
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Result<Array, Exception>;

    fn forward(&self, input: AttentionInput<'_, C>) -> Self::Output {
        let x = self.rms_norm1.forward(input.x)?;
        let attn_output = self.attention.forward(AttentionInput {
            x: &x,
            mask: input.mask,
            cache: input.cache,
        })?;
        let h = (input.x + &attn_output)?;
        let x_norm = self.rms_norm2.forward(&h)?;
        let mlp_output = self.mlp.forward(&x_norm)?;
        let output = (h + &mlp_output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LanguageModel {
    pub embedder: nn::QuantizedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: nn::RMSNorm,
    pub cfg: ModelArgs,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<C>,
}

impl<C> Module<ModelInput<'_, C>> for LanguageModel
where
    C: KeyValueCache + Default,
{
    type Output = Result<Array, Exception>;

    fn forward(&self, input: ModelInput<'_, C>) -> Self::Output {
        let ModelInput { inputs, mask, cache } = input;
        let h = self.embedder.forward(inputs)?;

        let mut h = h;
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = &mut cache[i];
            h = layer.forward(AttentionInput {
                x: &h,
                mask,
                cache: Some(cache),
            })?;
        }

        h = self.norm.forward(&h)?;
        let logits = self.embedder.as_linear()?.forward(&h)?;
        Ok(logits)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub model: LanguageModel,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Result<Array, Exception>;

    fn forward(&self, input: ModelInput<'_, C>) -> Self::Output {
        self.model.forward(input)
    }
}

// ==========================================
// Loading
// ==========================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    // Note: tokenizers library handles fast tokenizer initialization
    Ok(tokenizers::Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json"))
        .map_err(|e| Error::LoadError(format!("Failed to load tokenizer: {}", e)))?)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::LoadError(format!("Failed to read config.json: {}", e)))?;
    let cfg: ModelArgs = serde_json::from_str(&config_str)
        .map_err(|e| Error::LoadError(format!("Failed to parse config.json: {}", e)))?;
    Ok(cfg)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let safe_load = |path: &Path| -> Result<Array> {
        let content = std::fs::read(path)
            .map_err(|e| Error::LoadError(format!("Failed to read {}: {}", path.display(), e)))?;
        Array::from_slice(&[content], &[])
            .map_err(|e| Error::LoadError(format!("Failed to load weights from {}: {}", path.display(), e)))
    };

    let weight_map_path = model_dir.join("weight_map.json");
    let weight_map: WeightMap = if weight_map_path.exists() {
        let content = std::fs::read_to_string(&weight_map_path)
            .map_err(|e| Error::LoadError(format!("Failed to read weight_map.json: {}", e)))?;
        serde_json::from_str(&content)
            .map_err(|e| Error::LoadError(format!("Failed to parse weight_map.json: {}", e)))?
    } else {
        // No weight map - load all *.safetensors files
        let mut weights = HashMap::new();
        for entry in std::fs::read_dir(model_dir)
            .map_err(|e| Error::LoadError(format!("Failed to read model dir: {}", e)))?
        {
            let entry = entry.map_err(|e| Error::LoadError(format!("Dir entry error: {}", e)))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let arr = safe_load(&path)?;
                weights.insert(path.file_stem().unwrap().to_str().unwrap().to_string(), arr);
            }
        }
        return Ok(weights);
    };

    let mut all_weights = HashMap::new();
    for (weight_name, filename) in &weight_map.weight_map {
        let filepath = model_dir.join(filename);
        let file_weights = safe_load(&filepath)?;
        all_weights.insert(weight_name.clone(), file_weights);
    }
    Ok(all_weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::LoadError(format!("Weight not found: {}", key)))
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
    // Check if weights are already quantized (have scales)
    let scales_key = format!("{}.scales", prefix);
    let has_scales = weights.contains_key(&scales_key);

    if has_scales {
        // Load pre-quantized weights
        let qweight = get_weight(weights, &format!("{}.qweight", prefix))?;
        let scales = get_weight(weights, &format!("{}.scales", prefix))?;
        let biases = get_weight_optional(weights, &format!("{}.biases", prefix));

        // Rearrange for MLX: qweight shape may be [out_dim, in_dim/2] for 4-bit
        // MLX expects qweight shape [out_dim, in_dim * bits / 32, 32 / bits]
        let qweight = qweight.reshape(&[-1])?;
        let scales = scales.reshape(&[-1])?;
        let biases = biases.map(|b| b.reshape(&[-1]).unwrap());

        nn::QuantizedLinear::from_quantized(qweight, scales, biases, group_size, bits)
            .map_err(|e| Error::LoadError(format!("Failed to create quantized linear {}: {}", prefix, e)))
    } else {
        // Load float weights and quantize
        let weight = get_weight(weights, &format!("{}.weight", prefix))?;
        let (qweight, scales, biases) = quantize(&weight, group_size, bits)?;
        nn::QuantizedLinear::from_quantized(qweight, scales, biases, group_size, bits)
            .map_err(|e| Error::LoadError(format!("Failed to quantize linear {}: {}", prefix, e)))
    }
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let (qweight, scales, biases) = quantize(&weight, group_size, bits)?;

    nn::QuantizedEmbedding::from_quantized(qweight, scales, biases, group_size, bits)
        .map_err(|e| Error::LoadError(format!("Failed to quantize embedding {}: {}", prefix, e)))
}

fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let qweight = get_weight(weights, &format!("{}.qweight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight_optional(weights, &format!("{}.biases", prefix));
    Ok(QuantizedSwitchLinear::new(qweight, scales, biases, group_size, bits))
}

fn make_quantized_mlp(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MLP, Error> {
    let gate_proj = make_quantized_linear(weights, &format!("{}.gate_proj", prefix), group_size, bits)?;
    let up_proj = make_quantized_linear(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
    let down_proj = make_quantized_linear(weights, &format!("{}.down_proj", prefix), group_size, bits)?;
    Ok(MLP::new(gate_proj, up_proj, down_proj))
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let start = Instant::now();

    let cfg = get_model_args(model_dir)?;
    let weights = load_all_weights(model_dir)?;

    let group_size = cfg.quantization.as_ref().map(|q| q.group_size.unwrap_or(default_group_size())).unwrap_or(default_group_size());
    let bits = cfg.quantization.as_ref().map(|q| q.bits.unwrap_or(default_bits())).unwrap_or(default_bits());

    eprintln!("  Model config: {} layers, {} heads, {} hidden dim, {} experts, {} bits",
              cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size,
              cfg.num_experts.unwrap_or(0), bits);

    let embedder = make_quantized_embedding(&weights, "transformer.word_embeddings", group_size, bits)?;
    let norm = nn::RMSNorm::new(
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )?;
    if let Some(norm_weight) = get_weight_optional(&weights, "transformer.final_layernorm.weight") {
        norm.weight.set(norm_weight);
    }

    let num_hidden_layers = cfg.num_hidden_layers as usize;
    let num_experts = cfg.num_experts.unwrap_or(0);
    let first_k_dense_replace = cfg.first_k_dense_replace.unwrap_or(default_first_k_dense_replace());

    let mut layers = Vec::with_capacity(num_hidden_layers);
    for layer_idx in 0..num_hidden_layers {
        let attention = Attention::new(&cfg, layer_idx as i32, group_size, bits, &weights)?;

        let mlp = if layer_idx < first_k_dense_replace as usize {
            // Dense layer
            let prefix = format!("transformer.layers.{}.mlp", layer_idx);
            let gate_proj = make_quantized_linear(&weights, &format!("{}.gate_proj", prefix), group_size, bits)?;
            let up_proj = make_quantized_linear(&weights, &format!("{}.up_proj", prefix), group_size, bits)?;
            let down_proj = make_quantized_linear(&weights, &format!("{}.down_proj", prefix), group_size, bits)?;
            let shared_expert = MLP::new(gate_proj, up_proj, down_proj);

            // For dense layers, routed experts are identity
            let d_model = cfg.hidden_size;
            let routed_gate = make_quantized_switch_linear(
                &weights,
                &format!("transformer.layers.{}.mlp.shared_experts.gate", layer_idx),
                group_size, bits,
            )?;
            // ... dummy routed experts for dense layers
            let routed_up = make_quantized_switch_linear(
                &weights,
                &format!("transformer.layers.{}.mlp.shared_experts.up", layer_idx),
                group_size, bits,
            )?;
            let routed_down = make_quantized_switch_linear(
                &weights,
                &format!("transformer.layers.{}.mlp.shared_experts.down", layer_idx),
                group_size, bits,
            )?;

            let gate = MoEGate::new(&cfg, &weights)?;
            let routed_experts = SwitchGLU::new(routed_gate, routed_up, routed_down);

            MoE {
                shared_expert,
                gate,
                routed_experts,
                num_experts_per_tok: cfg.num_experts_per_tok.unwrap_or(default_num_experts_per_tok()),
                n_experts: num_experts,
                first_k_dense_replace,
            }
        } else {
            // Actual MoE layer
            let prefix = format!("transformer.layers.{}.mlp.shared_experts", layer_idx);
            let shared_gate = make_quantized_linear(&weights, &format!("{}.gate_proj", &prefix), group_size, bits)?;
            let shared_up = make_quantized_linear(&weights, &format!("{}.up_proj", &prefix), group_size, bits)?;
            let shared_down = make_quantized_linear(&weights, &format!("{}.down_proj", &prefix), group_size, bits)?;
            let shared_expert = MLP::new(shared_gate, shared_up, shared_down);

            // Routed experts (switch linear)
            let prefix = format!("transformer.layers.{}.mlp.experts", layer_idx);
            let routed_gate = make_quantized_switch_linear(&weights, &format!("{}.gate_proj", &prefix), group_size, bits)?;
            let routed_up = make_quantized_switch_linear(&weights, &format!("{}.up_proj", &prefix), group_size, bits)?;
            let routed_down = make_quantized_switch_linear(&weights, &format!("{}.down_proj", &prefix), group_size, bits)?;

            let gate = MoEGate::new(&cfg, &weights)?;
            let routed_experts = SwitchGLU::new(routed_gate, routed_up, routed_down);

            MoE {
                shared_expert,
                gate,
                routed_experts,
                num_experts_per_tok: cfg.num_experts_per_tok.unwrap_or(default_num_experts_per_tok()),
                n_experts: num_experts,
                first_k_dense_replace,
            }
        };

        let rms_norm1 = nn::RMSNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        if let Some(norm_weight) = get_weight_optional(&weights, &format!("transformer.layers.{}.attention_norm.weight", layer_idx)) {
            rms_norm1.weight.set(norm_weight);
        }

        let rms_norm2 = nn::RMSNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        if let Some(norm_weight) = get_weight_optional(&weights, &format!("transformer.layers.{}.ffn_norm.weight", layer_idx)) {
            rms_norm2.weight.set(norm_weight);
        }

        layers.push(DecoderLayer {
            attention,
            mlp,
            rms_norm1,
            rms_norm2,
            layer_idx: layer_idx as i32,
        });
    }

    let model = LanguageModel {
        embedder,
        layers,
        norm,
        cfg: cfg.clone(),
    };

    eprintln!("  Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    Ok(Model { model })
}

// ==========================================
// Generation
// ==========================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    if temp < 1e-6 {
        // Greedy
        let token = argmin_all(logits)?;
        Ok(Array::from_slice(&[token.item::<u32>()], &[1, 1])?)
    } else {
        let logits = logits / temp;
        let probs = softmax(&logits, -1)?;
        let token = probs.random_categorical(1)?;
        Ok(token)
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<C>,
    temp: f32,
    y: Array,
    step: usize,
    max_tokens: usize,
}

pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt: &Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            y: prompt.clone(),
            step: 0,
            max_tokens: 512,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn prefill(&mut self) -> Result<Array, Exception> {
        let input = ModelInput {
            inputs: &self.y,
            mask: None,
            cache: &mut self.cache,
        };
        let logits = self.model.forward(input)?;
        let next_token = sample(&logits.index((.., -1, ..))?, self.temp)?;
        self.y = next_token;
        self.step = 1;
        Ok(self.y.clone())
    }

    fn generate_step(&mut self) -> Result<Array, Exception> {
        let x = self.y.index((.., NewAxis))?;
        let input = ModelInput {
            inputs: &x,
            mask: None,
            cache: &mut self.cache,
        };
        let logits = self.model.forward(input)?;
        let next_token = sample(&logits, self.temp)?;
        self.y = next_token.clone();
        self.step += 1;
        Ok(next_token)
    }
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.max_tokens {
            return None;
        }
        if self.step == 0 {
            Some(self.prefill())
        } else {
            Some(self.generate_step())
        }
    }
}
