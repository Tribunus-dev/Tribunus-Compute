//! Mixtral MoE Model Implementation
//!
//! Mixtral is a Mixture of Experts (MoE) model from Mistral AI.
//! Uses top-k routing with softmax scores on selected experts.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    self as mx,
    array::Array,
    ops::{
        self,
        indexing::{IndexOp, NewAxis},
        quantization::{AffineQuantizeMode, QuantizedPrimitive},
        creation::{arange, full, zeros},
    },
    random,
    transforms::eval,
    quantized,
    nn::{
        self,
        module::{Module as ModuleTrait, ModuleParameters},
        Embedding, QuantizedEmbedding, QuantizedLinear,
    },
    stream::Stream,
    MetalDevice,
};
use mlx_rs_core::{
    cache::{KeyValueCache},
    error::{Error, Result},
    fused_swiglu,
    utils::{create_attention_mask, scaled_dot_product_attention, AttentionMask, SdpaMask},
    load_tokenizer,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

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

fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}
fn default_vocab_size() -> i32 {
    32000
}
fn default_hidden_size() -> i32 {
    4096
}
fn default_intermediate_size() -> i32 {
    14336
}
fn default_num_hidden_layers() -> i32 {
    32
}
fn default_num_attention_heads() -> i32 {
    32
}
fn default_num_experts_per_tok() -> i32 {
    2
}
fn default_num_local_experts() -> i32 {
    8
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    1e6
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default)]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: i32,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    fn kv_dim(&self) -> i32 {
        self.head_dim() * self.num_key_value_heads()
    }

    fn num_key_value_heads(&self) -> i32 {
        if self.num_key_value_heads > 0 {
            self.num_key_value_heads
        } else {
            self.num_attention_heads
        }
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    wq: QuantizedLinear,
    wk: QuantizedLinear,
    wv: QuantizedLinear,
    wo: QuantizedLinear,
    n_heads: i32,
    n_kv_heads: i32,
    n_rep: i32,
    head_dim: i32,
    scale: f32,
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Option<C>,
}

impl<C: KeyValueCache> Module<AttentionInput<'_, C>> for Attention {
    type Output = Array;

    fn module(&self, input: AttentionInput<'_, C>) -> Self::Output {
        let AttentionInput { x, mask, cache } = input;
        let (batch_size, seq_len, _) = x.shape3().unwrap();
        let (B, n_kv_head) = (batch_size, self.n_kv_heads as usize);

        let queries = self.wq.module(x); // (1, 1, hidden)
        let keys = self.wk.module(x);
        let values = self.wv.module(x);

        // Reshape for attention
        let queries = queries
            .reshape(&[B, seq_len, self.n_heads, self.head_dim])
            .unwrap();
        let keys = keys
            .reshape(&[B, seq_len, n_kv_head, self.head_dim])
            .unwrap();
        let values = values
            .reshape(&[B, seq_len, n_kv_head, self.head_dim])
            .unwrap();

        // Apply KV cache
        let (keys, values) = if let Some(cache) = cache {
            cache.update_and_fetch(&keys, &values)
        } else {
            (keys, values)
        };

        // Repeat KV heads
        let keys = ops::repeat(&keys, self.n_rep as usize, 2).unwrap();
        let values = ops::repeat(&values, self.n_rep as usize, 2).unwrap();

        // Transpose for SDPA
        let queries = queries.transpose(&[0, 2, 1, 3]).unwrap();
        let keys = keys.transpose(&[0, 2, 1, 3]).unwrap();
        let values = values.transpose(&[0, 2, 1, 3]).unwrap();

        // Scaled dot-product attention
        let output = scaled_dot_product_attention(&queries, &keys, &values, mask, self.scale);

        // Concatenate heads
        let output = output
            .transpose(&[0, 2, 1, 3])
            .unwrap()
            .reshape(&[B, seq_len, self.n_heads * self.head_dim])
            .unwrap();

        self.wo.module(&output)
    }
}

// ============================================================================
// Quantized Switch Linear for MoE
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
    in_features: i32,
    out_features: i32,
    num_experts: i32,
}

impl QuantizedSwitchLinear {
    fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        in_features: i32,
        out_features: i32,
        num_experts: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            in_features,
            out_features,
            num_experts,
        }
    }

    fn module(&self, x: &Array, expert_indices: &Array) -> Array {
        // Gather quantized weights for selected experts and dispatch
        let w = ops::gather(&self.weight, expert_indices, 0, 0).unwrap();
        let s = ops::gather(&self.scales, expert_indices, 0, 0).unwrap();
        let b = self.biases.as_ref().map(|b| ops::gather(b, expert_indices, 0, 0).unwrap());

        // Compute quantized matmul with gathered weights
        // gather_qmm: efficient batched quantized matmul for MoE dispatch
        let result = quantized::gather_qmm(
            x,
            &w,
            &s,
            b.as_ref(),
            self.group_size,
            self.bits,
            self.in_features,
            self.out_features,
        ).unwrap();

        result
    }
}

fn gather_sort(x: &Array, indices: &Array) -> std::result::Result<(Array, Array, Array), Exception> {
    let flat_indices = indices.reshape(&[-1])?;
    let sorted_indices = ops::argsort(&flat_indices, None)?;
    let inv_order = ops::argsort(&sorted_indices, None)?;
    let x_gathered = ops::gather(x, &sorted_indices, 0, 0)?;
    Ok((x_gathered, sorted_indices, inv_order))
}

fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> std::result::Result<Array, Exception> {
    let x_scattered = ops::gather(x, inv_order, 0, 0)?;
    x_scattered.reshape(original_shape)
}

// ============================================================================
// SwitchGLU MLP for routed experts
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    w1: QuantizedSwitchLinear,
    w2: QuantizedSwitchLinear,
    w3: QuantizedSwitchLinear,
}

impl SwitchGLU {
    fn module(&self, x: &Array, expert_indices: &Array) -> Array {
        let up = self.w3.module(x, expert_indices);
        let gate = self.w1.module(x, expert_indices);

        // Fused SwiGLU: silu(gate) * up — custom Metal kernel
        // This is 10-12x faster than separate operations per expert
        let activated = fused_swiglu(&up, &gate);

        self.w2.module(&activated, expert_indices)
    }
}

// ============================================================================
// Mixtral Sparse MoE Block
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct MixtralSparseMoeBlock {
    gate: QuantizedLinear,
    experts: SwitchGLU,
    num_experts_per_tok: i32,
    num_local_experts: i32,
}

impl Module<&Array> for MixtralSparseMoeBlock {
    type Output = Array;

    fn module(&self, x: &Array) -> Self::Output {
        let (batch_size, seq_len, hidden_size) = x.shape3().unwrap();

        // Gate logits
        let gate_logits = self.gate.module(x); // (B, S, num_local_experts)

        // Top-k routing
        let (gate_scores, expert_indices) = ops::topk(&gate_logits, self.num_experts_per_tok as usize, -1).unwrap();

        // Softmax over selected experts only
        let routing_weights = ops::softmax(&gate_scores, -1, None).unwrap();

        // Flatten batch and sequence dimensions for expert dispatch
        let x_flat = x.reshape(&[-1, hidden_size]).unwrap();
        let routing_flat = routing_weights.reshape(&[-1, self.num_experts_per_tok]).unwrap();
        let indices_flat = expert_indices.reshape(&[-1, self.num_experts_per_tok]).unwrap();

        // Initialize output buffer (zeros)
        let mut final_hidden_states = Array::zeros::<f32>(&[batch_size * seq_len, hidden_size]);

        // Dispatch to experts
        for i in 0..self.num_experts_per_tok {
            let indices_for_expert = indices_flat.index(NewAxis, Ellipsis, i..=i).unwrap();
            let routing_for_expert = routing_flat.index(NewAxis, Ellipsis, i..=i).unwrap();

            let expert_output = self.experts.module(&x_flat, &indices_for_expert);

            // Weighted sum
            let weighted = &expert_output * &routing_for_expert;
            final_hidden_states = &final_hidden_states + &weighted;
        }

        final_hidden_states.reshape(&[batch_size, seq_len, hidden_size]).unwrap()
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    self_attn: Attention,
    block_sparse_moe: MixtralSparseMoeBlock,
    input_layernorm: nn::RMSNorm,
    post_attention_layernorm: nn::RMSNorm,
}

impl<C: KeyValueCache> Module<AttentionInput<'_, C>> for DecoderLayer {
    type Output = Array;

    fn module(&self, input: AttentionInput<'_, C>) -> Self::Output {
        let x = input.x;

        // Self-attention with pre-norm
        let residual = x;
        let h = self.input_layernorm.module(x);
        let attn_input = AttentionInput {
            x: &h,
            mask: input.mask,
            cache: input.cache,
        };
        let h = self.self_attn.module(attn_input);
        let h = &h + residual;

        // MoE block with pre-norm
        let residual = &h;
        let h = self.post_attention_layernorm.module(&h);
        let h = self.block_sparse_moe.module(&h);
        &h + residual
    }
}

// ============================================================================
// Mixtral Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct MixtralModel {
    tok_embeddings: QuantizedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: nn::RMSNorm,
    output: QuantizedLinear,
    args: ModelArgs,
}

pub struct ModelInput<'a, C> {
    pub input_ids: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C: KeyValueCache + Default> Module<ModelInput<'_, C>> for MixtralModel {
    type Output = Array;

    fn module(&self, input: ModelInput<'_, C>) -> Self::Output {
        let ModelInput { input_ids, mask, cache } = input;

        let mut h = self.tok_embeddings.module(input_ids);

        for (layer, cache) in self.layers.iter().zip(cache.iter_mut()) {
            let attn_input = AttentionInput {
                x: &h,
                mask,
                cache,
            };
            h = layer.module(attn_input);
        }

        h = self.norm.module(&h);
        self.output.module(&h)
    }
}

// ============================================================================
// Full Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    model: MixtralModel,
    args: ModelArgs,
}

impl Model {
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C: KeyValueCache + Default> Module<ModelInput<'_, C>> for Model {
    type Output = Array;

    fn module(&self, input: ModelInput<'_, C>) -> Self::Output {
        self.model.module(input)
    }
}

// ============================================================================
// Model Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Io(format!("Failed to read config: {}", e)))?;
    let mut args: ModelArgs = serde_json::from_str(&config_str)
        .map_err(|e| Error::Io(format!("Failed to parse config: {}", e)))?;

    // Set num_key_value_heads if not present
    if args.num_key_value_heads == 0 {
        args.num_key_value_heads = args.num_attention_heads;
    }

    Ok(args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub filename: String,
    pub shape: Vec<i32>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let mut weights = HashMap::new();
    let weight_map_path = model_dir.join("weight_map.json");

    if weight_map_path.exists() {
        let weight_map: HashMap<String, WeightMap> = serde_json::from_reader(
            std::fs::File::open(&weight_map_path)
                .map_err(|e| Error::Io(format!("Failed to open weight map: {}", e)))?
        ).map_err(|e| Error::Io(format!("Failed to parse weight map: {}", e)))?;

        let mut shard_files: HashSet<String> = HashSet::new();
        for weight in weight_map.values() {
            shard_files.insert(weight.filename.clone());
        }

        for filename in &shard_files {
            let shard_path = model_dir.join(filename);
            let shard: HashMap<String, Array> = unsafe {
                safetensors::mmap_file(&shard_path)
                    .map_err(|e| Error::Io(format!("Failed to mmap {}: {}", filename, e)))?
            };

            for (name, array) in shard {
                weights.insert(name, array);
            }
        }
    } else {
        // Single file: model.safetensors
        let model_path = model_dir.join("model.safetensors");
        if model_path.exists() {
            let shard: HashMap<String, Array> = unsafe {
                safetensors::mmap_file(&model_path)
                    .map_err(|e| Error::Io(format!("Failed to mmap model.safetensors: {}", e)))?
            };
            weights = shard;
        }
    }

    Ok(weights)
}

fn sanitize_weights(weights: &mut HashMap<String, Array>, args: &ModelArgs) -> Result<(), Error> {
    // Handle fused QKV weights if present
    let n_kv_heads = args.num_key_value_heads();
    let n_q_heads = args.num_attention_heads;
    let head_dim = args.head_dim();

    // Check for fused qkv weights
    for layer_idx in 0..args.num_hidden_layers {
        let fused_key = format!("model.layers.{}.attention.wqkv.weight", layer_idx);
        if let Some(fused_weight) = weights.remove(&fused_key) {
            let q_size = n_q_heads * head_dim;
            let kv_size = n_kv_heads * head_dim;
            let q = fused_weight.index(NewAxis, Ellipsis, ..q_size).unwrap();
            let k = fused_weight.index(NewAxis, Ellipsis, q_size..q_size + kv_size).unwrap();
            let v = fused_weight.index(NewAxis, Ellipsis, q_size + kv_size..).unwrap();

            weights.insert(format!("model.layers.{}.attention.wq.weight", layer_idx), q);
            weights.insert(format!("model.layers.{}.attention.wk.weight", layer_idx), k);
            weights.insert(format!("model.layers.{}.attention.wv.weight", layer_idx), v);
        }
    }

    Ok(())
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key).cloned().ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = weights.get(&format!("{}.biases", prefix)).cloned();

    let in_features = weight.shape()[1];
    let out_features = weight.shape()[0];
    let group_size_val = if weight.shape()[1] % group_size != 0 {
        weight.shape()[1] / ((weight.shape()[1] + group_size - 1) / group_size)
    } else {
        group_size
    };

    Ok(nn::QuantizedLinear::from_quantized(
        weight,
        scales,
        biases,
        group_size_val,
        bits,
        in_features,
        out_features,
    ))
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;

    let num_embeddings = weight.shape()[0];
    let embedding_dim = weight.shape()[1];
    let group_size_val = if embedding_dim % group_size != 0 {
        embedding_dim / ((embedding_dim + group_size - 1) / group_size)
    } else {
        group_size
    };

    Ok(nn::QuantizedEmbedding::from_quantized(
        weight,
        scales,
        None,
        group_size_val,
        bits,
        num_embeddings,
        embedding_dim,
    ))
}

fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = weights.get(&format!("{}.biases", prefix)).cloned();

    let shape = weight.shape();
    let num_experts = shape[0];
    let out_features = shape[1];
    let in_features = shape[2];

    Ok(QuantizedSwitchLinear::new(
        weight, scales, biases, group_size, bits, in_features, out_features, num_experts,
    ))
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();

    let args = get_model_args(model_dir)?;
    let quantization = args.quantization.as_ref().unwrap();
    let group_size = quantization.group_size;
    let bits = quantization.bits;

    let mut weights = load_all_weights(model_dir)?;
    sanitize_weights(&mut weights, &args)?;

    // Build model
    let tok_embeddings = make_quantized_embedding(&weights, "model.tok_embeddings", group_size, bits)?;

    let mut layers = Vec::new();
    for layer_idx in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{}", layer_idx);

        let wq = make_quantized_linear(&weights, &format!("{}.attention.wq", prefix), group_size, bits)?;
        let wk = make_quantized_linear(&weights, &format!("{}.attention.wk", prefix), group_size, bits)?;
        let wv = make_quantized_linear(&weights, &format!("{}.attention.wv", prefix), group_size, bits)?;
        let wo = make_quantized_linear(&weights, &format!("{}.attention.wo", prefix), group_size, bits)?;

        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads();
        let n_rep = n_heads / n_kv_heads;
        let head_dim = args.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let self_attn = Attention {
            wq, wk, wv, wo,
            n_heads,
            n_kv_heads,
            n_rep,
            head_dim,
            scale,
        };

        let input_layernorm = nn::RMSNorm::new(
            args.hidden_size,
            args.rms_norm_eps,
            Some(get_weight(&weights, &format!("{}.input_layernorm.weight", prefix))?),
        );

        let post_attention_layernorm = nn::RMSNorm::new(
            args.hidden_size,
            args.rms_norm_eps,
            Some(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix))?),
        );

        // MoE experts
        let gate = make_quantized_linear(&weights, &format!("{}.block_sparse_moe.gate", prefix), group_size, bits)?;

        let w1 = make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.w1", prefix), group_size, bits)?;
        let w2 = make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.w2", prefix), group_size, bits)?;
        let w3 = make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.w3", prefix), group_size, bits)?;

        let block_sparse_moe = MixtralSparseMoeBlock {
            gate,
            experts: SwitchGLU { w1, w2, w3 },
            num_experts_per_tok: args.num_experts_per_tok,
            num_local_experts: args.num_local_experts,
        };

        layers.push(DecoderLayer {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    let norm = nn::RMSNorm::new(
        args.hidden_size,
        args.rms_norm_eps,
        Some(get_weight(&weights, "model.norm.weight")?),
    );

    let output = make_quantized_linear(&weights, "model.output", group_size, bits)?;

    let model = MixtralModel {
        tok_embeddings,
        layers,
        norm,
        output,
        args: args.clone(),
    };

    Ok(Model { model, args })
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, Exception> {
    if temp < 1e-6 {
        // Argmax sampling
        ops::argmax(logits, -1, true)
    } else {
        // Scaled softmax sampling
        let logits = &logits / temp;
        let probs = ops::softmax(&logits, -1, None)?;
        random::categorical(probs, 1, None, None)
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    tokens: Array,
    step: usize,
}

pub enum GenerateState<'a> {
    Prompt { pos: usize },
    Decode,
}

impl<'a, C: KeyValueCache + Default> Generate<'a, C> {
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt: &Array,
    ) -> Self {
        // Initialize cache for each layer
        if cache.is_empty() {
            *cache = (0..model.model.args.num_hidden_layers as usize)
                .map(|_| Some(C::default()))
                .collect();
        }

        Self {
            model,
            cache,
            temp,
            tokens: prompt.clone(),
            step: 0,
        }
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C: KeyValueCache + Default> Iterator for Generate<'a, C> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step == 0 {
            // Full prompt pass
            let mask = None; // Causal mask handled by SDPA
            let input = ModelInput {
                input_ids: &self.tokens,
                mask,
                cache: &mut self.cache,
            };

            let logits = tri!(self.model.module(input));
            let next_token = tri!(sample(&logits, self.temp));
            self.tokens = next_token;
            self.step += 1;
            Some(Ok(self.tokens.clone()))
        } else {
            // Decode: single token
            let mask = None;
            let input = ModelInput {
                input_ids: &self.tokens,
                mask,
                cache: &mut self.cache,
            };

            let logits = tri!(self.model.module(input));
            let next_token = tri!(sample(&logits, self.temp));
            self.tokens = next_token;
            self.step += 1;
            Some(Ok(self.tokens.clone()))
        }
    }
}
