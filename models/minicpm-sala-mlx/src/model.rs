use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Error};
use mlx_rs::fast;
use mlx_rs::module::{ModuleParamMut, ModuleParamRef, ModuleParameters};
use mlx_rs::nn::{
    self, Cache, Embedding, Linear, Module, ModuleParameters as ModuleParametersTrait, Quantizable,
    QuantizedLinear, RmsNorm, ScaledDotProductAttentionMask,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    self, arange, astype, expand, full, split, stack, transpose, triu, zeros, ArrayOps,
};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, DType, StreamOrDevice};
use tokenizers::Tokenizer;

use crate::attention::{
    create_layer_caches, HybridAttention, LayerCache, LightningCache, SparseKVCache,
};
use crate::config::ModelArgs;

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, ModuleParameters, Quantizable)]
#[module(root = mlx_rs)]
pub struct Mlp {
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
}

impl Mlp {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, mlx_rs::error::Exception> {
        Ok(Self {
            gate_proj: Linear::new(hidden_size, intermediate_size)?,
            down_proj: Linear::new(intermediate_size, hidden_size)?,
            up_proj: Linear::new(hidden_size, intermediate_size)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        // SwiGLU: silu(gate_proj(x)) * up_proj(x), then down_proj
        let gate = self.gate_proj.forward(x)?;
        let gate = ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let gated = gate * up;
        self.down_proj.forward(&gated)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DecoderLayer {
    self_attn: HybridAttention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs, layer_idx: usize) -> Result<Self, mlx_rs::error::Exception> {
        let hidden_size = args.hidden_size;
        let intermediate_size = args.intermediate_size;
        let rms_norm_eps = args.rms_norm_eps;

        Ok(Self {
            self_attn: HybridAttention::new(args, layer_idx)?,
            mlp: Mlp::new(hidden_size, intermediate_size)?,
            input_layernorm: RmsNorm::new(hidden_size, rms_norm_eps)?,
            post_attention_layernorm: RmsNorm::new(hidden_size, rms_norm_eps)?,
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array, mlx_rs::error::Exception> {
        // Pre-norm: input -> layernorm -> attention -> residual
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&h, mask, cache)?;
        let h = residual + attn_out;

        // Post-norm -> MLP -> residual
        let residual = &h;
        let h = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&h)?;
        Ok(residual + mlp_out)
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MiniCPMSALAModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    pub args: ModelArgs,
}

impl MiniCPMSALAModel {
    pub fn new(args: &ModelArgs) -> Result<Self, mlx_rs::error::Exception> {
        let hidden_size = args.hidden_size;
        let num_hidden_layers = args.num_hidden_layers;
        let rms_norm_eps = args.rms_norm_eps;

        let mut layers = Vec::with_capacity(num_hidden_layers as usize);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new(args, i as usize)?);
        }

        Ok(Self {
            embed_tokens: Embedding::new(args.vocab_size, hidden_size)?,
            layers,
            norm: RmsNorm::new(hidden_size, rms_norm_eps)?,
            args: args.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Array,
        mask: Option<&Array>,
        caches: &mut [LayerCache],
    ) -> Result<Array, mlx_rs::error::Exception> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids)?;

        // muP: scale embeddings
        if (self.args.scale_emb - 1.0).abs() > 1e-6 {
            h = h * self.args.scale_emb;
        }

        // Forward through each layer
        let num_layers = self.layers.len();
        for i in 0..num_layers {
            h = self.layers[i].forward(&h, mask, &mut caches[i])?;
        }

        // Final norm
        h = self.norm.forward(&h)?;
        Ok(h)
    }
}

// ============================================================================
// Sampling
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, mlx_rs::error::Exception> {
    if temp <= 0.0 {
        // Greedy: argmax
        let logits_last = logits.slice_axis(1, 0, 1)?;
        if let Some(idx) = ops::argmin(&(-&logits_last)?, -1, false)? {
            let token = ops::astype(&idx.reshape(&[])?, DType::Uint32)?;
            eval(&[&token])?;
            Ok(token)
        } else {
            Err(mlx_rs::error::Exception::from("no tokens"))
        }
    } else {
        // Temperature sampling: softmax(logits / temp)
        let scaled = logits / temp;
        let probs = ops::softmax(&scaled, -1)?;
        let probs_2d = probs.reshape(&[1, -1])?;
        let token = ops::random_categorical(&probs_2d, 1)?;
        let token = token.reshape(&[])?;
        eval(&[&token])?;
        Ok(token)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Load all weight tensors from a model directory.
fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    // Check for index.json (sharded weights)
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: HashMap<String, String> = serde_json::from_str(&index_content)
            .map(|v: serde_json::Value| {
                v.get("weight_map")
                    .and_then(|m| serde_json::from_value(m.clone()).ok())
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        let mut weights = HashMap::new();
        let mut shard_files: std::collections::HashSet<String> = index.values().cloned().collect();
        for shard in shard_files.drain() {
            let shard_path = model_dir.join(&shard);
            let shard_weights = Array::load_safetensors(&shard_path, None::<&str>)?;
            for (key, val) in shard_weights {
                weights.insert(key, val);
            }
        }
        Ok(weights)
    } else {
        // Single file
        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            Array::load_safetensors(&safetensors_path, None::<&str>)
                .map_err(|e| Error::msg(format!("Failed to load safetensors: {e}")))
        } else {
            Err(Error::msg("No safetensors files found"))
        }
    }
}

/// Simple weight map from config.json metadata (if present) or empty.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightMap {
    #[serde(default)]
    pub weight_map: HashMap<String, String>,
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {:?}", config_path))?;
    let args: ModelArgs =
        serde_json::from_str(&content).context("Failed to parse config.json")?;
    Ok(args)
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::msg(format!("Failed to load tokenizer: {e}")))
}

/// Load a non-quantized model with all weights from safetensors.
fn load_model_impl(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let weights = load_all_weights(model_dir)?;
    let mut model = Model {
        model: MiniCPMSALAModel::new(args)?,
        args: args.clone(),
    };

    // Set weights on the model using the ModuleParameters interface
    // mlx-rs handles parameter name matching automatically
    for (key, param) in model.model.parameters_mut().flatten() {
        // Map key: replace "model.layers" prefix used in safetensors to match mlx-rs naming
        // mlx-rs typically uses just "layers.0.self_attn.q_proj.weight" etc.
        let wt_key = key.replace("model.", "");
        if let Some(w) = weights.get(&wt_key) {
            param.update(w)?;
        } else if let Some(w) = weights.get(key) {
            param.update(w)?;
        }
    }

    // Evaluate after loading
    let params: Vec<&Array> = model.model.parameters().flatten().map(|(_, v)| v).collect();
    eval(&params)?;

    Ok(model)
}

/// Load quantized model weights.
fn load_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let weights = load_all_weights(model_dir)?;
    // For quantized models we need to manually construct QuantizedLinear layers
    // from the packed weight/scale tensors. This is handled by the model's
    // quantized loading path which reads "qweight", "scales", "biases" keys.
    // For now, fall back to standard loading and let the quantization happen
    // at the mlx-rs level.
    load_model_impl(model_dir, args)
}

/// Main entry point: load a model (auto-detects quantized vs non-quantized).
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    if args.quantization.is_some() {
        load_model_quantized(model_dir, &args)
    } else {
        load_model_impl(model_dir, &args)
    }
}

// ============================================================================
// Model wrapper (top-level)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Model {
    pub model: MiniCPMSALAModel,
    pub args: ModelArgs,
}

impl Model {
    /// Get the hidden representation from the model.
    pub fn hidden_states(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array, mlx_rs::error::Exception> {
        self.model.forward(input_ids, None, caches)
    }

    /// Generate a single token given input_ids and caches.
    /// Returns the logits for the last position.
    pub fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array, mlx_rs::error::Exception> {
        let h = self.model.forward(input_ids, None, caches)?;
        // Last token's hidden state
        let last_h = h.slice_axis(0, -1, 1)?;
        // LM head (tied or separate)
        let lm_head = self.model.embed_tokens.weight();
        let logits = ops::matmul(&last_h, &lm_head.transpose(&[1, 0])?)?;

        // muP logits scaling
        let scale = self.args.logits_scale();
        if (scale - 1.0).abs() > 1e-6 {
            Ok(logits / scale)
        } else {
            Ok(logits)
        }
    }
}
