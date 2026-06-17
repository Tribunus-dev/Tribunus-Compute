//! Talker model (28-layer Qwen3-style transformer) and Code Predictor (5-layer sub-talker).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    builder::Builder,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::indexing::IndexOp,
    ops::zeros,
    quantization::MaybeQuantized,
    transforms::eval,
    Array,
};

use crate::config::{CodePredictorConfig, QuantizationConfig, TalkerConfig};
use crate::error::{Error, Result};
use crate::sampling::sample_logits;
use mlx_rs_core::cache::{KVCache, KeyValueCache};
use mlx_rs_core::utils::{scaled_dot_product_attention, SdpaMask};

// ============================================================================
// Attention with MRoPE
// ============================================================================

pub struct TalkerAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope: nn::Rope,
    /// RoPE position speed factor: >1.0 makes the model's internal clock run faster.
    /// KV cache indexing is unaffected — only the RoPE rotation angles change.
    pub rope_speed_factor: f32,

    /// Merged QKV projection: one quantized_matmul instead of three.
    /// Output dim = n_heads*head_dim + 2*n_kv_heads*head_dim.
    pub qkv_proj: MaybeQuantized<nn::Linear>,
    pub q_dim: i32,  // n_heads * head_dim
    pub kv_dim: i32, // n_kv_heads * head_dim
    pub o_proj: MaybeQuantized<nn::Linear>,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl TalkerAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Batched QKV projection: one matmul instead of three
        let qkv = self.qkv_proj.forward(x)?;
        let qd = self.q_dim;
        let kvd = self.kv_dim;
        let queries = qkv.index((.., .., ..qd));
        let keys = qkv.index((.., .., qd..(qd + kvd)));
        let values = qkv.index((.., .., (qd + kvd)..));

        // Reshape to [B, L, heads, head_dim] then transpose to [B, heads, L, head_dim]
        let mut queries = self
            .q_norm
            .forward(
                &queries
                    .reshape(&[b, l, self.n_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let mut keys = self
            .k_norm
            .forward(
                &keys
                    .reshape(&[b, l, self.n_kv_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE with optional speed factor (makes model's internal clock run faster)
        let rope_offset = if (self.rope_speed_factor - 1.0).abs() < 1e-6 {
            cache.offset()
        } else {
            (cache.offset() as f32 * self.rope_speed_factor) as i32
        };
        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(rope_offset)
            .build()
            .unwrap();
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(rope_offset)
            .build()
            .unwrap();
        keys = self.rope.forward(k_input)?;

        // Update KV cache
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // Attention mask
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<KVCache>(
            queries, keys, values, None, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        Ok(self.o_proj.forward(&output)?)
    }
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

pub struct TalkerMlp {
    pub gate_proj: MaybeQuantized<nn::Linear>,
    pub up_proj: MaybeQuantized<nn::Linear>,
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl TalkerMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate_raw = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = mlx_rs_core::fused_swiglu(&up, &gate_raw)
            .map_err(|e| crate::error::Error::Model(format!("fused_swiglu: {e}")))?;
        Ok(self.down_proj.forward(&activated)?)
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

pub struct TalkerBlock {
    pub self_attn: TalkerAttention,
    pub mlp: TalkerMlp,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TalkerBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        // Fused: h = x + attn_out, normed = rmsnorm(h, weight)
        let (h, normed) = crate::metal_kernels::fused_residual_rmsnorm(
            &attn_out, x, &self.post_attention_layernorm.weight,
        ).map_err(|e| crate::error::Error::Model(format!("fused_residual_rmsnorm: {e}")))?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(mlp_out)?)
    }
}

// ============================================================================
// Text Projection (2-layer MLP)
// ============================================================================

pub struct TextProjection {
    pub fc1: MaybeQuantized<nn::Linear>,
    pub fc1_bias: Option<Array>,
    pub fc2: MaybeQuantized<nn::Linear>,
    pub fc2_bias: Option<Array>,
}

impl TextProjection {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut h = self.fc1.forward(x)?;
        if let Some(bias) = &self.fc1_bias {
            h = h.add(bias)?;
        }
        h = nn::silu(h)?;
        h = self.fc2.forward(&h)?;
        if let Some(bias) = &self.fc2_bias {
            h = h.add(bias)?;
        }
        Ok(h)
    }
}

// ============================================================================
// Code Predictor
// ============================================================================

pub struct CodePredictorAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    pub qkv_proj: MaybeQuantized<nn::Linear>,
    pub q_dim: i32,
    pub kv_dim: i32,
    pub o_proj: MaybeQuantized<nn::Linear>,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub rope: nn::Rope,
}

impl CodePredictorAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let qkv = self.qkv_proj.forward(x)?;
        let qd = self.q_dim;
        let kvd = self.kv_dim;
        let queries = qkv.index((.., .., ..qd));
        let keys = qkv.index((.., .., qd..(qd + kvd)));
        let values = qkv.index((.., .., (qd + kvd)..));

        let mut queries = self
            .q_norm
            .forward(
                &queries
                    .reshape(&[b, l, self.n_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let mut keys = self
            .k_norm
            .forward(
                &keys
                    .reshape(&[b, l, self.n_kv_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Standard RoPE for code predictor
        let offset = cache.offset();
        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(offset)
            .build()
            .unwrap(); // safe: Infallible error type
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(offset)
            .build()
            .unwrap(); // safe: Infallible error type
        keys = self.rope.forward(k_input)?;

        let (keys, values) = cache.update_and_fetch(keys, values)?;

        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<KVCache>(
            queries, keys, values, None, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        Ok(self.o_proj.forward(&output)?)
    }
}

pub struct CodePredictorBlock {
    pub self_attn: CodePredictorAttention,
    pub mlp: TalkerMlp,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl CodePredictorBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        // Fused: h = x + attn_out, normed = rmsnorm(h, weight)
        let (h, normed) = crate::metal_kernels::fused_residual_rmsnorm(
            &attn_out, x, &self.post_attention_layernorm.weight,
        ).map_err(|e| crate::error::Error::Model(format!("fused_residual_rmsnorm: {e}")))?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(mlp_out)?)
    }
}

pub struct CodePredictor {
    pub layers: Vec<CodePredictorBlock>,
    pub lm_heads: Vec<MaybeQuantized<nn::Linear>>,
    pub input_layernorm: nn::RmsNorm,
    pub norm: nn::RmsNorm,
}

impl CodePredictor {
    pub fn forward(
        &mut self,
        x: &Array,
        masks: &[Option<&Array>],
        caches: &mut [KVCache],
    ) -> Result<Vec<Array>> {
        let mut h = self.input_layernorm.forward(x)?;
        for (i, block) in self.layers.iter_mut().enumerate() {
            h = block.forward(&h, masks[i], &mut caches[i])?;
        }
        h = self.norm.forward(&h)?;

        // Generate logits for each codebook group
        let outputs: std::result::Result<Vec<_>, _> = self
            .lm_heads
            .iter_mut()
            .map(|head| head.forward(&h))
            .collect();
        outputs.map_err(Error::from)
    }

    /// Generate codebooks 1-15 from hidden state + code0 embedding.
    pub fn generate_codes(
        &mut self,
        hidden: &Array,    // [1, 1, hidden_size]
        code0_embed: &Array, // [1, 1, codec_dim]
    ) -> Result<Vec<u32>> {
        // Concatenate hidden + code0_embed along feature dim
        let input = Array::concatenate(&[hidden, code0_embed], -1)?;
        let logits = self.forward(&input, &[None; 6], &mut self.reset_caches())?;

        let mut codes = Vec::with_capacity(15);
        for (g, mut logit) in logits.into_iter().enumerate() {
            logit = logit.reshape(&[-1])?;
            eval(std::iter::once(&logit))?;
            let token = sample_logits(
                &logit,
                0.9,  // temperature
                50,   // top_k
                1.0,  // top_p
                1.0,  // repetition_penalty
                &[],
                None,
            )?;
            codes.push(token);
        }
        Ok(codes)
    }

    fn reset_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len())
            .map(|_| KVCache::new(1, 1, 0))
            .collect()
    }
}

// ============================================================================
// Full Talker Model
// ============================================================================

pub struct Talker {
    pub text_embedding: nn::Embedding,
    pub codec_embedding: nn::Embedding,
    pub text_projection: TextProjection,
    pub layers: Vec<TalkerBlock>,
    pub norm: nn::RmsNorm,
    pub lm_head: MaybeQuantized<nn::Linear>,
    pub code_predictor: CodePredictor,
    pub caches: Vec<KVCache>,
    pub config: TalkerConfig,
    /// RoPE speed factor for all attention layers. Applied at forward time
    /// so KV cache offsets remain in model-time units.
    pub rope_speed_factor: f32,
}

impl Talker {
    /// Set the RoPE speed factor for all attention layers.
    /// Values > 1.0 make the model's internal clock run faster (longer output),
    /// values < 1.0 make it slower (shorter output).
    pub fn set_rope_speed_factor(&mut self, factor: f32) {
        for layer in &mut self.layers {
            layer.self_attn.rope_speed_factor = factor;
        }
    }

    /// Forward pass for a full batch (prefill).
    pub fn forward_batch(&mut self, input_embeds: Array) -> Result<(Array, Array)> {
        let seq_len = input_embeds.dim(1);

        if seq_len > 1 {
            /reset KV caches
            for cache in &mut self.caches {
                cache.reset(0);
            }
        }

        // Build causal mask for prefill
        let mask = if seq_len > 1 {
            let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask_data[(i * seq_len + j) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            Some(Array::from_slice(&mask_data, &[1, 1, seq_len, seq_len]))
        } else {
            None
        };

        let mut h = input_embeds;

        for (i, block) in self.layers.iter_mut().enumerate() {
            h = block.forward(&h, mask.as_ref(), &mut self.caches[i])?;
        }

        h = self.norm.forward(&h)?;

        // LM head
        let logits = self.lm_head.forward(&h)?;

        Ok((logits, h))
    }

    /// Single-step forward (autoregressive).
    pub fn forward_step(&mut self, input_embed: Array) -> Result<(Array, Array)> {
        let mut h = input_embed;

        for (i, block) in self.layers.iter_mut().enumerate() {
            h = block.forward(&h, None, &mut self.caches[i])?;
        }

        h = self.norm.forward(&h)?;

        // LM head — take last position
        let last_hidden = h.index((.., -1.., ..));
        let logits = self.lm_head.forward(&last_hidden)?;

        Ok((logits, h))
    }

    /// Build generation-step embedding: text_proj(text_embed) + codec_embed(prev_codes).
    pub fn build_generation_embedding_with_text(
        &self,
        prev_codes: &[u32; 16],
        text_embed: &Array,
    ) -> Result<Array> {
        // Sum all codec embeddings
        let mut sum_codec_embed = None;
        for &code in prev_codes.iter() {
            let code_arr = Array::from_slice(&[code as i32], &[1, 1]);
            let emb = self.codec_embedding.forward(&code_arr)?;
            sum_codec_embed = Some(match sum_codec_embed {
                None => emb,
                Some(prev) => prev.add(&emb)?,
            });
        }
        let sum_codec_embed = sum_codec_embed.unwrap();

        // text_proj is already embedded + projected
        // text_embed shape: [1, 1, hidden_size]
        // codec_embed shape: [1, 1, codec_dim]
        // We need to sum them dim-wise (project codec to hidden first)
        // In Qwen3-TTS, the codec embedding is in the same space as text
        // (codec_dim == hidden_size for this model)
        let combined = text_embed.add(&sum_codec_embed)?;
        Ok(combined)
    }

    pub fn hidden_size(&self) -> i32 {
        self.config.hidden_size
    }
}

// ============================================================================
// Weight loading
// ============================================================================

/// Load all weights from safetensors file(s) in the model directory.
pub fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let mut weights = HashMap::new();

    // Try index file first (sharded weights)
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index_data: serde_json::Value = serde_json::from_str(&index_content)?;
        if let Some(weight_map) = index_data["weight_map"].as_object() {
            let mut shard_files: std::collections::BTreeSet<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            for shard in &shard_files {
                let shard_path = model_dir.join(shard);
                let shard_weights = mlx_rs::Array::load_safetensors(&shard_path)?;
                weights.extend(shard_weights);
            }
            return Ok(weights);
        }
    }

    // Single file
    let model_path = model_dir.join("model.safetensors");
    if model_path.exists() {
        weights = mlx_rs::Array::load_safetensors(&model_path)?;
    }

    Ok(weights)
}

fn load_weights_to_talker(
    weights: &HashMap<String, Array>,
    config: &TalkerConfig,
    quant: Option<&QuantizationConfig>,
) -> Result<Talker> {
    let hidden_size = config.hidden_size;
    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let rope_theta = config.rope_theta;
    let rms_norm_eps = config.rms_norm_eps;
    let vocab_size = config.vocab_size;
    let text_hidden_size = config.text_hidden_size;
    let text_vocab_size = config.text_vocab_size;
    let intermediate_size = config.intermediate_size;

    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let qkv_dim = q_dim + 2 * kv_dim;

    let rope_traditional = config
        .rope_scaling
        .as_ref()
        .map(|r| r.interleaved)
        .unwrap_or(true);

    // Text embedding
    let text_embedding = nn::Embedding::from_pretrained(
        text_vocab_size,
        text_hidden_size,
        weights.get("model.text_embedding.weight"),
    )?;

    // Codec embedding
    let codec_embedding = nn::Embedding::from_pretrained(
        vocab_size,
        hidden_size,
        weights.get("model.codec_embedding.weight"),
    )?;

    // Text projection (2-layer MLP)
    let text_projection = TextProjection {
        fc1: MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                text_hidden_size,
                hidden_size,
                weights.get("model.text_projection.fc1.weight"),
                weights.get("model.text_projection.fc1.bias"),
            )?,
            quant,
            "model.text_projection.fc1",
            weights,
        ),
        fc1_bias: weights
            .get("model.text_projection.fc1.bias")
            .cloned(),
        fc2: MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                hidden_size,
                weights.get("model.text_projection.fc2.weight"),
                weights.get("model.text_projection.fc2.bias"),
            )?,
            quant,
            "model.text_projection.fc2",
            weights,
        ),
        fc2_bias: weights
            .get("model.text_projection.fc2.bias")
            .cloned(),
    };

    // Transformer layers
    let mut layers = Vec::with_capacity(n_layers as usize);
    let mut caches = Vec::with_capacity(n_layers as usize);

    for i in 0..n_layers {
        let prefix = format!("model.layers.{i}");

        let qkv_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                qkv_dim,
                weights.get(&format!("{prefix}.self_attn.qkv_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.self_attn.qkv_proj"),
            weights,
        );

        let o_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                hidden_size,
                weights.get(&format!("{prefix}.self_attn.o_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.self_attn.o_proj"),
            weights,
        );

        let gate_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                intermediate_size,
                weights.get(&format!("{prefix}.mlp.gate_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.gate_proj"),
            weights,
        );

        let up_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                intermediate_size,
                weights.get(&format!("{prefix}.mlp.up_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.up_proj"),
            weights,
        );

        let down_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                intermediate_size,
                hidden_size,
                weights.get(&format!("{prefix}.mlp.down_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.down_proj"),
            weights,
        );

        let attn = TalkerAttention {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rope: nn::Rope::from_pretrained(
                n_kv_heads,
                head_dim,
                rope_theta,
                rope_traditional,
                weights.get(&format!("{prefix}.self_attn.rope.freqs")),
            )?,
            rope_speed_factor: 1.0,
            qkv_proj,
            q_dim,
            kv_dim,
            o_proj,
            q_norm: nn::RmsNorm::from_pretrained(
                q_dim,
                weights.get(&format!("{prefix}.self_attn.q_norm.weight")),
                rms_norm_eps,
            )?,
            k_norm: nn::RmsNorm::from_pretrained(
                kv_dim,
                weights.get(&format!("{prefix}.self_attn.k_norm.weight")),
                rms_norm_eps,
            )?,
        };

        let mlp = TalkerMlp {
            gate_proj,
            up_proj,
            down_proj,
        };

        layers.push(TalkerBlock {
            self_attn: attn,
            mlp,
            input_layernorm: nn::RmsNorm::from_pretrained(
                hidden_size,
                weights.get(&format!("{prefix}.input_layernorm.weight")),
                rms_norm_eps,
            )?,
            post_attention_layernorm: nn::RmsNorm::from_pretrained(
                hidden_size,
                weights.get(&format!("{prefix}.post_attention_layernorm.weight")),
                rms_norm_eps,
            )?,
        });

        caches.push(KVCache::new(n_kv_heads, head_dim, 0));
    }

    // Norm
    let norm = nn::RmsNorm::from_pretrained(
        hidden_size,
        weights.get("model.norm.weight"),
        rms_norm_eps,
    )?;

    // LM head
    let lm_head = MaybeQuantized::from_linear(
        nn::Linear::from_pretrained(
            hidden_size,
            vocab_size,
            weights.get("lm_head.weight"),
            None::<&Array>,
        )?,
        quant,
        "lm_head",
        weights,
    );

    // Code Predictor
    let code_predictor = load_code_predictor(weights, &config.code_predictor_config, quant)?;

    Ok(Talker {
        text_embedding,
        codec_embedding,
        text_projection,
        layers,
        norm,
        lm_head,
        code_predictor,
        caches,
        config: config.clone(),
        rope_speed_factor: 1.0,
    })
}

fn load_code_predictor(
    weights: &HashMap<String, Array>,
    config: &CodePredictorConfig,
    quant: Option<&QuantizationConfig>,
) -> Result<CodePredictor> {
    let hidden_size = config.hidden_size;
    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let rope_theta = config.rope_theta;
    let rms_norm_eps = config.rms_norm_eps;
    let intermediate_size = config.intermediate_size;
    let num_code_groups = config.num_code_groups;

    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;

    let input_prefix = "model.code_predictor";

    // Input layernorm
    let input_layernorm = nn::RmsNorm::from_pretrained(
        hidden_size + hidden_size, // codec embedding is also hidden_size
        weights.get(&format!("{input_prefix}.input_layernorm.weight")),
        rms_norm_eps,
    )?;

    // Blocks
    let mut layers = Vec::with_capacity(n_layers as usize);
    for i in 0..n_layers {
        let prefix = format!("{input_prefix}.layers.{i}");

        let qkv_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                qkv_dim,
                weights.get(&format!("{prefix}.self_attn.qkv_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.self_attn.qkv_proj"),
            weights,
        );

        let o_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                hidden_size,
                weights.get(&format!("{prefix}.self_attn.o_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.self_attn.o_proj"),
            weights,
        );

        let gate_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                intermediate_size,
                weights.get(&format!("{prefix}.mlp.gate_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.gate_proj"),
            weights,
        );

        let up_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                intermediate_size,
                weights.get(&format!("{prefix}.mlp.up_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.up_proj"),
            weights,
        );

        let down_proj = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                intermediate_size,
                hidden_size,
                weights.get(&format!("{prefix}.mlp.down_proj.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{prefix}.mlp.down_proj"),
            weights,
        );

        let attn = CodePredictorAttention {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            qkv_proj,
            q_dim,
            kv_dim,
            o_proj,
            q_norm: nn::RmsNorm::from_pretrained(
                q_dim,
                weights.get(&format!("{prefix}.self_attn.q_norm.weight")),
                rms_norm_eps,
            )?,
            k_norm: nn::RmsNorm::from_pretrained(
                kv_dim,
                weights.get(&format!("{prefix}.self_attn.k_norm.weight")),
                rms_norm_eps,
            )?,
            rope: nn::Rope::from_pretrained(
                n_kv_heads,
                head_dim,
                rope_theta,
                true,
                weights.get(&format!("{prefix}.self_attn.rope.freqs")),
            )?,
        };

        layers.push(CodePredictorBlock {
            self_attn: attn,
            mlp: TalkerMlp {
                gate_proj,
                up_proj,
                down_proj,
            },
            input_layernorm: nn::RmsNorm::from_pretrained(
                hidden_size,
                weights.get(&format!("{prefix}.input_layernorm.weight")),
                rms_norm_eps,
            )?,
            post_attention_layernorm: nn::RmsNorm::from_pretrained(
                hidden_size,
                weights.get(&format!("{prefix}.post_attention_layernorm.weight")),
                rms_norm_eps,
            )?,
        });
    }

    // Norm
    let norm = nn::RmsNorm::from_pretrained(
        hidden_size,
        weights.get(&format!("{input_prefix}.norm.weight")),
        rms_norm_eps,
    )?;

    // LM heads (one per codebook group)
    let mut lm_heads = Vec::with_capacity(num_code_groups as usize);
    for g in 0..num_code_groups {
        let lm_head = MaybeQuantized::from_linear(
            nn::Linear::from_pretrained(
                hidden_size,
                config.vocab_size,
                weights.get(&format!("{input_prefix}.lm_heads.{g}.weight")),
                None::<&Array>,
            )?,
            quant,
            &format!("{input_prefix}.lm_heads.{g}"),
            weights,
        );
        lm_heads.push(lm_head);
    }

    Ok(CodePredictor {
        layers,
        lm_heads,
        input_layernorm,
        norm,
    })
}

/// Load a fully initialized Talker from disk.
pub fn load_talker(
    model_dir: &Path,
    config: &TalkerConfig,
    quant: Option<&QuantizationConfig>,
    tts_pad_token_id: u32,
) -> Result<Talker> {
    let weights = load_all_weights(model_dir)?;
    let mut talker = load_weights_to_talker(&weights, config, quant)?;
    // Ensure caches are properly sized
    talker.caches = (0..config.num_hidden_layers)
        .map(|_| KVCache::new(config.num_key_value_heads, config.head_dim, 0))
        .collect();
    Ok(talker)
}
