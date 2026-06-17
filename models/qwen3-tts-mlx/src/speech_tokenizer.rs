//! Speech tokenizer decoder: converts 16-codebook discrete codes to 24kHz waveform.

use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    module::Module,
    nn,
    ops::{
        concatenate_axis,
        indexing::IndexOp,
        zeros,
    },
    transforms::eval,
    Array,
};

use crate::config::DecoderConfig;
use crate::error::{Error, Result};
use crate::pretrained::*;

// ============================================================================
// Helper: Causal Conv1d (left-padding)
// ============================================================================

pub struct CausalConv1d {
    pub conv: nn::Conv1d,
    pub pad: i32,
}

impl CausalConv1d {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C] - pad left by (kernel_size - 1) * dilation
        let x = if self.pad > 0 {
            // Manual left-padding: concat zeros on left along time axis
            let b = x.dim(0) as i32;
            let c = x.dim(2) as i32;
            let pad_zeros = zeros::<f32>(&[b, self.pad, c])?;
            concatenate_axis(&[&pad_zeros, x], 1)?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

// ============================================================================
// Helper: Causal ConvTranspose1d
// ============================================================================

pub struct CausalConvTranspose1d {
    pub conv_t: nn::ConvTranspose1d,
    pub trim_right: i32,
}

impl CausalConvTranspose1d {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut y = self.conv_t.forward(&x)?;
        // Trim right to maintain causal property
        if self.trim_right > 0 {
            let t = y.dim(1) as i32;
            let keep = t - self.trim_right;
            if keep > 0 {
                y = y.index((.., ..keep, ..));
            }
        }
        Ok(y)
    }
}

// ============================================================================
// SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)
// ============================================================================

pub struct SnakeBeta {
    pub alpha_exp: Array, // [1, 1, C] — pre-exponentiated at load time
    pub beta_exp: Array,  // [1, 1, C] — pre-exponentiated at load time
}

impl SnakeBeta {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // x: [B, T, C], alpha_exp/beta_exp: [1, 1, C] (exp already applied at load)
        // Formula: x + sin^2(alpha_exp * x) / (beta_exp + 1e-9)
        crate::metal_kernels::fused_snake_beta(x, &self.alpha_exp, &self.beta_exp)
            .map_err(|e| crate::error::Error::Model(format!("SnakeBeta kernel: {e}")))
    }
}

// ============================================================================
// Residual Unit
// ============================================================================

pub struct ResidualUnit {
    pub act1: SnakeBeta,
    pub conv1: CausalConv1d,
    pub act2: SnakeBeta,
    pub conv2: CausalConv1d,
}

impl ResidualUnit {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.act1.forward(x)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.act1", &h);
        let h = self.conv1.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.conv1", &h);
        let h = self.act2.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.act2", &h);
        let h = self.conv2.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.conv2", &h);
        Ok(x.add(h)?)
    }
}

// ============================================================================
// Decoder Block: SnakeBeta → ConvTranspose1d → 3 ResidualUnits
// ============================================================================

pub struct DecoderBlock {
    pub snake: SnakeBeta,
    pub conv_t: CausalConvTranspose1d,
    pub res_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut h = self.snake.forward(x)?;
        SpeechTokenizerDecoder::debug_tensor("  block.snake", &h);
        h = self.conv_t.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("  block.conv_t", &h);
        for (i, ru) in self.res_units.iter_mut().enumerate() {
            h = ru.forward(&h)?;
            SpeechTokenizerDecoder::debug_tensor(&format!("  block.res_unit_{i}"), &h);
        }
        Ok(h)
    }
}

// ============================================================================
// ConvNeXt Block
// ============================================================================

pub struct ConvNeXtBlock {
    pub dwconv: CausalConv1d,
    pub norm_weight: Array,
    pub norm_bias: Array,
    pub pwconv1_weight: Array,
    pub pwconv1_bias: Array,
    pub pwconv2_weight: Array,
    pub pwconv2_bias: Array,
    pub gamma: Array,
}

impl ConvNeXtBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();

        // Depthwise conv
        let mut h = self.dwconv.forward(x)?;

        // Layer norm along last dim
        let mean = h.mean_axis(-1, true)?;
        let var = h.var_axis(-1, true, None)?;
        let norm_weight = self.norm_weight.reshape(&[1, 1, -1])?;
        let norm_bias = self.norm_bias.reshape(&[1, 1, -1])?;
        let inv_std = var.add(array!(1e-5f32))?.rsqrt()?;
        h = h
            .subtract(&mean)?
            .multiply(&inv_std)?
            .multiply(&norm_weight)?
            .add(&norm_bias)?;

        // Pointwise MLP (implemented as matmul since these are Linear weights)
        h = h
            .matmul(&self.pwconv1_weight.t())?
            .add(&self.pwconv1_bias)?;
        h = nn::gelu(h)?;
        h = h
            .matmul(&self.pwconv2_weight.t())?
            .add(&self.pwconv2_bias)?;

        // Layer scale
        let gamma = self.gamma.reshape(&[1, 1, -1])?;
        h = h.multiply(&gamma)?;

        Ok(residual.add(h)?)
    }
}

// ============================================================================
// Decoder Transformer Layer (with LayerScale)
// ============================================================================

pub struct DecoderTransformerLayer {
    pub input_layernorm: nn::RmsNorm,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub attn_layer_scale: Array,
    pub post_attention_layernorm: nn::RmsNorm,
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
    pub mlp_layer_scale: Array,

    pub n_heads: i32,
    pub head_dim: i32,
    pub rope: nn::Rope,
}

impl DecoderTransformerLayer {
    #[allow(non_snake_case)]
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>, offset: i32) -> Result<Array> {
        let B = x.dim(0) as i32;
        let L = x.dim(1) as i32;
        let scale = (self.head_dim as f32).sqrt().recip();

        let normed = self.input_layernorm.forward(x)?;
        let q = self.q_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope.forward(nn::RopeInputBuilder::new(&q).offset(offset).build().unwrap())?;
        let k = self.rope.forward(nn::RopeInputBuilder::new(&k).offset(offset).build().unwrap())?;

        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            q, k, v, scale,
            mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        let attn_out = self.o_proj.forward(&attn_out)?;
        let attn_scale = self.attn_layer_scale.reshape(&[1, 1, -1])?;
        let attn_out = attn_out.multiply(&attn_scale)?;
        // Fused: h = x + attn_out, normed = rmsnorm(h, weight)
        let (h, normed) = crate::metal_kernels::fused_residual_rmsnorm(
            &attn_out, x, &self.post_attention_layernorm.weight,
        ).map_err(|e| crate::error::Error::Model(format!("fused_residual_rmsnorm: {e}")))?;
        let gate_raw = self.gate_proj.forward(&normed)?;
        let up = self.up_proj.forward(&normed)?;
        let activated = mlx_rs_core::fused_swiglu(&up, &gate_raw)
            .map_err(|e| crate::error::Error::Model(format!("fused_swiglu: {e}")))?;
        let mlp_out = self.down_proj.forward(&activated)?;
        let mlp_scale = self.mlp_layer_scale.reshape(&[1, 1, -1])?;
        let mlp_out = mlp_out.multiply(&mlp_scale)?;

        Ok(h.add(mlp_out)?)
    }
}

// ============================================================================
// Full Speech Tokenizer Decoder
// ============================================================================

pub struct SpeechTokenizerDecoder {
    pub semantic_codebook: Array,
    pub acoustic_codebooks: Vec<Array>,
    pub rvq_first_output_proj: nn::Conv1d,
    pub rvq_rest_output_proj: nn::Conv1d,

    pub pre_conv: CausalConv1d,

    pub pre_transformer_input_proj: nn::Linear,
    pub pre_transformer_output_proj: nn::Linear,
    pub pre_transformer_norm: nn::RmsNorm,
    pub pre_transformer_layers: Vec<DecoderTransformerLayer>,

    pub upsample_convs: Vec<CausalConvTranspose1d>,
    pub upsample_convnext: Vec<ConvNeXtBlock>,

    pub initial_conv: CausalConv1d,
    pub decoder_blocks: Vec<DecoderBlock>,
    pub final_snake: SnakeBeta,
    pub final_conv: CausalConv1d,

    pub config: DecoderConfig,
}

impl SpeechTokenizerDecoder {
    fn debug_tensor(name: &str, t: &Array) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        use mlx_rs::transforms::eval;
        let flat = t.flatten(0, -1).unwrap();
        let min = flat.min_axis(0, None).unwrap();
        let max = flat.max_axis(0, None).unwrap();
        let mean = flat.mean_axis(0, None).unwrap();
        eval([&min, &max, &mean]).unwrap();
        tracing::debug!(
            "  {} shape={:?} min={:.4} max={:.4} mean={:.4}",
            name,
            t.shape(),
            min.item::<f32>(),
            max.item::<f32>(),
            mean.item::<f32>(),
        );
    }

    /// Decode a batch of codec frames to audio samples.
    pub fn decode(&mut self, codes: &[[u32; 16]]) -> Result<Vec<f32>> {
        let n_frames = codes.len();
        if n_frames == 0 {
            return Ok(vec![]);
        }

        // Convert codes to arrays and lookup embeddings
        let mut semantic_codes: Vec<u32> = Vec::with_capacity(n_frames);
        let mut acoustic_codes: Vec<Vec<u32>> = vec![Vec::with_capacity(n_frames); 15];

        for frame in codes {
            semantic_codes.push(frame[0]);
            for (g, &code) in frame[1..].iter().enumerate() {
                acoustic_codes[g].push(code);
            }
        }

        // Lookup semantic codebook
        let sem_arr = Array::from_slice(&semantic_codes, &[1, 1, n_frames as i32]); // [1, 1, T]
        let sem_emb = self.semantic_codebook.index(sem_arr.index((0, 0, ..))); // [T, D]
        let sem_emb = sem_emb.reshape(&[1, n_frames as i32, -1])?; // [1, T, D]

        // Lookup acoustic codebooks and sum them
        let mut sum_code = sem_emb;
        for g in 0..15 {
            let codes_arr = Array::from_slice(&acoustic_codes[g], &[1, 1, n_frames as i32]);
            let emb = self.acoustic_codebooks[g].index(codes_arr.index((0, 0, ..)));
            let emb = emb.reshape(&[1, n_frames as i32, -1])?;
            sum_code = sum_code.add(&emb)?;
        }

        // RVQ output projection
        // First output
        let first_out = self.rvq_first_output_proj.forward(&sum_code)?;
        // Rest output (no projection for now — simplified)
        let _rest_out = self.rvq_rest_output_proj.forward(&sum_code)?;

        // Pre-convolution
        let h = self.pre_conv.forward(&first_out)?;

        // Pre-transformer
        let t = h.dim(1);
        let mut h = self.pre_transformer_input_proj.forward(&h)?;
        let mut offset = 0;
        for layer in &mut self.pre_transformer_layers {
            h = layer.forward(&h, None, offset)?;
            offset += t;
        }
        h = self.pre_transformer_norm.forward(&h)?;
        h = self.pre_transformer_output_proj.forward(&h)?;

        // Upsample stages (ConvTranspose → ConvNeXt)
        for (i, up_conv) in self.upsample_convs.iter_mut().enumerate() {
            h = up_conv.forward(&h)?;
            if i < self.upsample_convnext.len() {
                h = self.upsample_convnext[i].forward(&h)?;
            }
        }

        // Decoder blocks
        h = self.initial_conv.forward(&h)?;
        for block in &mut self.decoder_blocks {
            h = block.forward(&h)?;
        }

        // Final SnakeBeta + Conv1d
        h = self.final_snake.forward(&h)?;
        h = self.final_conv.forward(&h)?;

        // Extract audio samples from output
        let flat = h.flatten(None, None)?;
        eval(std::iter::once(&flat))?;
        let samples: Vec<f32> = flat.as_slice::<f32>().to_vec();
        Ok(samples)
    }
}

/// Load the speech tokenizer decoder from a model directory.
///
/// Expected file structure:
///   speech_tokenizer/config.json
///   speech_tokenizer/model.safetensors
pub fn load_speech_tokenizer(
    model_dir: &Path,
    config: &DecoderConfig,
) -> Result<SpeechTokenizerDecoder> {
    let st_dir = model_dir.join("speech_tokenizer");
    let model_path = st_dir.join("model.safetensors");
    let weights = mlx_rs::Array::load_safetensors(&model_path)?;

    let h = config.hidden_size;
    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads;
    let _n_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let rope_theta = config.rope_theta;
    let rms_norm_eps = config.rms_norm_eps;
    let intermediate_size = config.intermediate_size;

    // Codebooks
    let _sem_codebook_size = config.semantic_codebook_size;
    let codebook_dim = config.codebook_dim;
    let _num_quantizers = config.num_quantizers;
    let semantic_quantizers = config.num_semantic_quantizers;

    // Semantic codebook (quantizers 0..semantic_quantizers summed)
    let semantic_codebook = weights.get("semantic_codebook.weight")
        .or_else(|| weights.get("codebook.0.weight"))
        .cloned()
        .ok_or_else(|| Error::WeightNotFound("semantic codebook".to_string()))?;

    // Acoustic codebooks
    let mut acoustic_codebooks = Vec::new();
    for g in 0..15 {
        let idx = semantic_quantizers + g;
        let key = format!("codebook.{}.weight", idx);
        let cb = weights.get(&key)
            .or_else(|| {
                // Try staggered keys
                let alt = format!("acoustic_codebook.{}.weight", g);
                weights.get(&alt)
            })
            .cloned()
            .ok_or_else(|| Error::WeightNotFound(format!("codebook {idx}")))?;
        acoustic_codebooks.push(cb);
    }

    // RVQ output projections
    let rvq_first = nn::Conv1d::from_pretrained(
        codebook_dim,
        h,
        1, 1, false,
        weights.get("rvq_first_output_proj.weight")
            .or_else(|| weights.get("first_output_proj.weight")),
        None,
    )?;
    let rvq_rest = nn::Conv1d::from_pretrained(
        codebook_dim,
        h,
        1, 1, false,
        weights.get("rvq_rest_output_proj.weight")
            .or_else(|| weights.get("rest_output_proj.weight")),
        None,
    )?;

    // Pre-conv
    let pre_conv = CausalConv1d {
        conv: nn::Conv1d::from_pretrained(
            h, h, 7, 1, false,
            weights.get("pre_conv.weight"),
            None,
        )?,
        pad: 3,
    };

    // Pre-transformer
    let pre_transformer_input_proj = nn::Linear::from_pretrained(
        h, h,
        weights.get("pre_transformer_input_proj.weight"),
        None::<&Array>,
    )?;
    let pre_transformer_output_proj = nn::Linear::from_pretrained(
        h, h,
        weights.get("pre_transformer_output_proj.weight"),
        None::<&Array>,
    )?;
    let pre_transformer_norm = nn::RmsNorm::from_pretrained(
        h,
        weights.get("pre_transformer_norm.weight"),
        rms_norm_eps,
    )?;

    let _rope_traditional = true;
    let mut pre_transformer_layers = Vec::new();
    for i in 0..n_layers {
        let prefix = format!("pre_transformer.layers.{i}");
        let layer = DecoderTransformerLayer {
            input_layernorm: nn::RmsNorm::from_pretrained(
                h,
                weights.get(&format!("{prefix}.input_layernorm.weight")),
                rms_norm_eps,
            )?,
            q_proj: nn::Linear::from_pretrained(
                h, h,
                weights.get(&format!("{prefix}.q_proj.weight")),
                None::<&Array>,
            )?,
            k_proj: nn::Linear::from_pretrained(
                h, h,
                weights.get(&format!("{prefix}.k_proj.weight")),
                None::<&Array>,
            )?,
            v_proj: nn::Linear::from_pretrained(
                h, h,
                weights.get(&format!("{prefix}.v_proj.weight")),
                None::<&Array>,
            )?,
            o_proj: nn::Linear::from_pretrained(
                h, h,
                weights.get(&format!("{prefix}.o_proj.weight")),
                None::<&Array>,
            )?,
            attn_layer_scale: weights.get(&format!("{prefix}.attn_layer_scale"))
                .cloned()
                .unwrap_or_else(|| Array::from_slice(&[config.layer_scale_initial_scale], &[h])),
            post_attention_layernorm: nn::RmsNorm::from_pretrained(
                h,
                weights.get(&format!("{prefix}.post_attention_layernorm.weight")),
                rms_norm_eps,
            )?,
            gate_proj: nn::Linear::from_pretrained(
                h, intermediate_size,
                weights.get(&format!("{prefix}.gate_proj.weight")),
                None::<&Array>,
            )?,
            up_proj: nn::Linear::from_pretrained(
                h, intermediate_size,
                weights.get(&format!("{prefix}.up_proj.weight")),
                None::<&Array>,
            )?,
            down_proj: nn::Linear::from_pretrained(
                intermediate_size, h,
                weights.get(&format!("{prefix}.down_proj.weight")),
                None::<&Array>,
            )?,
            mlp_layer_scale: weights.get(&format!("{prefix}.mlp_layer_scale"))
                .cloned()
                .unwrap_or_else(|| Array::from_slice(&[config.layer_scale_initial_scale], &[h])),
            n_heads,
            head_dim,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(true)
                .base(rope_theta)
                .build()
                .expect("RopeBuilder::build should never fail (Infallible)"),
        };
        pre_transformer_layers.push(layer);
    }

    // Upsample convs and ConvNeXt blocks
    let upsample_rates = &config.upsample_rates;
    let upsampling_ratios = &config.upsampling_ratios;
    let mut upsample_convs = Vec::new();
    let mut upsample_convnext = Vec::new();

    for (i, &ratio) in upsampling_ratios.iter().enumerate() {
        let prefix = format!("upsample_convs.{i}");
        let in_ch = if i == 0 { h } else { upsample_rates[i - 1] };
        let out_ch = upsample_rates[i];
        let trim = (ratio - 1) / 2; // approximate causal trim for ConvTranspose

        let conv_t = CausalConvTranspose1d {
            conv_t: nn::ConvTranspose1d::from_pretrained(
                in_ch, out_ch, ratio as i32, ratio, 0, false,
                weights.get(&format!("{prefix}.weight")),
                None,
            )?,
            trim_right: trim,
        };
        upsample_convs.push(conv_t);

        // ConvNeXt block after each upsample
        let cn_prefix = format!("upsample_convnext.{i}");
        let dwconv = CausalConv1d {
            conv: nn::Conv1d::from_pretrained(
                out_ch, out_ch, 7, 1, false,
                weights.get(&format!("{cn_prefix}.dwconv.weight")),
                None,
            )?,
            pad: 3,
        };
        upsample_convnext.push(ConvNeXtBlock {
            dwconv,
            norm_weight: weights.get(&format!("{cn_prefix}.norm.weight"))
                .cloned()
                .unwrap_or_else(|| Array::ones::<f32>(&[out_ch]).expect("Array::ones failed")),
            norm_bias: weights.get(&format!("{cn_prefix}.norm.bias"))
                .cloned()
                .unwrap_or_else(|| Array::zeros::<f32>(&[out_ch]).expect("Array::zeros failed")),
            pwconv1_weight: weights.get(&format!("{cn_prefix}.pwconv1.weight"))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{cn_prefix}.pwconv1.weight")))?,
            pwconv1_bias: weights.get(&format!("{cn_prefix}.pwconv1.bias"))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{cn_prefix}.pwconv1.bias")))?,
            pwconv2_weight: weights.get(&format!("{cn_prefix}.pwconv2.weight"))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{cn_prefix}.pwconv2.weight")))?,
            pwconv2_bias: weights.get(&format!("{cn_prefix}.pwconv2.bias"))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{cn_prefix}.pwconv2.bias")))?,
            gamma: weights.get(&format!("{cn_prefix}.gamma"))
                .cloned()
                .unwrap_or_else(|| Array::from_slice(&[config.layer_scale_initial_scale], &[out_ch])),
        });
    }

    // Decoder blocks
    let block_out_ch = upsample_rates.last().copied().unwrap_or(h);
    let initial_conv = CausalConv1d {
        conv: nn::Conv1d::from_pretrained(
            block_out_ch, block_out_ch, 7, 1, false,
            weights.get("initial_conv.weight"),
            None,
        )?,
        pad: 3,
    };

    let final_out_ch = 1; // mono audio
    let final_conv = CausalConv1d {
        conv: nn::Conv1d::from_pretrained(
            block_out_ch, final_out_ch, 7, 1, false,
            weights.get("final_conv.weight"),
            None,
        )?,
        pad: 3,
    };

    // Decoder blocks: 4 blocks with SnakeBeta activation
    let mut decoder_blocks = Vec::new();
    for i in 0..4 {
        let prefix = format!("decoder_blocks.{i}");
        let alpha_prefix = if i == 0 {
            "snake.alpha_exp".to_string()
        } else {
            format!("decoder_blocks.{i}.snake.alpha_exp")
        };
        let beta_prefix = if i == 0 {
            "snake.beta_exp".to_string()
        } else {
            format!("decoder_blocks.{i}.snake.beta_exp")
        };

        let snake = SnakeBeta {
            alpha_exp: weights.get(&alpha_prefix)
                .or_else(|| weights.get(&format!("{prefix}.snake.alpha")))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(alpha_prefix.clone()))?,
            beta_exp: weights.get(&beta_prefix)
                .or_else(|| weights.get(&format!("{prefix}.snake.beta")))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(beta_prefix.clone()))?,
        };

        let conv_t = CausalConvTranspose1d {
            conv_t: nn::ConvTranspose1d::from_pretrained(
                block_out_ch, block_out_ch, 8, 4, 0, false,
                weights.get(&format!("{prefix}.conv_t.weight")),
                None,
            )?,
            trim_right: 2,
        };

        let mut res_units = Vec::new();
        for j in 0..3 {
            let ru_prefix = format!("{prefix}.res_units.{j}");
            let ru_alpha = weights.get(&format!("{ru_prefix}.act1.alpha_exp"))
                .or_else(|| weights.get(&format!("{ru_prefix}.act1.alpha")))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{ru_prefix}.act1.alpha")))?;
            let ru_beta = weights.get(&format!("{ru_prefix}.act1.beta_exp"))
                .or_else(|| weights.get(&format!("{ru_prefix}.act1.beta")))
                .cloned()
                .ok_or_else(|| Error::WeightNotFound(format!("{ru_prefix}.act1.beta")))?;
            res_units.push(ResidualUnit {
                act1: SnakeBeta { alpha_exp: ru_alpha, beta_exp: ru_beta },
                conv1: CausalConv1d {
                    conv: nn::Conv1d::from_pretrained(
                        block_out_ch, block_out_ch, 3, 1, false,
                        weights.get(&format!("{ru_prefix}.conv1.weight")),
                        None,
                    )?,
                    pad: 1,
                },
                act2: SnakeBeta {
                    alpha_exp: weights.get(&format!("{ru_prefix}.act2.alpha_exp"))
                        .or_else(|| weights.get(&format!("{ru_prefix}.act2.alpha")))
                        .cloned()
                        .ok_or_else(|| Error::WeightNotFound(format!("{ru_prefix}.act2.alpha")))?,
                    beta_exp: weights.get(&format!("{ru_prefix}.act2.beta_exp"))
                        .or_else(|| weights.get(&format!("{ru_prefix}.act2.beta")))
                        .cloned()
                        .ok_or_else(|| Error::WeightNotFound(format!("{ru_prefix}.act2.beta")))?,
                },
                conv2: CausalConv1d {
                    conv: nn::Conv1d::from_pretrained(
                        block_out_ch, block_out_ch, 1, 1, false,
                        weights.get(&format!("{ru_prefix}.conv2.weight")),
                        None,
                    )?,
                    pad: 0,
                },
            });
        }

        decoder_blocks.push(DecoderBlock {
            snake,
            conv_t,
            res_units,
        });
    }

    // Final SnakeBeta
    let final_snake = SnakeBeta {
        alpha_exp: weights.get("final_snake.alpha_exp")
            .or_else(|| weights.get("final_snake.alpha"))
            .cloned()
            .ok_or_else(|| Error::WeightNotFound("final_snake.alpha".to_string()))?,
        beta_exp: weights.get("final_snake.beta_exp")
            .or_else(|| weights.get("final_snake.beta"))
            .cloned()
            .ok_or_else(|| Error::WeightNotFound("final_snake.beta".to_string()))?,
    };

    Ok(SpeechTokenizerDecoder {
        semantic_codebook,
        acoustic_codebooks,
        rvq_first_output_proj: rvq_first,
        rvq_rest_output_proj: rvq_rest,
        pre_conv,
        pre_transformer_input_proj,
        pre_transformer_output_proj,
        pre_transformer_norm,
        pre_transformer_layers,
        upsample_convs,
        upsample_convnext,
        initial_conv,
        decoder_blocks,
        final_snake,
        final_conv,
        config: config.clone(),
    })
}
