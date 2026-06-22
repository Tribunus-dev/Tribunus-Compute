//! Audio encoder — processes mel spectrograms into feature embeddings.
//!
//! Uses the model's `audio_encoder.*` weights (conformer/transformer encoder).
//! Outputs a sequence of audio frame embeddings projected into the text model's
//! hidden dimension.

use crate::config::AudioArchitecture;
use crate::profiled_executor::LoadedProfiledModel;
use crate::quantized::QuantizedLinearBinding;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::Array;

/// One layer of the audio encoder (transformer block).
#[derive(Clone, Debug)]
pub struct AudioEncoderLayer {
    /// Self-attention QKV projection + output projection.
    pub q_proj: QuantizedLinearBinding,
    pub k_proj: QuantizedLinearBinding,
    pub v_proj: QuantizedLinearBinding,
    pub o_proj: QuantizedLinearBinding,
    /// Feed-forward network.
    pub gate_proj: QuantizedLinearBinding,
    pub up_proj: QuantizedLinearBinding,
    pub down_proj: QuantizedLinearBinding,
    /// Layer norms.
    pub input_layernorm: Array,
    pub post_attention_layernorm: Array,
    /// Hidden dimension for this layer.
    pub hidden_size: u32,
}

impl AudioEncoderLayer {
    /// Run one encoder layer forward pass.
    pub fn forward(&self, x: &Array) -> Result<Array, String> {
        // Self-attention with pre-norm.
        let residual = x;
        let normed = self.rms_norm(x, &self.input_layernorm)?;

        // QKV projections.
        let q = self
            .q_proj
            .forward(&normed)
            .map_err(|e| format!("q_proj: {:?}", e))?;
        let k = self
            .k_proj
            .forward(&normed)
            .map_err(|e| format!("k_proj: {:?}", e))?;
        let v = self
            .v_proj
            .forward(&normed)
            .map_err(|e| format!("v_proj: {:?}", e))?;

        // Self-attention.
        let _dim = self.hidden_size as i32;
        let n_heads = self.hidden_size as i32 / 64; // Default head_dim = 64
        let head_dim = 64i32;

        let q = q.reshape(&[-1, n_heads, head_dim])?;
        let k = k.reshape(&[-1, n_heads, head_dim])?;
        let v = v.reshape(&[-1, n_heads, head_dim])?;

        // Transpose to [n_heads, seq_len, head_dim].
        let q = ops::transpose_axes(&q, &[1, 0, 2])?;
        let k = ops::transpose_axes(&k, &[1, 0, 2])?;
        let v = ops::transpose_axes(&v, &[1, 0, 2])?;

        // Scaled dot-product attention.
        let scale = (head_dim as f32).sqrt();
        let scores = q
            .matmul(&ops::transpose_axes(&k, &[0, 2, 1])?)?
            .divide(&Array::from_f32(scale))?;
        let attn_weights =
            ops::softmax_axes(&scores, &[-1], None).map_err(|e| format!("softmax: {:?}", e))?;
        let attn_out = attn_weights
            .matmul(&v)?
            .reshape(&[-1, n_heads * head_dim])?;

        // Output projection.
        let attn_out = self
            .o_proj
            .forward(&attn_out)
            .map_err(|e| format!("o_proj: {:?}", e))?;

        // Residual + post-attention norm.
        let x = ops::add(residual, &attn_out).map_err(|e| format!("attn residual: {:?}", e))?;
        let normed = self.rms_norm(&x, &self.post_attention_layernorm)?;

        // Feed-forward (SwiGLU).
        let gate = self
            .gate_proj
            .forward(&normed)
            .map_err(|e| format!("gate_proj: {:?}", e))?;
        let up = self
            .up_proj
            .forward(&normed)
            .map_err(|e| format!("up_proj: {:?}", e))?;
        let gated =
            ops::multiply(&nn::silu(&gate)?, &up).map_err(|e| format!("silu*gate: {:?}", e))?;
        let ffn_out = self
            .down_proj
            .forward(&gated)
            .map_err(|e| format!("down_proj: {:?}", e))?;

        let result = ops::add(&x, &ffn_out).map_err(|e| format!("ffn residual: {:?}", e))?;
        result.eval()?;
        Ok(result)
    }

    /// RMS normalization.
    fn rms_norm(&self, x: &Array, weight: &Array) -> Result<Array, String> {
        let eps = 1e-6f32;
        use crate::primitives;
        primitives::rms_norm(x, weight, eps).map_err(|e| format!("rms_norm: {:?}", e))
    }
}

/// Audio encoder — processes waveforms into feature embeddings.
///
/// Uses the model's `audio_encoder.*` weights (typically a transformer or
/// conformer encoder). Outputs a sequence of audio frame embeddings
/// projected into the text model's hidden dimension.
pub struct AudioEncoder {
    /// Input projection: mel_bins → hidden_size
    pub input_proj: QuantizedLinearBinding,
    /// Encoder transformer layers.
    pub encoder_layers: Vec<AudioEncoderLayer>,
    /// Output projection: hidden_size → text_projection_dim (text hidden_dim).
    pub output_proj: Option<QuantizedLinearBinding>,
    /// Configuration.
    pub config: AudioArchitecture,
}

impl AudioEncoder {
    /// Load audio encoder weights from a loaded model.
    pub fn load(model: &LoadedProfiledModel) -> Result<Self, String> {
        let audio_prefix = "audio_encoder";
        let hidden_size = model.reader.manifest.architecture.hidden_size;

        // Load the audio config if available in the manifest. Without it,
        // use defaults derived from the audio_encoder tensor shapes.
        let audio_config = model.reader.manifest.audio_config.clone();

        // Helper to find a tensor entry and create a QuantizedLinearBinding.
        let load_side_tensor = |model: &LoadedProfiledModel,
                                name: &str,
                                segment: &std::sync::Arc<crate::mapped_image::MappedSegment>|
         -> Result<u64, String> {
            let entry = model
                .reader
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.name == name);
            match entry {
                Some(e) => {
                    let (arr, _) = crate::profiled_executor::load_tensor_from_mapped_segment(
                        segment, e, false,
                    )
                    .map_err(|err| format!("load side tensor {}: {:?}", name, err))?;
                    Ok(crate::bridge::ARRAY_REGISTRY.write().insert(arr, None))
                }
                None => {
                    // Create dummy zero scales/biases if not present.
                    let entry = model
                        .reader
                        .manifest
                        .tensor_table
                        .iter()
                        .find(|e| {
                            e.name
                                == name
                                    .replace(".scales", ".weight")
                                    .replace(".biases", ".weight")
                        })
                        .ok_or_else(|| {
                            format!("weight tensor not found for side tensor: {}", name)
                        })?;
                    let shape: Vec<i32> = entry.logical_shape.iter().map(|&d| d as i32).collect();
                    let out_dim = shape[0];
                    let n_groups = 1;
                    let dummy = Array::from_slice(
                        &vec![0.0f32; (out_dim * n_groups) as usize],
                        &[out_dim, n_groups],
                    );
                    let dummy_biases = Array::from_slice(
                        &vec![0.0f32; (out_dim * n_groups) as usize],
                        &[out_dim, n_groups],
                    );
                    if name.ends_with(".scales") {
                        Ok(crate::bridge::ARRAY_REGISTRY.write().insert(dummy, None))
                    } else {
                        Ok(crate::bridge::ARRAY_REGISTRY
                            .write()
                            .insert(dummy_biases, None))
                    }
                }
            }
        };

        let find_tensor = |name: &str| -> Result<(u64, u64, u64, u32, u32), String> {
            let entry = model
                .reader
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| format!("audio tensor not found: {}", name))?;

            let segment = model
                .mapped_image
                .segments
                .get(&entry.segment)
                .ok_or_else(|| format!("segment not found: {}", entry.segment))?;

            let (weight, _) =
                crate::profiled_executor::load_tensor_from_mapped_segment(segment, entry, false)
                    .map_err(|e| format!("load tensor {}: {:?}", name, e))?;

            let handle = crate::bridge::ARRAY_REGISTRY.write().insert(weight, None);
            let logical_shape = entry.logical_shape.as_slice();
            let out_features = *logical_shape.first().unwrap_or(&0);
            let in_features = *logical_shape.get(1).unwrap_or(&0);

            // Load scales and biases.
            let scales_name = name.replace(".weight", ".scales");
            let biases_name = name.replace(".weight", ".biases");

            let scales_handle = load_side_tensor(&model, &scales_name, segment)?;
            let biases_handle = load_side_tensor(&model, &biases_name, segment)?;

            Ok((
                handle,
                scales_handle,
                biases_handle,
                out_features,
                in_features,
            ))
        };

        // Input projection.
        let input_proj_name = format!("{}.input_proj.weight", audio_prefix);
        let (w_h, s_h, b_h, out_feat, in_feat) = find_tensor(&input_proj_name)?;

        let input_proj = QuantizedLinearBinding::new(w_h, s_h, b_h, out_feat, in_feat, 64, 8, true);

        // Determine encoder layer count from audio config or tensor names.
        let num_layers = audio_config
            .as_ref()
            .map(|c| c.num_hidden_layers)
            .unwrap_or_else(|| {
                // Count audio_encoder.layers.* blocks from tensor table.
                model
                    .reader
                    .manifest
                    .tensor_table
                    .iter()
                    .filter(|e| {
                        e.name.starts_with(&format!("{}.layers.", audio_prefix))
                            && e.name.contains(".self_attn.q_proj.weight")
                    })
                    .count() as u32
            });

        // Load encoder layers.
        let mut encoder_layers = Vec::with_capacity(num_layers as usize);
        let hsize = audio_config
            .as_ref()
            .map(|c| c.hidden_size)
            .unwrap_or(out_feat);

        for layer_idx in 0..num_layers {
            let layer_prefix = format!("{}.layers.{}", audio_prefix, layer_idx);

            let load_qkv = |proj: &str| -> Result<QuantizedLinearBinding, String> {
                let w_name = format!("{}.self_attn.{}.weight", layer_prefix, proj);
                let (w, s, b, out_f, in_f) = find_tensor(&w_name)?;
                Ok(QuantizedLinearBinding::new(
                    w, s, b, out_f, in_f, 64, 8, true,
                ))
            };

            let load_mlp = |proj: &str| -> Result<QuantizedLinearBinding, String> {
                let w_name = format!("{}.mlp.{}.weight", layer_prefix, proj);
                let (w, s, b, out_f, in_f) = find_tensor(&w_name)?;
                Ok(QuantizedLinearBinding::new(
                    w, s, b, out_f, in_f, 64, 8, true,
                ))
            };

            let load_norm = |norm: &str| -> Result<Array, String> {
                let n_name = format!("{}.{}.weight", layer_prefix, norm);
                let entry = model
                    .reader
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.name == n_name)
                    .ok_or_else(|| format!("tensor not found: {}", n_name))?;
                let segment = model
                    .mapped_image
                    .segments
                    .get(&entry.segment)
                    .ok_or_else(|| format!("segment not found: {}", entry.segment))?;
                let (arr, _) = crate::profiled_executor::load_tensor_from_mapped_segment(
                    segment, entry, false,
                )
                .map_err(|e| format!("load norm {}: {:?}", n_name, e))?;
                Ok(arr)
            };

            let layer = AudioEncoderLayer {
                q_proj: load_qkv("q_proj")?,
                k_proj: load_qkv("k_proj")?,
                v_proj: load_qkv("v_proj")?,
                o_proj: load_qkv("o_proj")?,
                gate_proj: load_mlp("gate_proj")?,
                up_proj: load_mlp("up_proj")?,
                down_proj: load_mlp("down_proj")?,
                input_layernorm: load_norm("input_layernorm")?,
                post_attention_layernorm: load_norm("post_attention_layernorm")?,
                hidden_size: hsize,
            };

            encoder_layers.push(layer);
        }

        // Output projection.
        let output_proj_name = format!("{}.output_proj.weight", audio_prefix);
        let (w_h, s_h, b_h, out_f, in_f) =
            find_tensor(&output_proj_name).unwrap_or_else(|_| (0, 0, 0, hidden_size, hsize));

        let output_proj = if w_h != 0 {
            Some(QuantizedLinearBinding::new(
                w_h, s_h, b_h, out_f, in_f, 64, 8, true,
            ))
        } else {
            // No output projection — pass through encoder output as-is.
            None
        };

        let config = audio_config.unwrap_or(AudioArchitecture {
            hidden_size: hsize,
            num_attention_heads: hsize / 64,
            num_hidden_layers: num_layers,
            intermediate_size: hsize * 4,
            sample_rate: 16000,
            num_mel_bins: 80,
            hop_length: 160,
            max_audio_length_s: 30,
            projection_dim: hidden_size,
        });

        Ok(Self {
            input_proj,
            encoder_layers,
            output_proj,
            config,
        })
    }

    /// Encode audio into feature tokens.
    ///
    /// `mel_spec` — mel spectrogram from preprocessing, shape `[1, num_mel_bins, num_frames]`.
    ///
    /// Returns `[num_frames, projection_dim]` array for text model injection.
    pub fn encode(&self, mel_spec: &Array) -> Result<Array, String> {
        let mel_ndim = mel_spec.ndim();
        if mel_ndim != 3 {
            return Err(format!(
                "mel_spec must be rank 3 [1, num_mel_bins, num_frames], got rank {}",
                mel_ndim
            ));
        }

        let n_mel = mel_spec.shape()[1];
        let n_frames = mel_spec.shape()[2];

        // Reshape to [n_frames, n_mel] — batch dim is 1, so just squeeze the batch dim.
        let x = mel_spec.reshape(&[n_mel as i32, n_frames as i32])?;
        let x = ops::transpose_axes(&x, &[1, 0])?;

        // Input projection: mel_bins → hidden_size.
        let mut x = self
            .input_proj
            .forward(&x)
            .map_err(|e| format!("input_proj: {:?}", e))?;

        // Encoder layers.
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }

        // Output projection: hidden_size -> projection_dim (text hidden size).
        if let Some(proj) = &self.output_proj {
            x = proj
                .forward(&x)
                .map_err(|e| format!("output_proj: {:?}", e))?;
        }

        x.eval()?;
        Ok(x)
    }
}
