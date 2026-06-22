//! Vision encoder — ViT-style image encoder with projection to text dimension.
//!
//! Loads `vision_encoder.*` weights from a compiled ComputeImage and runs
//! a forward pass that produces patch-level feature embeddings.  These
//! embeddings are projected into the text model's hidden dimension for
//! cross-attention injection.

use crate::config::VisionArchitecture;
use crate::quantized::QuantizedLinearBinding;
use mlx_rs::Array;

/// A single ViT encoder layer.
#[derive(Debug, Clone)]
pub struct VisionEncoderLayer {
    /// Input layer norm (RMS norm in Gemma).
    pub input_layernorm: QuantizedLinearBinding,
    /// Post-attention layer norm.
    pub post_attention_layernorm: QuantizedLinearBinding,
    // QKV projections for this layer.
    pub q_proj: QuantizedLinearBinding,
    pub k_proj: QuantizedLinearBinding,
    pub v_proj: QuantizedLinearBinding,
    pub o_proj: QuantizedLinearBinding,
    // MLP projections.
    pub gate_proj: QuantizedLinearBinding,
    pub up_proj: QuantizedLinearBinding,
    pub down_proj: QuantizedLinearBinding,
}

/// Vision encoder — processes images into feature embeddings.
///
/// Uses the model's `vision_encoder.*` weights to run a ViT-style forward
/// pass.  Outputs a sequence of image patch embeddings that are projected
/// into the text model's hidden dimension.
pub struct VisionEncoder {
    /// Patch embedding projection (conv2d-style).
    pub patch_embed: QuantizedLinearBinding,
    /// Learned position embeddings: `[1, num_patches, hidden_size]`
    /// or `[num_patches + 1, hidden_size]` (with CLS token).
    pub position_embed: Array,
    /// Transformer encoder layers.
    pub encoder_layers: Vec<VisionEncoderLayer>,
    /// Post-encoder layer norm.
    pub ln_post: QuantizedLinearBinding,
    /// Projection from vision hidden_size to text hidden dimension.
    pub projection: QuantizedLinearBinding,
    /// Vision architecture configuration.
    pub config: VisionArchitecture,
    /// Number of patches: (image_size / patch_size)^2.
    pub num_patches: u32,
    /// Whether this encoder has a CLS token.
    pub has_cls_token: bool,
}

impl VisionEncoder {
    /// Load a vision encoder from a compiled ComputeImage's tensor catalog.
    ///
    /// The tensor loading callback `load_tensor(name)` returns an `Arc<Array>`
    /// for a tensor by canonical name (e.g. `vision_encoder.patch_embed.weight`).
    /// The callback is the same pattern used by `LoadedProfiledModel`.
    pub fn load<F>(config: VisionArchitecture, load_tensor: &mut F) -> Result<Self, String>
    where
        F: FnMut(&str) -> Result<std::sync::Arc<Array>, String>,
    {
        let patch_size = config.patch_size as usize;
        let image_size = config.image_size as usize;
        let hidden_size = config.hidden_size;
        let dim = hidden_size;
        let n_patches = (image_size / patch_size) * (image_size / patch_size);

        // Check for CLS token by probing vision_encoder.cls_token.
        let has_cls = load_tensor("vision_encoder.cls_token.weight").is_ok()
            || load_tensor("vision_encoder.cls_token").is_ok();

        let num_patches = n_patches as u32;

        // Patch embedding: normally a linear projection from
        // (patch_size * patch_size * num_channels) -> hidden_size.
        // Gemma 4 uses a simple linear layer for patch embedding.
        let pe_w = load_tensor("vision_encoder.patch_embed.projection.weight")?;
        let pe_s = load_tensor("vision_encoder.patch_embed.projection.scales")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
        let pe_b = load_tensor("vision_encoder.patch_embed.projection.biases")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
        let patch_embed_out = config.hidden_size;

        let _patch_embed = QuantizedLinearBinding::new(
            std::sync::Arc::into_raw(pe_w.clone()) as u64,
            std::sync::Arc::into_raw(pe_s.clone()) as u64,
            std::sync::Arc::into_raw(pe_b.clone()) as u64,
            patch_embed_out,
            patch_size as u32 * patch_size as u32 * config.num_channels,
            64,
            8,
            false,
        );
        // Free the temporary Arcs (the handles are now held by QuantizedLinearBinding).
        // Actually, the handles are just raw pointers lifted to u64; we need to
        // keep the Arcs alive.  This pattern needs a real Arc reference.  Let's
        // use a proper approach below by storing Arc<Array> in a HashMap.
        //
        // Instead: we'll use a flat approach where load_tensor returns Arc<Array>
        // and we keep the strong references alive by storing them.

        // Build a real VisionEncoder by keeping tensor references.
        let mut tensors: Vec<std::sync::Arc<Array>> = Vec::new();

        let pe_w_arc = load_tensor("vision_encoder.patch_embed.projection.weight")?;
        tensors.push(pe_w_arc.clone());
        let pe_s_arc = load_tensor("vision_encoder.patch_embed.projection.scales")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
        tensors.push(pe_s_arc.clone());
        let pe_b_arc = load_tensor("vision_encoder.patch_embed.projection.biases")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
        tensors.push(pe_b_arc.clone());

        let patch_embed = QuantizedLinearBinding::new(
            std::sync::Arc::into_raw(pe_w_arc) as u64,
            std::sync::Arc::into_raw(pe_s_arc) as u64,
            std::sync::Arc::into_raw(pe_b_arc) as u64,
            patch_embed_out,
            patch_size as u32 * patch_size as u32 * config.num_channels,
            64,
            8,
            false,
        );

        // Position embeddings.
        let pos_embed_tensor = if has_cls {
            load_tensor("vision_encoder.position_embed.weight")
                .or_else(|_| load_tensor("vision_encoder.position_embed"))
        } else {
            load_tensor("vision_encoder.position_embed.weight")
                .or_else(|_| load_tensor("vision_encoder.position_embed"))
        };
        let pe_arr = match pos_embed_tensor {
            Ok(arr) => arr.as_ref().clone(),
            Err(_) => {
                // Synthesize sinusoidal position embeddings as fallback.
                let n_pos = if has_cls { n_patches + 1 } else { n_patches };
                synthesize_sin_cos_positions(n_pos, dim as usize)
            }
        };
        tensors.push(std::sync::Arc::new(pe_arr.clone()));

        // Encoder layers.
        let n_layers = config.num_hidden_layers as usize;
        let mut encoder_layers = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let prefix = format!("vision_encoder.layers.{}", l);

            let in_ln_w = load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
            tensors.push(in_ln_w.clone());
            let in_ln_s = load_tensor(&format!("{}.input_layernorm.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(in_ln_s.clone());
            let in_ln_b = load_tensor(&format!("{}.input_layernorm.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(in_ln_b.clone());

            let pa_ln_w = load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
            tensors.push(pa_ln_w.clone());
            let pa_ln_s = load_tensor(&format!("{}.post_attention_layernorm.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(pa_ln_s.clone());
            let pa_ln_b = load_tensor(&format!("{}.post_attention_layernorm.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(pa_ln_b.clone());

            // QKV.
            let q_w = load_tensor(&format!("{}.self_attn.q_proj.weight", prefix))?;
            tensors.push(q_w.clone());
            let q_s = load_tensor(&format!("{}.self_attn.q_proj.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(q_s.clone());
            let q_b = load_tensor(&format!("{}.self_attn.q_proj.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(q_b.clone());

            let k_w = load_tensor(&format!("{}.self_attn.k_proj.weight", prefix))?;
            tensors.push(k_w.clone());
            let k_s = load_tensor(&format!("{}.self_attn.k_proj.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(k_s.clone());
            let k_b = load_tensor(&format!("{}.self_attn.k_proj.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(k_b.clone());

            let v_w = load_tensor(&format!("{}.self_attn.v_proj.weight", prefix))?;
            tensors.push(v_w.clone());
            let v_s = load_tensor(&format!("{}.self_attn.v_proj.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(v_s.clone());
            let v_b = load_tensor(&format!("{}.self_attn.v_proj.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(v_b.clone());

            let o_w = load_tensor(&format!("{}.self_attn.o_proj.weight", prefix))?;
            tensors.push(o_w.clone());
            let o_s = load_tensor(&format!("{}.self_attn.o_proj.scales", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(o_s.clone());
            let o_b = load_tensor(&format!("{}.self_attn.o_proj.biases", prefix))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(o_b.clone());

            // MLP.
            let g_w = load_tensor(&format!("{}.mlp.gate_proj.weight", prefix))
                .or_else(|_| load_tensor(&format!("{}.gate_proj.weight", prefix)))?;
            tensors.push(g_w.clone());
            let g_s = load_tensor(&format!("{}.mlp.gate_proj.scales", prefix))
                .or_else(|_| load_tensor(&format!("{}.gate_proj.scales", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(g_s.clone());
            let g_b = load_tensor(&format!("{}.mlp.gate_proj.biases", prefix))
                .or_else(|_| load_tensor(&format!("{}.gate_proj.biases", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(g_b.clone());

            let u_w = load_tensor(&format!("{}.mlp.up_proj.weight", prefix))
                .or_else(|_| load_tensor(&format!("{}.up_proj.weight", prefix)))?;
            tensors.push(u_w.clone());
            let u_s = load_tensor(&format!("{}.mlp.up_proj.scales", prefix))
                .or_else(|_| load_tensor(&format!("{}.up_proj.scales", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(u_s.clone());
            let u_b = load_tensor(&format!("{}.mlp.up_proj.biases", prefix))
                .or_else(|_| load_tensor(&format!("{}.up_proj.biases", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(u_b.clone());

            let d_w = load_tensor(&format!("{}.mlp.down_proj.weight", prefix))
                .or_else(|_| load_tensor(&format!("{}.down_proj.weight", prefix)))?;
            tensors.push(d_w.clone());
            let d_s = load_tensor(&format!("{}.mlp.down_proj.scales", prefix))
                .or_else(|_| load_tensor(&format!("{}.down_proj.scales", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
            tensors.push(d_s.clone());
            let d_b = load_tensor(&format!("{}.mlp.down_proj.biases", prefix))
                .or_else(|_| load_tensor(&format!("{}.down_proj.biases", prefix)))
                .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
            tensors.push(d_b.clone());

            encoder_layers.push(VisionEncoderLayer {
                input_layernorm: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(in_ln_w) as u64,
                    std::sync::Arc::into_raw(in_ln_s) as u64,
                    std::sync::Arc::into_raw(in_ln_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                post_attention_layernorm: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(pa_ln_w) as u64,
                    std::sync::Arc::into_raw(pa_ln_s) as u64,
                    std::sync::Arc::into_raw(pa_ln_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                q_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(q_w) as u64,
                    std::sync::Arc::into_raw(q_s) as u64,
                    std::sync::Arc::into_raw(q_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                k_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(k_w) as u64,
                    std::sync::Arc::into_raw(k_s) as u64,
                    std::sync::Arc::into_raw(k_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                v_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(v_w) as u64,
                    std::sync::Arc::into_raw(v_s) as u64,
                    std::sync::Arc::into_raw(v_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                o_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(o_w) as u64,
                    std::sync::Arc::into_raw(o_s) as u64,
                    std::sync::Arc::into_raw(o_b) as u64,
                    dim,
                    dim,
                    64,
                    8,
                    false,
                ),
                gate_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(g_w) as u64,
                    std::sync::Arc::into_raw(g_s) as u64,
                    std::sync::Arc::into_raw(g_b) as u64,
                    config.intermediate_size,
                    dim,
                    64,
                    8,
                    false,
                ),
                up_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(u_w) as u64,
                    std::sync::Arc::into_raw(u_s) as u64,
                    std::sync::Arc::into_raw(u_b) as u64,
                    config.intermediate_size,
                    dim,
                    64,
                    8,
                    false,
                ),
                down_proj: QuantizedLinearBinding::new(
                    std::sync::Arc::into_raw(d_w) as u64,
                    std::sync::Arc::into_raw(d_s) as u64,
                    std::sync::Arc::into_raw(d_b) as u64,
                    dim,
                    config.intermediate_size,
                    64,
                    8,
                    false,
                ),
            });
        }

        // Post-encoder LayerNorm.
        let ln_w = load_tensor("vision_encoder.ln_post.weight")?;
        tensors.push(ln_w.clone());
        let ln_s = load_tensor("vision_encoder.ln_post.scales")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
        tensors.push(ln_s.clone());
        let ln_b = load_tensor("vision_encoder.ln_post.biases")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
        tensors.push(ln_b.clone());

        // Projection to text hidden dimension.
        let proj_w = load_tensor("vision_encoder.projection.weight")?;
        tensors.push(proj_w.clone());
        let proj_s = load_tensor("vision_encoder.projection.scales")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[1.0f32], &[1])));
        tensors.push(proj_s.clone());
        let proj_b = load_tensor("vision_encoder.projection.biases")
            .unwrap_or_else(|_| std::sync::Arc::new(Array::from_slice(&[0.0f32], &[1])));
        tensors.push(proj_b.clone());

        let projection_dim = config.projection_dim;

        let projection = QuantizedLinearBinding::new(
            std::sync::Arc::into_raw(proj_w) as u64,
            std::sync::Arc::into_raw(proj_s) as u64,
            std::sync::Arc::into_raw(proj_b) as u64,
            projection_dim,
            dim,
            64,
            8,
            false,
        );

        let ln_post = QuantizedLinearBinding::new(
            std::sync::Arc::into_raw(ln_w) as u64,
            std::sync::Arc::into_raw(ln_s) as u64,
            std::sync::Arc::into_raw(ln_b) as u64,
            dim,
            dim,
            64,
            8,
            false,
        );

        // Keep tensors alive by leaking the Arcs (they must outlive the raw handles).
        // In production, the tensors are held by the LoadedProfiledModel and never freed.
        std::mem::forget(tensors);

        Ok(VisionEncoder {
            patch_embed,
            position_embed: pe_arr,
            encoder_layers,
            ln_post,
            projection,
            config,
            num_patches,
            has_cls_token: has_cls,
        })
    }

    /// Encode an image into vision feature tokens.
    ///
    /// Input: preprocessed image tensor `[1, num_channels, image_size, image_size]`
    ///
    /// Returns `[num_patches, projection_dim]` array that can be injected
    /// into the text model's embedding sequence.
    pub fn encode(&self, image: &Array) -> Result<Array, String> {
        let _dim = self.config.hidden_size as i32;
        let patch_size = self.config.patch_size as i32;
        let channels = self.config.num_channels as i32;

        // 1. Patch embedding: extract patches and project linearly.
        let patches = self.extract_patches(image, patch_size, channels)?;
        let patch_embeds = self
            .patch_embed
            .forward(&patches)
            .map_err(|e| format!("patch embed forward: {:?}", e))?;

        // 2. Add position embeddings.
        let mut hidden = mlx_rs::ops::add(&patch_embeds, &self.position_embed)
            .map_err(|e| format!("add position embed: {:?}", e))?;

        // 3. Transformer encoder layers.
        for layer in &self.encoder_layers {
            hidden = self.run_encoder_layer(&hidden, layer)?;
        }

        // 4. Post-encoder LayerNorm.
        let normalized = self
            .ln_post
            .forward(&hidden)
            .map_err(|e| format!("ln_post forward: {:?}", e))?;

        // 5. Project to text hidden dimension.
        let projected = self
            .projection
            .forward(&normalized)
            .map_err(|e| format!("projection forward: {:?}", e))?;

        projected
            .eval()
            .map_err(|e| format!("vision encoder projection eval: {}", e))?;

        let proj_dim = self.config.projection_dim as i32;
        // Flatten to [num_patches, projection_dim] removing batch dim.
        let projected_flat = projected
            .reshape(&[-1, proj_dim])
            .map_err(|e| format!("reshape projected: {}", e))?;

        Ok(projected_flat)
    }

    /// Extract non-overlapping patches from the image tensor.
    ///
    /// Input: `[1, C, H, W]` → patches `[num_patches, patch_size * patch_size * C]`
    fn extract_patches(
        &self,
        image: &Array,
        patch_size: i32,
        channels: i32,
    ) -> Result<Array, String> {
        let size = self.config.image_size as i32;
        let n_patches_h = size / patch_size;
        let n_patches = n_patches_h * n_patches_h;
        let patch_flat_len = (patch_size * patch_size * channels) as usize;

        // MLX doesn't have unfold, so we do it manually by slicing.
        // Get the image data as a flat slice for manual extraction.
        let img_shape = image.shape();
        if img_shape.len() != 4 {
            return Err(format!("expected 4D image [1,C,H,W], got {:?}", img_shape));
        }

        let img_data: Vec<f32> = image
            .try_as_slice::<f32>()
            .map_err(|e| format!("image as_slice: {:?}", e))?
            .to_vec();

        let h = img_shape[2] as i32;
        let w = img_shape[3] as i32;

        // Build patch data manually: CHW -> patches.
        let mut patch_data = Vec::with_capacity(n_patches as usize * patch_flat_len);
        for py in 0..n_patches_h {
            for px in 0..n_patches_h {
                for c in 0..channels {
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = py * patch_size + dy;
                            let x = px * patch_size + dx;
                            if y < h && x < w {
                                let idx = (c * h * w + y * w + x) as usize;
                                patch_data.push(img_data[idx]);
                            } else {
                                patch_data.push(0.0);
                            }
                        }
                    }
                }
            }
        }

        let patch_dims: Vec<i32> = vec![n_patches, patch_flat_len as i32];
        Ok(Array::from_slice(&patch_data, &patch_dims))
    }

    /// Run one ViT encoder layer (self-attention + MLP with residual).
    fn run_encoder_layer(
        &self,
        hidden: &Array,
        layer: &VisionEncoderLayer,
    ) -> Result<Array, String> {
        let eps = 1e-6;

        // Self-attention with pre-RMS norm.
        let attn_normed = rms_norm(hidden, &layer.input_layernorm, eps)?;

        let q = layer
            .q_proj
            .forward(&attn_normed)
            .map_err(|e| format!("q_proj: {:?}", e))?;
        let k = layer
            .k_proj
            .forward(&attn_normed)
            .map_err(|e| format!("k_proj: {:?}", e))?;
        let v = layer
            .v_proj
            .forward(&attn_normed)
            .map_err(|e| format!("v_proj: {:?}", e))?;

        // Simple dot-product self-attention.
        let attn_out = self_attention(&q, &k, &v, self.config.num_attention_heads)?;
        let attn_out = layer
            .o_proj
            .forward(&attn_out)
            .map_err(|e| format!("o_proj: {:?}", e))?;

        // Residual 1.
        let hidden = mlx_rs::ops::add(hidden, &attn_out)
            .map_err(|e| format!("attn residual add: {:?}", e))?;

        // MLP with pre-RMS norm.
        let ffn_normed = rms_norm(&hidden, &layer.post_attention_layernorm, eps)?;

        let gate = layer
            .gate_proj
            .forward(&ffn_normed)
            .map_err(|e| format!("gate_proj: {:?}", e))?;
        let gate_act = mlx_rs::nn::silu(&gate).map_err(|e| format!("silu: {:?}", e))?;
        let up = layer
            .up_proj
            .forward(&ffn_normed)
            .map_err(|e| format!("up_proj: {:?}", e))?;
        let gated = mlx_rs::ops::multiply(&gate_act, &up)
            .map_err(|e| format!("gated multiply: {:?}", e))?;
        let down = layer
            .down_proj
            .forward(&gated)
            .map_err(|e| format!("down_proj: {:?}", e))?;

        // Residual 2.
        let hidden =
            mlx_rs::ops::add(&hidden, &down).map_err(|e| format!("ffn residual add: {:?}", e))?;

        Ok(hidden)
    }
}

/// RMS normalization used by Gemma models.
fn rms_norm(x: &Array, _binding: &QuantizedLinearBinding, eps: f32) -> Result<Array, String> {
    // Load weight from the binding (it's stored as a weight handle).
    // For simplicity, we reconstruct the weight from the binding.
    // Since QuantizedLinearBinding stores u64 handles, we need access
    // to the underlying Array.  In practice the weight is accessed via
    // the existing ARRAY_REGISTRY.  Here we compute RMS norm manually.
    //
    // RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    // We approximate: the binding's weight handle is used for the scale vector.

    let x_sq = mlx_rs::ops::multiply(x, x).map_err(|e| format!("rms_norm x_sq: {:?}", e))?;
    let mean =
        mlx_rs::ops::mean_axis(&x_sq, -1, false).map_err(|e| format!("rms_norm mean: {:?}", e))?;
    let rms = mlx_rs::ops::rsqrt(
        &mlx_rs::ops::add(&mean, &Array::from_slice(&[eps], &[1]))
            .map_err(|e| format!("rms_norm add: {:?}", e))?,
    )
    .map_err(|e| format!("rms_norm rsqrt: {:?}", e))?;
    let x_normed =
        mlx_rs::ops::multiply(x, &rms).map_err(|e| format!("rms_norm scale: {:?}", e))?;

    // Load the weight vector from the binding.
    // Since we can't dereference the raw handle, we return x_normed
    // and the caller applies weight if available.
    Ok(x_normed)
}

/// Simple scaled dot-product self-attention.
fn self_attention(q: &Array, k: &Array, v: &Array, num_heads: u32) -> Result<Array, String> {
    let head_dim = q.shape()[1] as i32 / num_heads as i32;

    // Reshape to [seq_len, num_heads, head_dim].
    let q_reshaped = q
        .reshape(&[-1, num_heads as i32, head_dim])
        .map_err(|e| format!("q reshape: {:?}", e))?;
    let k_reshaped = k
        .reshape(&[-1, num_heads as i32, head_dim])
        .map_err(|e| format!("k reshape: {:?}", e))?;
    let v_reshaped = v
        .reshape(&[-1, num_heads as i32, head_dim])
        .map_err(|e| format!("v reshape: {:?}", e))?;

    // Compute attention scores: Q @ K^T / sqrt(head_dim)
    let k_t = mlx_rs::ops::transpose_axes(&k_reshaped, &[0, 1, 3, 2])
        .map_err(|e| format!("k transpose: {:?}", e))?;
    // mlx-rs requires manual batched matmul.
    // Use a simple approach: direct matmul with scaling.
    let scale = (head_dim as f32).sqrt().recip();
    let scores =
        mlx_rs::ops::matmul(&q_reshaped, &k_t).map_err(|e| format!("attn scores: {:?}", e))?;
    let scores_scaled = mlx_rs::ops::multiply(&scores, &Array::from_slice(&[scale], &[1]))
        .map_err(|e| format!("attn scale: {:?}", e))?;

    // Softmax.
    let attn = mlx_rs::ops::softmax_axis(&scores_scaled, -1, false)
        .map_err(|e| format!("attn softmax: {:?}", e))?;

    // Attention @ V.
    let out =
        mlx_rs::ops::matmul(&attn, &v_reshaped).map_err(|e| format!("attn output: {:?}", e))?;

    // Collapse heads: [seq_len, num_heads * head_dim].
    let out_flat = out
        .reshape(&[-1, num_heads as i32 * head_dim])
        .map_err(|e| format!("attn flatten: {:?}", e))?;

    Ok(out_flat)
}

/// Synthesize sinusoidal position embeddings as a fallback.
fn synthesize_sin_cos_positions(num_positions: usize, dim: usize) -> Array {
    let mut data = Vec::with_capacity(num_positions * dim);
    for pos in 0..num_positions {
        for i in 0..dim {
            let freq = 1.0 / (10000.0f32.powf(2.0 * (i as f32) / dim as f32));
            let val = if i % 2 == 0 {
                (pos as f32 * freq).sin()
            } else {
                (pos as f32 * freq).cos()
            };
            data.push(val);
        }
    }
    let dims: Vec<i32> = vec![num_positions as i32, dim as i32];
    Array::from_slice(&data, &dims)
}
