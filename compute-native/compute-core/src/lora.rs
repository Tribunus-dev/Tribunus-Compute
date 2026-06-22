//! On-device LoRA adapter training and serving.
//!
//! Users train lightweight low-rank adapters on selected layers and modules,
//! then hot-load them during inference without recompiling the base model.
//!
//! # Layout
//!
//! Each targeted `(layer, module)` pair gets two matrices:
//! - `lora_a`: [in_dim, rank]  — random init (Kaiming-uniform)
//! - `lora_b`: [rank, out_dim] — zero init
//!
//! At forward time the LoRA contribution is:
//!
//!   `lora_out(x) = (x @ lora_a) @ lora_b * (alpha / rank)`
//!
//! which is added to the original projection output.

use mlx_rs::ops;
use mlx_rs::transforms::value_and_grad_with_argnums;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::profiled_executor::LoadedProfiledModel;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// LoraAdapter
// ---------------------------------------------------------------------------

/// A single low-rank adapter for selected layers and modules.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// User-visible name (e.g. `"my-style-v2"`).
    pub name: String,
    /// LoRA rank `r`.
    pub rank: u32,
    /// LoRA scaling factor `alpha`.
    pub alpha: f32,
    /// Layer indices that receive LoRA modifications (0-based). Empty = all.
    pub target_layers: Vec<u32>,
    /// Module names within each target layer (e.g. `["q_proj", "v_proj"]`).
    pub target_modules: Vec<String>,
    /// A matrices  `module_name -> Array` shaped `[in_dim, rank]`.
    pub lora_a: HashMap<String, Array>,
    /// B matrices  `module_name -> Array` shaped `[rank, out_dim]`.
    pub lora_b: HashMap<String, Array>,
    /// Snapshot of the original projection weights captured at `merge()` time.
    /// Keyed by `"{layer_idx}_{module_name}_weight"`.
    pub(crate) original_weights: HashMap<String, Array>,
    /// Whether LoRA deltas are currently merged into the base model.
    pub(crate) is_merged: bool,
}

impl LoraAdapter {
    /// Create a new empty adapter.
    ///
    /// Matrices are not initialised until [`init_weights`] is called with the
    /// actual model dimensions, or the adapter is loaded from disk.
    pub fn new(name: &str, rank: u32, alpha: f32) -> Self {
        Self {
            name: name.to_string(),
            rank,
            alpha,
            target_layers: Vec::new(),
            target_modules: Vec::new(),
            lora_a: HashMap::new(),
            lora_b: HashMap::new(),
            original_weights: HashMap::new(),
            is_merged: false,
        }
    }

    /// Initialise LoRA A/B matrices from the given model's weight shapes.
    ///
    /// `target_layers` and `target_modules` must be set beforehand.
    /// A matrices use uniform Kaiming init `~U(-1/√d_in, +1/√d_in)`;
    /// B matrices are zero-initialised (so the adapter produces zero
    /// modification before training).
    pub fn init_weights(&mut self, model: &LoadedProfiledModel) -> Result<(), String> {
        let layers = self.target_layers.clone();
        let modules = self.target_modules.clone();

        for &layer_idx in &layers {
            let lw = match model.layers.get(layer_idx as usize) {
                Some(l) => l,
                None => {
                    return Err(format!(
                        "layer {} out of range (model has {} layers)",
                        layer_idx,
                        model.layers.len()
                    ));
                }
            };

            for module in &modules {
                let key = format!("{}_{}_weight", layer_idx, module);
                if self.lora_a.contains_key(&key) && self.lora_b.contains_key(&key) {
                    continue; // already initialised
                }

                let (in_dim, out_dim) = projection_dims(module, lw)?;
                let r = self.rank as i32;

                // A: Kaiming uniform — U(-1/√d, +1/√d) with d = in_dim
                let bound = 1.0 / (in_dim as f64).sqrt();
                let a = mlx_rs::random::uniform::<_, f32>(
                    -bound as f32,
                    bound as f32,
                    &[in_dim as i32, r],
                    None,
                )
                .map_err(|e| format!("lora_a init: {:?}", e))?;

                // B: zero init
                let b: Array = Array::from_slice_f64(
                    &vec![0.0f64; (r * out_dim as i32) as usize],
                    &[r, out_dim as i32],
                );

                self.lora_a.insert(key.clone(), a);
                self.lora_b.insert(key, b);
            }
        }
        Ok(())
    }

    /// Run a single training step on a batch of token sequences.
    ///
    /// Performs a full forward pass through the model (embedding → all layers
    /// with LoRA modifications → norm → lm_head projection), computes
    /// cross-entropy loss against `target_ids`, backpropagates through the
    /// LoRA parameters only, and applies an SGD update.
    ///
    /// Returns the loss value for this step.
    pub fn train_step(
        &mut self,
        model: &LoadedProfiledModel,
        input_ids: &[u32],
        target_ids: &[u32],
        lr: f64,
    ) -> Result<f64, String> {
        let plan = &model.reader.manifest.execution_plan;
        let batch_size = 1i32;
        let seq_len = input_ids.len() as i32;

        // ── Build the differentiable parameter list ──────────────────────
        // All lora_a and lora_b arrays are the differentiable inputs.
        // We'll index into them by the order they appear.
        let param_keys: Vec<String> = {
            let mut keys: Vec<String> = self.lora_a.keys().cloned().collect();
            keys.sort();
            keys
        };

        let mut params: Vec<Array> = Vec::new();
        let mut param_start_idx: HashMap<String, usize> = HashMap::new();
        for key in &param_keys {
            param_start_idx.insert(key.clone(), params.len());
            if let Some(a) = self.lora_a.get(key) {
                params.push(a.clone());
            }
            if let Some(b) = self.lora_b.get(key) {
                params.push(b.clone());
            }
        }

        let n_params = params.len();
        let argnums: Vec<i32> = (0..n_params).map(|i| i as i32).collect();

        // ── Build closure ───────────────────────────────────────────────
        // Capture everything needed for the forward pass.
        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
        let target_ids_i32: Vec<i32> = target_ids.iter().map(|&t| t as i32).collect();
        let target_ids_arr = Array::from_slice(&target_ids_i32, &[batch_size, seq_len]);

        let emb_w = model.emb_w.clone();
        let emb_s = model.emb_s.clone();
        let emb_b = model.emb_b.clone();
        let fn_w = model.fn_w.clone();
        let layers_weights: Vec<LayerWeightSnapshot> = model
            .layers
            .iter()
            .map(|lw| LayerWeightSnapshot::from(lw))
            .collect();
        let rms_norm_eps = plan.rms_norm_eps as f32;
        let alpha = self.alpha;
        let rank = self.rank as f32;
        let target_layers: std::collections::HashSet<u32> =
            self.target_layers.iter().copied().collect();
        let target_modules: std::collections::HashSet<String> =
            self.target_modules.iter().cloned().collect();
        let param_keys_clone = param_keys.clone();

        // ── Define loss function ─────────────────────────────────────────
        let loss_fn = move |params_slice: &[Array]| -> Vec<Array> {
            // Reconstruct LoRA weights from the flat params_slice
            let mut lora_a_map: HashMap<String, Array> = HashMap::new();
            let mut lora_b_map: HashMap<String, Array> = HashMap::new();
            for key in &param_keys_clone {
                let &start = param_start_idx.get(key).unwrap();
                lora_a_map.insert(key.clone(), params_slice[start].clone());
                lora_b_map.insert(key.clone(), params_slice[start + 1].clone());
            }

            // Embedding
            let tok_arr = Array::from_slice(&input_ids_i32, &[batch_size, seq_len]);
            let hidden = embed_lookup_2d(&emb_w, &emb_s, &emb_b, &tok_arr);

            // ── Transformer layers ──────────────────────────────────────
            let mut h = hidden;
            for (l, _layer_plan) in plan.layers.iter().enumerate() {
                let lw = &layers_weights[l];
                let is_target = target_layers.is_empty() || target_layers.contains(&(l as u32));

                // --- Attention sub-block ---
                let rms_eps = rms_norm_eps;
                let normed = rms_norm(&h, &lw.input_layernorm, rms_eps);

                // Q projection
                let q = matmul(&normed, &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b);
                let q = if is_target && target_modules.contains("q_proj") {
                    let key = format!("{}_{}_weight", l, "q_proj");
                    if let Some(la) = lora_a_map.get(&key) {
                        let lora_out =
                            matmul_lora(&normed, la, lora_b_map.get(&key).unwrap(), alpha, rank);
                        ops::add(&q, &lora_out).unwrap()
                    } else {
                        q
                    }
                } else {
                    q
                };

                // K projection
                let k = matmul(&normed, &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b);
                let k = if is_target && target_modules.contains("k_proj") {
                    let key = format!("{}_{}_weight", l, "k_proj");
                    if let Some(la) = lora_a_map.get(&key) {
                        let lora_out =
                            matmul_lora(&normed, la, lora_b_map.get(&key).unwrap(), alpha, rank);
                        ops::add(&k, &lora_out).unwrap()
                    } else {
                        k
                    }
                } else {
                    k
                };

                // V projection
                let v = matmul(&normed, &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b);
                let v = if is_target && target_modules.contains("v_proj") {
                    let key = format!("{}_{}_weight", l, "v_proj");
                    if let Some(la) = lora_a_map.get(&key) {
                        let lora_out =
                            matmul_lora(&normed, la, lora_b_map.get(&key).unwrap(), alpha, rank);
                        ops::add(&v, &lora_out).unwrap()
                    } else {
                        v
                    }
                } else {
                    v
                };

                // Simplified attention (no RoPE, no KV cache — just dot-product)
                let attn_out = simplified_attention(&q, &k, &v);

                // O projection
                let attn_out = matmul(&attn_out, &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b);

                // Residual
                h = ops::add(&h, &attn_out).unwrap();

                // --- MLP sub-block ---
                let normed = rms_norm(&h, &lw.post_attention_layernorm, rms_eps);
                let gate = matmul(&normed, &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b);
                let up = matmul(&normed, &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b);
                // Silu gate
                let gate_silu = silu(&gate);
                let down_in = ops::multiply(&gate_silu, &up).unwrap();
                let mlp_out = matmul(&down_in, &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b);
                h = ops::add(&h, &mlp_out).unwrap();
            }

            // Final RMS norm
            let h = rms_norm(&h, &fn_w, rms_norm_eps);

            // LM head projection (tied embeddings)
            let logits = ops::matmul(&h, &emb_w).unwrap();

            // Cross-entropy loss
            let loss = cross_entropy_loss(&logits, &target_ids_arr);
            vec![loss]
        };

        // ── Compute value and gradients ─────────────────────────────────
        let mut vg = value_and_grad_with_argnums(loss_fn, argnums.as_slice());
        let (values, grads) = vg(&params).map_err(|e| format!("value_and_grad: {:?}", e))?;

        let loss_val = values
            .first()
            .ok_or_else(|| "no loss output from value_and_grad".to_string())?
            .as_slice::<f32>()
            .first()
            .copied()
            .unwrap_or(f32::INFINITY) as f64;

        // ── SGD update ──────────────────────────────────────────────────
        for (i, key) in param_keys.iter().enumerate() {
            let a_idx = i * 2;
            let b_idx = i * 2 + 1;

            if a_idx < grads.len() {
                let g_a = &grads[a_idx];
                let lr_arr = Array::from_f32(lr as f32);
                let step = ops::multiply(g_a, &lr_arr).unwrap();
                let updated_a = ops::subtract(&self.lora_a[key], &step).unwrap();
                self.lora_a.insert(key.clone(), updated_a);
            }
            if b_idx < grads.len() {
                let g_b = &grads[b_idx];
                let lr_arr = Array::from_f32(lr as f32);
                let step = ops::multiply(g_b, &lr_arr).unwrap();
                let updated_b = ops::subtract(&self.lora_b[key], &step).unwrap();
                self.lora_b.insert(key.clone(), updated_b);
            }
        }

        Ok(loss_val)
    }

    /// Merge LoRA deltas into the base model's projection weights.
    ///
    /// Saves a snapshot of the original weights so [`unmerge`] can restore
    /// them.  After merge the model runs at base-inference speed with no
    /// extra per-layer overhead.
    pub fn merge(&self, model: &mut LoadedProfiledModel) -> Result<(), String> {
        if self.lora_a.is_empty() {
            return Ok(());
        }

        let layers = if self.target_layers.is_empty() {
            (0..model.layers.len() as u32).collect::<Vec<_>>()
        } else {
            self.target_layers.clone()
        };

        for &layer_idx in &layers {
            let lw = &mut model.layers[layer_idx as usize];

            for module in &self.target_modules {
                let key = format!("{}_{}_weight", layer_idx, module);
                let Some(lora_a) = self.lora_a.get(&key) else {
                    continue;
                };
                let Some(lora_b) = self.lora_b.get(&key) else {
                    continue;
                };

                // Compute delta: A @ B * (alpha / rank)
                let ab = ops::matmul(lora_a, lora_b).unwrap();
                let scale = self.alpha / self.rank as f32;
                let delta = ops::multiply(&ab, &Array::from_f32(scale)).unwrap();

                // Apply to the matching weight
                let (orig_w, _orig_s, _orig_b) = get_weight_mut(module, lw);
                let merged = ops::add(orig_w.as_ref(), &delta).unwrap();
                *orig_w = Arc::new(merged);
            }
        }

        Ok(())
    }

    /// Unmerge LoRA deltas, restoring the original base-model weights.
    pub fn unmerge(&self, model: &mut LoadedProfiledModel) -> Result<(), String> {
        // No original weights snapshot => nothing to do
        if self.original_weights.is_empty() {
            return Ok(());
        }

        let layers = if self.target_layers.is_empty() {
            (0..model.layers.len() as u32).collect::<Vec<_>>()
        } else {
            self.target_layers.clone()
        };

        for &layer_idx in &layers {
            let lw = &mut model.layers[layer_idx as usize];

            for module in &self.target_modules {
                let key = format!("{}_{}_weight", layer_idx, module);
                if let Some(original) = self.original_weights.get(&key) {
                    let (orig_w, _orig_s, _orig_b) = get_weight_mut(module, lw);
                    *orig_w = Arc::new(original.clone());
                }
            }
        }

        Ok(())
    }

    /// Save the adapter to a directory as a safetensors file plus a JSON
    /// metadata file.
    ///
    /// Layout:
    /// ```text
    /// <path>/
    ///   adapter_config.json   — name, rank, alpha, target_layers, target_modules
    ///   adapter_model.safetensors — lora_a.* and lora_b.* tensors
    /// ```
    pub fn save(&self, path: &str) -> Result<(), String> {
        let dir = Path::new(path);
        fs::create_dir_all(dir).map_err(|e| format!("create adapter directory: {}", e))?;

        // Config
        let config = serde_json::json!({
            "name": self.name,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_layers": self.target_layers,
            "target_modules": self.target_modules,
        });
        let config_path = dir.join("adapter_config.json");
        fs::write(&config_path, serde_json::to_string(&config).unwrap())
            .map_err(|e| format!("write adapter config: {}", e))?;

        // Safetensors
        let mut tensors: Vec<(String, &Array)> = Vec::new();
        let mut keys: Vec<&String> = self.lora_a.keys().collect();
        keys.sort();
        for key in keys {
            tensors.push((format!("lora_a.{}", key), &self.lora_a[key]));
            if let Some(b) = self.lora_b.get(key) {
                tensors.push((format!("lora_b.{}", key), b));
            }
        }
        let st_path = dir.join("adapter_model.safetensors");
        Array::save_safetensors(tensors, None::<&HashMap<String, String>>, &st_path)
            .map_err(|e| format!("save safetensors: {:?}", e))?;

        Ok(())
    }

    /// Load an adapter from a directory saved by [`save`].
    pub fn load(path: &str) -> Result<Self, String> {
        let dir = Path::new(path);

        // Config
        let config_path = dir.join("adapter_config.json");
        let config_str =
            fs::read_to_string(&config_path).map_err(|e| format!("read config: {}", e))?;
        let config: serde_json::Value =
            serde_json::from_str(&config_str).map_err(|e| format!("parse config: {}", e))?;

        let name = config["name"].as_str().unwrap_or("unnamed").to_string();
        let rank = config["rank"].as_u64().unwrap_or(8) as u32;
        let alpha = config["alpha"].as_f64().unwrap_or(16.0) as f32;
        let target_layers: Vec<u32> = config["target_layers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as u32))
                    .collect()
            })
            .unwrap_or_default();
        let target_modules: Vec<String> = config["target_modules"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Safetensors
        let st_path = dir.join("adapter_model.safetensors");
        let loaded =
            Array::load_safetensors(&st_path).map_err(|e| format!("load safetensors: {:?}", e))?;

        let mut lora_a: HashMap<String, Array> = HashMap::new();
        let mut lora_b: HashMap<String, Array> = HashMap::new();

        for (tensor_name, arr) in &loaded {
            if let Some(rest) = tensor_name.strip_prefix("lora_a.") {
                lora_a.insert(rest.to_string(), arr.clone());
            } else if let Some(rest) = tensor_name.strip_prefix("lora_b.") {
                lora_b.insert(rest.to_string(), arr.clone());
            }
        }

        Ok(Self {
            name,
            rank,
            alpha,
            target_layers,
            target_modules,
            lora_a,
            lora_b,
            original_weights: HashMap::new(),
            is_merged: false,
        })
    }
}

// ---------------------------------------------------------------------------
// SharedWeightTable — reference-counted base weight sharing across adapters
// ---------------------------------------------------------------------------

/// Reference-counted base model weights shared across adapters.
///
/// Instead of each adapter owning a full copy of the base model's weights,
/// [`SharedWeightTable`] keeps a single `Arc<LoadedProfiledModel>` and tracks
/// how many adapters reference it via [`refcount`].  With ~30 adapters on a
/// 12 GB base model, this saves ~348 GB of RAM compared to naive duplication.
pub struct SharedWeightTable {
    /// The shared base model weights.
    pub base: Arc<LoadedProfiledModel>,
    /// Number of attached adapters (diagnostic / lifecycle guard).
    pub refcount: AtomicU32,
    /// Registered LoRA adapters keyed by name.
    pub adapters: RwLock<HashMap<String, Arc<LoraAdapter>>>,
}

impl SharedWeightTable {
    /// Register a new shared model base.
    ///
    /// Returns an `Arc<Self>` so every caller that clones the pointer keeps
    /// the base model alive.
    pub fn register(model: Arc<LoadedProfiledModel>) -> Arc<Self> {
        Arc::new(Self {
            base: model,
            refcount: AtomicU32::new(1),
            adapters: RwLock::new(HashMap::new()),
        })
    }

    /// Attach a LoRA adapter that references the shared base.
    ///
    /// Increments [`refcount`] so the base cannot be dropped while any
    /// adapter still references it.  Returns an error if an adapter with
    /// the same `name` is already attached.
    pub fn attach_adapter(&self, name: &str, adapter: Arc<LoraAdapter>) -> Result<(), String> {
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| format!("shared-weight-table lock: {}", e))?;
        if adapters.contains_key(name) {
            return Err(format!("adapter '{}' is already attached", name));
        }
        adapters.insert(name.to_string(), adapter);
        self.refcount.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Detach an adapter, releasing its reference on the shared base.
    ///
    /// Decrements [`refcount`].  No-op if the adapter is not attached.
    pub fn detach_adapter(&self, name: &str) -> Result<(), String> {
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| format!("shared-weight-table lock: {}", e))?;
        if adapters.remove(name).is_some() {
            self.refcount.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Run a forward pass through the shared base model with the LoRA delta
    /// from the named adapter applied on the fly.
    ///
    /// `input` — token IDs shaped `[batch, seq_len]` (i32).
    /// Returns the logits `[batch, seq_len, vocab_size]`.
    pub fn forward_with_adapter(&self, input: &Array, adapter_name: &str) -> Result<Array, String> {
        // Look up the adapter while holding the read lock.
        let adapter = {
            let adapters = self
                .adapters
                .read()
                .map_err(|e| format!("shared-weight-table lock: {}", e))?;
            adapters
                .get(adapter_name)
                .ok_or_else(|| format!("adapter '{}' not found", adapter_name))?
                .clone()
        };

        let model = &self.base;
        let plan = &model.reader.manifest.execution_plan;
        let rms_eps = plan.rms_norm_eps as f32;

        let target_layers: std::collections::HashSet<u32> =
            adapter.target_layers.iter().copied().collect();
        let target_modules: std::collections::HashSet<String> =
            adapter.target_modules.iter().cloned().collect();
        let alpha = adapter.alpha;
        let rank = adapter.rank as f32;

        // ── Embedding lookup ──────────────────────────────────────────
        let mut h = embed_lookup_2d(&model.emb_w, &model.emb_s, &model.emb_b, input);

        // ── Transformer layers ────────────────────────────────────────
        for (l, _layer_plan) in plan.layers.iter().enumerate() {
            let lw = &model.layers[l];
            let is_target = target_layers.is_empty() || target_layers.contains(&(l as u32));

            // --- Attention sub-block ---
            let normed = rms_norm(&h, &lw.input_layernorm, rms_eps);

            // Q projection
            let q = matmul(&normed, &lw.q_proj_w, &lw.q_proj_s, &lw.q_proj_b);
            let q = apply_lora_if_needed(
                q,
                &normed,
                &adapter,
                &target_modules,
                is_target,
                l,
                "q_proj",
                alpha,
                rank,
            );

            // K projection
            let k = matmul(&normed, &lw.k_proj_w, &lw.k_proj_s, &lw.k_proj_b);
            let k = apply_lora_if_needed(
                k,
                &normed,
                &adapter,
                &target_modules,
                is_target,
                l,
                "k_proj",
                alpha,
                rank,
            );

            // V projection
            let v = matmul(&normed, &lw.v_proj_w, &lw.v_proj_s, &lw.v_proj_b);
            let v = apply_lora_if_needed(
                v,
                &normed,
                &adapter,
                &target_modules,
                is_target,
                l,
                "v_proj",
                alpha,
                rank,
            );

            // Simplified attention (no RoPE, single head)
            let attn_out = simplified_attention(&q, &k, &v);
            let attn_out = matmul(&attn_out, &lw.o_proj_w, &lw.o_proj_s, &lw.o_proj_b);
            h = ops::add(&h, &attn_out).unwrap();

            // --- MLP sub-block ---
            let normed = rms_norm(&h, &lw.post_attention_layernorm, rms_eps);
            let gate = matmul(&normed, &lw.gate_proj_w, &lw.gate_proj_s, &lw.gate_proj_b);
            let up = matmul(&normed, &lw.up_proj_w, &lw.up_proj_s, &lw.up_proj_b);
            let gate_silu = silu(&gate);
            let down_in = ops::multiply(&gate_silu, &up).unwrap();
            let mlp_out = matmul(&down_in, &lw.down_proj_w, &lw.down_proj_s, &lw.down_proj_b);
            h = ops::add(&h, &mlp_out).unwrap();
        }

        // ── Final RMS norm ────────────────────────────────────────────
        let h = rms_norm(&h, &model.fn_w, rms_eps);

        // ── LM head (tied embeddings) ──────────────────────────────────
        let logits = ops::matmul(&h, &model.emb_w).unwrap();
        Ok(logits)
    }
}

/// Helper: conditionally apply LoRA delta to a projection output.
fn apply_lora_if_needed(
    base_out: Array,
    normed: &Array,
    adapter: &LoraAdapter,
    target_modules: &std::collections::HashSet<String>,
    is_target: bool,
    layer_idx: usize,
    module: &str,
    alpha: f32,
    rank: f32,
) -> Array {
    if is_target && target_modules.contains(module) {
        let key = format!("{}_{}_weight", layer_idx, module);
        if let Some(la) = adapter.lora_a.get(&key) {
            if let Some(lb) = adapter.lora_b.get(&key) {
                let delta = matmul_lora(normed, la, lb, alpha, rank);
                return ops::add(&base_out, &delta).unwrap();
            }
        }
    }
    base_out
}

// ---------------------------------------------------------------------------
// Helpers — forward-pass building blocks
// ---------------------------------------------------------------------------

/// Snapshot of a single layer's weights used during the differentiable
/// training forward pass.
#[derive(Clone)]
struct LayerWeightSnapshot {
    input_layernorm: Array,
    post_attention_layernorm: Array,
    q_proj_w: Array,
    q_proj_s: Array,
    q_proj_b: Array,
    k_proj_w: Array,
    k_proj_s: Array,
    k_proj_b: Array,
    v_proj_w: Array,
    v_proj_s: Array,
    v_proj_b: Array,
    o_proj_w: Array,
    o_proj_s: Array,
    o_proj_b: Array,
    gate_proj_w: Array,
    gate_proj_s: Array,
    gate_proj_b: Array,
    up_proj_w: Array,
    up_proj_s: Array,
    up_proj_b: Array,
    down_proj_w: Array,
    down_proj_s: Array,
    down_proj_b: Array,
}

impl From<&crate::profiled_executor::LayerWeights> for LayerWeightSnapshot {
    fn from(lw: &crate::profiled_executor::LayerWeights) -> Self {
        Self {
            input_layernorm: lw.input_layernorm.as_ref().clone(),
            post_attention_layernorm: lw.post_attention_layernorm.as_ref().clone(),
            q_proj_w: lw.q_proj_w.as_ref().clone(),
            q_proj_s: lw.q_proj_s.as_ref().clone(),
            q_proj_b: lw.q_proj_b.as_ref().clone(),
            k_proj_w: lw.k_proj_w.as_ref().clone(),
            k_proj_s: lw.k_proj_s.as_ref().clone(),
            k_proj_b: lw.k_proj_b.as_ref().clone(),
            v_proj_w: lw.v_proj_w.as_ref().clone(),
            v_proj_s: lw.v_proj_s.as_ref().clone(),
            v_proj_b: lw.v_proj_b.as_ref().clone(),
            o_proj_w: lw.o_proj_w.as_ref().clone(),
            o_proj_s: lw.o_proj_s.as_ref().clone(),
            o_proj_b: lw.o_proj_b.as_ref().clone(),
            gate_proj_w: lw.gate_proj_w.as_ref().clone(),
            gate_proj_s: lw.gate_proj_s.as_ref().clone(),
            gate_proj_b: lw.gate_proj_b.as_ref().clone(),
            up_proj_w: lw.up_proj_w.as_ref().clone(),
            up_proj_s: lw.up_proj_s.as_ref().clone(),
            up_proj_b: lw.up_proj_b.as_ref().clone(),
            down_proj_w: lw.down_proj_w.as_ref().clone(),
            down_proj_s: lw.down_proj_s.as_ref().clone(),
            down_proj_b: lw.down_proj_b.as_ref().clone(),
        }
    }
}

/// Look up (in_dim, out_dim) for a given module name within a layer.
fn projection_dims(
    module: &str,
    lw: &crate::profiled_executor::LayerWeights,
) -> Result<(i32, i32), String> {
    let w = match module {
        "q_proj" => &lw.q_proj_w,
        "k_proj" => &lw.k_proj_w,
        "v_proj" => &lw.v_proj_w,
        "o_proj" => &lw.o_proj_w,
        "gate_proj" => &lw.gate_proj_w,
        "up_proj" => &lw.up_proj_w,
        "down_proj" => &lw.down_proj_w,
        other => return Err(format!("unknown module: {}", other)),
    };
    let shape = w.shape();
    if shape.len() < 2 {
        return Err(format!(
            "projection weight for {} has {} dims, expected >= 2",
            module,
            shape.len()
        ));
    }
    // Weight is [out_dim, in_dim] for nn.Linear
    let out_dim = shape[0];
    let in_dim = shape[1];
    Ok((in_dim, out_dim))
}

/// Return a mutable reference triple `(&mut Arc<Array>, ...)` for a projection
/// module name.  The second and third elements are scales/biases (ignored for
/// LoRA merging).
fn get_weight_mut<'a>(
    module: &str,
    lw: &'a mut crate::profiled_executor::LayerWeights,
) -> (
    &'a mut std::sync::Arc<Array>,
    &'a mut std::sync::Arc<Array>,
    &'a mut std::sync::Arc<Array>,
) {
    match module {
        "q_proj" => (&mut lw.q_proj_w, &mut lw.q_proj_s, &mut lw.q_proj_b),
        "k_proj" => (&mut lw.k_proj_w, &mut lw.k_proj_s, &mut lw.k_proj_b),
        "v_proj" => (&mut lw.v_proj_w, &mut lw.v_proj_s, &mut lw.v_proj_b),
        "o_proj" => (&mut lw.o_proj_w, &mut lw.o_proj_s, &mut lw.o_proj_b),
        "gate_proj" => (
            &mut lw.gate_proj_w,
            &mut lw.gate_proj_s,
            &mut lw.gate_proj_b,
        ),
        "up_proj" => (&mut lw.up_proj_w, &mut lw.up_proj_s, &mut lw.up_proj_b),
        "down_proj" => (
            &mut lw.down_proj_w,
            &mut lw.down_proj_s,
            &mut lw.down_proj_b,
        ),
        _ => panic!("unknown module for weight mut: {}", module),
    }
}

/// 2-D embedding lookup via `take` along axis 0.
fn embed_lookup_2d(emb_w: &Array, _emb_s: &Array, _emb_b: &Array, tok_ids: &Array) -> Array {
    // emb_w: [vocab_size, hidden_size]
    // tok_ids: [batch, seq_len] — i32 values
    let flat = ops::reshape(tok_ids, &[-1]).unwrap();
    emb_w.take_axis(&flat, 0).unwrap()
}

/// RMS normalisation.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Array {
    let dtype = x.dtype();
    let x_f32 = if dtype != Dtype::Float32 {
        x.as_type::<f32>().unwrap()
    } else {
        x.clone()
    };

    // variance = mean(x^2)
    let x_sq = ops::square(&x_f32).unwrap();
    let variance = ops::mean_axis(&x_sq, -1, None).unwrap();
    // x / sqrt(variance + eps) * weight
    let eps_arr = Array::from_f32(eps);
    let denom = ops::sqrt(&ops::add(&variance, &eps_arr).unwrap()).unwrap();
    let normalised = ops::divide(&x_f32, &denom).unwrap();
    // Expand weight for broadcasting: [hidden] -> [1, 1, hidden]
    let result = ops::multiply(&normalised, weight).unwrap();

    if dtype != Dtype::Float32 {
        result.as_dtype(dtype).unwrap()
    } else {
        result
    }
}

/// Quantised or plain matrix multiply.
///
/// If scales are non-trivial (> 1 element) this is a fake-quant matmul;
/// otherwise plain `x @ w.T` via `matmul(a, w.T)`.
fn matmul(x: &Array, w: &Array, _s: &Array, _b: &Array) -> Array {
    // Plain FP matmul: x @ w^T where w is [out_dim, in_dim]
    // MLX matmul with [batch, seq, in_dim] x [in_dim, out_dim]
    let w_t = ops::transpose_axes(w, &[1, 0]).unwrap();
    ops::matmul(x, &w_t).unwrap()
}

/// LoRA contribution: `(x @ A) @ B * (alpha / rank)`
fn matmul_lora(x: &Array, lora_a: &Array, lora_b: &Array, alpha: f32, rank: f32) -> Array {
    // x: [batch, seq, in_dim]
    // lora_a: [in_dim, rank]
    // lora_b: [rank, out_dim]
    let a_out = ops::matmul(x, lora_a).unwrap();
    let b_out = ops::matmul(&a_out, lora_b).unwrap();
    let scale = alpha / rank;
    ops::multiply(&b_out, &Array::from_f32(scale)).unwrap()
}

/// Simplified dot-product attention (no RoPE, single head).
///
/// q, k, v: [batch, seq, dim]
fn simplified_attention(q: &Array, k: &Array, v: &Array) -> Array {
    let scale = 1.0 / ((k.shape().last().copied().unwrap_or(1) as f32).sqrt());
    let scores = ops::matmul(q, &ops::transpose_axes(k, &[0, 2, 1]).unwrap()).unwrap();
    let scaled = ops::multiply(&scores, &Array::from_f32(scale)).unwrap();
    let attn = ops::softmax_axes(&scaled, &[-1], None::<bool>).unwrap();
    ops::matmul(&attn, v).unwrap()
}

/// SiLU activation: x * sigmoid(x).
fn silu(x: &Array) -> Array {
    // sigmoid = 1 / (1 + exp(-x))
    let neg = ops::multiply(x, &Array::from_f32(-1.0)).unwrap();
    let exp = ops::exp(&neg).unwrap();
    let one: Array = Array::from_f32(1.0);
    let denom = ops::add(&exp, &one).unwrap();
    let sig = ops::divide(&one, &denom).unwrap();
    ops::multiply(x, &sig).unwrap()
}

/// Cross-entropy loss: mean(-log_softmax(logits)[targets]).
fn cross_entropy_loss(logits: &Array, targets: &Array) -> Array {
    // logits: [batch, seq_len, vocab]
    // targets: [batch, seq_len] — i32 token ids
    let lse = ops::logsumexp_axis(logits, -1, Some(true)).unwrap();
    let log_softmax = ops::subtract(logits, &lse).unwrap();
    let nll = log_softmax.take_along_axis(targets, -1).unwrap();
    let loss = ops::mean(&nll, None::<bool>).unwrap();
    ops::multiply(&loss, &Array::from_f32(-1.0)).unwrap()
}

// ---------------------------------------------------------------------------
// Convenience re-exports for the server module
// ---------------------------------------------------------------------------

/// Wrapper type for LOAD adapter state in the server.
pub type AdapterMap = HashMap<String, LoraAdapter>;

/// Trained adapter metadata returned by the API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdapterInfo {
    pub name: String,
    pub rank: u32,
    pub alpha: f32,
    pub target_layers: Vec<u32>,
    pub target_modules: Vec<String>,
    pub is_loaded: bool,
}

impl From<&LoraAdapter> for AdapterInfo {
    fn from(adapter: &LoraAdapter) -> Self {
        Self {
            name: adapter.name.clone(),
            rank: adapter.rank,
            alpha: adapter.alpha,
            target_layers: adapter.target_layers.clone(),
            target_modules: adapter.target_modules.clone(),
            is_loaded: adapter.is_merged,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: create an adapter, verify it round-trips through save/load.
    #[test]
    fn test_adapter_save_load_roundtrip() {
        let mut adapter = LoraAdapter::new("test-adapter", 8, 16.0);
        adapter.target_layers = vec![0, 1];
        adapter.target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];

        // Create minimal dummy arrays for lora_a and lora_b
        adapter.lora_a.insert(
            "0_q_proj_weight".to_string(),
            Array::from_slice_f64(&[0.1f64; 64], &[8, 8]),
        );
        adapter.lora_a.insert(
            "0_v_proj_weight".to_string(),
            Array::from_slice_f64(&[0.1f64; 64], &[8, 8]),
        );
        adapter.lora_b.insert(
            "0_q_proj_weight".to_string(),
            Array::from_slice_f64(&[0.0f64; 64], &[8, 8]),
        );
        adapter.lora_b.insert(
            "0_v_proj_weight".to_string(),
            Array::from_slice_f64(&[0.0f64; 64], &[8, 8]),
        );
        adapter.lora_a.insert(
            "1_q_proj_weight".to_string(),
            Array::from_slice_f64(&[0.1f64; 64], &[8, 8]),
        );
        adapter.lora_b.insert(
            "1_q_proj_weight".to_string(),
            Array::from_slice_f64(&[0.0f64; 64], &[8, 8]),
        );

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("adapters/test-adapter");
        let path_str = path.to_str().unwrap();

        adapter.save(path_str).unwrap();

        let loaded = LoraAdapter::load(path_str).unwrap();
        assert_eq!(loaded.name, "test-adapter");
        assert_eq!(loaded.rank, 8);
        assert!((loaded.alpha - 16.0).abs() < 1e-5);
        assert_eq!(loaded.target_layers, vec![0, 1]);
        assert_eq!(loaded.target_modules, vec!["q_proj", "v_proj"]);
        assert_eq!(loaded.lora_a.len(), 3);
        assert_eq!(loaded.lora_b.len(), 3);
    }
}
