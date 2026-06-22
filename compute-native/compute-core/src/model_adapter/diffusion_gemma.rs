use super::{
    CanonicalModel, CanonicalRole, ModelFamilyAdapter, NormalizationReport, SourceModel, TensorData,
};
use crate::config::{
    AttentionKind, ConfidenceType, DiffusionConfig, MoEConfig, NoiseScheduleType, RopeSpec,
    TextArchitecture,
};
use serde_json::Value;

/// DiffusionGemma model-family adapter.
///
/// Handles DiffusionGemma models which use a discrete-diffusion decoder
/// with optional MoE layers, extracted from the model's config.json and
/// diffusion_config sub-object.
#[derive(Debug, Clone, Copy)]
pub struct DiffusionGemmaAdapter;

impl ModelFamilyAdapter for DiffusionGemmaAdapter {
    fn family_name(&self) -> &'static str {
        "diffusion_gemma"
    }

    fn claimed_config_types(&self) -> &'static [&'static str] {
        &["diffusion_gemma"]
    }

    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool {
        let _ = tensor_names;
        let model_type = config.get("model_type").and_then(|v| v.as_str());
        if model_type == Some("diffusion_gemma") {
            return true;
        }
        if config.get("diffusion_config").is_some() {
            return true;
        }
        if config.get("_diffusion").is_some() {
            return true;
        }
        false
    }

    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport> {
        let cfg = &source.config;
        let h = num(cfg, "hidden_size");
        let n_layers = num(cfg, "num_hidden_layers");
        let n_heads = num(cfg, "num_attention_heads");
        let n_kv_heads = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
        let head_dim = num_opt(cfg, "head_dim").unwrap_or(128);
        let int_size = num(cfg, "intermediate_size");
        let vocab = num(cfg, "vocab_size");
        let max_pos = num(cfg, "max_position_embeddings");
        let slide = num_opt(cfg, "sliding_window").unwrap_or(0);
        let eps = f64_val(cfg, "rms_norm_eps").unwrap_or(1e-6);
        let tie = bool_val(cfg, "tie_word_embeddings").unwrap_or(true);
        let softcap = f64_val(cfg, "final_logit_softcapping");
        let rope_theta = f64_val(cfg, "rope_theta").unwrap_or(10000.0);

        // Determine layer types (bidirectional full attention for diffusion).
        let layer_types: Vec<AttentionKind> = (0..n_layers)
            .map(|_| AttentionKind::FullAttention)
            .collect();

        // Extract MoE config from "moe" or "moe_config" key.
        let moe_config = if let Some(moe) = cfg.get("moe").or_else(|| cfg.get("moe_config")) {
            Some(MoEConfig {
                num_experts: moe
                    .get("num_experts")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as u32,
                top_k_experts: moe
                    .get("top_k_experts")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8) as u32,
                intermediate_size: moe
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(int_size as u64) as u32,
                shared_experts: moe
                    .get("shared_experts")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            })
        } else {
            None
        };

        // Extract diffusion config — look in diffusion_config sub-object or _diffusion key.
        let dcfg = cfg
            .get("diffusion_config")
            .or_else(|| cfg.get("_diffusion"));
        let diffusion_config = dcfg.map(|d| DiffusionConfig {
            max_diffusion_tokens: d
                .get("max_diffusion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as u32,
            default_denoising_steps: d
                .get("default_denoising_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(6) as u32,
            noise_schedule: match d.get("noise_schedule").and_then(|v| v.as_str()) {
                Some("cosine") => NoiseScheduleType::Cosine,
                Some("sqrt") => NoiseScheduleType::Sqrt,
                Some("linear") => NoiseScheduleType::Linear,
                _ => NoiseScheduleType::Cosine,
            },
            parallel_token_generation: d
                .get("parallel_token_generation")
                .and_then(|v| v.as_u64())
                .unwrap_or(18) as u32,
            supports_images: d
                .get("supports_images")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            supports_video: d
                .get("supports_video")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            image_size: d.get("image_size").and_then(|v| v.as_u64()).unwrap_or(896) as u32,
            patch_size: d.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as u32,
            max_context_length: d
                .get("max_context_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(262_144) as u32,
            mask_token_id: d.get("mask_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            pad_token_id: d.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            eos_token_id: d.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            max_canvas_tokens: d
                .get("max_canvas_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as u32,
            timestep_embedding_dim: d
                .get("timestep_embedding_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(h as u64) as u32,
            confidence_type: match d.get("confidence_type").and_then(|v| v.as_str()) {
                Some("softmax_margin") => ConfidenceType::SoftmaxMargin,
                Some("normalized_entropy") => ConfidenceType::NormalizedEntropy,
                _ => ConfidenceType::LogProb,
            },
            default_confidence_threshold: d
                .get("default_confidence_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7) as f32,
            eos_collapse_enabled: d
                .get("eos_collapse_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        });

        // Build TextArchitecture (same base pattern as GemmaAdapter).
        let arch = TextArchitecture {
            hidden_size: h,
            intermediate_size: int_size,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv_heads,
            head_dim,
            global_head_dim: None,
            num_global_key_value_heads: None,
            num_hidden_layers: n_layers,
            vocab_size: vocab,
            sliding_window: slide,
            max_position_embeddings: max_pos,
            rms_norm_eps: eps,
            tie_word_embeddings: tie,
            attention_k_eq_v: true,
            final_logit_softcapping: softcap,
            hidden_size_per_layer_input: 0,
            model_type: "diffusion_gemma".into(),
            layer_types,
            rope_local: RopeSpec {
                theta: rope_theta,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            moe_config,
            diffusion_config,
        };

        // ── Tensor normalisation ────────────────────────────────────────
        let mut tensors = std::collections::HashMap::new();
        let mut missing = Vec::new();

        // Embedding
        if let Some(t) = source.tensors.get("model.embed_tokens.weight") {
            tensors.insert(
                CanonicalRole::Embedding,
                TensorData {
                    dtype: t.0.clone(),
                    shape: t.1.clone(),
                    data: t.2.clone(),
                },
            );
        } else {
            missing.push(CanonicalRole::Embedding);
        }

        // Final norm
        if let Some(t) = source.tensors.get("model.norm.weight") {
            tensors.insert(
                CanonicalRole::FinalNorm,
                TensorData {
                    dtype: t.0.clone(),
                    shape: t.1.clone(),
                    data: t.2.clone(),
                },
            );
        } else {
            missing.push(CanonicalRole::FinalNorm);
        }

        // Optional lm_head (may be tied to embeddings)
        if let Some(t) = source.tensors.get("model.lm_head.weight") {
            tensors.insert(
                CanonicalRole::LmHead,
                TensorData {
                    dtype: t.0.clone(),
                    shape: t.1.clone(),
                    data: t.2.clone(),
                },
            );
        }

        // Per-layer tensors
        for i in 0..n_layers {
            // Input layernorm
            let name = format!("model.layers.{i}.input_layernorm.weight");
            match source.tensors.get(&name) {
                Some(t) => {
                    tensors.insert(
                        CanonicalRole::AttnNorm(i),
                        TensorData {
                            dtype: t.0.clone(),
                            shape: t.1.clone(),
                            data: t.2.clone(),
                        },
                    );
                }
                None => {
                    missing.push(CanonicalRole::AttnNorm(i));
                }
            }

            // Attention projections (q, k, v, o)
            for proj in &["q", "k", "v", "o"] {
                let role = match *proj {
                    "q" => CanonicalRole::Q(i),
                    "k" => CanonicalRole::K(i),
                    "v" => CanonicalRole::V(i),
                    "o" => CanonicalRole::O(i),
                    _ => unreachable!(),
                };
                let name = format!("model.layers.{i}.self_attn.{proj}_proj.weight");
                match source.tensors.get(&name) {
                    Some(t) => {
                        tensors.insert(
                            role,
                            TensorData {
                                dtype: t.0.clone(),
                                shape: t.1.clone(),
                                data: t.2.clone(),
                            },
                        );
                    }
                    None => {
                        missing.push(role);
                    }
                }
            }

            // Post-attention layernorm
            let name = format!("model.layers.{i}.post_attention_layernorm.weight");
            match source.tensors.get(&name) {
                Some(t) => {
                    tensors.insert(
                        CanonicalRole::MlpNorm(i),
                        TensorData {
                            dtype: t.0.clone(),
                            shape: t.1.clone(),
                            data: t.2.clone(),
                        },
                    );
                }
                None => {
                    missing.push(CanonicalRole::MlpNorm(i));
                }
            }

            // MLP projections (gate, up, down)
            for proj in &["gate", "up", "down"] {
                let role = match *proj {
                    "gate" => CanonicalRole::Gate(i),
                    "up" => CanonicalRole::Up(i),
                    "down" => CanonicalRole::Down(i),
                    _ => unreachable!(),
                };
                let name = format!("model.layers.{i}.mlp.{proj}_proj.weight");
                match source.tensors.get(&name) {
                    Some(t) => {
                        tensors.insert(
                            role,
                            TensorData {
                                dtype: t.0.clone(),
                                shape: t.1.clone(),
                                data: t.2.clone(),
                            },
                        );
                    }
                    None => {
                        missing.push(role);
                    }
                }
            }

            // Q/K norms (DiffusionGemma-specific; optional)
            let qn = format!("model.layers.{i}.self_attn.q_norm.weight");
            if let Some(t) = source.tensors.get(&qn) {
                tensors.insert(
                    CanonicalRole::QNorm(i),
                    TensorData {
                        dtype: t.0.clone(),
                        shape: t.1.clone(),
                        data: t.2.clone(),
                    },
                );
            }
            let kn = format!("model.layers.{i}.self_attn.k_norm.weight");
            if let Some(t) = source.tensors.get(&kn) {
                tensors.insert(
                    CanonicalRole::KNorm(i),
                    TensorData {
                        dtype: t.0.clone(),
                        shape: t.1.clone(),
                        data: t.2.clone(),
                    },
                );
            }

            // MoE router weight (optional, present only when moe_config is active)
            if moe_config.is_some() {
                let router_name = format!("model.layers.{i}.mlp.router.weight");
                if let Some(t) = source.tensors.get(&router_name) {
                    tensors.insert(
                        CanonicalRole::Gate(i),
                        TensorData {
                            dtype: t.0.clone(),
                            shape: t.1.clone(),
                            data: t.2.clone(),
                        },
                    );
                }
            }
        }

        if !missing.is_empty() {
            return Err(NormalizationReport {
                family: "diffusion_gemma".into(),
                errors: vec![],
                missing_roles: missing,
                shape_mismatches: vec![],
            });
        }

        Ok(CanonicalModel {
            architecture: arch,
            tensors,
        })
    }
}

// ── Helper functions (same pattern as gemma.rs) ──────────────────────────────

fn num(v: &Value, key: &str) -> u32 {
    v.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as u32
}

fn num_opt(v: &Value, key: &str) -> Option<u32> {
    v.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn f64_val(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|v| v.as_f64())
}

fn bool_val(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(|v| v.as_bool())
}
