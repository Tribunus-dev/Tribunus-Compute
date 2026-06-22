use super::{
    CanonicalModel, CanonicalRole, ModelFamilyAdapter, NormalizationReport, SourceModel, TensorData,
};
use crate::config::{AttentionKind, RopeSpec, TextArchitecture};
use serde_json::Value;

/// Mistral model-family adapter.
#[derive(Debug, Clone, Copy)]
pub struct MistralAdapter;

impl ModelFamilyAdapter for MistralAdapter {
    fn family_name(&self) -> &'static str {
        "mistral"
    }

    fn claimed_config_types(&self) -> &'static [&'static str] {
        &["mistral"]
    }

    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool {
        let _ = tensor_names;
        config.get("model_type").and_then(|v| v.as_str()) == Some("mistral")
            && config
                .get("sliding_window")
                .and_then(|v| v.as_u64())
                .is_some()
    }

    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport> {
        let cfg = &source.config;
        let h = num(cfg, "hidden_size");
        let n_layers = num(cfg, "num_hidden_layers");
        let n_heads = num(cfg, "num_attention_heads");
        let n_kv_heads = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
        let head_dim = num_opt(cfg, "head_dim").unwrap_or(h / n_heads);
        let int_size = num(cfg, "intermediate_size");
        let vocab = num(cfg, "vocab_size");
        let max_pos = num(cfg, "max_position_embeddings");
        let slide = num(cfg, "sliding_window");
        let eps = f64_val(cfg, "rms_norm_eps").unwrap_or(1e-6);
        let tie = bool_val(cfg, "tie_word_embeddings").unwrap_or(true);

        let arch = TextArchitecture {
            diffusion_config: None,
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
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 0,
            model_type: "mistral".into(),
            layer_types: (0..n_layers)
                .map(|_| AttentionKind::SlidingAttention)
                .collect(),
            rope_local: RopeSpec {
                theta: f64_val(cfg, "rope_theta").unwrap_or(10000.0),
                rope_type: "default".into(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            moe_config: None,
        };

        let mut tensors = std::collections::HashMap::new();
        let mut missing = Vec::new();

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

        for i in 0..n_layers {
            // input layernorm
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

            // attention projections
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

            // post attention layernorm
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

            // mlp projections
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
        }

        if !missing.is_empty() {
            return Err(NormalizationReport {
                family: "mistral".into(),
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
