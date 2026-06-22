use super::{
    CanonicalModel, CanonicalRole, ModelFamilyAdapter, NormalizationReport, SourceModel, TensorData,
};
use crate::config::{AttentionKind, RopeSpec, TextArchitecture};
use serde_json::Value;

/// Gemma model-family adapter (handles gemma, gemma2).
#[derive(Debug, Clone, Copy)]
pub struct GemmaAdapter;

impl ModelFamilyAdapter for GemmaAdapter {
    fn family_name(&self) -> &'static str {
        "gemma"
    }

    fn claimed_config_types(&self) -> &'static [&'static str] {
        &["gemma", "gemma2"]
    }

    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool {
        config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|t| t.starts_with("gemma"))
            .unwrap_or(false)
            && tensor_names
                .iter()
                .any(|n| n.contains(".self_attn.q_proj.weight"))
    }

    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport> {
        let cfg = &source.config;
        let mt = cfg
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("gemma")
            .to_string();
        let h = num(cfg, "hidden_size");
        let n_layers = num(cfg, "num_hidden_layers");
        let n_heads = num(cfg, "num_attention_heads");
        let n_kv_heads = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
        let head_dim = num_opt(cfg, "head_dim").unwrap_or(256);
        let int_size = num(cfg, "intermediate_size");
        let vocab = num(cfg, "vocab_size");
        let max_pos = num(cfg, "max_position_embeddings");
        let slide = num_opt(cfg, "sliding_window").unwrap_or(0);
        let eps = f64_val(cfg, "rms_norm_eps").unwrap_or(1e-6);
        let tie = bool_val(cfg, "tie_word_embeddings").unwrap_or(true);
        let softcap = f64_val(cfg, "final_logit_softcapping");

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
            final_logit_softcapping: softcap,
            hidden_size_per_layer_input: 0,
            model_type: mt,
            layer_types: (0..n_layers)
                .map(|_| AttentionKind::FullAttention)
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

        // Gemma has optional lm_head (may be tied to embeddings).
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

            // Q/K norms (Gemma2-specific; Gemma 1 doesn't have them)
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
                family: "gemma".into(),
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
