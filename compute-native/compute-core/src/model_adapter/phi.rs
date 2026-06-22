use super::{
    CanonicalModel, CanonicalRole, ModelFamilyAdapter, NormalizationReport, SourceModel, TensorData,
};
use crate::config::{AttentionKind, RopeSpec, TextArchitecture};
use serde_json::Value;

/// Phi model-family adapter (handles phi, phi3, phimoe).
#[derive(Debug, Clone, Copy)]
pub struct PhiAdapter;

impl ModelFamilyAdapter for PhiAdapter {
    fn family_name(&self) -> &'static str {
        "phi"
    }

    fn claimed_config_types(&self) -> &'static [&'static str] {
        &["phi", "phi3", "phimoe"]
    }

    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool {
        let type_match = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|t| t.starts_with("phi"))
            .unwrap_or(false);

        if !type_match {
            return false;
        }

        let has_q_proj = tensor_names
            .iter()
            .any(|n| n.contains(".self_attn.q_proj.weight"));
        let has_transformer_h = tensor_names.iter().any(|n| n.contains("transformer.h."));

        has_q_proj || has_transformer_h
    }

    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport> {
        let cfg = &source.config;
        let mt = cfg
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("phi")
            .to_string();
        let h = num(cfg, "hidden_size");
        let n_layers = num(cfg, "num_hidden_layers");
        let n_heads = num(cfg, "num_attention_heads");
        let n_kv_heads = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
        let head_dim = num_opt(cfg, "head_dim").unwrap_or(h / n_heads);
        let int_size = num(cfg, "intermediate_size");
        let vocab = num(cfg, "vocab_size");
        let max_pos = num(cfg, "max_position_embeddings");
        let slide = num_opt(cfg, "sliding_window").unwrap_or(0);
        let eps = f64_val(cfg, "rms_norm_eps").unwrap_or(1e-6);
        let tie = bool_val(cfg, "tie_word_embeddings").unwrap_or(false);

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

        // Determine which namespace prefix the source uses.
        // Phi models may use "model" (phi-1) or "transformer" (phi-3) as root.
        let has_model_ns = source.tensors.contains_key("model.embed_tokens.weight")
            || source.tensors.contains_key("model.lm_head.weight")
            || source
                .tensors
                .keys()
                .any(|k| k.starts_with("model.layers."));
        let prefix = if has_model_ns { "model" } else { "transformer" };

        // Determine layer key pattern: "model.layers.{i}" vs "transformer.h.{i}"
        let layer_prefix = if prefix == "model" {
            "model.layers".to_string()
        } else {
            "transformer.h".to_string()
        };

        let mut tensors = std::collections::HashMap::new();
        let mut missing = Vec::new();

        // Embedding: try "{prefix}.embed_tokens.weight" or "{prefix}.wte.weight" (phi-1 naming).
        let embed_key = format!("{}.embed_tokens.weight", prefix);
        let wte_key = format!("{}.wte.weight", prefix);
        if let Some(t) = source
            .tensors
            .get(&embed_key)
            .or_else(|| source.tensors.get(&wte_key))
        {
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

        // Final norm: try "{prefix}.norm.weight" or "{prefix}.ln_f.weight" (phi-1 naming).
        let norm_key = format!("{}.norm.weight", prefix);
        let ln_f_key = format!("{}.ln_f.weight", prefix);
        if let Some(t) = source
            .tensors
            .get(&norm_key)
            .or_else(|| source.tensors.get(&ln_f_key))
        {
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

        // LmHead: try "{prefix}.lm_head.weight".
        let lm_head_key = format!("{}.lm_head.weight", prefix);
        if let Some(t) = source.tensors.get(&lm_head_key) {
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
            let name = format!("{layer_prefix}.{i}.input_layernorm.weight");
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
                let name = format!("{layer_prefix}.{i}.self_attn.{proj}_proj.weight");
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
            let name = format!("{layer_prefix}.{i}.post_attention_layernorm.weight");
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
                let name = format!("{layer_prefix}.{i}.mlp.{proj}_proj.weight");
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
                family: "phi".into(),
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
