//! Conformance tests for every ModelFamilyAdapter implementation.
//!
//! Each adapter family gets the same three-test pattern:
//!
//! 1. **registry_select** – verifies `AdapterRegistry::select()` picks the
//!    correct adapter based on `model_type` and tensor names.
//! 2. **normalize_success** – supplies a full set of correctly-named tensors
//!    and expects `Ok(CanonicalModel)`.
//! 3. **missing_fails** – supplies an empty tensor map and expects
//!    `Err(NormalizationReport)` with at least one missing role.

use super::*;
use std::collections::HashMap;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a source model with the given model_type, config overrides, and tensor
/// entries.
fn make_source(
    model_type: &str,
    extra_config: &serde_json::Value,
    tensor_names: Vec<String>,
    tensors: HashMap<String, (String, Vec<u32>, Vec<u8>)>,
) -> SourceModel {
    let mut cfg = serde_json::json!({
        "model_type": model_type,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
    });
    if let serde_json::Value::Object(ref mut m) = cfg {
        if let serde_json::Value::Object(extra) = extra_config {
            for (k, v) in extra {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    SourceModel {
        config: cfg,
        config_path: PathBuf::from("/tmp/test-model"),
        model_type: model_type.to_string(),
        tensor_names,
        tensors,
    }
}

/// Fill a tensor map with the canonical Qwen2/Llama/Mistral/Phi3 layout at
/// 2 layers, hidden_size=64, intermediate_size=256.
fn fill_standard_tensors(
    map: &mut HashMap<String, (String, Vec<u32>, Vec<u8>)>,
    hs: u32,
    hs4: u32,
    num_layers: u32,
    vocab_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
) {
    let qk_dim = hs; // each head's Q/K projection dimension
    let _qk_dim = hs; // each head's Q/K projection dimension
    let _vo_dim = head_dim * num_kv_heads; // V/O see KV heads

    map.insert(
        "model.embed_tokens.weight".into(),
        (
            "F32".into(),
            vec![vocab_size, hs],
            vec![0u8; (vocab_size * hs * 4) as usize],
        ),
    );

    for i in 0..num_layers {
        let p = format!("model.layers.{i}");
        // Norms
        map.insert(
            format!("{p}.input_layernorm.weight"),
            ("F32".into(), vec![hs], vec![0u8; (hs * 4) as usize]),
        );
        map.insert(
            format!("{p}.post_attention_layernorm.weight"),
            ("F32".into(), vec![hs], vec![0u8; (hs * 4) as usize]),
        );
        // QKV projections (each is [hidden, hidden] in the A.00 config)
        // Q: [hs, hs], K: [hs, hs], V: [hs, hs], O: [hs, hs]
        for proj in &["q", "k", "v", "o"] {
            let name = format!("{p}.self_attn.{proj}_proj.weight");
            map.insert(
                name,
                (
                    "F32".into(),
                    vec![hs, hs],
                    vec![0u8; (hs * hs * 4) as usize],
                ),
            );
        }
        // MLP
        for proj in &["gate", "up", "down"] {
            let (out_d, in_d) = if *proj == "down" {
                (hs, hs4) // down: [hidden, intermediate]
            } else {
                (hs4, hs) // gate/up: [intermediate, hidden]
            };
            let name = format!("{p}.mlp.{proj}_proj.weight");
            map.insert(
                name,
                (
                    "F32".into(),
                    vec![out_d, in_d],
                    vec![0u8; (out_d * in_d * 4) as usize],
                ),
            );
        }
    }

    map.insert(
        "model.norm.weight".into(),
        ("F32".into(), vec![hs], vec![0u8; (hs * 4) as usize]),
    );

    map.insert(
        "lm_head.weight".into(),
        (
            "F32".into(),
            vec![vocab_size, hs],
            vec![0u8; (vocab_size * hs * 4) as usize],
        ),
    );
}

/// Add Q/K norm tensors (Gemma-specific) to an existing tensor map.
fn add_qk_norms(map: &mut HashMap<String, (String, Vec<u32>, Vec<u8>)>, hs: u32, num_layers: u32) {
    for i in 0..num_layers {
        let p = format!("model.layers.{i}");
        map.insert(
            format!("{p}.self_attn.q_norm.weight"),
            ("F32".into(), vec![hs], vec![0u8; (hs * 4) as usize]),
        );
        map.insert(
            format!("{p}.self_attn.k_norm.weight"),
            ("F32".into(), vec![hs], vec![0u8; (hs * 4) as usize]),
        );
    }
}

/// Returns the standard tensor name list extracted from a filled map.
fn standard_names(map: &HashMap<String, (String, Vec<u32>, Vec<u8>)>) -> Vec<String> {
    map.keys().cloned().collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Registry
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_registry_selects_qwen2() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "qwen2"});
    let tensor_names = vec!["model.embed_tokens.weight".to_string()];
    let adapter = registry.select(&config, &tensor_names).unwrap();
    assert_eq!(adapter.family_name(), "qwen2");
}

#[test]
fn test_registry_selects_llama() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "llama"});
    let tensor_names = vec!["model.embed_tokens.weight".to_string()];
    let adapter = registry.select(&config, &tensor_names).unwrap();
    assert_eq!(adapter.family_name(), "llama");
}

#[test]
fn test_registry_selects_mistral() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "mistral"});
    let tensor_names = vec!["model.embed_tokens.weight".to_string()];
    let adapter = registry.select(&config, &tensor_names).unwrap();
    assert_eq!(adapter.family_name(), "mistral");
}

#[test]
fn test_registry_selects_gemma() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "gemma"});
    let tensor_names = vec!["model.embed_tokens.weight".to_string()];
    let adapter = registry.select(&config, &tensor_names).unwrap();
    assert_eq!(adapter.family_name(), "gemma");
}

#[test]
fn test_registry_selects_phi() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "phi3"});
    let tensor_names = vec!["model.embed_tokens.weight".to_string()];
    let adapter = registry.select(&config, &tensor_names).unwrap();
    assert_eq!(adapter.family_name(), "phi");
}

#[test]
fn test_registry_rejects_unknown() {
    let registry = AdapterRegistry::new();
    let config = serde_json::json!({"model_type": "unknown-architecture"});
    let tensor_names = vec!["foo.weight".to_string()];
    let err = match registry.select(&config, &tensor_names) {
        Err(e) => e,
        Ok(_) => panic!("expected Err for unknown model_type"),
    };
    assert!(
        err.contains("unsupported model_type 'unknown-architecture'"),
        "error message should mention unsupported model_type, got: {err}",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Qwen2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_qwen2_normalize_success() {
    let mut tensors = HashMap::new();
    fill_standard_tensors(&mut tensors, 64, 256, 2, 32000, 2, 16);
    let names = standard_names(&tensors);
    let source = make_source("qwen2", &serde_json::json!({}), names, tensors);
    let adapter = qwen2::Qwen2Adapter;
    let result = adapter.normalize(&source);
    assert!(
        result.is_ok(),
        "qwen2 normalize succeeded: {:?}",
        result.err()
    );
}

#[test]
fn test_qwen2_missing_fails() {
    let tensors = HashMap::new();
    let source = make_source("qwen2", &serde_json::json!({}), vec![], tensors);
    let adapter = qwen2::Qwen2Adapter;
    let err = adapter.normalize(&source).unwrap_err();
    assert!(!err.missing_roles.is_empty(), "expected missing roles");
}

// ═══════════════════════════════════════════════════════════════════════════
// Llama
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_llama_normalize_success() {
    let mut tensors = HashMap::new();
    fill_standard_tensors(&mut tensors, 64, 256, 2, 32000, 2, 16);
    let names = standard_names(&tensors);
    let source = make_source("llama", &serde_json::json!({}), names, tensors);
    let adapter = llama::LlamaAdapter;
    let result = adapter.normalize(&source);
    assert!(
        result.is_ok(),
        "llama normalize succeeded: {:?}",
        result.err()
    );
}

#[test]
fn test_llama_missing_fails() {
    let tensors = HashMap::new();
    let source = make_source("llama", &serde_json::json!({}), vec![], tensors);
    let adapter = llama::LlamaAdapter;
    let err = adapter.normalize(&source).unwrap_err();
    assert!(!err.missing_roles.is_empty(), "expected missing roles");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mistral
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mistral_normalize_success() {
    let mut tensors = HashMap::new();
    fill_standard_tensors(&mut tensors, 64, 256, 2, 32000, 2, 16);
    let names = standard_names(&tensors);
    let source = make_source(
        "mistral",
        &serde_json::json!({"sliding_window": 4096}),
        names,
        tensors,
    );
    let adapter = mistral::MistralAdapter;
    let result = adapter.normalize(&source);
    assert!(
        result.is_ok(),
        "mistral normalize succeeded: {:?}",
        result.err()
    );
}

#[test]
fn test_mistral_missing_fails() {
    let tensors = HashMap::new();
    let source = make_source(
        "mistral",
        &serde_json::json!({"sliding_window": 4096}),
        vec![],
        tensors,
    );
    let adapter = mistral::MistralAdapter;
    let err = adapter.normalize(&source).unwrap_err();
    assert!(!err.missing_roles.is_empty(), "expected missing roles");
}

// ═══════════════════════════════════════════════════════════════════════════
// Gemma
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_gemma_normalize_success() {
    let mut tensors = HashMap::new();
    fill_standard_tensors(&mut tensors, 64, 256, 2, 32000, 2, 16);
    add_qk_norms(&mut tensors, 64, 2);
    let names = standard_names(&tensors);
    let source = make_source("gemma", &serde_json::json!({}), names, tensors);
    let adapter = gemma::GemmaAdapter;
    let result = adapter.normalize(&source);
    assert!(
        result.is_ok(),
        "gemma normalize succeeded: {:?}",
        result.err()
    );
}

#[test]
fn test_gemma_missing_fails() {
    let tensors = HashMap::new();
    let source = make_source("gemma", &serde_json::json!({}), vec![], tensors);
    let adapter = gemma::GemmaAdapter;
    let err = adapter.normalize(&source).unwrap_err();
    assert!(!err.missing_roles.is_empty(), "expected missing roles");
}

// ═══════════════════════════════════════════════════════════════════════════
// Phi
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_phi_normalize_success() {
    let mut tensors = HashMap::new();
    fill_standard_tensors(&mut tensors, 64, 256, 2, 32000, 2, 16);
    let names = standard_names(&tensors);
    let source = make_source("phi3", &serde_json::json!({}), names, tensors);
    let adapter = phi::PhiAdapter;
    let result = adapter.normalize(&source);
    assert!(
        result.is_ok(),
        "phi normalize succeeded: {:?}",
        result.err()
    );
}

#[test]
fn test_phi_missing_fails() {
    let tensors = HashMap::new();
    let source = make_source("phi3", &serde_json::json!({}), vec![], tensors);
    let adapter = phi::PhiAdapter;
    let err = adapter.normalize(&source).unwrap_err();
    assert!(!err.missing_roles.is_empty(), "expected missing roles");
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixture end-to-end
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_fixture_generation_end_to_end() {
    use crate::model_adapter::fixtures;
    use safetensors::{Dtype, SafeTensors};

    let dir = tempfile::TempDir::new().unwrap();
    fixtures::generate_qwen2_fixture(dir.path());

    // Verify files exist
    let shard_path = dir.path().join("model.safetensors");
    let config_path = dir.path().join("config.json");
    assert!(shard_path.exists(), "shard file exists");
    assert!(config_path.exists(), "config file exists");

    // Parse config and verify model_type
    let config_raw = std::fs::read_to_string(&config_path).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_raw).unwrap();
    assert_eq!(config["model_type"], "qwen2");
    assert_eq!(config["hidden_size"], 64);
    assert_eq!(config["num_hidden_layers"], 2);

    // Read back safetensors with the safetensors crate
    let shard_bytes = std::fs::read(&shard_path).unwrap();
    let loaded = SafeTensors::deserialize(&shard_bytes).unwrap();

    // Check expected tensor names
    let expected_names: Vec<&str> = vec![
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    for name in &expected_names {
        let view = loaded
            .tensor(name)
            .unwrap_or_else(|_| panic!("tensor {name} accessible"));
        assert_eq!(view.dtype(), Dtype::F32, "tensor {name} is F32");
        let elts: usize = view.shape().iter().product();
        assert_eq!(view.data().len(), elts * 4, "tensor {name} byte count");
    }

    // Also verify the loaded names cover every expected name
    let loaded_names: Vec<&str> = loaded.names().iter().map(|s| s.as_str()).collect();
    assert_eq!(
        loaded_names.len(),
        expected_names.len(),
        "tensor count matches"
    );
    for name in &expected_names {
        assert!(
            loaded_names.contains(name),
            "missing expected tensor: {name}",
        );
    }
}
