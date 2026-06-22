//! Synthetic fixture generator for model-family adapter tests.
//!
//! Each `generate_*_fixture` writes a minimal but valid safetensors model
//! directory into the given path, including a `config.json` with realistic
//! architecture parameters and tensor shards with deterministic data.

use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Compute the number of F32 bytes for a given shape.
fn f32_byte_count(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>() * 4
}

/// Generate deterministic F32 data bytes for a tensor.
///
/// Uses a simple LCG seeded from the tensor index so data is reproducible
/// but distinct per tensor.
fn deterministic_f32_data(byte_count: usize, tensor_idx: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(byte_count);
    let mut state: u64 = tensor_idx as u64 ^ 0xDEAD_BEEF;
    for _ in 0..(byte_count / 4) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = (state >> 32) as u32;
        data.extend_from_slice(&val.to_le_bytes());
    }
    // Fill any remaining bytes (should not happen for properly aligned sizes)
    while data.len() < byte_count {
        data.push(0);
    }
    data
}

/// Build the JSON header string for a list of named tensors.
///
/// Each tuple is `(name, shape)`. Data offsets are computed sequentially
/// from `base_offset` (which is the start of the data section after the
/// header).
fn build_safetensors_header(tensors: &[(&str, &[u32])], base_offset: usize) -> String {
    let mut map = serde_json::Map::new();
    let mut offset = base_offset;

    for &(name, shape) in tensors {
        let byte_count = f32_byte_count(shape);
        let entry = json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        });
        map.insert(name.to_string(), entry);
        offset += byte_count;
    }

    serde_json::to_string(&serde_json::Value::Object(map)).expect("valid JSON header")
}

/// Write a valid `.safetensors` shard file with deterministic F32 data.
///
/// # Arguments
///
/// * `dir` - Directory to write into (must exist).
/// * `filename` - Shard filename (e.g. `"model-00001-of-00001.safetensors"`).
/// * `tensors` - Slice of `(name, shape)` tuples describing each tensor.
///
/// Returns the full path of the written file.
pub fn write_tiny_shard(dir: &Path, filename: &str, tensors: &[(&str, &[u32])]) -> PathBuf {
    let path = dir.join(filename);

    // Pre-compute data bytes with deterministic values.
    let mut all_data: Vec<u8> = Vec::new();
    let mut named: Vec<(&str, &[u32])> = Vec::new();
    for (idx, &(name, shape)) in tensors.iter().enumerate() {
        let byte_count = f32_byte_count(shape);
        let data = deterministic_f32_data(byte_count, idx);
        all_data.extend_from_slice(&data);
        named.push((name, shape));
    }

    let base_offset: usize = 8; // header length field is 8 bytes
    let header = build_safetensors_header(&named, base_offset);
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut buf = Vec::with_capacity(base_offset + header_bytes.len() + all_data.len());
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header_bytes);
    buf.extend_from_slice(&all_data);

    std::fs::write(&path, &buf).expect("write safetensors shard");
    path
}

/// Write a minimal `config.json` with the given key-value pairs.
fn write_config(dir: &Path, config: &serde_json::Value) {
    let path = dir.join("config.json");
    let contents = serde_json::to_string_pretty(config).expect("serialize config");
    std::fs::write(&path, &contents).expect("write config.json");
}

/// Write model index file (`model.safetensors.index.json`).
fn write_model_index(dir: &Path, shard_path: &str, tensor_names: &[&str]) {
    let mut index_map = serde_json::Map::new();
    index_map.insert("metadata".into(), serde_json::json!({"total_size": 0}));
    let weight_map: serde_json::Map<String, serde_json::Value> = tensor_names
        .iter()
        .map(|n| {
            (
                n.to_string(),
                serde_json::Value::String(shard_path.to_string()),
            )
        })
        .collect();
    index_map.insert("weight_map".into(), serde_json::Value::Object(weight_map));
    let index = serde_json::Value::Object(index_map);
    let path = dir.join("model.safetensors.index.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&index).expect("serialize index"),
    )
    .expect("write model index");
}

// ── Common tensor lists ────────────────────────────────────────────────────

/// Standard Qwen2/Llama/Mistral/Phi3 per-layer tensors (no Q/K norm).
fn standard_layer_tensors(prefix: &str, hs: u32, hs4: u32) -> Vec<(String, Vec<u32>)> {
    vec![
        (format!("{prefix}.input_layernorm.weight"), vec![hs]),
        (format!("{prefix}.self_attn.q_proj.weight"), vec![hs, hs]),
        (format!("{prefix}.self_attn.k_proj.weight"), vec![hs, hs]),
        (format!("{prefix}.self_attn.v_proj.weight"), vec![hs, hs]),
        (format!("{prefix}.self_attn.o_proj.weight"), vec![hs, hs]),
        (
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hs],
        ),
        (format!("{prefix}.mlp.gate_proj.weight"), vec![hs4, hs]),
        (format!("{prefix}.mlp.up_proj.weight"), vec![hs4, hs]),
        (format!("{prefix}.mlp.down_proj.weight"), vec![hs, hs4]),
    ]
}

/// Gemma per-layer tensors (includes Q/K norm).
fn gemma_layer_tensors(prefix: &str, hs: u32, hs4: u32) -> Vec<(String, Vec<u32>)> {
    let mut t = standard_layer_tensors(prefix, hs, hs4);
    t.push((format!("{prefix}.self_attn.q_norm.weight"), vec![hs]));
    t.push((format!("{prefix}.self_attn.k_norm.weight"), vec![hs]));
    t
}

// ── Fixture generators ─────────────────────────────────────────────────────

/// Generate a synthetic Qwen2 fixture at `dir`.
///
/// Architecture:
///   - 2 layers, hidden_size=64, intermediate_size=256
///   - 4 attention heads, 2 KV heads, head_dim=16
///   - vocab_size=32000, max_position_embeddings=32768
///   - No sliding window, no Q/K norms
///   - Shard: `model.safetensors`
pub fn generate_qwen2_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create fixture dir");

    let hs: u32 = 64;
    let hs4: u32 = 256;

    let mut tensor_list: Vec<(String, Vec<u32>)> = Vec::new();
    // Embedding
    tensor_list.push(("model.embed_tokens.weight".into(), vec![32000, hs]));
    // Layers
    for i in 0..2u32 {
        tensor_list.extend(standard_layer_tensors(
            &format!("model.layers.{i}"),
            hs,
            hs4,
        ));
    }
    // Final norm + LM head
    tensor_list.push(("model.norm.weight".into(), vec![hs]));
    tensor_list.push(("lm_head.weight".into(), vec![32000, hs]));

    // Write config
    write_config(
        dir,
        &json!({
            "model_type": "qwen2",
            "hidden_size": hs,
            "intermediate_size": hs4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
        }),
    );

    // Write shard
    let refs: Vec<(&str, &[u32])> = tensor_list
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    write_tiny_shard(dir, "model.safetensors", &refs);

    // Write index
    let names: Vec<&str> = tensor_list.iter().map(|(n, _)| n.as_str()).collect();
    write_model_index(dir, "model.safetensors", &names);
}

/// Generate a synthetic Llama fixture at `dir`.
///
/// Same architecture as Qwen2 but with `model_type = "llama"`.
pub fn generate_llama_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create fixture dir");

    let hs: u32 = 64;
    let hs4: u32 = 256;

    let mut tensor_list: Vec<(String, Vec<u32>)> = Vec::new();
    tensor_list.push(("model.embed_tokens.weight".into(), vec![32000, hs]));
    for i in 0..2u32 {
        tensor_list.extend(standard_layer_tensors(
            &format!("model.layers.{i}"),
            hs,
            hs4,
        ));
    }
    tensor_list.push(("model.norm.weight".into(), vec![hs]));
    tensor_list.push(("lm_head.weight".into(), vec![32000, hs]));

    write_config(
        dir,
        &json!({
            "model_type": "llama",
            "hidden_size": hs,
            "intermediate_size": hs4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
        }),
    );

    let refs: Vec<(&str, &[u32])> = tensor_list
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    write_tiny_shard(dir, "model.safetensors", &refs);

    let names: Vec<&str> = tensor_list.iter().map(|(n, _)| n.as_str()).collect();
    write_model_index(dir, "model.safetensors", &names);
}

/// Generate a synthetic Mistral fixture at `dir`.
///
/// Same as Qwen2 but includes `sliding_window` in config.
pub fn generate_mistral_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create fixture dir");

    let hs: u32 = 64;
    let hs4: u32 = 256;

    let mut tensor_list: Vec<(String, Vec<u32>)> = Vec::new();
    tensor_list.push(("model.embed_tokens.weight".into(), vec![32000, hs]));
    for i in 0..2u32 {
        tensor_list.extend(standard_layer_tensors(
            &format!("model.layers.{i}"),
            hs,
            hs4,
        ));
    }
    tensor_list.push(("model.norm.weight".into(), vec![hs]));
    tensor_list.push(("lm_head.weight".into(), vec![32000, hs]));

    write_config(
        dir,
        &json!({
            "model_type": "mistral",
            "hidden_size": hs,
            "intermediate_size": hs4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "sliding_window": 4096,
        }),
    );

    let refs: Vec<(&str, &[u32])> = tensor_list
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    write_tiny_shard(dir, "model.safetensors", &refs);

    let names: Vec<&str> = tensor_list.iter().map(|(n, _)| n.as_str()).collect();
    write_model_index(dir, "model.safetensors", &names);
}

/// Generate a synthetic Gemma fixture at `dir`.
///
/// Gemma adds Q/K norm tensors per layer.
pub fn generate_gemma_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create fixture dir");

    let hs: u32 = 64;
    let hs4: u32 = 256;

    let mut tensor_list: Vec<(String, Vec<u32>)> = Vec::new();
    // Gemma uses `embed` not `embed_tokens` and a slightly different prefix:
    tensor_list.push(("model.embed_tokens.weight".into(), vec![32000, hs]));
    for i in 0..2u32 {
        tensor_list.extend(gemma_layer_tensors(&format!("model.layers.{i}"), hs, hs4));
    }
    tensor_list.push(("model.norm.weight".into(), vec![hs]));

    write_config(
        dir,
        &json!({
            "model_type": "gemma",
            "hidden_size": hs,
            "intermediate_size": hs4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
        }),
    );

    let refs: Vec<(&str, &[u32])> = tensor_list
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    write_tiny_shard(dir, "model.safetensors", &refs);

    let names: Vec<&str> = tensor_list.iter().map(|(n, _)| n.as_str()).collect();
    write_model_index(dir, "model.safetensors", &names);
}

/// Generate a synthetic Phi-3 fixture at `dir`.
///
/// Same tensor layout as Qwen2 but `model_type = "phi3"`.
pub fn generate_phi_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create fixture dir");

    let hs: u32 = 64;
    let hs4: u32 = 256;

    let mut tensor_list: Vec<(String, Vec<u32>)> = Vec::new();
    tensor_list.push(("model.embed_tokens.weight".into(), vec![32000, hs]));
    for i in 0..2u32 {
        tensor_list.extend(standard_layer_tensors(
            &format!("model.layers.{i}"),
            hs,
            hs4,
        ));
    }
    tensor_list.push(("model.norm.weight".into(), vec![hs]));
    tensor_list.push(("lm_head.weight".into(), vec![32000, hs]));

    write_config(
        dir,
        &json!({
            "model_type": "phi3",
            "hidden_size": hs,
            "intermediate_size": hs4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
        }),
    );

    let refs: Vec<(&str, &[u32])> = tensor_list
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    write_tiny_shard(dir, "model.safetensors", &refs);

    let names: Vec<&str> = tensor_list.iter().map(|(n, _)| n.as_str()).collect();
    write_model_index(dir, "model.safetensors", &names);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_write_tiny_shard_creates_valid_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let tensors = [("test.weight", &[4_u32, 4] as &[u32])];
        let path = write_tiny_shard(dir.path(), "test.safetensors", &tensors);
        assert!(path.exists(), "shard file exists");

        let bytes = std::fs::read(&path).unwrap();
        // Parse header length
        assert!(bytes.len() >= 8, "at least header length field");
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert!(header_len > 0, "header length > 0");
        assert!(
            bytes.len() >= 8 + header_len + 4 * 4 * 4,
            "file large enough for header + 64 F32 bytes"
        );

        // Verify JSON header parses
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let test_entry = &header["test.weight"];
        assert_eq!(test_entry["dtype"], "F32");

        // Verify data section is non-zero
        let data_start = 8 + header_len;
        let slice = &bytes[data_start..data_start + 64];
        assert!(slice.iter().any(|&b| b != 0), "data is non-zero");
    }

    #[test]
    fn test_generate_qwen2_fixture_structure() {
        let dir = tempfile::TempDir::new().unwrap();
        generate_qwen2_fixture(dir.path());

        assert!(dir.path().join("config.json").exists());
        assert!(dir.path().join("model.safetensors").exists());
        assert!(dir.path().join("model.safetensors.index.json").exists());

        // Verify config content
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "qwen2");
        assert_eq!(cfg["hidden_size"], 64);
        assert_eq!(cfg["num_hidden_layers"], 2);
    }

    #[test]
    fn test_generate_llama_fixture_model_type() {
        let dir = tempfile::TempDir::new().unwrap();
        generate_llama_fixture(dir.path());
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "llama");
    }

    #[test]
    fn test_generate_mistral_has_sliding_window() {
        let dir = tempfile::TempDir::new().unwrap();
        generate_mistral_fixture(dir.path());
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "mistral");
        assert_eq!(cfg["sliding_window"], 4096);
    }

    #[test]
    fn test_generate_gemma_fixture_model_type() {
        let dir = tempfile::TempDir::new().unwrap();
        generate_gemma_fixture(dir.path());
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "gemma");
    }

    #[test]
    fn test_generate_phi_fixture_model_type() {
        let dir = tempfile::TempDir::new().unwrap();
        generate_phi_fixture(dir.path());
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "phi3");
    }
}
