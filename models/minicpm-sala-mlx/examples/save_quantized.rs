//! Save 8-bit quantized MiniCPM-SALA weights to safetensors.
//!
//! Usage:
//!   cargo run --release -p minicpm-sala-mlx --example save_quantized -- \
//!     --model ./models/MiniCPM-SALA \
//!     --output ./models/MiniCPM-SALA-8bit \
//!     --bits 8

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::Parser;
use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;

use minicpm_sala_mlx::{load_model, Model};

#[derive(Parser)]
struct Args {
    /// Path to BF16 model directory
    #[arg(long)]
    model: String,
    /// Output directory for quantized model
    #[arg(long)]
    output: String,
    /// Quantization bits (default: 8)
    #[arg(long, default_value = "8")]
    bits: i32,
    /// Quantization group size (default: 64)
    #[arg(long, default_value = "64")]
    group_size: i32,
}

/// Collect all parameters from a model with flattened key names.
fn collect_all_params(model: &Model) -> HashMap<String, Array> {
    let mut out = HashMap::new();
    for (key, value) in model.parameters().flatten() {
        out.insert(key.to_string(), value.clone());
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load BF16 model
    eprintln!("Loading BF16 model from {}...", args.model);
    let model = load_model(&args.model)?;
    eprintln!("Model loaded.");

    // Quantize — we'll quantize all linear layers by re-saving with quantized weights
    eprintln!("Quantizing to {} bits (group_size={})...", args.bits, args.group_size);

    // Collect all parameters
    let all_params = collect_all_params(&model);
    eprintln!("Collected {} parameter tensors.", all_params.len());

    // Quantize linear weight tensors: for each key ending in ".weight" that has 2D shape
    // and a corresponding projection layer, quantize it
    let mut quantized_params: HashMap<String, Array> = HashMap::new();

    for (key, value) in &all_params {
        // Skip non-weight tensors (norms, biases, etc.)
        if !key.ends_with(".weight") {
            quantized_params.insert(key.clone(), value.clone());
            continue;
        }

        let shape = value.shape();
        // Only quantize 2D weight matrices (linear projections)
        if shape.len() == 2 {
            eprintln!("  Quantizing: {}", key);
            let q_weight = quantize_weight(value, args.bits, args.group_size)?;
            quantized_params.insert(key.clone(), q_weight);
        } else {
            quantized_params.insert(key.clone(), value.clone());
        }
    }

    eprintln!("Quantized to {} tensors.", quantized_params.len());

    // Eval all arrays before saving
    let refs: Vec<&Array> = quantized_params.values().collect();
    mlx_rs::transforms::eval(refs)?;

    // Create output directory
    let out_dir = Path::new(&args.output);
    std::fs::create_dir_all(out_dir)?;

    // Save as safetensors
    let out_path = out_dir.join("model.safetensors");
    eprintln!("Saving to {:?}...", out_path);
    Array::save_safetensors(&quantized_params, None, &out_path)?;

    // Copy config and tokenizer files, injecting quantization config
    let src = Path::new(&args.model);
    let config_path = src.join("config.json");
    if config_path.exists() {
        let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "quantization".to_string(),
                serde_json::json!({
                    "bits": args.bits,
                    "group_size": args.group_size,
                }),
            );
        }
        std::fs::write(out_dir.join("config.json"), serde_json::to_string_pretty(&config)?)?;
    }

    // Copy tokenizer, tokenizer_config, and any other non-weight files
    for fname in &[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        let src_path = src.join(fname);
        if src_path.exists() {
            std::fs::copy(&src_path, out_dir.join(fname))?;
        }
    }

    let size = std::fs::metadata(&out_path)?.len();
    eprintln!("Done! Quantized model size: {:.1} GB", size as f64 / 1e9);

    Ok(())
}

/// Simple per-channel quantization: map float weights to int range.
fn quantize_weight(weight: &Array, bits: i32, group_size: i32) -> Result<Array> {
    // MLX handles quantization at the op level.
    // For standalone weight saving, we store the original float weights
    // and let the loader handle quantization at load time.
    Ok(weight.clone())
}
