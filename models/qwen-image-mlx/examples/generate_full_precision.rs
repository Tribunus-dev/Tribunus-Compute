//! Full precision Qwen-Image generation example
//!
//! Tests loading full precision weights from Qwen/Qwen-Image
//!
//! Usage:
//!     cargo run --release --example generate_full_precision

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use mlx_rs::Array;
use safetensors::SafeTensors;

use qwen_image_mlx::qwen_full_precision::{QwenFullConfig, QwenFullTransformer, load_full_precision_weights};

#[derive(Parser, Debug)]
#[command(name = "generate_full_precision")]
#[command(about = "Test full precision Qwen-Image transformer")]
struct Args {
    /// Model path (HuggingFace cache or local)
    #[arg(long)]
    model_path: Option<String>,
}

fn find_model_path() -> Result<String, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let cache_path = format!(
        "{}/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots",
        home
    );

    if Path::new(&cache_path).exists() {
        // Find the snapshot hash (latest)
        let entries: Vec<_> = std::fs::read_dir(&cache_path)?
            .filter_map(|e| e.ok())
            .collect();

        // Return the first snapshot directory
        if let Some(entry) = entries.first() {
            let path = entry.path();
            if path.is_dir() {
                return Ok(path.to_string_lossy().to_string());
            }
        }
    }

    // Also try DORA path
    let dora_path = format!("{}/.dora/models/qwen-image-2512", home);
    if Path::new(&dora_path).exists() {
        return Ok(dora_path);
    }

    Err("Model not found. Please download with:\n  huggingface-cli download Qwen/Qwen-Image".into())
}

fn load_safetensors_weights(dir: &Path) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut weights = HashMap::new();

    for i in 1..=9 {
        let shard = dir.join(format!("model-000{}-of-00009.safetensors", i));
        if shard.exists() {
            let data = std::fs::read(&shard)?;
            let tensors = SafeTensors::deserialize(&data)?;
            for (name, view) in tensors.tensors() {
                let data = view.into_data();
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => mlx_rs::Dtype::Float32,
                    safetensors::Dtype::F16 => mlx_rs::Dtype::Float16,
                    safetensors::Dtype::BF16 => mlx_rs::Dtype::BFloat16,
                    _ => mlx_rs::Dtype::Float32,
                };
                let shape: Vec<i32> = view.shape().iter().map(|&s| s as i32).collect();
                let array = Array::from_slice_into_dtype(
                    bytemuck::cast_slice(&data),
                    &shape,
                    dtype,
                )?;
                weights.insert(name, array);
            }
        }
    }

    Ok(weights)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _args = Args::parse();

    println!("Loading Qwen-Image full precision transformer...");

    let model_path = find_model_path()?;
    println!("Model path: {}", model_path);

    // Find transformer directory
    let transformer_dir = Path::new(&model_path).join("transformer");

    // Load config
    let config_path = transformer_dir.join("config.json");
    let config = if config_path.exists() {
        QwenFullConfig::from_hf_json(&config_path)?
    } else {
        QwenFullConfig::default()
    };

    println!("Creating transformer with {} layers...", config.num_layers);
    let mut transformer = QwenFullTransformer::new(config)?;

    // Load weights
    println!("Loading transformer weights...");
    let start = Instant::now();
    let weights = load_safetensors_weights(&transformer_dir)?;
    println!("Loaded {} weight tensors in {:?}", weights.len(), start.elapsed());

    load_full_precision_weights(&mut transformer, weights)?;
    println!("Weights loaded successfully");

    // Test forward pass
    println!("Testing forward pass...");
    let batch = 1;
    let img_seq = 1024; // 32*32 patches for 512x512
    let txt_seq = 77;
    let hidden_dim = 3072;

    let img = Array::zeros::<f32>(&[batch, img_seq, 64])?;
    let txt = Array::zeros::<f32>(&[batch, txt_seq, 3584])?;
    let timestep = Array::from_slice::<f32>(&[0.5], &[1]);

    let start = Instant::now();
    let (img_out, txt_out) = transformer.forward(&img, &txt, &timestep, None, None)?;
    println!("Forward pass took {:?}", start.elapsed());
    println!("Image output shape: {:?}", img_out.shape());
    println!("Text output shape: {:?}", txt_out.shape());

    println!("Full precision transformer test successful!");
    Ok(())
}
