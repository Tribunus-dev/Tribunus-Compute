//! Full Precision Qwen-Image generation example
//!
//! Uses full precision BF16 weights for highest quality.
//!
//! Model path can be set via environment variable:
//!   export DORA_MODELS_PATH=~/.OminiX/models
//!
//! Expected directory structure:
//!   $DORA_MODELS_PATH/qwen-image-2512/
//!   ├── transformer/    (full precision BF16 weights)
//!   ├── text_encoder/   (MLX format)
//!   ├── vae/            (MLX format)
//!   └── tokenizer/
//!
//! Usage:
//!   cargo run --release --example generate_fp32 -- --prompt "a fluffy cat"

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use image::{ImageBuffer, Rgb};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

// For cache clearing
extern crate mlx_rs_core;

use qwen_image_mlx::qwen_full_precision::{QwenFullConfig, QwenFullTransformer, load_full_precision_weights};

#[derive(Parser, Debug)]
#[command(name = "generate_fp32")]
#[command(about = "Generate images with full precision Qwen-Image")]
struct Args {
    /// Text prompt for image generation
    #[arg(short = 'p', long)]
    prompt: String,

    /// Output image path
    #[arg(short = 'o', long, default_value = "output.png")]
    output: String,

    /// Image width
    #[arg(short = 'W', long, default_value_t = 1024)]
    width: i32,

    /// Image height
    #[arg(short = 'H', long, default_value_t = 1024)]
    height: i32,

    /// Number of diffusion steps
    #[arg(short = 's', long, default_value_t = 20)]
    steps: i32,

    /// Classifier-free guidance scale
    #[arg(short = 'g', long, default_value_t = 4.0)]
    guidance: f32,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
}

/// Get model directory from DORA_MODELS_PATH environment variable or default location
fn get_model_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var("DORA_MODELS_PATH") {
        let dir = PathBuf::from(&path).join("qwen-image-2512");
        if dir.exists() {
            return Ok(dir);
        }
    }

    let home = std::env::var("HOME")?;
    let default = PathBuf::from(format!("{}/.dora/models/qwen-image-2512", home));
    if default.exists() {
        return Ok(default);
    }

    // Try HuggingFace cache
    let hf_cache = PathBuf::from(format!(
        "{}/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots",
        home
    ));
    if hf_cache.exists() {
        let entries: Vec<_> = std::fs::read_dir(&hf_cache)?
            .filter_map(|e| e.ok())
            .collect();
        if let Some(entry) = entries.first() {
            let path = entry.path();
            if path.is_dir() {
                return Ok(path);
            }
        }
    }

    Err("Model not found. Set DORA_MODELS_PATH or download model:\n  huggingface-cli download Qwen/Qwen-Image --local-dir ~/.dora/models/qwen-image-2512".into())
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

fn load_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&data)?;
    let mut weights = HashMap::new();

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

    Ok(weights)
}

fn load_sharded_weights(paths: &[PathBuf]) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut all_weights = HashMap::new();
    for path in paths {
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }
    Ok(all_weights)
}

fn load_tokenizer(model_dir: &std::path::Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        Ok(Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer error: {}", e))?)
    } else {
        Err(format!("tokenizer.json not found in {}", model_dir.display()).into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Qwen-Image Full Precision Generator");
    println!("====================================");
    println!("Prompt: {}", args.prompt);
    println!("Resolution: {}x{}", args.width, args.height);
    println!("Steps: {}", args.steps);
    println!("Guidance: {}", args.guidance);
    println!();

    // Find model
    let model_dir = get_model_dir()?;
    println!("Model directory: {}", model_dir.display());

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = load_tokenizer(&model_dir)?;

    // Encode prompt
    let template = "<|im_start|>system\nDescribe the image in detail.<|im_end|>\n<|im_start|>user\n";
    let prompt_text = format!("{}{}", template, args.prompt);
    let encoding = tokenizer.encode(prompt_text.as_str(), true).map_err(|e| format!("Tokenization error: {}", e))?;

    let input_ids = encoding.get_ids();
    let seq_len = input_ids.len().min(111);
    let input_ids: Vec<i32> = input_ids[input_ids.len().saturating_sub(seq_len)..]
        .iter()
        .map(|&id| id as i32)
        .collect();

    println!("Input tokens: {}", input_ids.len());

    // Setup text encoder
    let text_config = qwen_image_mlx::text_encoder::TextEncoderConfig::default();
    let text_encoder_path = model_dir.join("text_encoder");
    let mut text_encoder = if text_encoder_path.exists() {
        qwen_image_mlx::text_encoder::load_text_encoder(&model_dir)?
    } else {
        qwen_image_mlx::text_encoder::QwenTextEncoder::new(text_config)?
    };

    // Encode text
    println!("Encoding text...");
    let input_ids_arr = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);
    let txt_embedding = text_encoder.forward(&input_ids_arr, None, None)?;
    let txt_embedding = txt_embedding.index(&[.., ..1, ..])?;
    println!("Text embedding shape: {:?}", txt_embedding.shape());

    // Release text encoder
    drop(text_encoder);

    // Load VAE
    println!("Loading VAE...");
    let vae_dir = model_dir.join("vae");

    // Load transformer
    let transformer_dir = model_dir.join("transformer");
    println!("Loading transformer config...");
    let config_path = transformer_dir.join("config.json");
    let config = if config_path.exists() {
        QwenFullConfig::from_hf_json(&config_path)?
    } else {
        QwenFullConfig::default()
    };

    println!("Creating transformer ({} layers)...", config.num_layers);
    let mut transformer = QwenFullTransformer::new(config)?;

    println!("Loading transformer weights...");
    let start = Instant::now();
    let weights = load_safetensors_weights(&transformer_dir)?;
    println!("Loaded {} weight tensors in {:?}", weights.len(), start.elapsed());
    load_full_precision_weights(&mut transformer, weights)?;

    // Generate image
    println!("Generating image...");

    let batch = 1;
    let img_seq = (args.width / 16) * (args.height / 16);
    let txt_seq = 77;
    let in_channels = 64;

    // Create initial latent noise
    let noise = Array::zeros::<f32>(&[batch, img_seq, in_channels])?;

    // Create text condition (zeros for now)
    let txt = Array::zeros::<f32>(&[batch, txt_seq, config.caption_projection_dim])?;

    // Create timestep
    let timestep = Array::from_slice::<f32>(&[0.5], &[1]);

    // Forward pass
    let start = Instant::now();
    let (_img_out, _txt_out) = transformer.forward(&noise, &txt, &timestep, None, None)?;
    println!("Forward pass: {:?}", start.elapsed());

    // TODO: Full diffusion loop with CFG

    println!("Done!");
    Ok(())
}
