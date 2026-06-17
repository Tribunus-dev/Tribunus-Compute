//! Qwen-Image generation example (4-bit quantized)
//!
//! Model path can be set via environment variable:
//!   export DORA_MODELS_PATH=~/.OminiX/models
//!
//! Expected directory structure:
//!   $DORA_MODELS_PATH/qwen-image-2512-4bit/
//!   ├── transformer/    (4-bit quantized)
//!   ├── text_encoder/   (full precision)
//!   ├── vae/            (full precision)
//!   └── tokenizer/
//!
//! Falls back to HuggingFace cache if not found.
//!
//! Usage:
//!   cargo run --release --example generate_qwen_image -- --prompt "a cat sitting on a couch"

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "generate_qwen_image")]
#[command(about = "Generate images with Qwen-Image")]
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

    /// Use 8-bit quantization instead of 4-bit
    #[arg(long)]
    use_8bit: bool,
}

fn get_hf_cache_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Check DORA_MODELS_PATH first
    if let Ok(path) = std::env::var("DORA_MODELS_PATH") {
        dirs.push(PathBuf::from(path));
    }

    // Check common HF cache locations
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(format!("{}/.cache/huggingface/hub", home)));
        dirs.push(PathBuf::from(format!("{}/.dora/models", home)));
    }

    dirs
}

/// Get model directory from DORA_MODELS_PATH or HuggingFace cache
fn get_model_dir(repo_id: &str) -> std::io::Result<PathBuf> {
    let cache_dirs = get_hf_cache_dirs();

    for base in &cache_dirs {
        // Check direct path
        let direct = base.join(repo_id);
        if direct.join("transformer").exists() || direct.join("config.json").exists() {
            return Ok(direct);
        }

        // Check HF snapshot format: .../hub/models--repo--id/snapshots/<hash>/
        let hf_path = base.join(format!("models--{}", repo_id.replace('/', "--")));
        if hf_path.exists() {
            let snapshots = hf_path.join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if path.join("transformer").exists() || path.join("config.json").exists() {
                            return Ok(path);
                        }
                    }
                }
            }
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("Model '{}' not found in any cache directory", repo_id),
    ))
}

/// Load safetensors weights from a single file
fn load_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path.as_ref())?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;
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

/// Load safetensors weights from multiple shards
fn load_sharded_weights(paths: &[PathBuf]) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut all_weights = HashMap::new();
    for path in paths {
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }
    Ok(all_weights)
}

/// Load tokenizer from model directory
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

    println!("Initializing Qwen-Image generation...");

    // Find model directory
    let repo_id = if args.use_8bit {
        "mlx-community/Qwen-Image-2512-8bit"
    } else {
        "mlx-community/Qwen-Image-2512-4bit"
    };

    let model_dir = get_model_dir(repo_id)?;
    println!("Model directory: {}", model_dir.display());

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = load_tokenizer(&model_dir)?;

    // Encode prompt
    let template = "<|im_start|>system\nDescribe the image in detail.<|im_end|>\n<|im_start|>user\n";
    let prompt_text = format!("{}{}", template, args.prompt);
    let encoding = tokenizer.encode(prompt_text.as_str(), true).map_err(|e| format!("Tokenization error: {}", e))?;

    let input_ids = encoding.get_ids();
    let seq_len = input_ids.len().min(111); // 77 + 34

    // Trim to max length
    let input_ids: Vec<i32> = input_ids[input_ids.len().saturating_sub(seq_len)..]
        .iter()
        .map(|&id| id as i32)
        .collect();

    println!("Input tokens: {}", input_ids.len());

    // Setup text encoder config
    let text_config = qwen_image_mlx::text_encoder::TextEncoderConfig::default();

    // Load text encoder
    println!("Loading text encoder...");
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
    let txt_embedding = txt_embedding.index(&[.., ..1, ..])?; // Use [EOS] token embedding
    println!("Text embedding shape: {:?}", txt_embedding.shape());

    // Release text encoder memory
    drop(text_encoder);

    // Setup VAE
    println!("Loading VAE...");
    let vae = qwen_image_mlx::vae::load_vae_from_dir(&model_dir)?;

    // Setup transformer config
    let config_path = model_dir.join("transformer").join("config.json");
    let qwen_config = if config_path.exists() {
        qwen_image_mlx::qwen_quantized::QwenConfig::from_hf_json(&config_path)?
    } else {
        qwen_image_mlx::qwen_quantized::QwenConfig::default()
    };

    let quantized_bits = if args.use_8bit { 8 } else { 4 };
    let config = qwen_image_mlx::qwen_quantized::QwenConfig {
        quantized_bits,
        quantized_group_size: 64,
        ..qwen_config
    };

    // Load transformer
    println!("Loading transformer ({}-bit)...", config.quantized_bits);
    let mut transformer = qwen_image_mlx::qwen_quantized::QwenQuantizedTransformer::new(config)?;

    // Load transformer weights
    let transformer_dir = model_dir.join("transformer");
    let mut shard_files: Vec<PathBuf> = std::fs::read_dir(&transformer_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "safetensors").unwrap_or(false))
        .collect();
    shard_files.sort();

    if shard_files.is_empty() {
        // Try single file
        let single = transformer_dir.join("model.safetensors");
        if single.exists() {
            shard_files.push(single);
        }
    }

    if !shard_files.is_empty() {
        let weights = load_sharded_weights(&shard_files)?;
        qwen_image_mlx::qwen_quantized::load_transformer_weights(&mut transformer, weights)?;
    }

    // Create pipeline
    let pipe_config = qwen_image_mlx::pipeline::QwenImageConfig {
        height: args.height,
        width: args.width,
        num_steps: args.steps,
        guidance_scale: args.guidance,
    };

    let mut pipeline = qwen_image_mlx::pipeline::QwenImagePipeline::new(
        vae,
        qwen_image_mlx::text_encoder::QwenTextEncoder::new(text_config)?,
        pipe_config,
    );

    pipeline.transformer = Some(transformer);
    pipeline.text_encoder = qwen_image_mlx::text_encoder::QwenTextEncoder::new(text_config)?;

    // Generate image
    println!("Generating image...");
    let image = pipeline.generate(args.prompt.as_str())?;

    // Save image
    println!("Saving to {}...", args.output);
    // TODO: Convert MLX Array to image and save

    println!("Done!");
    Ok(())
}
