use std::io::Write;
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, SpeculativeDecoder,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(
    name = "speculative_generate",
    about = "MiniCPM-SALA with self-speculative decoding"
)]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Prompt text
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Number of layers for draft model
    #[arg(long, default_value_t = 8)]
    draft_layers: usize,

    /// Number of draft tokens per speculation round
    #[arg(long, default_value_t = 4)]
    num_draft: usize,

    /// Raw completion mode (no chat template)
    #[arg(long)]
    raw: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load model
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", args.model_dir);
    let model_args = get_model_args(&args.model_dir)?;
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;
    eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // Prepare prompt
    let prompt = if args.raw {
        args.prompt.clone()
    } else {
        format_chat_prompt(&args.system, &args.prompt)
    };

    let encoded = tokenizer.encode(prompt.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
    let input_ids = mlx_rs::Array::from_slice(&encoded.get_ids(), &[1, encoded.len() as i32])?;

    eprintln!("Prompt: {} tokens", encoded.len());
    eprintln!(
        "Speculative decoding: draft={} layers, {} draft tokens",
        args.draft_layers, args.num_draft
    );
    eprintln!("Generating...\n");

    // Prefill
    let prefill_start = Instant::now();
    let mut caches = create_layer_caches(&model.args);
    let logits = model.forward(&input_ids, &mut caches)?;
    eval(&[&logits])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();
    eprintln!("Prefill: {:.2}s ({:.1} tok/s)", prefill_time, encoded.len() as f32 / prefill_time);

    // Speculative decoding loop
    let decoder = SpeculativeDecoder::new(args.draft_layers, args.num_draft, args.temperature);
    let mut last_token = sample(&logits, args.temperature)?;
    let mut generated_tokens = Vec::new();
    let decode_start = Instant::now();

    while generated_tokens.len() < args.max_tokens {
        let result = decoder.step(&mut model, &mut caches, &last_token)?;

        for &token in &result.tokens {
            if is_stop_token(token) {
                break;
            }
            generated_tokens.push(token);
        }

        if generated_tokens.len() >= args.max_tokens {
            break;
        }

        // Update last_token for next speculation round
        last_token = mlx_rs::Array::from_slice(&[generated_tokens.last().copied().unwrap_or(2)], &[1])?;
    }

    let decode_time = decode_start.elapsed().as_secs_f32();
    let text = tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", text);

    eprintln!(
        "\nDecode: {:.2}s ({:.1} tok/s, {} tokens)",
        decode_time,
        generated_tokens.len() as f32 / decode_time,
        generated_tokens.len()
    );
    eprintln!("Total: {:.2}s", load_start.elapsed().as_secs_f32());

    Ok(())
}
