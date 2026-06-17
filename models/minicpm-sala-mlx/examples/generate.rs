use std::io::Write;
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, ThinkFilter,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "generate", about = "MiniCPM-SALA text generation")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Prompt text
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Raw completion mode (no chat template)
    #[arg(long)]
    raw: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Hide <think>...</think> reasoning and only show final answer
    #[arg(long)]
    no_think: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load model
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", args.model_dir);
    let model_args = get_model_args(&args.model_dir)?;
    eprintln!(
        "  Args: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
        model_args.hidden_size,
        model_args.num_hidden_layers,
        model_args.num_attention_heads,
        model_args.num_key_value_heads,
        model_args.vocab_size
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;
    let mut caches = create_layer_caches(&model.args);

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
    eprintln!("Generating...\n");

    // Forward pass for prefill
    let prefill_start = Instant::now();
    let logits = model.forward(&input_ids, &mut caches)?;
    eval(&[&logits])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();
    eprintln!("Prefill: {:.2}s ({:.1} tok/s)", prefill_time, encoded.len() as f32 / prefill_time);

    // Generate tokens
    let mut think_filter = ThinkFilter::new(args.no_think);
    let mut full_text = String::new();
    let mut last_token = sample(&logits, args.temperature)?;
    let mut token_count = 0usize;

    let decode_start = Instant::now();

    for i in 0..args.max_tokens {
        if is_stop_token(last_token.item::<u32>()?) {
            break;
        }

        // Decode token
        let token_slice = last_token.reshape(&[1, 1])?;

        let logits = model.forward(&token_slice, &mut caches)?;
        eval(&[&logits])?;
        last_token = sample(&logits, args.temperature)?;

        token_count = i + 1;

        // Incremental decode using tokenizer
        let text = tokenizer.decode(&[last_token.item::<u32>()?], true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        full_text.push_str(&text);

        // Apply think filter and print
        let display = think_filter.next(&full_text);
        if !display.is_empty() {
            print!("{display}");
            std::io::stdout().flush()?;
        }
    }

    if !args.no_think {
        // Print any remaining filtered text
        let remaining = think_filter.next(&full_text);
        if !remaining.is_empty() {
            print!("{remaining}");
        }
    }

    let decode_time = decode_start.elapsed().as_secs_f32();
    eprintln!("\n\nDecode: {:.2}s ({:.1} tok/s)", decode_time, token_count as f32 / decode_time);
    eprintln!("Total: {:.2}s", load_start.elapsed().as_secs_f32());

    Ok(())
}
