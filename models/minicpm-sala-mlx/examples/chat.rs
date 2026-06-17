use std::io::{self, BufRead, Write};
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, ThinkFilter,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "chat", about = "Interactive chat with MiniCPM-SALA")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Maximum tokens per response
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// System prompt
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Hide <think>...</think> reasoning
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
    eprintln!("System: {}", args.system);
    eprintln!("Type your message and press Enter. Type 'quit' or Ctrl-D to exit.\n");

    let stdin = io::stdin();
    let mut turn_num = 0usize;
    let mut last_response_text = String::new();

    loop {
        // Read user input
        eprint!("You: ");
        std::io::stderr().flush()?;

        let mut user_input = String::new();
        if stdin.lock().read_line(&mut user_input).is_err() || user_input.trim().is_empty() {
            break;
        }
        let user_input = user_input.trim().to_string();

        if user_input == "quit" {
            break;
        }

        // Build prompt
        let turn_prompt = if turn_num == 0 {
            format_chat_prompt(&args.system, &user_input)
        } else {
            // Multi-turn: continue from last assistant response
            format!("{}<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n", last_response_text)
        };

        let encoded = tokenizer.encode(&turn_prompt, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let input_ids = mlx_rs::Array::from_slice(&encoded.get_ids(), &[1, encoded.len() as i32])?;

        eprint!("Assistant: ");
        std::io::stderr().flush()?;

        // Forward
        let logits = model.forward(&input_ids, &mut caches)?;
        eval(&[&logits])?;

        // Generate
        let mut think_filter = ThinkFilter::new(args.no_think);
        let mut full_text = String::new();
        let mut last_token = sample(&logits, args.temperature)?;

        for _ in 0..args.max_tokens {
            if is_stop_token(last_token.item::<u32>()?) {
                break;
            }

            let token_slice = last_token.reshape(&[1, 1])?;
            let logits = model.forward(&token_slice, &mut caches)?;
            eval(&[&logits])?;
            last_token = sample(&logits, args.temperature)?;

            let text = tokenizer.decode(&[last_token.item::<u32>()?], true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            full_text.push_str(&text);

            let display = think_filter.next(&full_text);
            if !display.is_empty() {
                print!("{display}");
                std::io::stdout().flush()?;
            }
        }

        println!();
        last_response_text = full_text;
        turn_num += 1;
    }

    Ok(())
}
