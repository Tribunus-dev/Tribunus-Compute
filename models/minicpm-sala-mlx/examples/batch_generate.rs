use std::io::Write;
use std::time::Instant;

use minicpm_sala_mlx::{
    create_layer_caches, get_model_args, is_stop_token, load_model, load_tokenizer, sample,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

fn main() -> anyhow::Result<()> {
    eprintln!("Loading model...");
    let load_start = Instant::now();
    let model_dir = std::env::args().nth(1).expect("Usage: batch_generate <model_dir>");
    let model_args = get_model_args(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;
    eprintln!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let prompts = vec![
        "Explain quantum entanglement in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What is the difference between TCP and UDP?",
    ];

    for prompt in &prompts {
        eprintln!("\n--- Prompt: {} ---", prompt);
        let encoded = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let input_ids = mlx_rs::Array::from_slice(&encoded.get_ids(), &[1, encoded.len() as i32])?;
        let mut caches = create_layer_caches(&model.args);

        // Prefill
        let start = Instant::now();
        let mut logits = model.forward(&input_ids, &mut caches)?;
        eval(&[&logits])?;
        let prefill_tok_s = encoded.len() as f32 / start.elapsed().as_secs_f32();
        eprintln!("  Prefill: {:.1} tok/s", prefill_tok_s);

        // Decode tokens
        let mut last_token = sample(&logits, 0.7)?;
        let mut tokens = Vec::new();
        let decode_start = Instant::now();

        for _ in 0..200 {
            if is_stop_token(last_token.item::<u32>()?) {
                break;
            }
            tokens.push(last_token.item::<u32>()?);

            let token_slice = last_token.reshape(&[1, 1])?;
            logits = model.forward(&token_slice, &mut caches)?;
            eval(&[&logits])?;
            last_token = sample(&logits, 0.7)?;
        }

        let decode_time = decode_start.elapsed().as_secs_f32();
        let decode_tok_s = tokens.len() as f32 / decode_time;
        eprintln!("  Decode: {:.1} tok/s ({} tokens)", decode_tok_s, tokens.len());

        let text = tokenizer.decode(&tokens, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("  Output: {}", text);
    }

    Ok(())
}
