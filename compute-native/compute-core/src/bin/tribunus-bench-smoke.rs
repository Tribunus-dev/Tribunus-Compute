//! Benchmark harness for smoke-testing the worker IPC path end-to-end.
//!
//! Spawns a worker process, loads the model, tokenizes a prompt, runs
//! generation, and reports metrics as CSV to stdout.
//!
//! Usage:
//!   tribunus-bench-smoke --model-path <path> \
//!                        [--prompt <text>] \
//!                        [--max-tokens <N>]
//!
//! The worker binary is resolved from `TRIBUNUS_WORKER_BINARY` if set,
//! otherwise discovered alongside the current executable.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tribunus_compute_core::engine_policy::qualification_policy;
use tribunus_compute_core::logging::{log_error, log_info};
use tribunus_compute_core::streaming::GenerationEvent;
use tribunus_compute_core::tokenizer::TribunusTokenizer;
use tribunus_compute_core::worker_protocol::StartGenerationPayload;
use tribunus_compute_core::worker_supervisor::WorkerSupervisor;

fn print_usage() {
    eprintln!(
        "Usage: tribunus-bench-smoke --model-path <path> [--prompt <text>] [--max-tokens <N>]"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model-path <dir>   Path to model image directory (required)");
    eprintln!("  --prompt <text>      Prompt text (default: Hello, world!)");
    eprintln!("  --max-tokens <N>     Max output tokens (default: 50)");
    eprintln!("  --help, -h           Show this help");
}

fn main() {
    // macOS workaround: unset MallocStackLogging inherited from Xcode/LLDB
    // to suppress "can't turn off malloc stack logging because it was not enabled"
    // on stderr during process exit, which corrupts terminal output.
    unsafe {
        std::env::set_var("MallocStackLogging", "0");
        std::env::set_var("MallocStackLoggingNoCompact", "0");
    }

    // ── Parse CLI args ───────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut prompt = "Hello, world!".to_string();
    let mut max_tokens: u32 = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    model_path = Some(p.clone());
                }
            }
            "--prompt" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    prompt = p.clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    max_tokens = s.parse().unwrap_or(50);
                }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("error: --model-path is required");
            print_usage();
            std::process::exit(1);
        }
    };

    // Print CSV header and fixed fields.
    println!("metric,value,unit");
    println!("model_path,{},", &model_path);
    println!("prompt,{},", &prompt);
    println!("max_tokens,{},", max_tokens);

    // ── Validate model path ──────────────────────────────────────────
    let path = Path::new(&model_path);
    if !path.exists() {
        log_error!("Model path not found");
        println!("status,error,");
        println!("error,model path not found: {},", &model_path);
        return;
    }

    // ── Load tokenizer ───────────────────────────────────────────────
    let tokenizer_load_start = Instant::now();
    let tokenizer = match TribunusTokenizer::from_dir(path) {
        Ok(tok) => {
            let elapsed_ms = tokenizer_load_start.elapsed().as_millis() as u64;
            log_info!("Tokenizer loaded in {}ms", elapsed_ms);
            println!("tokenizer_load_ms,{},ms", elapsed_ms);
            println!("tokenizer_available,true,");
            Some(tok)
        }
        Err(e) => {
            log_error!("Failed to load tokenizer: {}", e);
            println!("tokenizer_load_ms,0,ms");
            println!("tokenizer_available,false,");
            None
        }
    };

    // ── Resolve worker binary ────────────────────────────────────────
    let worker_binary: PathBuf = std::env::var("TRIBUNUS_WORKER_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Look for the worker alongside the current binary.
            let exe = std::env::current_exe()
                .unwrap_or_else(|_| PathBuf::from("tribunus-compute-worker"));
            let dir = exe.parent().unwrap_or(Path::new("."));
            dir.join("tribunus-compute-worker")
        });

    if !worker_binary.exists() {
        log_error!("Worker binary not found: {}", worker_binary.display());
        println!("status,error,");
        println!(
            "error,worker binary not found: {},",
            worker_binary.display()
        );
        return;
    }

    // ── Spawn worker and handshake ───────────────────────────────────
    let worker_start = Instant::now();
    let policy = qualification_policy();
    let image_hash = "";
    let worker_id = uuid::Uuid::new_v4().to_string();

    let supervisor = match WorkerSupervisor::launch_and_handshake(
        policy,
        &worker_binary,
        path,
        image_hash,
        &worker_id,
    ) {
        Ok(sup) => {
            let elapsed_ms = worker_start.elapsed().as_millis() as u64;
            log_info!(
                "Worker spawned (pid {}) in {}ms",
                sup.process_ctrl.pid(),
                elapsed_ms
            );
            println!("worker_start_ms,{},ms", elapsed_ms);
            sup
        }
        Err(e) => {
            log_error!("Worker launch failed: {:?}", e);
            println!("status,error,");
            println!("error,worker launch failed: {:?},", e);
            return;
        }
    };

    // ── Load model on worker ─────────────────────────────────────────
    let model_load_start = Instant::now();
    match supervisor.load_model(image_hash) {
        Ok(()) => {
            let elapsed_ms = model_load_start.elapsed().as_millis() as u64;
            log_info!("Model loaded in {}ms", elapsed_ms);
            println!("model_load_ms,{},ms", elapsed_ms);
        }
        Err(e) => {
            log_error!("Model load failed: {:?}", e);
            println!("status,error,");
            println!("error,model load failed: {:?},", e);
            return;
        }
    }

    // ── Tokenize prompt ──────────────────────────────────────────────
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = &tokenizer {
        match tok.encode(&prompt) {
            Ok(ids) => {
                log_info!("Encoded prompt: {} tokens", ids.len());
                ids
            }
            Err(e) => {
                log_error!(
                    "Tokenizer encode failed: {}, falling back to byte tokenizer",
                    e
                );
                prompt.bytes().map(|b| b as u32).collect()
            }
        }
    } else {
        // Byte-level fallback when no tokenizer is loaded.
        prompt.bytes().map(|b| b as u32).collect()
    };

    if prompt_tokens.is_empty() {
        log_error!("Empty prompt after tokenization");
        println!("status,error,");
        println!("error,empty prompt after tokenization,");
        return;
    }

    println!("prompt_token_count,{},tokens", prompt_tokens.len());

    // ── Start generation ─────────────────────────────────────────────
    let request_id = uuid::Uuid::new_v4().to_string();
    let payload = StartGenerationPayload {
        generation_regime: Default::default(),
        denoising_steps: None,
        confidence_threshold: None,
        canvas_tokens: None,
        prompt_token_ids: prompt_tokens,
        max_output_tokens: max_tokens,
        deadline_ms: 120_000, // 2-minute wall-clock deadline
        request_id,
        temperature: None,
        top_k: None,
        top_p: None,
        seed: None,
        stop_token_ids: Vec::new(),
    };

    let mut handle = match supervisor.start_generation(&payload) {
        Ok(h) => h,
        Err(e) => {
            log_error!("Start generation failed: {:?}", e);
            println!("status,error,");
            println!("error,start generation failed: {:?},", e);
            return;
        }
    };

    // ── Collect events ───────────────────────────────────────────────
    let mut tokens: Vec<u32> = Vec::new();
    let mut ttft_ms: Option<u64> = None;
    let gen_start = Instant::now();

    loop {
        // Use match ergonomics: borrow the scrutinee.
        match &mut handle.stream.recv() {
            Some(GenerationEvent::Started) => {
                log_info!("Generation started");
            }
            Some(GenerationEvent::Token(tok)) => {
                if ttft_ms.is_none() {
                    ttft_ms = Some(gen_start.elapsed().as_millis() as u64);
                }
                tokens.push(*tok);
            }
            Some(GenerationEvent::Done) => {
                log_info!("Generation completed: {} tokens", tokens.len());
                break;
            }
            Some(GenerationEvent::Error(msg)) => {
                log_error!("Generation error: {}", msg);
                println!("status,error,");
                println!("error,generation error: {},", msg);
                return;
            }
            Some(GenerationEvent::Cancelled) => {
                log_error!("Generation cancelled");
                println!("status,error,");
                println!("error,generation cancelled,");
                return;
            }
            Some(_) => {
                // Ignore other event types (Chunk, Progress, Metrics, etc.).
            }
            None => {
                // Stream closed without terminal event.
                log_error!("Stream closed unexpectedly");
                println!("status,error,");
                println!("error,stream closed without terminal event,");
                return;
            }
        }
    }

    let gen_elapsed = gen_start.elapsed();
    let token_count = tokens.len() as u64;

    // ── Compute and print metrics ────────────────────────────────────
    println!("status,ok,");
    println!("token_count,{},tokens", token_count);
    if let Some(ttft) = ttft_ms {
        println!("ttft_ms,{},ms", ttft);
    } else {
        println!("ttft_ms,0,ms");
    }
    println!("gen_duration_ms,{},ms", gen_elapsed.as_millis() as u64);

    if token_count > 0 && gen_elapsed.as_secs_f64() > 0.0 {
        let tps = token_count as f64 / gen_elapsed.as_secs_f64();
        println!("tokens_per_sec,{:.2},tok/s", tps);
    }

    // Supervisor and handle are dropped here, triggering worker shutdown.
    log_info!("Benchmark complete");
}
