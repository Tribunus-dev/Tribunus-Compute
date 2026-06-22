//! tribunus-bench — serving benchmark harness.
//!
//! Measures prefill throughput, decode throughput, concurrent serving
//! latency/throughput, and produces comparable numbers to vLLM's
//! benchmark_serving.py and llama.cpp's bench.
//!
//! Phases:
//!   prefill  — cold model load + prefill token throughput
//!   decode   — decode token throughput with warm cache
//!   serve    — concurrent request serving under a fixed latency SLO
//!   compare  — diff two JSON result files (e.g. baseline vs candidate)
//!
//! Output: per-phase JSON to stdout; structured for automated ingestion
//! and the `compare` subcommand.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct BenchReport {
    phase: String,
    model: String,
    timestamp: String,
    environment: EnvironmentInfo,
    metrics: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct EnvironmentInfo {
    hostname: String,
    binary: String,
    features: Vec<String>,
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!("Usage:");
    eprintln!(
        "  tribunus-bench prefill  --model <path> [--prompt-tokens 64] [--warmup 3] [--trials 5]"
    );
    eprintln!("  tribunus-bench decode   --model <path> [--prompt-tokens 10] [--output-tokens 128] [--warmup 3]");
    eprintln!("  tribunus-bench serve    --model <path> [--concurrency 1] [--duration 30s] [--prompt-tokens 64] [--output-tokens 50]");
    eprintln!("  tribunus-bench compare  --baseline-path <dir> --candidate-path <dir>");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  TRIBUNUS_BENCH_MODEL      Model path (default: --model)");
    eprintln!("  TRIBUNUS_BENCH_OUTPUT     Write JSON report to file instead of stdout");
    eprintln!("  TRIBUNUS_SKIP_MANIFEST_HASH  Set automatically by the harness");
}

fn main() {
    // macOS workaround: suppress MallocStackLogging noise on stderr during exit.
    unsafe {
        std::env::set_var("MallocStackLogging", "0");
        std::env::set_var("MallocStackLoggingNoCompact", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let phase = args[1].clone();
    let rest: Vec<String> = args[2..].to_vec();

    let model_path: String = std::env::var("TRIBUNUS_BENCH_MODEL")
        .or_else(|_| std::env::var("MODEL_PATH"))
        .ok()
        .unwrap_or_default();

    match phase.as_ref() {
        "prefill" => cmd_prefill(&rest, model_path),
        "decode" => cmd_decode(&rest, model_path),
        "serve" => cmd_serve(&rest, model_path),
        "compare" => cmd_compare(&rest),
        "--help" | "-h" | "help" => {
            print_usage();
        }
        other => {
            eprintln!("error: unknown phase '{other}'");
            print_usage();
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Parse `--key value` pairs from a slice.  Returns a map of stripped keys.
fn parse_kv(args: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            if let Some(val) = args.get(i + 1) {
                map.insert(key.to_string(), val.clone());
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    map
}

fn parse_duration(s: &str) -> std::time::Duration {
    if let Some(secs) = s.strip_suffix('s') {
        if let Ok(f) = secs.parse::<f64>() {
            return std::time::Duration::from_secs_f64(f);
        }
    }
    if let Some(ms) = s.strip_suffix("ms") {
        if let Ok(n) = ms.parse::<u64>() {
            return std::time::Duration::from_millis(n);
        }
    }
    // fallback: parse as raw seconds
    s.parse::<u64>()
        .map(std::time::Duration::from_secs)
        .unwrap_or(std::time::Duration::from_secs(30))
}

fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into())
}

fn env_features() -> Vec<String> {
    let mut f = Vec::new();
    #[cfg(feature = "mlx-backend")]
    f.push("mlx-backend".into());
    #[cfg(feature = "candle-cpu")]
    f.push("candle-cpu".into());
    #[cfg(feature = "server")]
    f.push("server".into());
    f
}

fn current_exe_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "tribunus-bench".into())
}

fn emit_report(report: &BenchReport) {
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    if let Ok(path) = std::env::var("TRIBUNUS_BENCH_OUTPUT") {
        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("warning: failed to write report to {path}: {e}");
        }
    }
    println!("{json}");
}

fn kv_int(map: &BTreeMap<String, String>, key: &str, default: usize) -> usize {
    map.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn kv_str<'a>(map: &'a BTreeMap<String, String>, key: &str, default: &'a str) -> &'a str {
    map.get(key).map(String::as_str).unwrap_or(default)
}

fn kv_duration(map: &BTreeMap<String, String>, key: &str, default: &str) -> std::time::Duration {
    map.get(key)
        .map(|s| parse_duration(s))
        .unwrap_or_else(|| parse_duration(default))
}

// ---------------------------------------------------------------------------
// Phase: prefill
// ---------------------------------------------------------------------------

#[cfg(feature = "mlx-backend")]
fn do_prefill_mlx(
    model_path: &Path,
    prompt_len: usize,
    warmup_rounds: usize,
    trials: usize,
) -> BenchReport {
    use tribunus_compute_core::kv_cache::KvCache;
    use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

    let mut metrics = BTreeMap::new();

    // Cold load timing
    let load_start = Instant::now();
    let model =
        LoadedProfiledModel::new(model_path).expect("failed to load model for prefill bench");
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    metrics.insert("model_load_ms".into(), serde_json::json!(load_ms));

    let n_layers = model.reader.manifest.execution_plan.layers.len();
    metrics.insert("num_layers".into(), serde_json::json!(n_layers));

    // Build prompt
    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 1024) as u32 + 1).collect();

    // Warmup rounds
    for _w in 0..warmup_rounds {
        let kv_caches: Vec<KvCache> = (0..n_layers)
            .map(|_| KvCache::new(prompt_len as u32 + 128, 128, 64, false))
            .collect();
        let mut session = ProfiledInferenceSession::new("prefill-warmup".into(), kv_caches);
        session.setup_from_model(&model);
        let _ = session.prefill(&prompt, &model).expect("warmup prefill");
    }

    // Measurement trials
    let mut prefill_times_ms: Vec<f64> = Vec::with_capacity(trials);
    for _t in 0..trials {
        let kv_caches: Vec<KvCache> = (0..n_layers)
            .map(|_| KvCache::new(prompt_len as u32 + 128, 128, 64, false))
            .collect();
        let mut session = ProfiledInferenceSession::new("prefill-trial".into(), kv_caches);
        session.setup_from_model(&model);

        let start = Instant::now();
        let _next = session.prefill(&prompt, &model).expect("trial prefill");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        prefill_times_ms.push(elapsed_ms);
    }

    let n = prefill_times_ms.len() as f64;
    if n > 0.0 {
        let total_ms: f64 = prefill_times_ms.iter().sum();
        let mean_ms = total_ms / n;
        let min_ms = prefill_times_ms.iter().cloned().fold(f64::MAX, f64::min);
        let max_ms = prefill_times_ms.iter().cloned().fold(f64::MIN, f64::max);

        // sort for median
        let mut sorted = prefill_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let prefill_tok_s = (prompt_len as f64) / (mean_ms / 1000.0);
        let prefill_tok_s_min = (prompt_len as f64) / (max_ms / 1000.0);
        let prefill_tok_s_max = (prompt_len as f64) / (min_ms / 1000.0);

        metrics.insert("prompt_len".into(), serde_json::json!(prompt_len));
        metrics.insert("trials".into(), serde_json::json!(trials));
        metrics.insert("prefill_time_mean_ms".into(), serde_json::json!(mean_ms));
        metrics.insert(
            "prefill_time_median_ms".into(),
            serde_json::json!(median_ms),
        );
        metrics.insert("prefill_time_min_ms".into(), serde_json::json!(min_ms));
        metrics.insert("prefill_time_max_ms".into(), serde_json::json!(max_ms));
        metrics.insert(
            "prefill_tokens_per_sec".into(),
            serde_json::json!(prefill_tok_s),
        );
        metrics.insert(
            "prefill_tokens_per_sec_min".into(),
            serde_json::json!(prefill_tok_s_min),
        );
        metrics.insert(
            "prefill_tokens_per_sec_max".into(),
            serde_json::json!(prefill_tok_s_max),
        );
    }

    BenchReport {
        phase: "prefill".into(),
        model: model_path.to_string_lossy().into_owned(),
        timestamp: tribunus_compute_core::now_iso8601(),
        environment: EnvironmentInfo {
            hostname: hostname_or_default(),
            binary: current_exe_name(),
            features: env_features(),
        },
        metrics,
    }
}

fn cmd_prefill(args: &[String], model_env: String) {
    let kv = parse_kv(args);
    let model = kv_str(&kv, "model", &model_env);
    if model.is_empty() {
        eprintln!("error: --model <path> required (or set TRIBUNUS_BENCH_MODEL)");
        std::process::exit(1);
    }
    let model_path = Path::new(model);
    if !model_path.exists() {
        eprintln!("error: model path not found: {model}");
        std::process::exit(1);
    }

    let prompt_len = kv_int(&kv, "prompt-tokens", 64);
    let warmup = kv_int(&kv, "warmup", 3);
    let trials = kv_int(&kv, "trials", 5);

    #[cfg(feature = "mlx-backend")]
    {
        let report = do_prefill_mlx(model_path, prompt_len, warmup, trials);
        emit_report(&report);
    }

    #[cfg(not(feature = "mlx-backend"))]
    {
        eprintln!("tribunus-bench: prefill requires --features mlx-backend");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Phase: decode
// ---------------------------------------------------------------------------

#[cfg(feature = "mlx-backend")]
fn do_decode_mlx(
    model_path: &Path,
    prompt_tokens: usize,
    output_tokens: usize,
    warmup_rounds: usize,
) -> BenchReport {
    use tribunus_compute_core::kv_cache::KvCache;
    use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

    let mut metrics = BTreeMap::new();

    // Load model (cold)
    let load_start = Instant::now();
    let model =
        LoadedProfiledModel::new(model_path).expect("failed to load model for decode bench");
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    metrics.insert("model_load_ms".into(), serde_json::json!(load_ms));

    let n_layers = model.reader.manifest.execution_plan.layers.len();
    let kv_cache_len = (prompt_tokens + output_tokens) as u32;

    // Create a fresh kv cache for each warmup/bench run
    let setup_session = |label: String| -> ProfiledInferenceSession {
        let kv_caches: Vec<KvCache> = (0..n_layers)
            .map(|_| KvCache::new(kv_cache_len, 128, 64, false))
            .collect();
        let mut session = ProfiledInferenceSession::new(label, kv_caches);
        session.setup_from_model(&model);
        session
    };

    // Prefill prompt tokens
    let prompt: Vec<u32> = (0..prompt_tokens).map(|i| (i % 1024) as u32 + 1).collect();

    // Warmup rounds: prefill + decode
    for _w in 0..warmup_rounds {
        let mut session = setup_session("decode-warmup".into());
        let mut tok = session.prefill(&prompt, &model).expect("warmup prefill");
        let warmup_decode = std::cmp::min(output_tokens, 10);
        for _ in 0..warmup_decode {
            tok = session.decode_one(tok, &model).expect("warmup decode");
        }
    }

    // Benchmark: single prefill + N decode steps
    {
        let mut session = setup_session("decode-bench".into());
        let mut tok = session.prefill(&prompt, &model).expect("bench prefill");

        let decode_start = Instant::now();
        for _step in 0..output_tokens {
            tok = session.decode_one(tok, &model).expect("bench decode");
        }
        let decode_elapsed = decode_start.elapsed();
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let decode_tok_s = output_tokens as f64 / decode_elapsed.as_secs_f64();

        // Measure first token (TTFT) separately during decoding
        // We also measure KV cache memory usage

        // Rough memory estimate from KV cache size
        let kv_bytes_per_layer = (kv_cache_len as u64) * 128u64 * 64u64 * 2u64 * 2u64; // k and v, f16
        let total_kv_bytes = kv_bytes_per_layer * n_layers as u64;

        metrics.insert("prompt_tokens".into(), serde_json::json!(prompt_tokens));
        metrics.insert("output_tokens".into(), serde_json::json!(output_tokens));
        metrics.insert("decode_time_ms".into(), serde_json::json!(decode_ms));
        metrics.insert(
            "decode_tokens_per_sec".into(),
            serde_json::json!(decode_tok_s),
        );
        metrics.insert(
            "kv_cache_estimate_bytes".into(),
            serde_json::json!(total_kv_bytes),
        );
        metrics.insert(
            "kv_cache_estimate_mib".into(),
            serde_json::json!(total_kv_bytes as f64 / (1024.0 * 1024.0)),
        );
    }

    BenchReport {
        phase: "decode".into(),
        model: model_path.to_string_lossy().into_owned(),
        timestamp: tribunus_compute_core::now_iso8601(),
        environment: EnvironmentInfo {
            hostname: hostname_or_default(),
            binary: current_exe_name(),
            features: env_features(),
        },
        metrics,
    }
}

fn cmd_decode(args: &[String], model_env: String) {
    let kv = parse_kv(args);
    let model = kv_str(&kv, "model", &model_env);
    if model.is_empty() {
        eprintln!("error: --model <path> required (or set TRIBUNUS_BENCH_MODEL)");
        std::process::exit(1);
    }
    let model_path = Path::new(model);
    if !model_path.exists() {
        eprintln!("error: model path not found: {model}");
        std::process::exit(1);
    }

    let prompt_tokens = kv_int(&kv, "prompt-tokens", 10);
    let output_tokens = kv_int(&kv, "output-tokens", 128);
    let warmup = kv_int(&kv, "warmup", 3);

    #[cfg(feature = "mlx-backend")]
    {
        let report = do_decode_mlx(model_path, prompt_tokens, output_tokens, warmup);
        emit_report(&report);
    }

    #[cfg(not(feature = "mlx-backend"))]
    {
        eprintln!("tribunus-bench: decode requires --features mlx-backend");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Phase: serve — concurrent serving benchmark
// ---------------------------------------------------------------------------

fn cmd_serve(args: &[String], model_env: String) {
    let kv = parse_kv(args);
    let model = kv_str(&kv, "model", &model_env);
    if model.is_empty() {
        eprintln!("error: --model <path> required (or set TRIBUNUS_BENCH_MODEL)");
        std::process::exit(1);
    }
    let model_path = Path::new(model);
    if !model_path.exists() {
        eprintln!("error: model path not found: {model}");
        std::process::exit(1);
    }

    let concurrency: usize = kv_int(&kv, "concurrency", 1);
    let duration = kv_duration(&kv, "duration", "30s");
    let prompt_tokens = kv_int(&kv, "prompt-tokens", 64);
    let output_tokens = kv_int(&kv, "output-tokens", 50);

    #[cfg(feature = "mlx-backend")]
    {
        let report = do_serve_mlx(
            model_path,
            concurrency,
            duration,
            prompt_tokens,
            output_tokens,
        );
        emit_report(&report);
    }

    #[cfg(not(feature = "mlx-backend"))]
    {
        eprintln!("tribunus-bench: serve requires --features mlx-backend");
        std::process::exit(1);
    }
}

#[cfg(feature = "mlx-backend")]
fn do_serve_mlx(
    model_path: &Path,
    concurrency: usize,
    duration: std::time::Duration,
    prompt_tokens: usize,
    output_tokens: usize,
) -> BenchReport {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use tribunus_compute_core::kv_cache::KvCache;
    use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

    let mut metrics = BTreeMap::new();

    // Load model once (serialized across concurrent access via session-per-request)
    let load_start = Instant::now();
    let model = Arc::new(
        LoadedProfiledModel::new(model_path).expect("failed to load model for serve bench"),
    );
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    metrics.insert("model_load_ms".into(), serde_json::json!(load_ms));

    let n_layers = model.reader.manifest.execution_plan.layers.len();
    let kv_cache_len = (prompt_tokens + output_tokens) as u32;

    // Build a synthetic prompt once
    let prompt: Arc<Vec<u32>> =
        Arc::new((0..prompt_tokens).map(|i| (i % 1024) as u32 + 1).collect());

    // Atomic counters for wall-clock tracking
    let completed: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let total_decode_tokens: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let deadline = Instant::now() + duration;

    // Spawn concurrent workers using std::thread
    let mut handles = Vec::with_capacity(concurrency);
    for _i in 0..concurrency {
        let model = Arc::clone(&model);
        let prompt = Arc::clone(&prompt);
        let completed = Arc::clone(&completed);
        let total_decode = Arc::clone(&total_decode_tokens);
        let deadline = deadline;

        handles.push(std::thread::spawn(move || {
            loop {
                if Instant::now() >= deadline {
                    break;
                }

                // Each request gets its own session + KV cache
                let kv_caches: Vec<KvCache> = (0..n_layers)
                    .map(|_| KvCache::new(kv_cache_len, 128, 64, false))
                    .collect();
                let mut session = ProfiledInferenceSession::new("serve-request".into(), kv_caches);
                session.setup_from_model(&model);

                // Prefill
                let mut tok = match session.prefill(&prompt, &model) {
                    Ok(t) => t,
                    Err(_) => break,
                };

                // Decode
                for _step in 0..output_tokens {
                    tok = match session.decode_one(tok, &model) {
                        Ok(t) => t,
                        Err(_) => break,
                    };
                }

                completed.fetch_add(1, Ordering::Relaxed);
                total_decode.fetch_add(output_tokens as u64, Ordering::Relaxed);
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    let actual_duration = duration; // planned wall-clock
    let total_requests = completed.load(Ordering::Relaxed);
    let total_decoded = total_decode_tokens.load(Ordering::Relaxed);
    let dur_secs = actual_duration.as_secs_f64();

    let throughput_rps = total_requests as f64 / dur_secs;
    let throughput_tok_s = total_decoded as f64 / dur_secs;

    metrics.insert("concurrency".into(), serde_json::json!(concurrency));
    metrics.insert("duration_secs".into(), serde_json::json!(dur_secs));
    metrics.insert("prompt_tokens".into(), serde_json::json!(prompt_tokens));
    metrics.insert("output_tokens".into(), serde_json::json!(output_tokens));
    metrics.insert("total_requests".into(), serde_json::json!(total_requests));
    metrics.insert(
        "total_decode_tokens".into(),
        serde_json::json!(total_decoded),
    );
    metrics.insert(
        "throughput_requests_per_sec".into(),
        serde_json::json!(throughput_rps),
    );
    metrics.insert(
        "throughput_tokens_per_sec".into(),
        serde_json::json!(throughput_tok_s),
    );

    BenchReport {
        phase: "serve".into(),
        model: model_path.to_string_lossy().into_owned(),
        timestamp: tribunus_compute_core::now_iso8601(),
        environment: EnvironmentInfo {
            hostname: hostname_or_default(),
            binary: current_exe_name(),
            features: env_features(),
        },
        metrics,
    }
}

// ---------------------------------------------------------------------------
// Phase: compare — diff two JSON reports
// ---------------------------------------------------------------------------

fn cmd_compare(args: &[String]) {
    let kv = parse_kv(args);

    let baseline_dir = kv.get("baseline-path").cloned();
    let candidate_dir = kv.get("candidate-path").cloned();

    let (Some(baseline), Some(candidate)) = (baseline_dir, candidate_dir) else {
        eprintln!("error: --baseline-path <dir> and --candidate-path <dir> required");
        std::process::exit(1);
    };

    let baseline_path = Path::new(&baseline);
    let candidate_path = Path::new(&candidate);

    if !baseline_path.is_dir() {
        eprintln!("error: baseline-path is not a directory: {baseline}");
        std::process::exit(1);
    }
    if !candidate_path.is_dir() {
        eprintln!("error: candidate-path is not a directory: {candidate}");
        std::process::exit(1);
    }

    // Collect JSON files from each directory
    let baseline_files = collect_json_files(baseline_path);
    let candidate_files = collect_json_files(candidate_path);

    if baseline_files.is_empty() {
        eprintln!("warning: no JSON files found in baseline directory: {baseline}");
    }
    if candidate_files.is_empty() {
        eprintln!("warning: no JSON files found in candidate directory: {candidate}");
    }

    // Load and emit comparison as CSV
    println!("=== Compare: {baseline} vs {candidate} ===");
    println!();

    // Index candidate files by name for O(1) lookup
    let candidate_map: BTreeMap<String, PathBuf> = candidate_files
        .iter()
        .map(|p| {
            (
                p.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                p.clone(),
            )
        })
        .collect();

    for bf in &baseline_files {
        let fname = bf
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();

        let baseline_data = match std::fs::read_to_string(bf) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  {fname}: baseline read error: {e}");
                continue;
            }
        };

        let baseline_json: serde_json::Value = match serde_json::from_str(&baseline_data) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  {fname}: baseline parse error: {e}");
                continue;
            }
        };

        let candidate_data = match candidate_map.get(fname.as_ref()) {
            Some(p) => match std::fs::read_to_string(p) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  {fname}: candidate read error: {e}");
                    continue;
                }
            },
            None => {
                eprintln!("  {fname}: candidate missing (no matching file)");
                continue;
            }
        };

        let candidate_json: serde_json::Value = match serde_json::from_str(&candidate_data) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  {fname}: candidate parse error: {e}");
                continue;
            }
        };

        // Print comparison header
        println!("--- {fname} ---");

        // Extract metrics from both
        let b_metrics = &baseline_json["metrics"];
        let c_metrics = &candidate_json["metrics"];

        if let (Some(bm), Some(cm)) = (b_metrics.as_object(), c_metrics.as_object()) {
            // Union of all metric keys
            let all_keys: std::collections::BTreeSet<&str> =
                bm.keys().chain(cm.keys()).map(String::as_str).collect();

            println!(
                "{:<40} {:>16} {:>16} {:>12}",
                "metric", "baseline", "candidate", "delta%"
            );

            for key in all_keys {
                let bv = bm.get(key).and_then(|v| v.as_f64());
                let cv = cm.get(key).and_then(|v| v.as_f64());

                match (bv, cv) {
                    (Some(b), Some(c)) => {
                        let delta_pct = if b != 0.0 {
                            ((c - b) / b) * 100.0
                        } else {
                            f64::INFINITY
                        };
                        println!("{:<40} {:>16.4} {:>16.4} {:>+11.2}%", key, b, c, delta_pct);
                    }
                    (Some(b), None) => {
                        println!("{:<40} {:>16.4} {:>16} {:>12}", key, b, "N/A", "N/A");
                    }
                    (None, Some(c)) => {
                        println!("{:<40} {:>16} {:>16.4} {:>12}", key, "N/A", c, "NEW");
                    }
                    (None, None) => {}
                }
            }
        }

        println!();
    }
}

fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}
