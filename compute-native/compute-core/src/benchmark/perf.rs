//! Performance benchmarks: throughput, TTFT, prompt-length sweep, and latency
//! distribution measurement against a [`ProfiledInferenceSession`].
//!
//! These benchmarks delegate to the harness methods in [`super::BenchmarkHarness`].
//! Standalone convenience functions are provided for callers that want to run
//! a specific benchmark without constructing a full harness.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

use super::{BenchmarkConfig, BenchmarkResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn tokenize(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

fn build_prompt(prompt_len: u32) -> String {
    let base = "The quick brown fox jumps over the lazy dog. ";
    let repeat_count = (prompt_len as usize / base.len()).max(1);
    let mut s: String = (0..repeat_count).map(|_| base).collect();
    s.truncate(prompt_len as usize);
    s
}

fn hardware_label() -> String {
    let hw = crate::scheduling::HardwareConfig::detect();
    format!("{}gpu {} cores", if hw.gpu_cores > 0 { "" } else { "" }, hw.gpu_cores)
}

/// Run a full warmup cycle (prefill + decode).
fn warmup(
    session: &Arc<Mutex<ProfiledInferenceSession>>,
    model: &Arc<LoadedProfiledModel>,
    prompt: &str,
    max_tokens: u32,
) -> Result<(), String> {
    let prompt_ids = tokenize(prompt);
    let mut sess = session.try_lock().map_err(|e| e.to_string())?;
    let first = sess
        .prefill(&prompt_ids, model)
        .map_err(|e| format!("warmup prefill: {:?}", e))?;
    let mut current = first;
    for _ in 1..max_tokens {
        match sess.decode_one(current, model) {
            Ok(tok) => current = tok,
            Err(e) => return Err(format!("warmup decode: {:?}", e)),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Standalone top-level benchmark functions
// ---------------------------------------------------------------------------

/// Measure token-generation throughput for a single prompt length.
///
/// Generates `max_tokens` after prefill and returns tokens/s plus latency
/// breakdown.
pub fn measure_throughput(
    session: &Arc<Mutex<ProfiledInferenceSession>>,
    model: &Arc<LoadedProfiledModel>,
    prompt_len: u32,
    max_tokens: u32,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult, String> {
    let prompt = build_prompt(prompt_len);
    let prompt_ids = tokenize(&prompt);

    // Warmup
    for _ in 0..config.warmup_runs {
        warmup(session, model, &prompt, 16)?;
    }

    // Timed run
    let mut latencies: Vec<f64> = Vec::with_capacity(max_tokens as usize);
    let start = Instant::now();

    let mut sess = session.try_lock().map_err(|e| e.to_string())?;

    // Prefill — TTFT
    let ttft_start = Instant::now();
    let first_token = sess
        .prefill(&prompt_ids, model)
        .map_err(|e| format!("prefill: {:?}", e))?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    latencies.push(ttft_ms);

    // Decode loop
    let mut current = first_token;
    for _ in 1..max_tokens {
        let step_start = Instant::now();
        match sess.decode_one(current, model) {
            Ok(next) => {
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                latencies.push(step_ms);
                current = next;
            }
            Err(e) => return Err(format!("decode: {:?}", e)),
        }
    }

    let total_elapsed = start.elapsed().as_secs_f64();
    let tokens_generated = max_tokens as u64;

    // Compute percentiles
    let mut sorted = latencies.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    let p50 = if len > 0 { sorted[len / 2] } else { 0.0 };
    let p95 = if len > 0 {
        sorted[(len as f64 * 0.95).ceil() as usize - 1].min(*sorted.last().unwrap())
    } else {
        0.0
    };
    let p99 = if len > 0 {
        sorted[(len as f64 * 0.99).ceil() as usize - 1].min(*sorted.last().unwrap())
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        test_name: format!("throughput/prompt_len={prompt_len}"),
        tokens_generated,
        prompt_tokens: prompt_len as u64,
        total_time_s: total_elapsed,
        tokens_per_second: if total_elapsed > 0.0 {
            tokens_generated as f64 / total_elapsed
        } else {
            0.0
        },
        time_to_first_token_ms: ttft_ms,
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        latency_p99_ms: p99,
        model: config.model_name.clone(),
        quantization: config.quantization.clone(),
        hardware: hardware_label(),
    })
}

/// Measure time-to-first-token for a given prompt length.
pub fn measure_ttft(
    session: &Arc<Mutex<ProfiledInferenceSession>>,
    model: &Arc<LoadedProfiledModel>,
    prompt_len: u32,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult, String> {
    let prompt = build_prompt(prompt_len);
    let prompt_ids = tokenize(&prompt);

    // Warmup
    for _ in 0..config.warmup_runs {
        warmup(session, model, &prompt, 1)?;
    }

    // Measure TTFT over 3 runs
    let mut ttft_values = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let mut sess = session.try_lock().map_err(|e| e.to_string())?;
        let _first = sess
            .prefill(&prompt_ids, model)
            .map_err(|e| format!("ttft prefill: {:?}", e))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        ttft_values.push(elapsed);
    }

    let avg_ttft = ttft_values.iter().sum::<f64>() / ttft_values.len() as f64;

    Ok(BenchmarkResult {
        test_name: format!("ttft/prompt_len={prompt_len}"),
        tokens_generated: 1,
        prompt_tokens: prompt_len as u64,
        total_time_s: avg_ttft / 1000.0,
        tokens_per_second: 0.0,
        time_to_first_token_ms: avg_ttft,
        latency_p50_ms: avg_ttft,
        latency_p95_ms: avg_ttft,
        latency_p99_ms: avg_ttft,
        model: config.model_name.clone(),
        quantization: config.quantization.clone(),
        hardware: hardware_label(),
    })
}

/// Measure per-token latency distribution (P50/P95/P99) by timing each decode
/// step in isolation.
pub fn measure_latency_distribution(
    session: &Arc<Mutex<ProfiledInferenceSession>>,
    model: &Arc<LoadedProfiledModel>,
    prompt_len: u32,
    max_tokens: u32,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult, String> {
    let prompt = build_prompt(prompt_len);
    let prompt_ids = tokenize(&prompt);

    // Warmup
    for _ in 0..config.warmup_runs {
        warmup(session, model, &prompt, 16)?;
    }

    // Collect per-token latencies
    let mut latencies: Vec<f64> = Vec::with_capacity(max_tokens as usize);
    let start = Instant::now();

    let mut sess = session.try_lock().map_err(|e| e.to_string())?;

    let ttft_start = Instant::now();
    let first = sess
        .prefill(&prompt_ids, model)
        .map_err(|e| format!("latency prefill: {:?}", e))?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    latencies.push(ttft_ms);

    let mut current = first;
    for _ in 1..max_tokens {
        let step_start = Instant::now();
        match sess.decode_one(current, model) {
            Ok(next) => {
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                latencies.push(step_ms);
                current = next;
            }
            Err(e) => return Err(format!("latency decode: {:?}", e)),
        }
    }

    let total_time_s = start.elapsed().as_secs_f64();
    let tokens_generated = max_tokens as u64;

    // Compute percentiles
    let mut sorted = latencies.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    let p50 = if len > 0 { sorted[len / 2] } else { 0.0 };
    let p95 = if len > 0 {
        sorted[(len as f64 * 0.95).ceil() as usize - 1].min(*sorted.last().unwrap())
    } else {
        0.0
    };
    let p99 = if len > 0 {
        sorted[(len as f64 * 0.99).ceil() as usize - 1].min(*sorted.last().unwrap())
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        test_name: format!("latency_distribution/prompt_len={prompt_len}/max_tokens={max_tokens}"),
        tokens_generated,
        prompt_tokens: prompt_len as u64,
        total_time_s,
        tokens_per_second: if total_time_s > 0.0 {
            tokens_generated as f64 / total_time_s
        } else {
            0.0
        },
        time_to_first_token_ms: ttft_ms,
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        latency_p99_ms: p99,
        model: config.model_name.clone(),
        quantization: config.quantization.clone(),
        hardware: hardware_label(),
    })
}

/// Sweep across multiple prompt lengths and return throughput for each.
pub fn sweep_prompt_length(
    session: &Arc<Mutex<ProfiledInferenceSession>>,
    model: &Arc<LoadedProfiledModel>,
    prompt_lengths: &[u32],
    max_tokens: u32,
    config: &BenchmarkConfig,
) -> Result<Vec<BenchmarkResult>, String> {
    let mut results = Vec::new();
    for &pl in prompt_lengths {
        let r = measure_throughput(session, model, pl, max_tokens, config)?;
        results.push(r);
    }
    Ok(results)
}
