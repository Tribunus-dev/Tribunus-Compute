//! Benchmark harness for running performance and accuracy benchmarks against
//! an inference session.
//!
//! Performance benchmarks measure tokens/s, TTFT, and latency distribution.
//! Accuracy benchmarks evaluate model quality using small built-in test sets
//! (MMLU, GSM8K, HellaSwag).
//!
//! # Organization
//!
//! * [`mod.rs`](self) — shared types: [`BenchmarkHarness`], [`BenchmarkResult`],
//!   [`BenchmarkConfig`], and the core [`run_inference_timed`] helper.
//! * [`perf`] — throughput, TTFT, prompt-length sweep, latency distribution.
//! * [`intel`] — accuracy/intelligence benchmarks (MMLU, GSM8K, etc.).

pub mod perf;
pub mod intel;
pub mod admission;

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Results from a single benchmark test.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub tokens_generated: u64,
    pub prompt_tokens: u64,
    pub total_time_s: f64,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub model: String,
    pub quantization: String,
    pub hardware: String,
}

impl BenchmarkResult {
    fn new(
        test_name: impl Into<String>,
        prompt_tokens: u64,
        tokens_generated: u64,
        total_time_s: f64,
        token_latencies: &[f64],
        ttft_ms: f64,
        model: impl Into<String>,
        quantization: impl Into<String>,
        hardware: impl Into<String>,
    ) -> Self {
        let mut sorted = token_latencies.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let tokens_per_second = if total_time_s > 0.0 {
            tokens_generated as f64 / total_time_s
        } else {
            0.0
        };

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

        Self {
            test_name: test_name.into(),
            tokens_generated,
            prompt_tokens,
            total_time_s,
            tokens_per_second,
            time_to_first_token_ms: ttft_ms,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
            model: model.into(),
            quantization: quantization.into(),
            hardware: hardware.into(),
        }
    }
}

/// Configuration for benchmarks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkConfig {
    /// Prompt token lengths to test (e.g. [1024, 4096, 16384]).
    pub prompt_lengths: Vec<u32>,
    /// Batch sizes for continuous batching throughput tests.
    pub batch_sizes: Vec<u32>,
    /// Number of new tokens to generate per test.
    pub max_tokens: u32,
    /// Warmup iterations before measurement.
    pub warmup_runs: u32,
    /// Override model identifier for result metadata.
    pub model_name: String,
    /// Override quantization label for result metadata.
    pub quantization: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            prompt_lengths: vec![1024, 4096],
            batch_sizes: vec![1, 4],
            max_tokens: 128,
            warmup_runs: 2,
            model_name: String::from("unknown"),
            quantization: String::from("unknown"),
        }
    }
}

/// Benchmark harness: runs tests against a [`ProfiledInferenceSession`].
pub struct BenchmarkHarness {
    pub session: Arc<Mutex<ProfiledInferenceSession>>,
    pub model: Arc<LoadedProfiledModel>,
    pub config: BenchmarkConfig,
    last_results: Vec<BenchmarkResult>,
}

impl BenchmarkHarness {
    /// Create a new harness from an existing session and model.
    pub fn new(
        session: Arc<Mutex<ProfiledInferenceSession>>,
        model: Arc<LoadedProfiledModel>,
    ) -> Self {
        let config = BenchmarkConfig::default();
        Self {
            session,
            model,
            config,
            last_results: Vec::new(),
        }
    }

    /// Run all configured benchmarks (perf + intel).
    pub fn run_all(&mut self) -> Result<Vec<BenchmarkResult>, String> {
        let mut results = Vec::new();

        // Performance benchmarks
        for &prompt_len in &self.config.prompt_lengths {
            let r = self.measure_throughput(prompt_len, self.config.max_tokens)?;
            results.push(r);
        }

        // Latency distribution
        if let Some(pl) = self.config.prompt_lengths.first() {
            let lat_result =
                self.measure_latency_distribution(*pl, self.config.max_tokens)?;
            results.push(lat_result);
        }

        // Prompt length sweep (TTFT)
        for &pl in &self.config.prompt_lengths {
            let ttft = self.measure_ttft(pl)?;
            results.push(ttft);
        }

        // Accuracy benchmarks (scores stored in tokens_per_second field as a
        // convenience — the result is a fraction in [0, 1]).
        let mmlu_score = intel::run_mmlu(self)?;
        results.push(BenchmarkResult {
            test_name: "mmlu_accuracy".into(),
            tokens_generated: 0,
            prompt_tokens: 0,
            total_time_s: 0.0,
            tokens_per_second: mmlu_score,
            time_to_first_token_ms: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            model: self.config.model_name.clone(),
            quantization: self.config.quantization.clone(),
            hardware: hardware_label(),
        });

        let gsm8k_score = intel::run_gsm8k(self)?;
        results.push(BenchmarkResult {
            test_name: "gsm8k_accuracy".into(),
            tokens_generated: 0,
            prompt_tokens: 0,
            total_time_s: 0.0,
            tokens_per_second: gsm8k_score,
            time_to_first_token_ms: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            model: self.config.model_name.clone(),
            quantization: self.config.quantization.clone(),
            hardware: hardware_label(),
        });

        let hs_score = intel::run_hellaswag(self)?;
        results.push(BenchmarkResult {
            test_name: "hellaswag_accuracy".into(),
            tokens_generated: 0,
            prompt_tokens: 0,
            total_time_s: 0.0,
            tokens_per_second: hs_score,
            time_to_first_token_ms: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            model: self.config.model_name.clone(),
            quantization: self.config.quantization.clone(),
            hardware: hardware_label(),
        });

        self.last_results = results.clone();
        Ok(results)
    }

    /// Return the most recently collected results.
    pub fn last_results(&self) -> &[BenchmarkResult] {
        &self.last_results
    }

    /// Tokenize prompt text into token IDs (byte-level tokenization matching
    /// the server's `run_inference`).
    fn tokenize(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    // ── Performance helpers ────────────────────────────────────────────────

    /// Build a dummy prompt of approximately `prompt_len` bytes.
    fn build_prompt(&self, prompt_len: u32) -> String {
        let base = "The quick brown fox jumps over the lazy dog. ";
        let repeat_count = (prompt_len as usize / base.len()).max(1);
        let mut s: String = (0..repeat_count).map(|_| base).collect();
        s.truncate(prompt_len as usize);
        s
    }

    /// Run a warmup iteration.
    fn warmup(&self, prompt: &str, max_tokens: u32) -> Result<(), String> {
        let prompt_ids = self.tokenize(prompt);
        let mut sess = self.session.try_lock().map_err(|e| e.to_string())?;
        let first = sess
            .prefill(&prompt_ids, &self.model)
            .map_err(|e| format!("warmup prefill: {:?}", e))?;
        let mut current = first;
        for _ in 1..max_tokens {
            match sess.decode_one(current, &self.model) {
                Ok(tok) => current = tok,
                Err(e) => {
                    return Err(format!("warmup decode: {:?}", e));
                }
            }
        }
        Ok(())
    }

    /// Measure throughput: generate `max_tokens` and report tokens/s with
    /// per-token latency distribution.
    pub fn measure_throughput(
        &self,
        prompt_len: u32,
        max_tokens: u32,
    ) -> Result<BenchmarkResult, String> {
        let prompt = self.build_prompt(prompt_len);
        let prompt_ids = self.tokenize(&prompt);

        // Warmup
        for _ in 0..self.config.warmup_runs {
            self.warmup(&prompt, 16)?;
        }

        // Timed run with per-token latency tracking
        let mut latencies: Vec<f64> = Vec::with_capacity(max_tokens as usize);

        let start = Instant::now();

        let mut sess = self.session.try_lock().map_err(|e| e.to_string())?;

        // Prefill — measure TTFT
        let ttft_start = Instant::now();
        let first_token = sess
            .prefill(&prompt_ids, &self.model)
            .map_err(|e| format!("prefill: {:?}", e))?;
        let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(ttft_ms);

        // Decode loop
        let mut current = first_token;
        for _ in 1..max_tokens {
            let step_start = Instant::now();
            match sess.decode_one(current, &self.model) {
                Ok(next) => {
                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(step_ms);
                    current = next;
                }
                Err(e) => {
                    return Err(format!("decode: {:?}", e));
                }
            }
        }

        let total_elapsed = start.elapsed().as_secs_f64();
        let tokens_generated = max_tokens as u64;

        Ok(BenchmarkResult::new(
            format!("throughput/prompt_len={prompt_len}"),
            prompt_len as u64,
            tokens_generated,
            total_elapsed,
            &latencies,
            ttft_ms,
            &self.config.model_name,
            &self.config.quantization,
            hardware_label(),
        ))
    }

    /// Measure time-to-first-token.
    pub fn measure_ttft(&self, prompt_len: u32) -> Result<BenchmarkResult, String> {
        let prompt = self.build_prompt(prompt_len);
        let prompt_ids = self.tokenize(&prompt);

        // Warmup
        for _ in 0..self.config.warmup_runs {
            self.warmup(&prompt, 1)?;
        }

        // Measure TTFT over 3 runs, take average
        let mut ttft_values = Vec::new();
        for _ in 0..3 {
            let ttft_start = Instant::now();
            let mut sess = self.session.try_lock().map_err(|e| e.to_string())?;
            let _first = sess
                .prefill(&prompt_ids, &self.model)
                .map_err(|e| format!("ttft prefill: {:?}", e))?;
            let elapsed = ttft_start.elapsed().as_secs_f64() * 1000.0;
            ttft_values.push(elapsed);
        }

        let avg_ttft =
            ttft_values.iter().sum::<f64>() / ttft_values.len() as f64;

        Ok(BenchmarkResult::new(
            format!("ttft/prompt_len={prompt_len}"),
            prompt_len as u64,
            1,
            avg_ttft / 1000.0,
            &[avg_ttft],
            avg_ttft,
            &self.config.model_name,
            &self.config.quantization,
            hardware_label(),
        ))
    }

    /// Measure per-token latency distribution (P50/P95/P99).
    pub fn measure_latency_distribution(
        &self,
        prompt_len: u32,
        max_tokens: u32,
    ) -> Result<BenchmarkResult, String> {
        let prompt = self.build_prompt(prompt_len);
        let prompt_ids = self.tokenize(&prompt);

        // Warmup
        for _ in 0..self.config.warmup_runs {
            self.warmup(&prompt, 16)?;
        }

        // Collect per-token latencies
        let mut latencies: Vec<f64> = Vec::with_capacity(max_tokens as usize);
        let start = Instant::now();

        let mut sess = self.session.try_lock().map_err(|e| e.to_string())?;

        // Prefill
        let ttft_start = Instant::now();
        let first = sess
            .prefill(&prompt_ids, &self.model)
            .map_err(|e| format!("latency prefill: {:?}", e))?;
        let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(ttft_ms);

        let mut current = first;
        for _ in 1..max_tokens {
            let step_start = Instant::now();
            match sess.decode_one(current, &self.model) {
                Ok(next) => {
                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(step_ms);
                    current = next;
                }
                Err(e) => {
                    return Err(format!("latency decode: {:?}", e));
                }
            }
        }

        let total_time_s = start.elapsed().as_secs_f64();
        let tokens_generated = max_tokens as u64;

        Ok(BenchmarkResult::new(
            format!("latency_distribution/prompt_len={prompt_len}/max_tokens={max_tokens}"),
            prompt_len as u64,
            tokens_generated,
            total_time_s,
            &latencies,
            ttft_ms,
            &self.config.model_name,
            &self.config.quantization,
            hardware_label(),
        ))
    }

    /// Sweep prompt lengths and measure throughput for each.
    pub fn sweep_prompt_length(
        &self,
        prompt_lengths: &[u32],
        max_tokens: u32,
    ) -> Result<Vec<BenchmarkResult>, String> {
        let mut results = Vec::new();
        for &pl in prompt_lengths {
            let r = self.measure_throughput(pl, max_tokens)?;
            results.push(r);
        }
        Ok(results)
    }

    /// Run a complete inference cycle (prefill + decode) and return the
    /// generated text.  Used by accuracy benchmarks.
    pub fn run_inference_for_text(
        &self,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String, String> {
        let prompt_ids = self.tokenize(prompt);
        let mut sess = self.session.try_lock().map_err(|e| e.to_string())?;

        let first = sess
            .prefill(&prompt_ids, &self.model)
            .map_err(|e| format!("inference prefill: {:?}", e))?;

        let mut generated = vec![first];
        let mut current = first;
        for _ in 1..max_tokens {
            match sess.decode_one(current, &self.model) {
                Ok(next) => {
                    if next == 0 {
                        // EOS sentinel
                        break;
                    }
                    generated.push(next);
                    current = next;
                }
                Err(e) => {
                    return Err(format!("inference decode: {:?}", e));
                }
            }
        }

        // Detokenize: filter to printable ASCII range.
        let text: String = generated
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();
        Ok(text)
    }
}

/// Detect a short hardware label for benchmark metadata.
fn hardware_label() -> String {
    let hw = crate::scheduling::HardwareConfig::detect();
    format!("{}gpu {} cores", if hw.gpu_cores > 0 { "" } else { "" }, hw.gpu_cores)
}
