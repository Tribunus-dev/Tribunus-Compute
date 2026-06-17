//! Production telemetry and observability for the inference engine.
//!
//! Aggregates per-token latency, cache hit rates, KV compression ratios,
//! grammar success rates, memory usage, and more.  Exposes metrics in
//! Prometheus text format via the `/metrics` endpoint.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// AtomicF64 — lock‑free f64 via bit‑casting
// ---------------------------------------------------------------------------

/// Thin wrapper around `AtomicU64` that stores `f64` values via `to_bits` /
/// `from_bits`.  Provides `load`, `store`, `swap`, and `fetch_max`.
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    pub const fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    pub fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(self.0.load(order))
    }

    pub fn store(&self, val: f64, order: Ordering) {
        self.0.store(val.to_bits(), order);
    }

    pub fn swap(&self, val: f64, order: Ordering) -> f64 {
        f64::from_bits(self.0.swap(val.to_bits(), order))
    }

    /// Store `val` if it exceeds the current value.  Returns the previous
    /// value.
    pub fn fetch_max(&self, val: f64, order: Ordering) -> f64 {
        loop {
            let cur = self.0.load(Ordering::Relaxed);
            let cur_f = f64::from_bits(cur);
            if val <= cur_f {
                return cur_f;
            }
            if self
                .0
                .compare_exchange_weak(cur, val.to_bits(), order, Ordering::Relaxed)
                .is_ok()
            {
                return cur_f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CacheKind
// ---------------------------------------------------------------------------

/// Which cache layer recorded a hit or miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    Model,
    Prefix,
    KVCache,
}

// ---------------------------------------------------------------------------
// Histogram — bucketed, Prometheus‑compatible
// ---------------------------------------------------------------------------

static DEFAULT_LATENCY_BUCKETS_MS: &[f64] = &[
    1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0,
];

static DEFAULT_TTFB_BUCKETS_MS: &[f64] = &[
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0, 30_000.0,
];

static DEFAULT_TPUT_BUCKETS: &[f64] = &[
    1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0,
];

/// A fixed‑bucket histogram with atomic per‑bucket counters.
///
/// All values below the smallest bucket are counted in bucket 0; values
/// above the largest bucket are counted in the implicit `+Inf` bucket
/// (not stored — they go into the last bucket and the total count).
pub struct Histogram {
    /// Bucket upper bounds (sorted ascending).  The final element is the
    /// largest explicit bound; the implicit `+Inf` bucket is derived from
    /// `self.count - cumulative(self.buckets.len() - 1)`.
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with the given bucket bounds.
    /// Bounds MUST be sorted ascending and non‑empty.
    pub fn new(buckets: &[f64]) -> Self {
        assert!(!buckets.is_empty(), "Histogram needs at least one bucket");
        Self {
            buckets: buckets.to_vec(),
            counts: (0..buckets.len()).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Create a latency histogram with sensible default buckets (ms).
    pub fn latency_ms() -> Self {
        Self::new(DEFAULT_LATENCY_BUCKETS_MS)
    }

    /// Create a time‑to‑first‑token histogram (ms).
    pub fn time_to_first_token_ms() -> Self {
        Self::new(DEFAULT_TTFB_BUCKETS_MS)
    }

    /// Create a throughput histogram (tok/s).
    pub fn tokens_per_second() -> Self {
        Self::new(DEFAULT_TPUT_BUCKETS)
    }

    /// Record a single observation.  `value` is in the histogram's native
    /// unit (e.g. milliseconds for latency, tokens/second for throughput).
    pub fn observe(&self, value: f64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum
            .fetch_add((value * 1000.0) as u64, Ordering::Relaxed);
        // Find the bucket index: the smallest upper bound ≥ value.
        // We scan from the highest bucket down for a tight loop since the
        // bucket list is small (≤ 15).  Linear scan is faster than binary
        // search at this size.
        for (i, bound) in self.buckets.iter().enumerate() {
            if value <= *bound {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Falls into the implicit +Inf bucket — no dedicated counter;
        // the total count captures it.
    }

    // ── percentile helpers ────────────────────────────────────────────

    /// Cumulative count up to (and including) bucket index `idx`.
    fn cumulative(&self, idx: usize) -> u64 {
        let mut total: u64 = 0;
        for c in &self.counts[..=idx] {
            total += c.load(Ordering::Relaxed);
        }
        total
    }

    /// Total observation count.
    pub fn total_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Total sum of all observations (in native unit × 1000).
    pub fn total_sum_raw(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Estimate the P**p**th percentile value by linear interpolation
    /// within the bucket that contains the rank.  Returns `0.0` when
    /// no data has been recorded.
    pub fn percentile(&self, p: f64) -> f64 {
        let n = self.count.load(Ordering::Relaxed);
        if n == 0 {
            return 0.0;
        }
        let rank = (p / 100.0) * n as f64;
        let mut cum: u64 = 0;
        for (i, bound) in self.buckets.iter().enumerate() {
            let bucket_count = self.counts[i].load(Ordering::Relaxed);
            cum += bucket_count;
            if rank <= cum as f64 {
                if bucket_count == 0 {
                    return *bound;
                }
                let prev_bound = if i == 0 { 0.0 } else { self.buckets[i - 1] };
                let prev_cum = cum - bucket_count;
                let frac = (rank - prev_cum as f64) / bucket_count as f64;
                return prev_bound + frac * (bound - prev_bound);
            }
        }
        // Above the last explicit bucket — extrapolate.
        let last = *self.buckets.last().unwrap();
        last * (rank / cum as f64)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        for c in &self.counts {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// Append Prometheus histogram text to `buf`.
    pub fn write_prometheus(&self, buf: &mut String, name: &str) {
        let _ = writeln!(buf, "{name}_count {}", self.count.load(Ordering::Relaxed));
        let sum_raw = self.sum.load(Ordering::Relaxed);
        let sum_sec = if sum_raw == 0 {
            0.0
        } else {
            sum_raw as f64 / 1000.0
        };
        let _ = writeln!(buf, "{name}_sum {sum_sec}");
        // Bucket lines (excluding +Inf, which Prometheus infers).
        for (i, bound) in self.buckets.iter().enumerate() {
            let cumulative = if i == 0 {
                self.counts[0].load(Ordering::Relaxed)
            } else {
                self.cumulative(i)
            };
            // Use integer representation to avoid floating‑point formatting
            // noise.  Show bound as integer when it is one.
            if *bound == bound.trunc() {
                let _ = writeln!(
                    buf,
                    "{name}_bucket{{le=\"{}\"}} {cumulative}",
                    *bound as u64
                );
            } else {
                let _ = writeln!(buf, "{name}_bucket{{le=\"{bound}\"}} {cumulative}");
            }
        }
        // +Inf bucket
        let total = self.count.load(Ordering::Relaxed);
        let _ = writeln!(buf, "{name}_bucket{{le=\"+Inf\"}} {total}");
    }
}

// ---------------------------------------------------------------------------
// InferenceTelemetry
// ---------------------------------------------------------------------------

/// Real‑time metrics aggregator for the inference engine.
///
/// All fields are atomics or histograms, safe to share across threads via
/// `&self`.  Call `record_*` methods from any hot path.
pub struct InferenceTelemetry {
    // ── Token generation ───────────────────────────────────────────────
    pub tokens_generated: AtomicU64,
    pub tokens_per_second: Histogram,
    pub latency_ms: Histogram,
    pub time_to_first_token_ms: Histogram,

    // ── Cache ──────────────────────────────────────────────────────────
    pub model_cache_hits: AtomicU64,
    pub model_cache_misses: AtomicU64,
    pub prefix_cache_hits: AtomicU64,
    pub prefix_cache_misses: AtomicU64,
    pub kv_compression_ratio: AtomicF64,

    // ── Backend invocations ────────────────────────────────────────────
    pub mlx_invocations: AtomicU64,
    pub ane_invocations: AtomicU64,
    pub accelerate_invocations: AtomicU64,
    pub grammar_accepted: AtomicU64,
    pub grammar_rejected: AtomicU64,

    // ── Memory ─────────────────────────────────────────────────────────
    pub peak_memory_bytes: AtomicU64,
    pub current_memory_bytes: AtomicU64,

    // ── Speculative decoding ───────────────────────────────────────────
    pub draft_acceptance_rate: AtomicF64,
    pub spec_tokens_saved: AtomicU64,

    // ── Anomaly ────────────────────────────────────────────────────────
    pub anomalies_detected: AtomicU64,
    pub patches_applied: AtomicU64,
}

impl InferenceTelemetry {
    pub fn new() -> Self {
        Self {
            tokens_generated: AtomicU64::new(0),
            tokens_per_second: Histogram::tokens_per_second(),
            latency_ms: Histogram::latency_ms(),
            time_to_first_token_ms: Histogram::time_to_first_token_ms(),

            model_cache_hits: AtomicU64::new(0),
            model_cache_misses: AtomicU64::new(0),
            prefix_cache_hits: AtomicU64::new(0),
            prefix_cache_misses: AtomicU64::new(0),
            kv_compression_ratio: AtomicF64::new(1.0),

            mlx_invocations: AtomicU64::new(0),
            ane_invocations: AtomicU64::new(0),
            accelerate_invocations: AtomicU64::new(0),
            grammar_accepted: AtomicU64::new(0),
            grammar_rejected: AtomicU64::new(0),

            peak_memory_bytes: AtomicU64::new(0),
            current_memory_bytes: AtomicU64::new(0),

            draft_acceptance_rate: AtomicF64::new(0.0),
            spec_tokens_saved: AtomicU64::new(0),

            anomalies_detected: AtomicU64::new(0),
            patches_applied: AtomicU64::new(0),
        }
    }

    /// Record a single token generation with its end‑to‑end latency (ms).
    pub fn record_token(&self, latency_ms: f64) {
        self.tokens_generated.fetch_add(1, Ordering::Relaxed);
        self.latency_ms.observe(latency_ms);
        // Infer throughput from the last 10 observations using an EWMA-ish
        // approach — here we simply record instantaneous tok/s.
        let tps = if latency_ms > 0.0 {
            1000.0 / latency_ms
        } else {
            0.0
        };
        if tps > 0.0 {
            self.tokens_per_second.observe(tps);
        }
    }

    /// Record the time from request start to first token emitted (ms).
    pub fn record_time_to_first_token(&self, latency_ms: f64) {
        self.time_to_first_token_ms.observe(latency_ms);
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self, kind: CacheKind) {
        match kind {
            CacheKind::Model => {
                self.model_cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            CacheKind::Prefix => {
                self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            CacheKind::KVCache => {
                // No dedicated counter; KV hits are reflected in
                // `model_cache_hits` for simplicity.
                self.model_cache_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self, kind: CacheKind) {
        match kind {
            CacheKind::Model => {
                self.model_cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            CacheKind::Prefix => {
                self.prefix_cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            CacheKind::KVCache => {
                self.model_cache_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a draft‑token acceptance rate observation (0.0 – 1.0).
    pub fn record_draft_acceptance(&self, rate: f64) {
        self.draft_acceptance_rate.store(rate, Ordering::Relaxed);
    }

    /// Track peak memory (idempotent — only affects the max).
    pub fn observe_memory(&self, current_bytes: u64) {
        self.current_memory_bytes
            .store(current_bytes, Ordering::Relaxed);
        let prev = self
            .peak_memory_bytes
            .fetch_max(current_bytes, Ordering::Relaxed);
        if current_bytes > prev {
            self.peak_memory_bytes
                .store(current_bytes, Ordering::Relaxed);
        }
    }

    /// Render all metrics in Prometheus text format (one big string).
    ///
    /// Counters get `_total` suffix per convention.  Each metric is
    /// preceded by `# HELP` and `# TYPE` comments.
    pub fn to_prometheus(&self) -> String {
        let mut buf = String::with_capacity(4096);

        // ── Help / type header ─────────────────────────────────────────
        let _ = writeln!(
            buf,
            "# HELP tribunus_tokens_generated_total Total tokens generated"
        );
        let _ = writeln!(buf, "# TYPE tribunus_tokens_generated_total counter");
        let _ = writeln!(
            buf,
            "tribunus_tokens_generated_total {}",
            self.tokens_generated.load(Ordering::Relaxed)
        );

        // ── Latency histogram ──────────────────────────────────────────
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_latency_ms Token generation latency in ms"
        );
        let _ = writeln!(buf, "# TYPE tribunus_latency_ms histogram");
        self.latency_ms
            .write_prometheus(&mut buf, "tribunus_latency_ms");

        // ── Time to first token ────────────────────────────────────────
        let _ = writeln!(buf, "\n# HELP tribunus_ttft_ms Time to first token in ms");
        let _ = writeln!(buf, "# TYPE tribunus_ttft_ms histogram");
        self.time_to_first_token_ms
            .write_prometheus(&mut buf, "tribunus_ttft_ms");

        // ── Throughput ─────────────────────────────────────────────────
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_tokens_per_second Token throughput (tok/s)"
        );
        let _ = writeln!(buf, "# TYPE tribunus_tokens_per_second histogram");
        self.tokens_per_second
            .write_prometheus(&mut buf, "tribunus_tokens_per_second");

        // ── Model cache ────────────────────────────────────────────────
        let hits = self.model_cache_hits.load(Ordering::Relaxed);
        let misses = self.model_cache_misses.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_model_cache_hits_total Model cache hits"
        );
        let _ = writeln!(buf, "# TYPE tribunus_model_cache_hits_total counter");
        let _ = writeln!(buf, "tribunus_model_cache_hits_total {hits}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_model_cache_misses_total Model cache misses"
        );
        let _ = writeln!(buf, "# TYPE tribunus_model_cache_misses_total counter");
        let _ = writeln!(buf, "tribunus_model_cache_misses_total {misses}");
        if hits + misses > 0 {
            let hit_ratio = hits as f64 / (hits + misses) as f64;
            let _ = writeln!(
                buf,
                "\n# HELP tribunus_model_cache_hit_ratio Model cache hit ratio"
            );
            let _ = writeln!(buf, "# TYPE tribunus_model_cache_hit_ratio gauge");
            let _ = writeln!(buf, "tribunus_model_cache_hit_ratio {hit_ratio:.4}");
        }

        // ── Prefix cache ───────────────────────────────────────────────
        let phits = self.prefix_cache_hits.load(Ordering::Relaxed);
        let pmisses = self.prefix_cache_misses.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_prefix_cache_hits_total Prefix cache hits"
        );
        let _ = writeln!(buf, "# TYPE tribunus_prefix_cache_hits_total counter");
        let _ = writeln!(buf, "tribunus_prefix_cache_hits_total {phits}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_prefix_cache_misses_total Prefix cache misses"
        );
        let _ = writeln!(buf, "# TYPE tribunus_prefix_cache_misses_total counter");
        let _ = writeln!(buf, "tribunus_prefix_cache_misses_total {pmisses}");

        // ── KV compression ─────────────────────────────────────────────
        let kv_ratio = self.kv_compression_ratio.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_kv_compression_ratio KV cache compression ratio"
        );
        let _ = writeln!(buf, "# TYPE tribunus_kv_compression_ratio gauge");
        let _ = writeln!(buf, "tribunus_kv_compression_ratio {kv_ratio:.4}");

        // ── Backend invocations ────────────────────────────────────────
        let mlx = self.mlx_invocations.load(Ordering::Relaxed);
        let ane_inv = self.ane_invocations.load(Ordering::Relaxed);
        let accel = self.accelerate_invocations.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_mlx_invocations_total MLX backend invocations"
        );
        let _ = writeln!(buf, "# TYPE tribunus_mlx_invocations_total counter");
        let _ = writeln!(buf, "tribunus_mlx_invocations_total {mlx}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_ane_invocations_total ANE backend invocations"
        );
        let _ = writeln!(buf, "# TYPE tribunus_ane_invocations_total counter");
        let _ = writeln!(buf, "tribunus_ane_invocations_total {ane_inv}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_accelerate_invocations_total Accelerate backend invocations"
        );
        let _ = writeln!(buf, "# TYPE tribunus_accelerate_invocations_total counter");
        let _ = writeln!(buf, "tribunus_accelerate_invocations_total {accel}");

        // ── Grammar ────────────────────────────────────────────────────
        let g_ok = self.grammar_accepted.load(Ordering::Relaxed);
        let g_rej = self.grammar_rejected.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_grammar_accepted_total Grammar accepted tokens"
        );
        let _ = writeln!(buf, "# TYPE tribunus_grammar_accepted_total counter");
        let _ = writeln!(buf, "tribunus_grammar_accepted_total {g_ok}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_grammar_rejected_total Grammar rejected tokens"
        );
        let _ = writeln!(buf, "# TYPE tribunus_grammar_rejected_total counter");
        let _ = writeln!(buf, "tribunus_grammar_rejected_total {g_rej}");
        if g_ok + g_rej > 0 {
            let g_ratio = g_ok as f64 / (g_ok + g_rej) as f64;
            let _ = writeln!(
                buf,
                "\n# HELP tribunus_grammar_success_rate Grammar success rate"
            );
            let _ = writeln!(buf, "# TYPE tribunus_grammar_success_rate gauge");
            let _ = writeln!(buf, "tribunus_grammar_success_rate {g_ratio:.4}");
        }

        // ── Memory ─────────────────────────────────────────────────────
        let peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        let cur = self.current_memory_bytes.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_peak_memory_bytes Peak memory usage in bytes"
        );
        let _ = writeln!(buf, "# TYPE tribunus_peak_memory_bytes gauge");
        let _ = writeln!(buf, "tribunus_peak_memory_bytes {peak}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_current_memory_bytes Current memory usage in bytes"
        );
        let _ = writeln!(buf, "# TYPE tribunus_current_memory_bytes gauge");
        let _ = writeln!(buf, "tribunus_current_memory_bytes {cur}");

        // ── Speculative decoding ──────────────────────────────────────
        let spec_saved = self.spec_tokens_saved.load(Ordering::Relaxed);
        let draft_rate = self.draft_acceptance_rate.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_draft_acceptance_rate Speculative draft acceptance rate"
        );
        let _ = writeln!(buf, "# TYPE tribunus_draft_acceptance_rate gauge");
        let _ = writeln!(buf, "tribunus_draft_acceptance_rate {draft_rate:.4}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_spec_tokens_saved_total Tokens saved via speculative decoding"
        );
        let _ = writeln!(buf, "# TYPE tribunus_spec_tokens_saved_total counter");
        let _ = writeln!(buf, "tribunus_spec_tokens_saved_total {spec_saved}");

        // ── Anomaly ────────────────────────────────────────────────────
        let anom = self.anomalies_detected.load(Ordering::Relaxed);
        let patches = self.patches_applied.load(Ordering::Relaxed);
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_anomalies_detected_total Anomalies detected"
        );
        let _ = writeln!(buf, "# TYPE tribunus_anomalies_detected_total counter");
        let _ = writeln!(buf, "tribunus_anomalies_detected_total {anom}");
        let _ = writeln!(
            buf,
            "\n# HELP tribunus_patches_applied_total Self‑healing patches applied"
        );
        let _ = writeln!(buf, "# TYPE tribunus_patches_applied_total counter");
        let _ = writeln!(buf, "tribunus_patches_applied_total {patches}");

        buf
    }
}

impl Default for InferenceTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_empty() {
        let h = Histogram::latency_ms();
        assert_eq!(h.total_count(), 0);
        assert_eq!(h.percentile(50.0), 0.0);
    }

    #[test]
    fn test_histogram_single_observation() {
        let h = Histogram::latency_ms();
        h.observe(42.0);
        assert_eq!(h.total_count(), 1);
        assert!(h.percentile(50.0) >= 25.0);
        assert!(h.percentile(50.0) <= 50.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let h = Histogram::latency_ms();
        // Insert many observations in bucket 10..25
        for _ in 0..100 {
            h.observe(15.0);
        }
        // And a few in bucket 25..50
        for _ in 0..50 {
            h.observe(30.0);
        }
        // P50 should be around bucket 10..25
        let p50 = h.percentile(50.0);
        assert!(p50 >= 10.0);
        assert!(p50 <= 25.0);

        let p95 = h.percentile(95.0);
        assert!(p95 >= 25.0);
        assert!(p95 <= 50.0);
    }

    #[test]
    fn test_histogram_reset() {
        let h = Histogram::latency_ms();
        h.observe(100.0);
        assert_eq!(h.total_count(), 1);
        h.reset();
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn test_atomic_f64() {
        let v = AtomicF64::new(3.14);
        assert!((v.load(Ordering::Relaxed) - 3.14).abs() < 1e-10);
        v.store(2.71, Ordering::Relaxed);
        assert!((v.load(Ordering::Relaxed) - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_telemetry_new() {
        let t = InferenceTelemetry::new();
        assert_eq!(t.tokens_generated.load(Ordering::Relaxed), 0);
        assert_eq!(t.latency_ms.total_count(), 0);
    }

    #[test]
    fn test_record_token() {
        let t = InferenceTelemetry::new();
        t.record_token(25.0);
        assert_eq!(t.tokens_generated.load(Ordering::Relaxed), 1);
        assert_eq!(t.latency_ms.total_count(), 1);
    }

    #[test]
    fn test_cache_kind() {
        let t = InferenceTelemetry::new();
        t.record_cache_hit(CacheKind::Model);
        assert_eq!(t.model_cache_hits.load(Ordering::Relaxed), 1);
        t.record_cache_miss(CacheKind::Prefix);
        assert_eq!(t.prefix_cache_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_prometheus_output() {
        let t = InferenceTelemetry::new();
        t.record_token(15.0);
        t.record_token(22.0);
        t.record_cache_hit(CacheKind::Model);
        t.record_cache_miss(CacheKind::Model);
        t.record_draft_acceptance(0.75);
        t.observe_memory(1_073_741_824);

        let out = t.to_prometheus();
        assert!(out.contains("tribunus_tokens_generated_total 2"));
        assert!(out.contains("tribunus_latency_ms_count 2"));
        assert!(out.contains("tribunus_model_cache_hits_total 1"));
        assert!(out.contains("tribunus_model_cache_misses_total 1"));
        assert!(out.contains("tribunus_draft_acceptance_rate 0.7500"));
        assert!(out.contains("tribunus_peak_memory_bytes"));
        assert!(out.contains("tribunus_current_memory_bytes"));
    }

    #[test]
    fn test_histogram_write_prometheus() {
        let h = Histogram::latency_ms();
        h.observe(5.0);
        h.observe(50.0);
        h.observe(200.0);

        let mut buf = String::new();
        h.write_prometheus(&mut buf, "test_latency_ms");
        assert!(buf.contains("test_latency_ms_count 3"));
        assert!(buf.contains("test_latency_ms_sum"));
        assert!(buf.contains("le=\"+Inf\""));
        // Bucket 5 should have at least 1
        assert!(buf.contains("le=\"5\""));
    }
}
