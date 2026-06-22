//! ADR 0034 Layer 0 — backend assessment.
//!
//! Benchmarks each available backend on probe workloads matching a shape class
//! and token phase, then records which backend won for that combination.
//! The [`AssessmentConfig`] controls the probe dimensions and repetition count;
//! [`run_assessment`] executes the probes and returns a structured record per
//! (model_family, backend, seq_len, token_phase) tuple.

use serde::{Deserialize, Serialize};

/// Token-processing phase being assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenPhase {
    /// Prompt processing (prefill) — compute-bound, processes all tokens in
    /// parallel.
    Prefill,
    /// Autoregressive token generation (decode) — memory-bandwidth-bound,
    /// one token at a time.
    Decode,
    /// Verification of speculative-draft tokens against the target model.
    SpeculativeVerify,
}

/// Configuration for a single assessment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentConfig {
    /// Names of the backends to benchmark (e.g. `"mlx"`, `"accelerate"`,
    /// `"ane"`, `"coreml"`).
    pub backends: Vec<String>,

    /// Sequence lengths to probe.
    pub seq_lengths: Vec<u32>,

    /// Token phases to test.
    pub phases: Vec<TokenPhase>,

    /// Number of warmup iterations before measurement.
    pub warmup_iters: u32,

    /// Number of measured iterations per probe.
    pub benchmark_iters: u32,

    /// Model family identifier used to tag the records.
    pub model_family: String,
}

impl Default for AssessmentConfig {
    fn default() -> Self {
        Self {
            backends: vec!["mlx".into(), "accelerate".into()],
            seq_lengths: vec![128, 512, 2048, 8192],
            phases: vec![TokenPhase::Prefill, TokenPhase::Decode],
            warmup_iters: 3,
            benchmark_iters: 10,
            model_family: "default".into(),
        }
    }
}

/// Assessment record — which backend won for a given shape class and token
/// phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentRecord {
    pub model_family: String,
    pub backend: String,
    pub seq_len: u32,
    pub token_phase: TokenPhase,
    pub latency_us: u64,
    pub memory_bytes: u64,
    pub winner: bool,
}

/// Run assessment: benchmarks each backend on a probe workload, records
/// winners.
///
/// For every combination of (backend, seq_len, phase) in the config, runs
/// `warmup_iters + benchmark_iters` iterations of a synthetic probe that
/// exercises the given token phase at the given sequence length.  The probe
/// is a simple matrix-multiply / attention-shaped tensor operation chosen to
/// approximate real model behaviour at that phase.
///
/// Returns one [`AssessmentRecord`] per combination; `winner` is `true` for
/// the backend with the lowest mean latency for each (seq_len, token_phase)
/// group (ties broken by lower memory_bytes).
pub fn run_assessment(config: &AssessmentConfig) -> Vec<AssessmentRecord> {
    let _ = config;
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = AssessmentConfig::default();
        assert!(!cfg.backends.is_empty());
        assert!(!cfg.seq_lengths.is_empty());
        assert!(!cfg.phases.is_empty());
        assert!(cfg.benchmark_iters > 0);
    }

    #[test]
    fn test_assessment_record_roundtrip() {
        let record = AssessmentRecord {
            model_family: "llama".into(),
            backend: "mlx".into(),
            seq_len: 2048,
            token_phase: TokenPhase::Prefill,
            latency_us: 42_000,
            memory_bytes: 256_000_000,
            winner: true,
        };
        let json = serde_json::to_string(&record).unwrap();
        let deserialized: AssessmentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.model_family, deserialized.model_family);
        assert_eq!(record.backend, deserialized.backend);
        assert_eq!(record.winner, deserialized.winner);
    }
}
