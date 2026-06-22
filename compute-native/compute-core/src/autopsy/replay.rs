//! Diagnostic replay results.
//!
//! Contains the [`ReplayResult`] type returned by the anomaly tracer's
//! `replay_weight` method. This captures the outcome of dequantizing a
//! suspected weight tensor, running a reference forward pass, and comparing
//! against the expected output.

/// Result of a diagnostic weight replay.
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub weight_name: String,
    pub segment: String,
    pub original_hash: String,
    pub computed_hash: String,
    pub hash_match: bool,
    pub reference_mse: f64,
    pub reference_cosine: f64,
    pub elapsed_ms: u64,
    pub layer: u32,
    pub tensor_name: Option<String>,
    pub family: String,
}
