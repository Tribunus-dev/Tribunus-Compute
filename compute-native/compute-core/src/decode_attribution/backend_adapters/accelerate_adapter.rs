//! Accelerate domain adapter — hand-written stub for compilation.
//!
//! The original source was structurally incomplete (36 unclosed brace pairs).
//! This stub provides the same public API.  Replace with the full port when
//! the decode-attribution module is brought up to parity with the monorepo.

use crate::decode_attribution::backend_adapters::BackendSupportTier;

/// Simplified result until the full port lands.
#[derive(Clone)]
pub struct AccelerateDomainResult {
    pub output: Vec<f32>,
    pub duration_ns: u64,
    pub execution_kind: String,
    pub execution_proof: crate::decode_attribution::receipt::ExecutionProof,
}

/// Return the support tier for a given operation family.
pub fn support_tier(_family_name: &str) -> BackendSupportTier {
    BackendSupportTier::UnsupportedGraph
}

/// Run an operation family through the Accelerate domain adapter.
pub fn run_family(
    _family_name: &str,
    _input_data: &[f32],
    _weights: &[f32],
    _profile: &crate::decode_attribution::shape_profiles::ShapeProfile,
) -> Result<AccelerateDomainResult, String> {
    Err("Accelerate domain adapter not yet ported".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_tier_matmul() {
        assert_eq!(support_tier("matmul"), BackendSupportTier::UnsupportedGraph);
    }

    #[test]
    fn support_tier_unsupported() {
        assert_eq!(
            support_tier("nonexistent"),
            BackendSupportTier::UnsupportedGraph
        );
    }
}
