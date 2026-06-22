//! Fusion receipts — evidence that a fused kernel actually executed.
//!
//! Every fused kernel dispatch produces a [`FusedMetalExecutionEvidence`]
//! that identifies the exact artifact, the launch parameters, and whether
//! the dispatch succeeded or fell back.  The scheduler attaches this to
//! the phase receipt.

use crate::compute_image::fusion_abi::{MetalLaunchContract, SealedMetalFusionArtifact};
use serde::{Deserialize, Serialize};

/// Evidence that a fused Metal kernel was dispatched and completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedMetalExecutionEvidence {
    /// Which fusion region this evidence corresponds to.
    pub region_id: String,
    /// The artifact name that was dispatched.
    pub artifact_name: String,
    /// SHA-256 hash of the .metallib that was loaded.
    pub metallib_hash: String,
    /// The launch contract used for dispatch.
    pub launch_contract: MetalLaunchContract,
    /// Timestamp when the kernel was dispatched.
    pub performed_at: String,
    /// Kernel execution time in microseconds.
    pub duration_us: u64,
    /// Optional numerical summary (min/max/mean of output).
    pub numerical_summary: Option<NumericalSummary>,
    /// If non-None, the fused kernel did NOT run; the fallback was used.
    pub fallback_reason: Option<FusedMetalFallbackReason>,
}

/// Summary statistics of a kernel's output tensor (for qualification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

/// Why a fused kernel was not used at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusedMetalFallbackReason {
    /// xcrun / Metal compiler toolchain not available at compile time.
    XcrunNotAvailable,
    /// Metal source compilation failed.
    CompileFailed,
    /// Artifact seal verification failed.
    SealMismatch,
    /// Current hardware does not support the artifact.
    HardwareIncompatible,
    /// Numerical parity with unfused path failed.
    NumericalMismatch,
    /// Benchmark admission gate rejected this artifact.
    BenchmarkGateRejected,
    /// The unfused decomposition was used instead.
    DecompositionUsed,
}

impl FusedMetalExecutionEvidence {
    /// Create evidence for a successful fused kernel dispatch.
    pub fn from_artifact(
        artifact: &SealedMetalFusionArtifact,
        duration_us: u64,
    ) -> Self {
        Self {
            region_id: artifact.region_id.clone(),
            artifact_name: artifact.artifact_name.clone(),
            metallib_hash: artifact.metallib_hash.sha256.clone(),
            launch_contract: artifact.launch_contract.clone(),
            performed_at: crate::now_iso8601(),
            duration_us,
            numerical_summary: None,
            fallback_reason: None,
        }
    }

    /// Create evidence for a fallback path (fused kernel was not used).
    pub fn fallback(
        artifact_name: &str,
        reason: FusedMetalFallbackReason,
    ) -> Self {
        Self {
            region_id: String::new(),
            artifact_name: artifact_name.to_string(),
            metallib_hash: String::new(),
            launch_contract: MetalLaunchContract {
                entry_point: String::new(),
                threads_per_threadgroup: [0, 0, 0],
                threadgroups_per_grid: [0, 0, 0],
                buffer_bindings: std::collections::HashMap::new(),
            },
            performed_at: crate::now_iso8601(),
            duration_us: 0,
            numerical_summary: None,
            fallback_reason: Some(reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::fusion_abi::{
        ArtifactHash, MetalFusionFamily, QuantizationContract,
    };
    use std::collections::HashMap;

    fn dummy_artifact() -> SealedMetalFusionArtifact {
        SealedMetalFusionArtifact::new(
            "qkv_proj",
            MetalFusionFamily::QkvProj,
            ArtifactHash { sha256: "abc".into(), byte_length: 128 },
            MetalLaunchContract {
                entry_point: "qkv_proj_kernel".into(),
                threads_per_threadgroup: [32, 32, 1],
                threadgroups_per_grid: [1, 1, 1],
                buffer_bindings: HashMap::new(),
            },
            Some(QuantizationContract {
                scheme: "fp16".into(), group_size: 32, bits: 16,
            }),
        )
    }

    #[test]
    fn test_evidence_from_artifact() {
        let artifact = dummy_artifact();
        let ev = FusedMetalExecutionEvidence::from_artifact(&artifact, 42);
        assert_eq!(ev.region_id, "qkv_proj");
        assert_eq!(ev.duration_us, 42);
        assert!(ev.fallback_reason.is_none());
        assert_eq!(ev.launch_contract.entry_point, "qkv_proj_kernel");
    }

    #[test]
    fn test_evidence_fallback() {
        let ev = FusedMetalExecutionEvidence::fallback(
            "qkv_proj",
            FusedMetalFallbackReason::XcrunNotAvailable,
        );
        assert_eq!(ev.artifact_name, "qkv_proj");
        assert!(matches!(ev.fallback_reason, Some(FusedMetalFallbackReason::XcrunNotAvailable)));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let ev = FusedMetalExecutionEvidence::from_artifact(&dummy_artifact(), 123);
        let json = serde_json::to_string(&ev).unwrap();
        let deser: FusedMetalExecutionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.duration_us, 123);
        assert_eq!(deser.region_id, "qkv_proj");
    }
}
