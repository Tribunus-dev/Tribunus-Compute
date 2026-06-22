//! Fusion ABI — sealed artifact types for fused Metal kernels.
//!
//! A [`SealedMetalFusionArtifact`] is the compiler's binding commitment:
//! the .metallib content hash, the dispatch geometry, and the quantization
//! contract are all baked at compile time and verified at load time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which fused kernel family this artifact belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalFusionFamily {
    QkvProj,
    AttnOut,
    GateUpProj,
    SiluMul,
    DownProj,
    RmsNormResidual,
    SelfAttn,
}

/// Content hash of a sealed artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactHash {
    /// Hex-encoded SHA-256 of the .metallib bytes.
    pub sha256: String,
    /// Length of the .metallib in bytes.
    pub byte_length: u64,
}

/// Quantization contract — what format the weights use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationContract {
    pub scheme: String,       // "nf4", "int8", "fp16"
    pub group_size: u32,
    pub bits: u8,
}

/// Metal launch geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalLaunchContract {
    pub entry_point: String,
    pub threads_per_threadgroup: [u32; 3],
    pub threadgroups_per_grid: [u32; 3],
    pub buffer_bindings: HashMap<u32, String>,
}

/// A sealed, immutable fused-kernel artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedMetalFusionArtifact {
    pub region_id: String,
    pub family: MetalFusionFamily,
    pub artifact_name: String,
    pub op: String,
    pub metallib_hash: ArtifactHash,
    pub launch_contract: MetalLaunchContract,
    pub quantization_contract: Option<QuantizationContract>,
    pub seal_version: u32,
    pub seal_timestamp: String,
}

impl SealedMetalFusionArtifact {
    /// Create a new sealed artifact with a fresh timestamp.
    pub fn new(
        region_id: &str,
        family: MetalFusionFamily,
        metallib_hash: ArtifactHash,
        launch_contract: MetalLaunchContract,
        quantization_contract: Option<QuantizationContract>,
    ) -> Self {
        Self {
            region_id: region_id.to_string(),
            family,
            artifact_name: format!("fused_{}", region_id),
            op: format!("fused_{}", region_id),
            metallib_hash,
            launch_contract,
            quantization_contract,
            seal_version: 1,
            seal_timestamp: crate::now_iso8601(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sealed_artifact_creation() {
        let hash = ArtifactHash {
            sha256: "abcdef".into(),
            byte_length: 1024,
        };
        let launch = MetalLaunchContract {
            entry_point: "qkv_proj_kernel".into(),
            threads_per_threadgroup: [32, 32, 1],
            threadgroups_per_grid: [1, 1, 1],
            buffer_bindings: HashMap::new(),
        };
        let qc = QuantizationContract {
            scheme: "fp16".into(),
            group_size: 32,
            bits: 16,
        };
        let artifact = SealedMetalFusionArtifact::new(
            "qkv_proj", MetalFusionFamily::QkvProj, hash, launch, Some(qc),
        );
        assert_eq!(artifact.seal_version, 1);
        assert_eq!(artifact.family, MetalFusionFamily::QkvProj);
        assert_eq!(artifact.artifact_name, "fused_qkv_proj");
    }

    #[test]
    fn test_serialize_roundtrip() {
        let a = SealedMetalFusionArtifact::new(
            "test",
            MetalFusionFamily::SiluMul,
            ArtifactHash { sha256: "aabb".into(), byte_length: 64 },
            MetalLaunchContract {
                entry_point: "kernel".into(),
                threads_per_threadgroup: [1, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                buffer_bindings: HashMap::new(),
            },
            None,
        );
        let json = serde_json::to_string(&a).unwrap();
        let b: SealedMetalFusionArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(b.artifact_name, a.artifact_name);
        assert_eq!(b.seal_version, a.seal_version);
    }
}
