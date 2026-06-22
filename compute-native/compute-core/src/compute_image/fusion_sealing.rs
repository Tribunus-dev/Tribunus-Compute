//! Fusion sealing — create and verify seals on fused Metal artifacts.
//!
//! Sealing proves that a .metallib's bytes have not been tampered with
//! since compilation.  The seal is a SHA-256 hash of the metallib bytes
//! stored alongside the artifact metadata.

use crate::compute_image::fusion_abi::{ArtifactHash, SealedMetalFusionArtifact};
use sha2::{Digest, Sha256};

/// Seal a fused Metal artifact by computing its content hash.
///
/// Returns the artifact with the hash filled in and seal_timestamp updated.
pub fn seal_fusion_artifact(
    mut artifact: SealedMetalFusionArtifact,
    metallib_bytes: &[u8],
) -> SealedMetalFusionArtifact {
    let mut hasher = Sha256::new();
    hasher.update(metallib_bytes);
    let sha256 = format!("{:x}", hasher.finalize());

    artifact.metallib_hash = ArtifactHash {
        sha256,
        byte_length: metallib_bytes.len() as u64,
    };
    artifact.seal_version = 1;
    artifact.seal_timestamp = crate::now_iso8601();
    artifact
}

/// Verify that a sealed artifact's content hash matches the given bytes.
///
/// Returns `true` if the hash matches, `false` otherwise.
pub fn verify_seal(artifact: &SealedMetalFusionArtifact, metallib_bytes: &[u8]) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(metallib_bytes);
    let computed = format!("{:x}", hasher.finalize());
    computed == artifact.metallib_hash.sha256
}

/// Verify seal with explicit hash (no artifact needed).
pub fn verify_hash(expected: &str, metallib_bytes: &[u8]) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(metallib_bytes);
    let computed = format!("{:x}", hasher.finalize());
    computed == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::fusion_abi::{
        MetalFusionFamily, MetalLaunchContract, QuantizationContract,
    };
    use std::collections::HashMap;

    fn dummy_artifact() -> SealedMetalFusionArtifact {
        SealedMetalFusionArtifact::new(
            "test",
            MetalFusionFamily::SiluMul,
            ArtifactHash {
                sha256: String::new(),
                byte_length: 0,
            },
            MetalLaunchContract {
                entry_point: "kernel".into(),
                threads_per_threadgroup: [1, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                buffer_bindings: HashMap::new(),
            },
            None,
        )
    }

    #[test]
    fn test_seal_and_verify() {
        let bytes = b"MTLBtest_metallib_content";
        let artifact = dummy_artifact();
        let sealed = seal_fusion_artifact(artifact, bytes);
        assert!(!sealed.metallib_hash.sha256.is_empty());
        assert!(verify_seal(&sealed, bytes));
    }

    #[test]
    fn test_verify_rejects_tampered() {
        let bytes = b"MTLBtest_metallib_content";
        let artifact = dummy_artifact();
        let sealed = seal_fusion_artifact(artifact, bytes);
        assert!(!verify_seal(&sealed, b"MTLBtampered_content"));
    }

    #[test]
    fn test_verify_hash_direct() {
        let bytes = b"hello_metallib";
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let expected = format!("{:x}", hasher.finalize());
        assert!(verify_hash(&expected, bytes));
        assert!(!verify_hash(&expected, b"wrong_bytes"));
    }
}
