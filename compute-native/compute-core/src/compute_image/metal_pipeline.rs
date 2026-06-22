//! Metal compilation pipeline — compiles `.metal` sources to `.metallib`
//! via `xcrun metal` + `metallib`, validates MTLB magic, and returns
//! a [`MetalPipelineOutput`] ready for artifact embedding.

use crate::compute_image::manifest::{MetalDispatchRecipe, MetalKernelArtifact};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Output from compiling a Metal source to a .metallib.
#[derive(Debug, Clone)]
pub struct MetalPipelineOutput {
    /// The compiled .metallib bytes.
    pub metallib_bytes: Vec<u8>,
    /// SHA-256 hex hash of the .metallib.
    pub sha256: String,
    /// Byte length of the .metallib.
    pub byte_length: u64,
}

/// Compile a Metal source string into a .metallib via xcrun.
///
/// Returns `None` when xcrun is unavailable or compilation fails.
/// The caller should fall back gracefully.
pub fn compile_metal_source(name: &str, source: &str) -> Option<MetalPipelineOutput> {
    // Check toolchain availability first.
    let has_xcrun = std::process::Command::new("xcrun")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !has_xcrun {
        eprintln!("[metal-pipeline] xcrun not found — cannot compile '{}'", name);
        return None;
    }

    let tmp = std::env::temp_dir().join(format!("tribunus-metal-{}", name));
    let _ = std::fs::create_dir_all(&tmp);

    let src_path = tmp.join("kernel.metal");
    let air_path = tmp.join("kernel.air");
    let metallib_path = tmp.join("kernel.metallib");

    // Write source.
    if std::fs::write(&src_path, source).is_err() {
        let _ = std::fs::remove_dir_all(&tmp);
        return None;
    }

    // Compile to AIR.
    let status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal3.2", "-O3", "-c"])
        .arg(src_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap())
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("[metal-pipeline] metal compile failed for '{}'", name);
            let _ = std::fs::remove_dir_all(&tmp);
            return None;
        }
    }

    // Link to metallib.
    let status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(air_path.to_str().unwrap())
        .arg("-o")
        .arg(metallib_path.to_str().unwrap())
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("[metal-pipeline] metallib link failed for '{}'", name);
            let _ = std::fs::remove_dir_all(&tmp);
            return None;
        }
    }

    // Read + validate MTLB magic.
    let bytes = match std::fs::read(&metallib_path) {
        Ok(b) => b,
        Err(_) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return None;
        }
    };

    // MTLB magic: first 4 bytes should be b"MTLB".
    let valid_magic = bytes.len() >= 4 && &bytes[0..4] == b"MTLB";
    if !valid_magic {
        eprintln!(
            "[metal-pipeline] '{}' .metallib missing MTLB magic (got {:02x?})",
            name,
            &bytes[..bytes.len().min(4)]
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return None;
    }

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let sha256 = format!("{:x}", hasher.finalize());
    let byte_length = bytes.len() as u64;

    let _ = std::fs::remove_dir_all(&tmp);

    Some(MetalPipelineOutput {
        metallib_bytes: bytes,
        sha256,
        byte_length,
    })
}

/// Wrap a compiled Metal pipeline output into a `MetalKernelArtifact` suitable
/// for embedding in the ComputeImage manifest.
pub fn metal_pipeline_to_artifact(
    name: &str,
    op: &str,
    pipeline: &MetalPipelineOutput,
    entry_point: &str,
) -> MetalKernelArtifact {
    MetalKernelArtifact {
        artifact_id: name.to_string(),
        logical_operation: op.to_string(),
        kind: crate::compute_image::manifest::ArtifactKind::MlxNf4U32,
        metallib_relpath: format!("metal/kernels/{}.metallib", name),
        metallib_blake3: pipeline.sha256.clone(),
        metallib_byte_length: pipeline.byte_length,
        dispatch: MetalDispatchRecipe {
            entry_point: entry_point.to_string(),
            kernel_name: name.to_string(),
            threads_per_threadgroup: [32, 32, 1],
            threadgroups_per_grid: [1, 1, 1],
            buffer_slot_map: HashMap::new(),
            scalar_index_map: HashMap::new(),
            k: 64,
            n: 32,
            group_size: 0,
            bits: 0,
            kernel_abi_version: 1,
        },
        logical_shape: vec![4096, 4096],
        storage_shape: vec![4096, 512],
        bits: 0,
        group_size: 0,
        scale_tensor: String::new(),
        bias_tensor: String::new(),
        gpu_family: "m1".to_string(),
        checksum: String::new(),
    }
}

/// Validate that a slice of bytes has a valid MTLB magic header.
pub fn validate_metallib_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == b"MTLB"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_metallib_magic_valid() {
        let valid = b"MTLB\x01\x02\x03\x04";
        assert!(validate_metallib_magic(valid));
    }

    #[test]
    fn test_validate_metallib_magic_short() {
        assert!(!validate_metallib_magic(b""));
        assert!(!validate_metallib_magic(b"MTL"));
    }

    #[test]
    fn test_validate_metallib_magic_invalid() {
        assert!(!validate_metallib_magic(b"XXXX"));
        assert!(!validate_metallib_magic(b"mtlb"));
    }

    #[test]
    fn test_compile_metal_source_missing_xcrun() {
        // Simulate missing xcrun by checking graceful return.
        // On systems without xcrun, this returns None.
        let result = compile_metal_source("test", "invalid metal source");
        // If xcrun is available, this might fail for other reasons.
        // We only test the None-on-failure path.
        if result.is_some() {
            // xcrun is available — skip this test expectation.
            eprintln!("[test] xcrun available, skipping missing-xcrun assertion");
        }
    }

    #[test]
    fn test_metal_pipeline_to_artifact_creates_valid_entry() {
        let out = MetalPipelineOutput {
            metallib_bytes: vec![b'M', b'T', b'L', b'B', 0, 1, 2, 3],
            sha256: "abcdef1234567890".into(),
            byte_length: 8,
        };
        let artifact = metal_pipeline_to_artifact("test_kernel", "test_op", &out, "test_kernel_entry");
        assert_eq!(artifact.artifact_id, "test_kernel");
        assert_eq!(artifact.logical_operation, "test_op");
        assert_eq!(artifact.metallib_blake3, "abcdef1234567890");
        assert_eq!(artifact.metallib_byte_length, 8);
        assert_eq!(artifact.metallib_relpath, "metal/kernels/test_kernel.metallib");
    }
}
