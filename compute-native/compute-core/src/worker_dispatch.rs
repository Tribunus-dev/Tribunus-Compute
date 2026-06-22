//! Worker-side dispatch for pre-compiled Metal kernel artifacts.
//!
//! At inference time, the worker loads the compiled .metallib from the
//! ComputeImage, creates Metal pipeline states, binds buffers from the
//! Tribunus unified arena, and dispatches kernels directly — without
//! calling any MLX runtime functions.
//!
//! This is a vertical-slice implementation supporting one NF4 quantized
//! projection kernel.  Extension to other operations follows the same
//! pattern: load artifact → create pipeline → bind arena buffers → dispatch.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::compute_image::manifest::{MetalDispatchRecipe, MetalKernelArtifact};

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// A loaded and pipeline-state-ready Metal kernel ready for dispatch.
#[derive(Clone)]
pub struct LoadedMetalKernel {
    pub artifact: MetalKernelArtifact,
    pipeline_state: MetalPipelineState,
}

#[derive(Clone)]
pub struct MetalPipelineState {
    library_data: Vec<u8>,
    function_name: String,
    // Future: MTLComputePipelineState, MTLDevice reference, etc.
}

impl LoadedMetalKernel {
    /// Accessor for the pipeline state metadata.
    pub fn pipeline_state(&self) -> &MetalPipelineState {
        &self.pipeline_state
    }

    /// Access the loaded .metallib bytes.
    pub fn library_data(&self) -> &[u8] {
        &self.pipeline_state.library_data
    }

    /// Return the entry point function name.
    pub fn function_name(&self) -> &str {
        &self.pipeline_state.function_name
    }

    /// Load a Metal kernel artifact from a ComputeImage directory.
    /// Reads the .metallib, creates the Metal pipeline state if a device
    /// is available, and validates the artifact checksum.
    pub fn load(image_dir: &Path, artifact: &MetalKernelArtifact) -> Result<Self, String> {
        let metallib_path = image_dir.join(&artifact.metallib_relpath);
        let library_data = std::fs::read(&metallib_path).map_err(|e| {
            format!(
                "failed to read .metallib at {}: {}",
                metallib_path.display(),
                e
            )
        })?;

        // Validate checksum if provided
        if !artifact.metallib_blake3.is_empty() {
            let hash = blake3::hash(&library_data);
            let hash_hex = hash.to_hex().to_string();
            if hash_hex != artifact.metallib_blake3 {
                return Err(format!(
                    "Metal library checksum mismatch for {}: expected {} got {}",
                    artifact.artifact_id, artifact.metallib_blake3, hash_hex
                ));
            }
        }

        // ABI version check
        if artifact.dispatch.kernel_abi_version != 1 {
            return Err(format!(
                "unsupported Metal kernel ABI version {} for {}",
                artifact.dispatch.kernel_abi_version, artifact.artifact_id
            ));
        }

        // TODO: Create MTLDevice, compile library into MTLComputePipelineState.
        // For the vertical slice, this is a placeholder that validates
        // the artifact structure without requiring actual Metal objects.

        Ok(Self {
            artifact: artifact.clone(),
            pipeline_state: MetalPipelineState {
                library_data,
                function_name: artifact.dispatch.entry_point.clone(),
            },
        })
    }
}

/// Collection of all loaded Metal kernels for a ComputeImage.
pub struct MetalKernelRegistry {
    kernels: HashMap<String, Arc<LoadedMetalKernel>>,
}

impl MetalKernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Load all Metal kernel artifacts from a manifest.
    pub fn load_all(image_dir: &Path, artifacts: &[MetalKernelArtifact]) -> Result<Self, String> {
        let mut kernels = HashMap::new();
        for artifact in artifacts {
            let loaded = LoadedMetalKernel::load(image_dir, artifact)?;
            kernels.insert(artifact.artifact_id.clone(), Arc::new(loaded));
        }
        Ok(Self { kernels })
    }

    /// Get a loaded kernel by artifact ID.
    pub fn get(&self, id: &str) -> Option<Arc<LoadedMetalKernel>> {
        self.kernels.get(id).cloned()
    }

    /// Number of loaded kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }
}


impl MetalKernelRegistry {
    /// Consume the registry and return the loaded kernels as a Vec.
    pub fn into_vec(self) -> Vec<LoadedMetalKernel> {
        self.kernels
            .into_values()
            .map(|arc| Arc::unwrap_or_clone(arc))
            .collect()
    }
}
