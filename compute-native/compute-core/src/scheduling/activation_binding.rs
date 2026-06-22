//! Activation binding — canonical activation state carrier.
//!
//! # Direct Handoff Criteria
//!
//! A route may record `DirectSharedBacking` only when all of the following hold:
//! - The producer and consumer identify the same backing resource or an explicitly
//!   documented alias.
//! - The consumer accepts the producer's physical layout, dtype, alignment, and
//!   representation.
//! - No CPU read or write materialization occurs.
//! - No Metal blit or compute materialization occurs.
//! - No MLX contiguous temporary is allocated.
//! - The producer completion is synchronized before consumer dispatch.
//! - The receipt records backing identity, offsets or plane, byte range, and lease
//!   generation.

use crate::compute_image::phase_graph::{PhaseId, TensorId, TensorLayoutContract};
use serde::{Deserialize, Serialize};

#[cfg(feature = "mlx-backend")]
use mlx_rs::Array;

/// Arena binding describing a memory region within the activation arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaBinding {
    pub slot_id: String,
    pub offset: u64,
    pub byte_size: u64,
    pub generation: u64,
}

/// Activation generation tracker for ordering and freshness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationGeneration(pub u64);

/// Tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I32,
    U32,
}

/// Physical representation of an activation tensor.
#[derive(Debug, Clone)]
pub enum ActivationRepresentation {
    MetalBuffer(MetalBufferBinding),
    MetalTexture(MetalTextureBinding),
    IOSurfaceTexture(IOSurfaceTextureBinding),
    CpuAccessibleSharedMemory(CpuMemoryBinding),
    MlxArrayCompatibility(MlxArrayBinding),
    CoreMlTensor(CoreMlTensorBinding),
}

#[derive(Debug, Clone)]
pub struct MetalBufferBinding {
    pub label: String,
    pub length: u64,
    pub buffer_id: String,
}

#[derive(Debug, Clone)]
pub struct MetalTextureBinding {
    pub label: String,
    pub width: u64,
    pub height: u64,
    pub pixel_format: String,
}

#[derive(Debug, Clone)]
pub struct IOSurfaceTextureBinding {
    pub surface_id: u32,
    pub width: u64,
    pub height: u64,
}

#[derive(Debug, Clone)]
pub struct CpuMemoryBinding {
    pub address: u64,
    pub length: u64,
}

#[derive(Debug, Clone)]
pub struct MlxArrayBinding {
    pub dtype: String,
    pub shape: Vec<i32>,
    pub mlx_array_id: u64,
}

#[derive(Debug, Clone)]
pub struct CoreMlTensorBinding {
    pub name: String,
    pub multi_array_id: u64,
}

/// Materialization kinds for explicit copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationKind {
    MetalTextureToMetalBuffer,
    MetalBufferToMetalTexture,
    CpuAccelerateRepack,
    CoreMlBoundaryReformat,
    MlxCompatibilityViewCreation,
    QuantizedLayoutExpansion,
}

/// Record of a materialization operation.
#[derive(Debug, Clone)]
pub struct MaterializationReceipt {
    pub kind: MaterializationKind,
    pub source_representation: String,
    pub destination_representation: String,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub duration_us: u64,
    pub reason: String,
}

/// The canonical hidden state carrier.
///
/// Replaces `Option<mlx_rs::Array>` as the activation type flowing through
/// the PhaseEngine. Tracks provenance and representation.
pub struct CurrentActivation {
    pub tensor_id: TensorId,
    pub arena_binding: Option<ArenaBinding>,
    pub representation: ActivationRepresentation,
    pub dtype: TensorDType,
    pub layout: TensorLayoutContract,
    pub generation: ActivationGeneration,
    pub producer_phase: PhaseId,
    /// Legacy MLX compatibility view — temporary, will be removed.
    #[cfg(feature = "mlx-backend")]
    pub mlx_compatibility_view: Option<mlx_rs::Array>,
}

impl CurrentActivation {
    pub fn new(
        tensor_id: TensorId,
        representation: ActivationRepresentation,
        dtype: TensorDType,
        layout: TensorLayoutContract,
        producer_phase: PhaseId,
    ) -> Self {
        Self {
            tensor_id,
            arena_binding: None,
            representation,
            dtype,
            layout,
            generation: ActivationGeneration(0),
            producer_phase,
            #[cfg(feature = "mlx-backend")]
            mlx_compatibility_view: None,
        }
    }
}

/// Record of a tensor binding for receipts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTensorBinding {
    pub tensor_id: String,
    pub representation: String,
    pub byte_size: u64,
    pub generation: u64,
}
