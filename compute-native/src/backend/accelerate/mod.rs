//! Accelerate CPU inference backend for Tribunus.
//!
//! This module provides a **native-ready scaffold** for a first-class Accelerate backend.
//! It includes capability discovery, operation support classification, lowering decisions,
//! and execution receipts with truthful evidence reporting.
//!
//! # Implementation Status
//!
//! **Current State (HARDEN-ACCELERATE-PR2-TRUTHFUL-EVIDENCE-0001)**:
//!
//! - ✅ **Modeling Complete**: Full pipeline model with capability discovery, support classification,
//!   lowering decisions, and execution receipts.
//! - ✅ **Reference Backend**: All operations execute via reference implementations on all platforms.
//! - ✅ **Truthful Evidence**: Execution receipts distinguish between `lowering_subsystem` (intended)
//!   and `executed_subsystem` (actual), with mandatory fallback reasons.
//! - ✅ **Portable API**: All public types available on all platforms; only FFI linkage gated by cfg.
//! - ❌ **Native Calls**: No direct Accelerate.framework calls yet (vDSP, BLAS, BNNS not bound).
//!
//! **What This Means**:
//!
//! - On macOS: Lowering decisions target vDSP/BLAS/BNNS, but execution uses reference fallback.
//! - On Linux: Same behavior, with explicit "Accelerate unavailable" fallback reasons.
//! - BackendClassification::Pass means **native execution succeeded** (not yet achievable).
//! - BackendClassification::Fallback means reference was used (current state for all ops).
//!
//! # Architecture
//!
//! The backend is organized into five conceptual layers:
//!
//! 1. **Capability Discovery**: Platform, architecture, Accelerate availability,
//!    enabled subsystems, supported dtypes, supported layouts, threading policy.
//!
//! 2. **Canonical Op Coverage**: Support classification for each canonical operation
//!    with structured `AccelerateSupport` records.
//!
//! 3. **Lowering**: Compiler choices between BNNS, BLAS, vDSP/vForce, or reference fallback.
//!    Currently all operations lower to their intended subsystems but execute via reference.
//!
//! 4. **Runtime Execution**: `AccelerateExecutionPlan` that takes canonical phase IR
//!    plus input buffers and returns output buffers plus execution receipts.
//!
//! 5. **Evidence**: Reference-checked numerical validation with JSON evidence output.
//!    Evidence now includes both lowering_subsystem and executed_subsystem for truthful reporting.
//!
//! # Next Steps
//!
//! The next PR should implement actual native Accelerate calls:
//! - Bind vDSP_vadd, vDSP_vmul for add/multiply
//! - Bind cblas_sgemm for matmul
//! - Update kernels to call these when available
//! - Then BackendClassification::Pass will be achievable

pub mod activation;
pub mod capabilities;
pub mod conformance;
pub mod dtype;
pub mod evidence;
pub mod execution;
pub mod ffi;
pub mod kernels;
pub mod layout;
pub mod layout_handler;
pub mod lowering;
pub mod ops;
pub mod subsystem;
pub mod support;

// Re-export the main types for convenience
pub use activation::{ActivationContext, ActivationDispatcher, ActivationResult};
pub use capabilities::AccelerateBackendCapabilities;
pub use conformance::{BackendClassification, BackendId, BackendMatrix, BackendRegistry, ConformanceResult, AccelerateConformanceRunner};
pub use dtype::AccelerateDType;
pub use evidence::AccelerateEvidence;
pub use execution::{AccelerateExecutionPlan, AccelerateExecutionReceipt, NumericalStatus};
pub use ffi::{AccelerateHandle, AccelerateLinkage, AccelerateResult};
pub use kernels::{KernelContext, KernelDispatcher, KernelResult};
pub use layout::AccelerateLayout;
pub use layout_handler::{BlasLayoutAnalyzer, BlasLayoutDecision, LayoutHandler, LayoutTransform, TensorLayout};
pub use lowering::AccelerateLoweringKind;
pub use ops::CanonicalOp;
pub use subsystem::AccelerateSubsystem;
pub use support::AccelerateSupport;

/// Mission identifier for this Accelerate inference pipeline implementation.
pub const MISSION_ACCELERATE_INFERENCE_PIPELINE_V0: &str = "ACCELERATE-INFERENCE-PIPELINE-V0";

/// Backend identifier string for Accelerate.
pub const BACKEND_ACCELERATE: &str = "accelerate";

/// Default numerical tolerance for f32 elementwise operations.
pub const DEFAULT_F32_ELEMENTWISE_TOLERANCE: f32 = 1e-4;

/// Default numerical tolerance for f32 identity/reshape/transpose operations.
pub const DEFAULT_F32_IDENTITY_TOLERANCE: f32 = 1e-6;

/// Default cosine similarity threshold for larger matmul/softmax cases.
pub const DEFAULT_COSINE_SIMILARITY_THRESHOLD: f32 = 0.9999;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mission_identifier() {
        assert_eq!(
            MISSION_ACCELERATE_INFERENCE_PIPELINE_V0,
            "ACCELERATE-INFERENCE-PIPELINE-V0"
        );
    }

    #[test]
    fn test_backend_identifier() {
        assert_eq!(BACKEND_ACCELERATE, "accelerate");
    }

    #[test]
    fn test_tolerance_constants() {
        assert!(DEFAULT_F32_ELEMENTWISE_TOLERANCE > 0.0);
        assert!(DEFAULT_F32_IDENTITY_TOLERANCE > 0.0);
        assert!(DEFAULT_F32_IDENTITY_TOLERANCE < DEFAULT_F32_ELEMENTWISE_TOLERANCE);
        assert!(DEFAULT_COSINE_SIMILARITY_THRESHOLD > 0.99);
    }
}
