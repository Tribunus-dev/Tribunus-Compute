//! Accelerate operation support definitions.
//!
//! This module defines the support classification for each canonical operation,
//! including whether it's supported, which subsystem would execute it,
//! and detailed support information.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{dtype::DTypePolicy, layout::LayoutPolicy, lowering::AccelerateLoweringKind, ops::CanonicalOp, subsystem::AccelerateSubsystem, layout::ShapeConstraints, layout::AllocationPolicy};

/// Support classification for a canonical operation.
///
/// Each op should return an `AccelerateSupport` record with fields like:
/// - supported: whether the operation is supported
/// - subsystem: which Accelerate subsystem would execute it
/// - dtype_policy: how dtypes are handled
/// - layout_policy: how layouts are handled
/// - shape_constraints: any shape requirements
/// - allocation_policy: memory allocation requirements
/// - determinism_notes: notes about deterministic execution
/// - fallback_reason: reason if unsupported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelerateSupport {
    /// The canonical operation this support record describes.
    pub op: CanonicalOp,
    /// Whether this operation is supported by Accelerate.
    pub supported: bool,
    /// The primary subsystem that would execute this operation.
    pub subsystem: AccelerateSubsystem,
    /// The lowering kind for this operation.
    pub lowering_kind: AccelerateLoweringKind,
    /// How dtypes are handled for this operation.
    pub dtype_policy: DTypePolicy,
    /// How layouts are handled for this operation.
    pub layout_policy: LayoutPolicy,
    /// Shape constraints for this operation.
    pub shape_constraints: ShapeConstraints,
    /// Memory allocation policy for this operation.
    pub allocation_policy: AllocationPolicy,
    /// Notes about determinism for this operation.
    pub determinism_notes: String,
    /// Reason for unsupported status (if applicable).
    pub fallback_reason: Option<String>,
    /// Priority score for this support (higher is better).
    pub priority: u32,
}

impl AccelerateSupport {
    /// Creates a new supported operation record.
    pub fn supported(
        op: CanonicalOp,
        subsystem: AccelerateSubsystem,
        lowering_kind: AccelerateLoweringKind,
    ) -> Self {
        Self {
            op,
            supported: true,
            subsystem,
            lowering_kind,
            dtype_policy: DTypePolicy::Native,
            layout_policy: LayoutPolicy::Native,
            shape_constraints: ShapeConstraints::new(),
            allocation_policy: AllocationPolicy::NoAllocation,
            determinism_notes: String::new(),
            fallback_reason: None,
            priority: 100,
        }
    }

    /// Creates a new unsupported operation record.
    pub fn unsupported(op: CanonicalOp, reason: &str) -> Self {
        Self {
            op,
            supported: false,
            subsystem: AccelerateSubsystem::Reference,
            lowering_kind: AccelerateLoweringKind::Reference,
            dtype_policy: DTypePolicy::Unsupported,
            layout_policy: LayoutPolicy::Unsupported,
            shape_constraints: ShapeConstraints::new(),
            allocation_policy: AllocationPolicy::NoAllocation,
            determinism_notes: String::new(),
            fallback_reason: Some(reason.to_string()),
            priority: 0,
        }
    }

    /// Creates a new fallback operation record.
    pub fn fallback(op: CanonicalOp, subsystem: AccelerateSubsystem, reason: &str) -> Self {
        Self {
            op,
            supported: true, // Fallback is still "supported" in the sense that it will work
            subsystem,
            lowering_kind: AccelerateLoweringKind::Reference,
            dtype_policy: DTypePolicy::Native,
            layout_policy: LayoutPolicy::Native,
            shape_constraints: ShapeConstraints::new(),
            allocation_policy: AllocationPolicy::NoAllocation,
            determinism_notes: "Fallback implementation - deterministic".to_string(),
            fallback_reason: Some(reason.to_string()),
            priority: 10, // Lower priority than native
        }
    }

    /// Returns true if this operation is natively supported (not a fallback).
    pub fn is_native(&self) -> bool {
        self.supported && self.lowering_kind != AccelerateLoweringKind::Reference
    }

    /// Returns true if this operation requires fallback.
    pub fn requires_fallback(&self) -> bool {
        !self.supported || self.lowering_kind == AccelerateLoweringKind::Reference
    }

    /// Returns the support status as a string.
    pub fn status(&self) -> &'static str {
        if !self.supported {
            "unsupported"
        } else if self.is_native() {
            "native"
        } else {
            "fallback"
        }
    }

    /// Builder method to set dtype policy.
    pub fn with_dtype_policy(mut self, policy: DTypePolicy) -> Self {
        self.dtype_policy = policy;
        self
    }

    /// Builder method to set layout policy.
    pub fn with_layout_policy(mut self, policy: LayoutPolicy) -> Self {
        self.layout_policy = policy;
        self
    }

    /// Builder method to set shape constraints.
    pub fn with_shape_constraints(mut self, constraints: ShapeConstraints) -> Self {
        self.shape_constraints = constraints;
        self
    }

    /// Builder method to set allocation policy.
    pub fn with_allocation_policy(mut self, policy: AllocationPolicy) -> Self {
        self.allocation_policy = policy;
        self
    }

    /// Builder method to set determinism notes.
    pub fn with_determinism_notes(mut self, notes: &str) -> Self {
        self.determinism_notes = notes.to_string();
        self
    }

    /// Builder method to set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

impl fmt::Display for AccelerateSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] -> {} (dtype:{}, layout:{}, priority:{})",
            self.op,
            self.status(),
            self.subsystem,
            self.dtype_policy,
            self.layout_policy,
            self.priority
        )
    }
}

/// Support table for all canonical operations.
///
/// This provides the v0 support classification for each operation.
#[derive(Debug, Clone)]
pub struct OpSupportTable {
    /// Support records for all canonical operations.
    pub supports: Vec<AccelerateSupport>,
}

impl OpSupportTable {
    /// Creates a new empty support table.
    pub fn new() -> Self {
        Self { supports: Vec::new() }
    }

    /// Creates the v0 support table with classifications for all canonical ops.
    pub fn v0() -> Self {
        let mut table = Self::new();

        // Memory operations
        table.add(AccelerateSupport::supported(
            CanonicalOp::ConstantRoundtrip,
            AccelerateSubsystem::Reference,
            AccelerateLoweringKind::MemoryCopy,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - memory copy"));

        table.add(AccelerateSupport::supported(
            CanonicalOp::Identity,
            AccelerateSubsystem::Reference,
            AccelerateLoweringKind::LayoutTransform,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - no-op"));

        // Elementwise operations -> vDSP
        table.add(AccelerateSupport::supported(
            CanonicalOp::Add,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - vDSP vector add"));

        table.add(AccelerateSupport::supported(
            CanonicalOp::Multiply,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - vDSP vector multiply"));

        // Activation functions -> vDSP/vForce
        table.add(AccelerateSupport::supported(
            CanonicalOp::Sigmoid,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - vDSP vector sigmoid"));

        // SiLU is a composite: sigmoid(x) * x
        table.add(AccelerateSupport::supported(
            CanonicalOp::Silu,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_determinism_notes("Deterministic - composite sigmoid * multiply"));

        // Matrix multiplication -> BLAS
        let mut matmul_support = AccelerateSupport::supported(
            CanonicalOp::Matmul,
            AccelerateSubsystem::Blas,
            AccelerateLoweringKind::BlasGemm,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Convert) // May need layout conversion
        .with_determinism_notes("Deterministic - BLAS GEMM");

        // Matmul requires at least 2 dimensions
        let mut constraints = ShapeConstraints::new();
        constraints.min_dims = Some(2);
        constraints.must_be_matrix = true;
        matmul_support = matmul_support.with_shape_constraints(constraints);

        table.add(matmul_support);

        // Layout operations
        table.add(AccelerateSupport::supported(
            CanonicalOp::Reshape,
            AccelerateSubsystem::Reference,
            AccelerateLoweringKind::LayoutTransform,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_allocation_policy(AllocationPolicy::NoAllocation)
        .with_determinism_notes("Deterministic - metadata only"));

        table.add(AccelerateSupport::supported(
            CanonicalOp::Transpose,
            AccelerateSubsystem::Reference,
            AccelerateLoweringKind::LayoutTransform,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Convert) // May need materialization
        .with_allocation_policy(AllocationPolicy::Temporary) // May need temp storage
        .with_determinism_notes("Deterministic - metadata or materialized"));

        // Softmax -> vDSP (exp, max, sum, divide)
        let mut softmax_support = AccelerateSupport::supported(
            CanonicalOp::Softmax,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspReduction,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_allocation_policy(AllocationPolicy::TemporaryAndOutput)
        .with_determinism_notes("Deterministic - vDSP exp/reduce/divide");

        // Softmax requires at least 1 dimension
        let mut softmax_constraints = ShapeConstraints::new();
        softmax_constraints.min_dims = Some(1);
        softmax_support = softmax_support.with_shape_constraints(softmax_constraints);

        table.add(softmax_support);

        // KV-cache operations (not yet supported in v0)
        table.add(AccelerateSupport::unsupported(
            CanonicalOp::KvCacheView,
            "KV-cache operations not yet implemented in v0",
        ));
        table.add(AccelerateSupport::unsupported(
            CanonicalOp::KvCacheWrite,
            "KV-cache operations not yet implemented in v0",
        ));
        table.add(AccelerateSupport::unsupported(
            CanonicalOp::KvCacheAppend,
            "KV-cache operations not yet implemented in v0",
        ));

        table
    }

    /// Adds a support record to the table.
    pub fn add(&mut self, support: AccelerateSupport) -> &mut Self {
        self.supports.push(support);
        self
    }

    /// Gets support information for a specific operation.
    pub fn get(&self, op: CanonicalOp) -> Option<&AccelerateSupport> {
        self.supports.iter().find(|s| s.op == op)
    }

    /// Returns true if the given operation is supported.
    pub fn is_supported(&self, op: CanonicalOp) -> bool {
        self.get(op).map(|s| s.supported).unwrap_or(false)
    }

    /// Returns the support status for all operations as a summary.
    pub fn summary(&self) -> Vec<(CanonicalOp, bool, AccelerateSubsystem)> {
        self.supports
            .iter()
            .map(|s| (s.op, s.supported, s.subsystem))
            .collect()
    }

    /// Returns only the supported operations.
    pub fn supported_ops(&self) -> Vec<CanonicalOp> {
        self.supports
            .iter()
            .filter(|s| s.supported)
            .map(|s| s.op)
            .collect()
    }

    /// Returns only the unsupported operations.
    pub fn unsupported_ops(&self) -> Vec<CanonicalOp> {
        self.supports
            .iter()
            .filter(|s| !s.supported)
            .map(|s| s.op)
            .collect()
    }
}

impl Default for OpSupportTable {
    fn default() -> Self {
        Self::v0()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_display() {
        let support = AccelerateSupport::supported(
            CanonicalOp::Add,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        );
        let display = format!("{}", support);
        assert!(display.contains("add"));
        assert!(display.contains("native"));
        assert!(display.contains("vdsp"));
    }

    #[test]
    fn test_support_status() {
        let supported = AccelerateSupport::supported(
            CanonicalOp::Add,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        );
        assert_eq!(supported.status(), "native");
        assert!(supported.is_native());
        assert!(!supported.requires_fallback());

        let unsupported = AccelerateSupport::unsupported(CanonicalOp::KvCacheView, "not implemented");
        assert_eq!(unsupported.status(), "unsupported");
        assert!(!unsupported.is_native());
        assert!(unsupported.requires_fallback());
    }

    #[test]
    fn test_support_builder() {
        let support = AccelerateSupport::supported(
            CanonicalOp::Add,
            AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::VdspVector,
        )
        .with_dtype_policy(DTypePolicy::Native)
        .with_layout_policy(LayoutPolicy::Native)
        .with_priority(200);

        assert_eq!(support.dtype_policy, DTypePolicy::Native);
        assert_eq!(support.layout_policy, LayoutPolicy::Native);
        assert_eq!(support.priority, 200);
    }

    #[test]
    fn test_support_table_v0() {
        let table = OpSupportTable::v0();
        
        // Check that all v0 core ops are supported
        for op in CanonicalOp::v0_core() {
            assert!(table.is_supported(*op), "{} should be supported", op);
        }

        // Check that KV-cache ops are not supported
        for op in CanonicalOp::v1_kv_cache() {
            assert!(!table.is_supported(*op), "{} should not be supported in v0", op);
        }
    }

    #[test]
    fn test_support_table_summary() {
        let table = OpSupportTable::v0();
        let summary = table.summary();
        
        assert!(!summary.is_empty());
        assert_eq!(summary.len(), CanonicalOp::all().len());
    }

    #[test]
    fn test_support_table_filtering() {
        let table = OpSupportTable::v0();
        
        let supported = table.supported_ops();
        let unsupported = table.unsupported_ops();
        
        assert!(!supported.is_empty());
        assert!(!unsupported.is_empty());
        assert!(supported.contains(&CanonicalOp::Add));
        assert!(unsupported.contains(&CanonicalOp::KvCacheView));
    }

    #[test]
    fn test_support_table_default() {
        let table = OpSupportTable::default();
        assert!(!table.supports.is_empty());
    }
}
