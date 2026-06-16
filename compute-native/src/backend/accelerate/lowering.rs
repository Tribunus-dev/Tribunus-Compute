//! Accelerate lowering definitions.
//!
//! This module defines the lowering kinds and decisions for mapping canonical
//! operations to Accelerate subsystems.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::subsystem::AccelerateSubsystem;

/// Lowering kinds for Accelerate operations.
///
/// Each lowering kind represents a specific execution path within Accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerateLoweringKind {
    /// Lower to BNNS (Basic Neural Network Subroutines)
    Bnns,
    /// Lower to BNNSGraph for compiled graph execution
    BnnsGraph,
    /// Lower to BLAS GEMM (General Matrix Multiply)
    BlasGemm,
    /// Lower to BLAS other operations (e.g., TRSM, etc.)
    BlasOther,
    /// Lower to LAPACK operations
    Lapack,
    /// Lower to vDSP vector operations
    VdspVector,
    /// Lower to vDSP reduction operations
    VdspReduction,
    /// Lower to vDSP matrix operations
    VdspMatrix,
    /// Lower to vForce vector math
    Vforce,
    /// Lower to reference/fallback implementation
    Reference,
    /// Layout transformation only (no compute)
    LayoutTransform,
    /// Memory copy/movement only
    MemoryCopy,
}

impl fmt::Display for AccelerateLoweringKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateLoweringKind::Bnns => write!(f, "bnns"),
            AccelerateLoweringKind::BnnsGraph => write!(f, "bnns_graph"),
            AccelerateLoweringKind::BlasGemm => write!(f, "blas_gemm"),
            AccelerateLoweringKind::BlasOther => write!(f, "blas_other"),
            AccelerateLoweringKind::Lapack => write!(f, "lapack"),
            AccelerateLoweringKind::VdspVector => write!(f, "vdsp_vector"),
            AccelerateLoweringKind::VdspReduction => write!(f, "vdsp_reduction"),
            AccelerateLoweringKind::VdspMatrix => write!(f, "vdsp_matrix"),
            AccelerateLoweringKind::Vforce => write!(f, "vforce"),
            AccelerateLoweringKind::Reference => write!(f, "reference"),
            AccelerateLoweringKind::LayoutTransform => write!(f, "layout_transform"),
            AccelerateLoweringKind::MemoryCopy => write!(f, "memory_copy"),
        }
    }
}

impl AccelerateLoweringKind {
    /// Returns the corresponding Accelerate subsystem for this lowering kind.
    pub fn subsystem(&self) -> AccelerateSubsystem {
        match self {
            AccelerateLoweringKind::Bnns => AccelerateSubsystem::Bnns,
            AccelerateLoweringKind::BnnsGraph => AccelerateSubsystem::BnnsGraph,
            AccelerateLoweringKind::BlasGemm | AccelerateLoweringKind::BlasOther => {
                AccelerateSubsystem::Blas
            }
            AccelerateLoweringKind::Lapack => AccelerateSubsystem::Lapack,
            AccelerateLoweringKind::VdspVector
            | AccelerateLoweringKind::VdspReduction
            | AccelerateLoweringKind::VdspMatrix => AccelerateSubsystem::Vdsp,
            AccelerateLoweringKind::Vforce => AccelerateSubsystem::Vforce,
            AccelerateLoweringKind::Reference => AccelerateSubsystem::Reference,
            AccelerateLoweringKind::LayoutTransform | AccelerateLoweringKind::MemoryCopy => {
                AccelerateSubsystem::Reference
            }
        }
    }

    /// Returns true if this is a BLAS lowering.
    pub fn is_blas(&self) -> bool {
        matches!(
            self,
            AccelerateLoweringKind::BlasGemm | AccelerateLoweringKind::BlasOther
        )
    }

    /// Returns true if this is a vDSP lowering.
    pub fn is_vdsp(&self) -> bool {
        matches!(
            self,
            AccelerateLoweringKind::VdspVector
                | AccelerateLoweringKind::VdspReduction
                | AccelerateLoweringKind::VdspMatrix
        )
    }

    /// Returns true if this is a neural network lowering.
    pub fn is_neural_network(&self) -> bool {
        matches!(
            self,
            AccelerateLoweringKind::Bnns | AccelerateLoweringKind::BnnsGraph
        )
    }

    /// Returns true if this is a reference/fallback lowering.
    pub fn is_reference(&self) -> bool {
        matches!(self, AccelerateLoweringKind::Reference)
    }

    /// Returns true if this is a layout/memory operation.
    pub fn is_layout_or_memory(&self) -> bool {
        matches!(
            self,
            AccelerateLoweringKind::LayoutTransform | AccelerateLoweringKind::MemoryCopy
        )
    }

    /// Returns the lowering kind as a string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            AccelerateLoweringKind::Bnns => "bnns",
            AccelerateLoweringKind::BnnsGraph => "bnns_graph",
            AccelerateLoweringKind::BlasGemm => "blas_gemm",
            AccelerateLoweringKind::BlasOther => "blas_other",
            AccelerateLoweringKind::Lapack => "lapack",
            AccelerateLoweringKind::VdspVector => "vdsp_vector",
            AccelerateLoweringKind::VdspReduction => "vdsp_reduction",
            AccelerateLoweringKind::VdspMatrix => "vdsp_matrix",
            AccelerateLoweringKind::Vforce => "vforce",
            AccelerateLoweringKind::Reference => "reference",
            AccelerateLoweringKind::LayoutTransform => "layout_transform",
            AccelerateLoweringKind::MemoryCopy => "memory_copy",
        }
    }

    /// Returns all lowering kinds.
    pub fn all() -> &'static [AccelerateLoweringKind] {
        &[
            AccelerateLoweringKind::Bnns,
            AccelerateLoweringKind::BnnsGraph,
            AccelerateLoweringKind::BlasGemm,
            AccelerateLoweringKind::BlasOther,
            AccelerateLoweringKind::Lapack,
            AccelerateLoweringKind::VdspVector,
            AccelerateLoweringKind::VdspReduction,
            AccelerateLoweringKind::VdspMatrix,
            AccelerateLoweringKind::Vforce,
            AccelerateLoweringKind::Reference,
            AccelerateLoweringKind::LayoutTransform,
            AccelerateLoweringKind::MemoryCopy,
        ]
    }
}

impl Default for AccelerateLoweringKind {
    fn default() -> Self {
        AccelerateLoweringKind::Reference
    }
}

/// Lowering decision with priority and reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweringDecision {
    /// The chosen lowering kind.
    pub kind: AccelerateLoweringKind,
    /// Priority score (higher is better).
    pub priority: u32,
    /// Reason for this decision.
    pub reason: String,
    /// Whether this is a fallback decision.
    pub is_fallback: bool,
}

impl LoweringDecision {
    pub fn new(kind: AccelerateLoweringKind, priority: u32, reason: &str) -> Self {
        Self {
            kind,
            priority,
            reason: reason.to_string(),
            is_fallback: false,
        }
    }

    pub fn fallback(kind: AccelerateLoweringKind, reason: &str) -> Self {
        Self {
            kind,
            priority: 0,
            reason: reason.to_string(),
            is_fallback: true,
        }
    }
}

impl Default for LoweringDecision {
    fn default() -> Self {
        Self {
            kind: AccelerateLoweringKind::default(),
            priority: 0,
            reason: String::new(),
            is_fallback: false,
        }
    }
}

/// Lowering strategy for choosing between multiple possible lowering paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoweringStrategy {
    /// Prefer performance over all else.
    Performance,
    /// Prefer numerical accuracy/stability.
    Accuracy,
    /// Prefer minimal memory usage.
    Memory,
    /// Prefer deterministic execution.
    Deterministic,
    /// Use default strategy (balanced).
    Default,
}

impl fmt::Display for LoweringStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringStrategy::Performance => write!(f, "performance"),
            LoweringStrategy::Accuracy => write!(f, "accuracy"),
            LoweringStrategy::Memory => write!(f, "memory"),
            LoweringStrategy::Deterministic => write!(f, "deterministic"),
            LoweringStrategy::Default => write!(f, "default"),
        }
    }
}

impl Default for LoweringStrategy {
    fn default() -> Self {
        LoweringStrategy::Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowering_kind_display() {
        assert_eq!(AccelerateLoweringKind::Bnns.to_string(), "bnns");
        assert_eq!(AccelerateLoweringKind::BnnsGraph.to_string(), "bnns_graph");
        assert_eq!(AccelerateLoweringKind::BlasGemm.to_string(), "blas_gemm");
        assert_eq!(AccelerateLoweringKind::VdspVector.to_string(), "vdsp_vector");
        assert_eq!(AccelerateLoweringKind::Reference.to_string(), "reference");
    }

    #[test]
    fn test_lowering_kind_subsystem() {
        assert_eq!(
            AccelerateLoweringKind::Bnns.subsystem(),
            AccelerateSubsystem::Bnns
        );
        assert_eq!(
            AccelerateLoweringKind::BlasGemm.subsystem(),
            AccelerateSubsystem::Blas
        );
        assert_eq!(
            AccelerateLoweringKind::VdspVector.subsystem(),
            AccelerateSubsystem::Vdsp
        );
        assert_eq!(
            AccelerateLoweringKind::Reference.subsystem(),
            AccelerateSubsystem::Reference
        );
    }

    #[test]
    fn test_lowering_kind_categories() {
        assert!(AccelerateLoweringKind::BlasGemm.is_blas());
        assert!(AccelerateLoweringKind::BlasOther.is_blas());
        assert!(!AccelerateLoweringKind::VdspVector.is_blas());

        assert!(AccelerateLoweringKind::VdspVector.is_vdsp());
        assert!(AccelerateLoweringKind::VdspReduction.is_vdsp());
        assert!(!AccelerateLoweringKind::BlasGemm.is_vdsp());

        assert!(AccelerateLoweringKind::Bnns.is_neural_network());
        assert!(AccelerateLoweringKind::BnnsGraph.is_neural_network());
        assert!(!AccelerateLoweringKind::BlasGemm.is_neural_network());

        assert!(AccelerateLoweringKind::Reference.is_reference());
        assert!(!AccelerateLoweringKind::BlasGemm.is_reference());

        assert!(AccelerateLoweringKind::LayoutTransform.is_layout_or_memory());
        assert!(AccelerateLoweringKind::MemoryCopy.is_layout_or_memory());
        assert!(!AccelerateLoweringKind::BlasGemm.is_layout_or_memory());
    }

    #[test]
    fn test_lowering_kind_as_str() {
        assert_eq!(AccelerateLoweringKind::Bnns.as_str(), "bnns");
        assert_eq!(AccelerateLoweringKind::BlasGemm.as_str(), "blas_gemm");
    }

    #[test]
    fn test_all_lowering_kinds() {
        let all = AccelerateLoweringKind::all();
        assert_eq!(all.len(), 13);
        assert!(all.contains(&AccelerateLoweringKind::Bnns));
        assert!(all.contains(&AccelerateLoweringKind::Reference));
    }

    #[test]
    fn test_lowering_kind_default() {
        assert_eq!(
            AccelerateLoweringKind::default(),
            AccelerateLoweringKind::Reference
        );
    }

    #[test]
    fn test_lowering_decision() {
        let decision = LoweringDecision::new(AccelerateLoweringKind::BlasGemm, 100, "optimal");
        assert_eq!(decision.kind, AccelerateLoweringKind::BlasGemm);
        assert_eq!(decision.priority, 100);
        assert_eq!(decision.reason, "optimal");
        assert!(!decision.is_fallback);

        let fallback = LoweringDecision::fallback(AccelerateLoweringKind::Reference, "unsupported");
        assert!(fallback.is_fallback);
        assert_eq!(fallback.priority, 0);
    }

    #[test]
    fn test_lowering_decision_default() {
        let decision = LoweringDecision::default();
        assert_eq!(decision.kind, AccelerateLoweringKind::Reference);
        assert_eq!(decision.priority, 0);
        assert!(!decision.is_fallback);
    }

    #[test]
    fn test_lowering_strategy_display() {
        assert_eq!(LoweringStrategy::Performance.to_string(), "performance");
        assert_eq!(LoweringStrategy::Accuracy.to_string(), "accuracy");
        assert_eq!(LoweringStrategy::Memory.to_string(), "memory");
        assert_eq!(LoweringStrategy::Deterministic.to_string(), "deterministic");
        assert_eq!(LoweringStrategy::Default.to_string(), "default");
    }

    #[test]
    fn test_lowering_strategy_default() {
        assert_eq!(LoweringStrategy::default(), LoweringStrategy::Default);
    }
}
