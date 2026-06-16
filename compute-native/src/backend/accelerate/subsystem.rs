//! Accelerate subsystem definitions.
//!
//! This module defines the different Accelerate subsystems that can execute
//! canonical operations. Each subsystem represents a distinct execution path
//! within the Accelerate framework.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Accelerate subsystems available for operation execution.
///
/// Each subsystem represents a distinct execution path within Apple's Accelerate framework:
///
/// - **BNNS**: Basic Neural Network Subroutines - Apple's neural network library for training and inference
/// - **BNNSGraph**: CPU graph execution path that can execute compiled model graphs
/// - **BLAS**: Basic Linear Algebra Subprograms for dense linear algebra operations
/// - **LAPACK**: Linear Algebra Package for more complex linear algebra operations
/// - **vDSP**: Vector Digital Signal Processing library for vector elementwise/reduction/math kernels
/// - **vForce**: Vector Force - optimized vector math operations (part of vDSP family)
/// - **Reference**: Fallback scalar/reference kernels for unsupported operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerateSubsystem {
    Bnns,
    BnnsGraph,
    Blas,
    Lapack,
    Vdsp,
    Vforce,
    Reference,
}

impl fmt::Display for AccelerateSubsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateSubsystem::Bnns => write!(f, "bnns"),
            AccelerateSubsystem::BnnsGraph => write!(f, "bnns_graph"),
            AccelerateSubsystem::Blas => write!(f, "blas"),
            AccelerateSubsystem::Lapack => write!(f, "lapack"),
            AccelerateSubsystem::Vdsp => write!(f, "vdsp"),
            AccelerateSubsystem::Vforce => write!(f, "vforce"),
            AccelerateSubsystem::Reference => write!(f, "reference"),
        }
    }
}

impl AccelerateSubsystem {
    pub fn is_neural_network(&self) -> bool {
        matches!(self, AccelerateSubsystem::Bnns | AccelerateSubsystem::BnnsGraph)
    }

    pub fn is_linear_algebra(&self) -> bool {
        matches!(self, AccelerateSubsystem::Blas | AccelerateSubsystem::Lapack)
    }

    pub fn is_vector_math(&self) -> bool {
        matches!(self, AccelerateSubsystem::Vdsp | AccelerateSubsystem::Vforce)
    }

    pub fn is_reference(&self) -> bool {
        matches!(self, AccelerateSubsystem::Reference)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AccelerateSubsystem::Bnns => "bnns",
            AccelerateSubsystem::BnnsGraph => "bnns_graph",
            AccelerateSubsystem::Blas => "blas",
            AccelerateSubsystem::Lapack => "lapack",
            AccelerateSubsystem::Vdsp => "vdsp",
            AccelerateSubsystem::Vforce => "vforce",
            AccelerateSubsystem::Reference => "reference",
        }
    }

    pub fn all() -> &'static [AccelerateSubsystem] {
        &[
            AccelerateSubsystem::Bnns,
            AccelerateSubsystem::BnnsGraph,
            AccelerateSubsystem::Blas,
            AccelerateSubsystem::Lapack,
            AccelerateSubsystem::Vdsp,
            AccelerateSubsystem::Vforce,
            AccelerateSubsystem::Reference,
        ]
    }

    pub fn default_for_category(category: &str) -> AccelerateSubsystem {
        match category {
            "matmul" | "gemm" | "linear_algebra" => AccelerateSubsystem::Blas,
            "elementwise" | "activation" => AccelerateSubsystem::Vdsp,
            "reduction" => AccelerateSubsystem::Vdsp,
            "neural_network" | "graph" => AccelerateSubsystem::BnnsGraph,
            "layout" | "memory" => AccelerateSubsystem::Reference,
            _ => AccelerateSubsystem::Reference,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadingPolicy {
    SingleThreaded,
    MultiThreaded,
    SubsystemDefault,
}

impl fmt::Display for ThreadingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreadingPolicy::SingleThreaded => write!(f, "single_threaded"),
            ThreadingPolicy::MultiThreaded => write!(f, "multi_threaded"),
            ThreadingPolicy::SubsystemDefault => write!(f, "subsystem_default"),
        }
    }
}

impl Default for ThreadingPolicy {
    fn default() -> Self {
        ThreadingPolicy::SubsystemDefault
    }
}