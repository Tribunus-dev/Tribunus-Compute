use serde::{Deserialize, Serialize};

/// Program version identifier for a phase graph.
///
/// Incremented when the compiler produces a semantically different graph
/// for the same model architecture (e.g., different fusion selection,
/// different lane binding, different fallback topology).
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhaseProgramVersion(pub u32);

impl PhaseProgramVersion {
    pub fn current() -> Self {
        Self(1)
    }
}
