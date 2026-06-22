use crate::compute_image::phase_graph::{DeclaredFallback, EdgeSemanticKind, PhaseId};
use serde::{Deserialize, Serialize};

/// A fallback decomposition entry in the phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseFallbackEntry {
    pub phase_id: PhaseId,
    pub fallback: DeclaredFallback,
}

/// Registry of fallback decompositions for a phase graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseFallbackRegistry {
    pub entries: Vec<PhaseFallbackEntry>,
}
