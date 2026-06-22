//! Runtime-visible read-only view of a serialized phase program.
//!
//! The runtime receives a deserialized program but must not
//! mutate or reconstruct it.  These views provide safe read
//! access to the pre-compiled program structure.

#[derive(Debug, Clone)]
pub struct RuntimePhaseProgramView {
    pub program_id: String,
    pub phase_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone)]
pub struct RuntimePhaseView {
    pub phase_id: String,
    pub lane: String,
    pub artifact_id: String,
}
