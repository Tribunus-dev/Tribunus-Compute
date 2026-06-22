use crate::compute_image::phase_dag::{EmittedPhase, EmittedPhaseGraph};
use crate::compute_image::phase_graph::{EmittedPhaseGraphV2, PhaseId};
use std::collections::{HashMap, HashSet};

/// Computes the ready set of phases given completed phases.
///
/// A phase is ready when all its predecessor edges have been satisfied
/// (the source phases are in the completed set).
pub struct ReadinessChecker;

impl ReadinessChecker {
    pub fn new() -> Self {
        Self
    }

    /// Compute ready phases for a V1 graph.
    pub fn ready_phases(
        &self,
        graph: &EmittedPhaseGraph,
        completed: &HashSet<String>,
    ) -> Vec<String> {
        let mut ready = Vec::new();
        'outer: for phase in &graph.phases {
            if completed.contains(&phase.phase_id) {
                continue;
            }
            // All predecessors must be in completed
            for edge in &graph.edges {
                if edge.to_phase == phase.phase_id && !completed.contains(&edge.from_phase) {
                    continue 'outer;
                }
            }
            ready.push(phase.phase_id.clone());
        }
        ready
    }

    /// Compute ready phases for a V2 graph.
    pub fn ready_phases_v2(
        &self,
        graph: &EmittedPhaseGraphV2,
        completed: &HashSet<PhaseId>,
    ) -> Vec<PhaseId> {
        let mut ready = Vec::new();
        'outer: for phase in &graph.phases {
            if completed.contains(&phase.id) {
                continue;
            }
            for edge in &graph.edges {
                if edge.to_phase == phase.id && !completed.contains(&edge.from_phase) {
                    continue 'outer;
                }
            }
            ready.push(phase.id.clone());
        }
        ready
    }
}
