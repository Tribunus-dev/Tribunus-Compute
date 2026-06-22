//! Ready queue — given a phase DAG and a set of completed phases, determine
//! which phases have all their dependencies satisfied and are ready to run.

use crate::compute_image::phase_dag::EmittedPhaseGraph;

/// Tracks which phases are ready based on the DAG and completed set.
pub struct ReadyQueue<'a> {
    dag: &'a EmittedPhaseGraph,
}

impl<'a> ReadyQueue<'a> {
    pub fn new(dag: &'a EmittedPhaseGraph) -> Self {
        Self { dag }
    }

    /// Return all phases whose predecessors are all in the `completed` set.
    /// Phases already in `completed` are excluded.
    pub fn ready_phases(&self, completed: &std::collections::HashSet<String>) -> Vec<&'a crate::compute_image::phase_dag::EmittedPhase> {
        self.dag
            .phases
            .iter()
            .filter(|phase| {
                if completed.contains(&phase.phase_id) {
                    return false;
                }
                let preds = self.dag.predecessors(&phase.phase_id);
                preds.iter().all(|p| completed.contains(&p.phase_id))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::phase_dag::{
        ComputeLane, EmittedArenaPlan, EmittedConcurrencyPlan, EmittedPhase,
        EmittedPhaseEdge, PhaseKind, SemanticKind,
    };
    use std::collections::HashMap;

    fn make_phase(id: &str) -> EmittedPhase {
        EmittedPhase {
            phase_id: id.into(),
            kind: PhaseKind::MlxDecode,
            lane: ComputeLane::Metal,
            ops: vec![],
            arena_slots: vec![],
            tensor_reads: vec![],
            tensor_writes: vec!["out".into()],
            estimated_ops: 0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_ready_returns_phases_with_no_deps() {
        let dag = EmittedPhaseGraph {
            phases: vec![make_phase("p0"), make_phase("p1")],
            edges: vec![],
            arena_plan: EmittedArenaPlan { total_bytes: 0, slots: vec![] },
            concurrency_plan: EmittedConcurrencyPlan { independent_sets: vec![] },
            compiler_version: "test".into(),
        };
        let rq = ReadyQueue::new(&dag);
        let completed = std::collections::HashSet::new();
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_ready_respects_dependencies() {
        let dag = EmittedPhaseGraph {
            phases: vec![make_phase("a"), make_phase("b")],
            edges: vec![EmittedPhaseEdge {
                from_phase: "a".into(), to_phase: "b".into(),
                semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
            }],
            arena_plan: EmittedArenaPlan { total_bytes: 0, slots: vec![] },
            concurrency_plan: EmittedConcurrencyPlan { independent_sets: vec![] },
            compiler_version: "test".into(),
        };
        let rq = ReadyQueue::new(&dag);
        let mut completed = std::collections::HashSet::new();

        // Initially only 'a' is ready
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "a");

        // After 'a' completes, 'b' becomes ready
        completed.insert("a".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "b");
    }

    #[test]
    fn test_ready_excludes_completed() {
        let dag = EmittedPhaseGraph {
            phases: vec![make_phase("a"), make_phase("b")],
            edges: vec![],
            arena_plan: EmittedArenaPlan { total_bytes: 0, slots: vec![] },
            concurrency_plan: EmittedConcurrencyPlan { independent_sets: vec![] },
            compiler_version: "test".into(),
        };
        let rq = ReadyQueue::new(&dag);
        let mut completed = std::collections::HashSet::new();
        completed.insert("a".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "b");
    }
}
