#![cfg(test)]

use super::phase_dag::*;
use serde_json;
use std::collections::HashMap;

fn simple_arena() -> EmittedArenaPlan {
    EmittedArenaPlan {
        total_bytes: 32,
        slots: vec![ArenaSlotRef {
            slot_id: "slot0".into(),
            byte_size: 32,
            alignment: 64,
            lane: ComputeLane::Metal,
        }],
    }
}

fn empty_concurrency() -> EmittedConcurrencyPlan {
    EmittedConcurrencyPlan {
        independent_sets: vec![],
    }
}

fn make_phase(id: &str, kind: PhaseKind) -> EmittedPhase {
    EmittedPhase {
        phase_id: id.into(),
        kind,
        lane: ComputeLane::Metal,
        ops: vec![format!("op_{}", id)],
        arena_slots: vec![],
        tensor_reads: vec![],
        tensor_writes: vec!["output".into()],
        estimated_ops: 100,
        metadata: HashMap::new(),
    }
}

// ── Roundtrip ─────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_simple() {
    let phases = vec![
        make_phase("prologue", PhaseKind::MlxDecode),
        make_phase("decode", PhaseKind::MlxDecode),
        make_phase("epilogue", PhaseKind::MlxDecode),
    ];
    let edges = vec![
        EmittedPhaseEdge {
            from_phase: "prologue".into(),
            to_phase: "decode".into(),
            semantic_kind: SemanticKind::Data,
            label: Some("activation".into()),
            metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "decode".into(),
            to_phase: "epilogue".into(),
            semantic_kind: SemanticKind::Data,
            label: Some("hidden".into()),
            metadata: HashMap::new(),
        },
    ];
    let dag = EmittedPhaseGraph {
        phases,
        edges,
        arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };

    assert!(dag.validate().is_ok());

    let json = serde_json::to_string(&dag).expect("serialize");
    let restored: EmittedPhaseGraph = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored.phases.len(), 3);
    assert_eq!(restored.edges.len(), 2);
    assert_eq!(restored.arena_plan.total_bytes, 32);
    assert_eq!(restored.compiler_version, "tribunus-phase-dag-v1");
    assert_eq!(restored.phases[0].phase_id, "prologue");
    assert_eq!(restored.phases[1].phase_id, "decode");
    assert_eq!(restored.phases[2].phase_id, "epilogue");
}

// ── Cycle detection ───────────────────────────────────────────────────────

#[test]
fn test_validate_rejects_cycle() {
    let phases = vec![
        make_phase("a", PhaseKind::MlxDecode),
        make_phase("b", PhaseKind::MlxDecode),
        make_phase("c", PhaseKind::MlxDecode),
    ];
    let edges = vec![
        EmittedPhaseEdge {
            from_phase: "a".into(), to_phase: "b".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "b".into(), to_phase: "c".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "c".into(), to_phase: "a".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
    ];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    assert!(dag.validate().is_err());
}

// ── Missing phase in edge ─────────────────────────────────────────────────

#[test]
fn test_validate_rejects_missing_phase() {
    let phases = vec![make_phase("a", PhaseKind::MlxDecode)];
    let edges = vec![EmittedPhaseEdge {
        from_phase: "a".into(), to_phase: "nonexistent".into(),
        semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
    }];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    assert!(dag.validate().is_err());
}

// ── Duplicate phase_id ────────────────────────────────────────────────────

#[test]
fn test_validate_rejects_duplicate_id() {
    let phases = vec![
        make_phase("dup", PhaseKind::MlxDecode),
        make_phase("dup", PhaseKind::MlxDecode),
    ];
    let edges = vec![];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    assert!(dag.validate().is_err());
}

// ── Topological order ─────────────────────────────────────────────────────

#[test]
fn test_topological_order_valid() {
    // A → B, A → C, B → D, C → D
    let phases = vec![
        make_phase("a", PhaseKind::MlxDecode),
        make_phase("b", PhaseKind::MetalFusedKernel),
        make_phase("c", PhaseKind::CoreMlGraph),
        make_phase("d", PhaseKind::MlxDecode),
    ];
    let edges = vec![
        EmittedPhaseEdge {
            from_phase: "a".into(), to_phase: "b".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "a".into(), to_phase: "c".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "b".into(), to_phase: "d".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "c".into(), to_phase: "d".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
    ];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    let order = dag.topological_order().expect("acyclic");
    assert_eq!(order.len(), 4);
    // A must be first, D must be last
    assert_eq!(order[0].phase_id, "a");
    assert_eq!(order[3].phase_id, "d");
}

// ── Topological order rejects cycle ───────────────────────────────────────

#[test]
fn test_topological_order_rejects_cycle() {
    let phases = vec![
        make_phase("a", PhaseKind::MlxDecode),
        make_phase("b", PhaseKind::MlxDecode),
        make_phase("c", PhaseKind::MlxDecode),
    ];
    let edges = vec![
        EmittedPhaseEdge {
            from_phase: "a".into(), to_phase: "b".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "b".into(), to_phase: "c".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "c".into(), to_phase: "a".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
    ];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    assert!(dag.topological_order().is_err());
}

// ── Display format ────────────────────────────────────────────────────────

#[test]
fn test_display_format() {
    let phases = vec![make_phase("a", PhaseKind::MlxDecode)];
    let dag = EmittedPhaseGraph {
        phases, edges: vec![], arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    let s = format!("{}", dag);
    assert!(s.contains("PhaseGraph("));
    assert!(s.contains("1 phases"));
    assert!(s.contains("Arena: 32 bytes"));
}

// ── All phase kinds roundtrip ─────────────────────────────────────────────

#[test]
fn test_all_phase_kinds_roundtrip() {
    let kinds = vec![
        PhaseKind::MlxDecode,
        PhaseKind::MetalFusedKernel,
        PhaseKind::CoreMlGraph,
        PhaseKind::AccelMatMul,
        PhaseKind::AccelElementWise,
        PhaseKind::ArenaAlloc,
        PhaseKind::SyncBarrier,
        PhaseKind::Transfer,
        PhaseKind::ResidualRmsNorm,
    ];
    let phases: Vec<EmittedPhase> = kinds
        .into_iter()
        .enumerate()
        .map(|(i, kind)| {
            let mut p = make_phase(&format!("p{}", i), kind);
            // Some kinds might not write tensors; add a write for validation
            if p.tensor_writes.is_empty() {
                p.tensor_writes.push("out".into());
            }
            p
        })
        .collect();

    let dag = EmittedPhaseGraph {
        phases,
        edges: vec![],
        arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    assert!(dag.validate().is_ok());

    let json = serde_json::to_string(&dag).expect("serialize");
    let restored: EmittedPhaseGraph = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.phases.len(), 9);
}

// ── Predecessors/Successors ───────────────────────────────────────────────

#[test]
fn test_predecessors_and_successors() {
    let phases = vec![
        make_phase("a", PhaseKind::MlxDecode),
        make_phase("b", PhaseKind::MlxDecode),
        make_phase("c", PhaseKind::MlxDecode),
    ];
    let edges = vec![
        EmittedPhaseEdge {
            from_phase: "a".into(), to_phase: "b".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
        EmittedPhaseEdge {
            from_phase: "b".into(), to_phase: "c".into(),
            semantic_kind: SemanticKind::Data, label: None, metadata: HashMap::new(),
        },
    ];
    let dag = EmittedPhaseGraph {
        phases, edges, arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    let preds_b: Vec<&str> = dag.predecessors("b").iter().map(|p| p.phase_id.as_str()).collect();
    assert_eq!(preds_b, vec!["a"]);

    let succ_a: Vec<&str> = dag.successors("a").iter().map(|p| p.phase_id.as_str()).collect();
    assert_eq!(succ_a, vec!["b"]);

    assert!(dag.predecessors("a").is_empty());
    assert!(dag.successors("c").is_empty());
}

// ── Edge validation: data edges need label ────────────────────────────────

// (Validation rule 7 about Data edges needing a label is documented but not
// enforced by the current validate() impl — this test documents current behavior.)

#[test]
fn test_validation_requires_tensor_writes() {
    let mut phase = make_phase("orphan", PhaseKind::MlxDecode);
    phase.tensor_writes.clear();
    let phases = vec![phase];
    let dag = EmittedPhaseGraph {
        phases,
        edges: vec![],
        arena_plan: simple_arena(),
        concurrency_plan: empty_concurrency(),
        compiler_version: "tribunus-phase-dag-v1".into(),
    };
    // Every phase must have at least one tensor_write (validation rule 5)
    assert!(dag.validate().is_err());
}
