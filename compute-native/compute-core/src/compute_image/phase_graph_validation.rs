use crate::compute_image::phase_dag::EmittedPhaseGraph;
use crate::compute_image::phase_graph::{
    EdgeSemanticKind, EmittedPhaseGraphV2, EmittedPhaseKind, PhaseId,
};
use std::collections::{HashMap, HashSet};

/// Result of validating a phase graph.
#[derive(Debug)]
pub struct GraphValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
}

/// A validation error describing what rule was violated.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub code: ValidationErrorCode,
    pub message: String,
    pub phase_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    CyclicDependency,
    MissingPhase,
    MissingEdge,
    DuplicatePhaseId,
    TensorReadWithoutProducer,
    TensorWriteContractMismatch,
    WeightResidencyMissing,
    FusedArtifactMissing,
    ZeroCopyContractMismatch,
    KvGenerationMissing,
    FallbackPathMissing,
    CancellationBarrierViolation,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}",
            match self.code {
                ValidationErrorCode::CyclicDependency => "E001",
                ValidationErrorCode::MissingPhase => "E002",
                ValidationErrorCode::MissingEdge => "E003",
                ValidationErrorCode::DuplicatePhaseId => "E004",
                ValidationErrorCode::TensorReadWithoutProducer => "E005",
                ValidationErrorCode::TensorWriteContractMismatch => "E006",
                ValidationErrorCode::WeightResidencyMissing => "E007",
                ValidationErrorCode::FusedArtifactMissing => "E008",
                ValidationErrorCode::ZeroCopyContractMismatch => "E009",
                ValidationErrorCode::KvGenerationMissing => "E010",
                ValidationErrorCode::FallbackPathMissing => "E011",
                ValidationErrorCode::CancellationBarrierViolation => "E012",
            },
            self.message
        )
    }
}

/// Validate a V2 phase graph against all compiler rules.
pub fn validate_phase_graph_v2(graph: &EmittedPhaseGraphV2) -> GraphValidationResult {
    let mut errors = Vec::new();

    // 1. No duplicate phase IDs
    let mut seen_ids = HashSet::new();
    for phase in &graph.phases {
        if !seen_ids.insert(&phase.id) {
            errors.push(ValidationError {
                code: ValidationErrorCode::DuplicatePhaseId,
                message: format!("duplicate phase id: {:?}", phase.id),
                phase_id: Some(phase.id.0.clone()),
            });
        }
    }

    // Build phase id set for edge validation
    let phase_ids: HashSet<&PhaseId> = graph.phases.iter().map(|p| &p.id).collect();

    // 2. Every edge from/to must reference existing phases
    for edge in &graph.edges {
        if !phase_ids.contains(&edge.from_phase) {
            errors.push(ValidationError {
                code: ValidationErrorCode::MissingPhase,
                message: format!("edge from_phase {:?} not found in graph", edge.from_phase),
                phase_id: Some(edge.from_phase.0.clone()),
            });
        }
        if !phase_ids.contains(&edge.to_phase) {
            errors.push(ValidationError {
                code: ValidationErrorCode::MissingPhase,
                message: format!("edge to_phase {:?} not found in graph", edge.to_phase),
                phase_id: Some(edge.to_phase.0.clone()),
            });
        }
    }

    // 3. No cycles via Kahn's algorithm
    let mut in_degree: HashMap<&PhaseId, usize> = HashMap::new();
    let mut adjacency: HashMap<&PhaseId, Vec<&PhaseId>> = HashMap::new();
    for phase in &graph.phases {
        in_degree.entry(&phase.id).or_insert(0);
        adjacency.entry(&phase.id).or_insert_with(Vec::new);
    }
    for edge in &graph.edges {
        adjacency
            .entry(&edge.from_phase)
            .or_default()
            .push(&edge.to_phase);
        *in_degree.entry(&edge.to_phase).or_insert(0) += 1;
    }
    let mut queue: Vec<&PhaseId> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut visited = 0;
    while let Some(id) = queue.pop() {
        visited += 1;
        if let Some(neighbors) = adjacency.get(id) {
            for next in neighbors {
                if let Some(deg) = in_degree.get_mut(next) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(next);
                    }
                }
            }
        }
    }
    if visited != graph.phases.len() {
        errors.push(ValidationError {
            code: ValidationErrorCode::CyclicDependency,
            message: format!(
                "graph has a cycle: visited {} of {} phases",
                visited,
                graph.phases.len()
            ),
            phase_id: None,
        });
    }

    // 4. Fused phases must have artifact bindings
    for phase in &graph.phases {
        if phase.kind == EmittedPhaseKind::FusedMetalKernel && phase.artifact_binding.is_none() {
            errors.push(ValidationError {
                code: ValidationErrorCode::FusedArtifactMissing,
                message: format!("fused phase {:?} lacks artifact binding", phase.id),
                phase_id: Some(phase.id.0.clone()),
            });
        }
    }

    // 5. Weight phases must have required_weights declared
    for phase in &graph.phases {
        if phase.kind == EmittedPhaseKind::WeightResidency && phase.required_weights.is_none() {
            errors.push(ValidationError {
                code: ValidationErrorCode::WeightResidencyMissing,
                message: format!(
                    "weight residency phase {:?} missing required_weights",
                    phase.id
                ),
                phase_id: Some(phase.id.0.clone()),
            });
        }
    }

    // 6. Core ML subgraphs must have artifact bindings
    for phase in &graph.phases {
        if phase.kind == EmittedPhaseKind::CoreMlSubgraph && phase.artifact_binding.is_none() {
            errors.push(ValidationError {
                code: ValidationErrorCode::FusedArtifactMissing,
                message: format!(
                    "Core ML subgraph phase {:?} lacks artifact binding",
                    phase.id
                ),
                phase_id: Some(phase.id.0.clone()),
            });
        }
    }

    // 7. KvGeneration edges must have a source that writes the KV state
    for edge in &graph.edges {
        if edge.semantic_kind == EdgeSemanticKind::KvGeneration {
            if let Some(from_phase) = graph.phases.iter().find(|p| p.id == edge.from_phase) {
                if from_phase.state_writes.is_empty() {
                    errors.push(ValidationError {
                        code: ValidationErrorCode::KvGenerationMissing,
                        message: format!(
                            "KvGeneration edge from {:?} but source has no state_writes",
                            edge.from_phase
                        ),
                        phase_id: Some(edge.from_phase.0.clone()),
                    });
                }
            }
        }
    }

    GraphValidationResult {
        valid: errors.is_empty(),
        errors,
    }
}

/// Validate a V1 phase graph (re-exports phase_dag's validation).
pub fn validate_phase_graph_v1(graph: &EmittedPhaseGraph) -> GraphValidationResult {
    match graph.validate() {
        Ok(()) => GraphValidationResult {
            valid: true,
            errors: vec![],
        },
        Err(e) => GraphValidationResult {
            valid: false,
            errors: vec![ValidationError {
                code: ValidationErrorCode::CyclicDependency,
                message: e,
                phase_id: None,
            }],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::manifest::SourceIdentity;
    use crate::compute_image::phase_graph::{
        CancellationClass, DeclaredFallback, EmittedEdgeV2, EmittedPhaseKind, EmittedPhaseV2,
        ExecutionClass, LaneBinding,
    };

    fn make_valid_graph() -> EmittedPhaseGraphV2 {
        let p1 = PhaseId("a".to_string());
        let p2 = PhaseId("b".to_string());
        EmittedPhaseGraphV2 {
            phases: vec![
                EmittedPhaseV2 {
                    id: p1.clone(),
                    kind: EmittedPhaseKind::Prologue,
                    layer_index: None,
                    lane_binding: LaneBinding {
                        primary_lane: "mlx".into(),
                        fallback_lanes: vec![],
                    },
                    operations: vec![],
                    tensor_reads: vec![],
                    tensor_writes: vec![],
                    state_reads: vec![],
                    state_writes: vec![],
                    required_weights: None,
                    input_contracts: vec![],
                    output_contracts: vec![],
                    artifact_binding: None,
                    fallback: None,
                    cancellation_class: CancellationClass::Preemptible,
                    execution_class: ExecutionClass::Required,
                },
                EmittedPhaseV2 {
                    id: p2.clone(),
                    kind: EmittedPhaseKind::LayerAttention,
                    layer_index: Some(0),
                    lane_binding: LaneBinding {
                        primary_lane: "mlx".into(),
                        fallback_lanes: vec![],
                    },
                    operations: vec![],
                    tensor_reads: vec![],
                    tensor_writes: vec![],
                    state_reads: vec![],
                    state_writes: vec![],
                    required_weights: None,
                    input_contracts: vec![],
                    output_contracts: vec![],
                    artifact_binding: None,
                    fallback: None,
                    cancellation_class: CancellationClass::Preemptible,
                    execution_class: ExecutionClass::Required,
                },
            ],
            edges: vec![EmittedEdgeV2 {
                from_phase: p1,
                to_phase: p2,
                semantic_kind: EdgeSemanticKind::TensorData,
                label: Some("hidden".into()),
                metadata: std::collections::HashMap::new(),
            }],
            compiler_version: "test".into(),
        }
    }

    #[test]
    fn test_valid_graph() {
        let result = validate_phase_graph_v2(&make_valid_graph());
        assert!(
            result.valid,
            "expected valid, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_cycle_detected() {
        let mut graph = make_valid_graph();
        let a_id = PhaseId("a".to_string());
        let b_id = PhaseId("b".to_string());
        graph.edges.push(EmittedEdgeV2 {
            from_phase: b_id,
            to_phase: a_id,
            semantic_kind: EdgeSemanticKind::TensorData,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::CyclicDependency));
    }

    #[test]
    fn test_duplicate_phase_id() {
        let mut graph = make_valid_graph();
        graph.phases.push(graph.phases[0].clone());
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::DuplicatePhaseId));
    }

    #[test]
    fn test_missing_phase_in_edge() {
        let mut graph = make_valid_graph();
        graph.edges.push(EmittedEdgeV2 {
            from_phase: PhaseId("nonexistent".to_string()),
            to_phase: PhaseId("b".to_string()),
            semantic_kind: EdgeSemanticKind::TensorData,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::MissingPhase));
    }

    #[test]
    fn test_fused_artifact_missing() {
        let mut graph = make_valid_graph();
        graph.phases.push(EmittedPhaseV2 {
            id: PhaseId("fused".to_string()),
            kind: EmittedPhaseKind::FusedMetalKernel,
            layer_index: None,
            lane_binding: LaneBinding {
                primary_lane: "metal".into(),
                fallback_lanes: vec![],
            },
            operations: vec![],
            tensor_reads: vec![],
            tensor_writes: vec![],
            state_reads: vec![],
            state_writes: vec![],
            required_weights: None,
            input_contracts: vec![],
            output_contracts: vec![],
            artifact_binding: None,
            fallback: None,
            cancellation_class: CancellationClass::Preemptible,
            execution_class: ExecutionClass::Required,
        });
        // Add a dummy edge to keep DAG acyclic
        graph.edges.push(EmittedEdgeV2 {
            from_phase: PhaseId("fused".to_string()),
            to_phase: PhaseId("a".to_string()),
            semantic_kind: EdgeSemanticKind::ProducerCompletion,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        // Also need to make fused reachable — add incoming edge
        graph.edges.push(EmittedEdgeV2 {
            from_phase: PhaseId("b".to_string()),
            to_phase: PhaseId("fused".to_string()),
            semantic_kind: EdgeSemanticKind::TensorData,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::FusedArtifactMissing));
    }

    #[test]
    fn test_coreml_subgraph_needs_artifact() {
        let mut graph = make_valid_graph();
        // Add a CoreMlSubgraph phase without artifact binding
        let coreml_id = PhaseId("coreml".to_string());
        graph.phases.push(EmittedPhaseV2 {
            id: coreml_id.clone(),
            kind: EmittedPhaseKind::CoreMlSubgraph,
            layer_index: None,
            lane_binding: LaneBinding {
                primary_lane: "coreml".into(),
                fallback_lanes: vec![],
            },
            operations: vec![],
            tensor_reads: vec![],
            tensor_writes: vec![],
            state_reads: vec![],
            state_writes: vec![],
            required_weights: None,
            input_contracts: vec![],
            output_contracts: vec![],
            artifact_binding: None,
            fallback: None,
            cancellation_class: CancellationClass::Preemptible,
            execution_class: ExecutionClass::Required,
        });
        // Wire it into the DAG (bidirectional so it's reachable)
        graph.edges.push(EmittedEdgeV2 {
            from_phase: PhaseId("b".to_string()),
            to_phase: coreml_id.clone(),
            semantic_kind: EdgeSemanticKind::TensorData,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        graph.edges.push(EmittedEdgeV2 {
            from_phase: coreml_id.clone(),
            to_phase: PhaseId("a".to_string()),
            semantic_kind: EdgeSemanticKind::ProducerCompletion,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert_eq!(
            result
                .errors
                .iter()
                .filter(|e| e.code == ValidationErrorCode::FusedArtifactMissing)
                .count(),
            1
        );
    }

    #[test]
    fn test_kv_generation_edge_missing_state_writes() {
        let mut graph = make_valid_graph();
        // Add edge with KvGeneration semantic from a phase with no state_writes
        graph.edges.push(EmittedEdgeV2 {
            from_phase: PhaseId("a".to_string()),
            to_phase: PhaseId("b".to_string()),
            semantic_kind: EdgeSemanticKind::KvGeneration,
            label: None,
            metadata: std::collections::HashMap::new(),
        });
        let result = validate_phase_graph_v2(&graph);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::KvGenerationMissing));
    }
}
