use crate::compute_image::phase_dag::{
    ComputeLane, EmittedArenaPlan, EmittedConcurrencyPlan, EmittedPhase, EmittedPhaseEdge,
    EmittedPhaseGraph, PhaseKind, SemanticKind,
};
use crate::compute_image::phase_graph::{
    CancellationClass, EdgeSemanticKind, EmittedEdgeV2, EmittedPhaseGraphV2, EmittedPhaseKind,
    EmittedPhaseV2, ExecutionClass, LaneBinding, PhaseId,
};
use std::collections::HashMap;

/// Builder for constructing layer-granular phase graphs.
///
/// Transforms model metadata (num layers, hidden dims, etc.) into
/// a complete EmittedPhaseGraphV2 with edges for:
/// - Prologue -> LayerAttention[0] -> LayerMlp[0] -> ... -> Epilogue -> Sampling
/// - Fallback decomposition edges for fused phases
pub struct PhaseGraphBuilder {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    has_prologue: bool,
    has_epilogue: bool,
}

impl PhaseGraphBuilder {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            hidden_size: 0,
            num_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            has_prologue: true,
            has_epilogue: true,
        }
    }

    pub fn with_dimensions(
        mut self,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Self {
        self.hidden_size = hidden_size;
        self.num_heads = num_heads;
        self.head_dim = head_dim;
        self.intermediate_size = intermediate_size;
        self
    }

    /// Build a standard decoder-layer phase graph.
    /// Topology: ArenaAlloc -> Prologue -> LayerAttention[0] -> LayerMlp[0] -> ... -> Epilogue ->
    /// Sampling
    pub fn build_v2(&self) -> EmittedPhaseGraphV2 {
        let mut phases = Vec::new();
        let mut edges = Vec::new();

        let mut prev_id: Option<PhaseId> = None;

        // Arena allocation phase
        let arena_id = PhaseId("arena_alloc".to_string());
        phases.push(EmittedPhaseV2 {
            id: arena_id.clone(),
            kind: EmittedPhaseKind::ArenaAlloc,
            layer_index: None,
            lane_binding: LaneBinding {
                primary_lane: "arena".into(),
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
            cancellation_class: CancellationClass::Barrier,
            execution_class: ExecutionClass::Required,
        });
        prev_id = Some(arena_id);

        // Prologue
        if self.has_prologue {
            let prologue_id = PhaseId("prologue".to_string());
            if let Some(p) = &prev_id {
                edges.push(EmittedEdgeV2 {
                    from_phase: p.clone(),
                    to_phase: prologue_id.clone(),
                    semantic_kind: EdgeSemanticKind::ProducerCompletion,
                    label: Some("arena_ready".into()),
                    metadata: HashMap::new(),
                });
            }
            phases.push(EmittedPhaseV2 {
                id: prologue_id.clone(),
                kind: EmittedPhaseKind::Prologue,
                layer_index: None,
                lane_binding: LaneBinding {
                    primary_lane: "mlx".into(),
                    fallback_lanes: vec!["accelerate".into()],
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
                cancellation_class: CancellationClass::Barrier,
                execution_class: ExecutionClass::Required,
            });
            prev_id = Some(prologue_id);
        }

        // Per-layer attention + MLP phases
        for layer in 0..self.num_layers {
            // Attention phase
            let attn_id = PhaseId(format!("layer_{}_attn", layer));
            if let Some(p) = &prev_id {
                edges.push(EmittedEdgeV2 {
                    from_phase: p.clone(),
                    to_phase: attn_id.clone(),
                    semantic_kind: EdgeSemanticKind::TensorData,
                    label: Some("hidden".into()),
                    metadata: HashMap::new(),
                });
            }
            phases.push(EmittedPhaseV2 {
                id: attn_id.clone(),
                kind: EmittedPhaseKind::LayerAttention,
                layer_index: Some(layer),
                lane_binding: LaneBinding {
                    primary_lane: "mlx".into(),
                    fallback_lanes: vec!["metal".into()],
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
            prev_id = Some(attn_id.clone());

            // MLP phase
            let mlp_id = PhaseId(format!("layer_{}_mlp", layer));
            edges.push(EmittedEdgeV2 {
                from_phase: attn_id.clone(),
                to_phase: mlp_id.clone(),
                semantic_kind: EdgeSemanticKind::TensorData,
                label: Some("hidden".into()),
                metadata: HashMap::new(),
            });
            phases.push(EmittedPhaseV2 {
                id: mlp_id.clone(),
                kind: EmittedPhaseKind::LayerMlp,
                layer_index: Some(layer),
                lane_binding: LaneBinding {
                    primary_lane: "mlx".into(),
                    fallback_lanes: vec!["accelerate".into()],
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
            prev_id = Some(mlp_id);
        }

        // Epilogue
        if self.has_epilogue {
            let epilogue_id = PhaseId("epilogue".to_string());
            if let Some(p) = &prev_id {
                edges.push(EmittedEdgeV2 {
                    from_phase: p.clone(),
                    to_phase: epilogue_id.clone(),
                    semantic_kind: EdgeSemanticKind::TensorData,
                    label: Some("hidden".into()),
                    metadata: HashMap::new(),
                });
            }
            phases.push(EmittedPhaseV2 {
                id: epilogue_id.clone(),
                kind: EmittedPhaseKind::Epilogue,
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
                cancellation_class: CancellationClass::Barrier,
                execution_class: ExecutionClass::Required,
            });
            prev_id = Some(epilogue_id);
        }

        // Sampling
        let sampling_id = PhaseId("sampling".to_string());
        if let Some(p) = &prev_id {
            edges.push(EmittedEdgeV2 {
                from_phase: p.clone(),
                to_phase: sampling_id.clone(),
                semantic_kind: EdgeSemanticKind::TensorData,
                label: Some("logits".into()),
                metadata: HashMap::new(),
            });
        }
        phases.push(EmittedPhaseV2 {
            id: sampling_id,
            kind: EmittedPhaseKind::Sampling,
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
            cancellation_class: CancellationClass::Barrier,
            execution_class: ExecutionClass::Required,
        });

        EmittedPhaseGraphV2 {
            phases,
            edges,
            compiler_version: "tribunus-phase-graph-v2".into(),
        }
    }

    /// Build a backward-compatible V1 graph from the V2 layout.
    /// This bridges old PhaseEngine with new builder.
    pub fn build_v1(&self) -> EmittedPhaseGraph {
        let v2 = self.build_v2();
        let mut phases = Vec::new();
        let mut dag_edges = Vec::new();

        for pv2 in &v2.phases {
            let kind = map_kind_to_v1(pv2.kind);
            phases.push(EmittedPhase {
                phase_id: pv2.id.0.clone(),
                kind,
                lane: ComputeLane::Metal,
                ops: pv2.operations.iter().map(|o| o.0.clone()).collect(),
                arena_slots: vec![],
                tensor_reads: pv2.tensor_reads.iter().map(|t| t.0.clone()).collect(),
                tensor_writes: pv2.tensor_writes.iter().map(|t| t.0.clone()).collect(),
                estimated_ops: 100,
                metadata: HashMap::new(),
            });
        }

        for ev2 in &v2.edges {
            dag_edges.push(EmittedPhaseEdge {
                from_phase: ev2.from_phase.0.clone(),
                to_phase: ev2.to_phase.0.clone(),
                semantic_kind: SemanticKind::Data,
                label: ev2.label.clone(),
                metadata: HashMap::new(),
            });
        }

        EmittedPhaseGraph {
            phases,
            edges: dag_edges,
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: vec![],
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: vec![],
            },
            compiler_version: "tribunus-phase-graph-v2-built".into(),
        }
    }
}

fn map_kind_to_v1(kind: EmittedPhaseKind) -> PhaseKind {
    match kind {
        EmittedPhaseKind::Prologue => PhaseKind::MlxDecode,
        EmittedPhaseKind::LayerAttention => PhaseKind::MlxDecode,
        EmittedPhaseKind::LayerMlp => PhaseKind::MlxDecode,
        EmittedPhaseKind::Epilogue => PhaseKind::MlxDecode,
        EmittedPhaseKind::Sampling => PhaseKind::MlxDecode,
        EmittedPhaseKind::ArenaAlloc => PhaseKind::ArenaAlloc,
        EmittedPhaseKind::MemoryPlanApply => PhaseKind::ArenaAlloc,
        EmittedPhaseKind::WeightResidency => PhaseKind::Transfer,
        EmittedPhaseKind::ExplicitMaterialization => PhaseKind::Transfer,
        EmittedPhaseKind::Synchronization => PhaseKind::SyncBarrier,
        EmittedPhaseKind::FusedMetalKernel => PhaseKind::MetalFusedKernel,
        EmittedPhaseKind::CoreMlSubgraph => PhaseKind::CoreMlGraph,
        EmittedPhaseKind::AccelerateBlock => PhaseKind::AccelMatMul,
        EmittedPhaseKind::LegacyMlxLayer => PhaseKind::MlxDecode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_two_layer_graph() {
        let builder = PhaseGraphBuilder::new(2).with_dimensions(4096, 32, 128, 14336);
        let graph = builder.build_v2();
        // Expected: arena_alloc + prologue + (layer_0_attn + layer_0_mlp) + (layer_1_attn +
        // layer_1_mlp) + epilogue + sampling = 8 phases
        assert_eq!(graph.phases.len(), 8);
        // Edges: 7 connections (arena->prologue, prologue->attn0, attn0->mlp0, mlp0->attn1,
        // attn1->mlp1, mlp1->epilogue, epilogue->sampling)
        assert_eq!(graph.edges.len(), 7);
    }

    #[test]
    fn test_build_single_layer() {
        let builder = PhaseGraphBuilder::new(1);
        let graph = builder.build_v2();
        // 1 + 1 + (1+1) + 1 + 1 = 5 phases
        assert_eq!(graph.phases.len(), 5);
    }

    #[test]
    fn test_v1_conversion() {
        let builder = PhaseGraphBuilder::new(2);
        let v1 = builder.build_v1();
        assert_eq!(v1.phases.len(), 8);
        assert_eq!(v1.edges.len(), 7);
    }
}
