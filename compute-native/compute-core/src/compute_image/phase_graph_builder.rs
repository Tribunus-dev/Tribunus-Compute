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

        // WeightResidency phase (only when layers exist)
        if self.num_layers > 0 {
            let wr_id = PhaseId("weight_residency".to_string());
            phases.push(EmittedPhaseV2 {
                id: wr_id.clone(),
                kind: EmittedPhaseKind::WeightResidency,
                layer_index: None,
                lane_binding: LaneBinding {
                    primary_lane: "accelerate".into(),
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
            if let Some(p) = &prev_id {
                edges.push(EmittedEdgeV2 {
                    from_phase: p.clone(),
                    to_phase: wr_id.clone(),
                    semantic_kind: EdgeSemanticKind::ProducerCompletion,
                    label: Some("arena_ready".into()),
                    metadata: HashMap::new(),
                });
            }
            prev_id = Some(wr_id);
        }

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

            // Residual RMSNorm phase (between attention and MLP) — runs on Accelerate for small element-wise ops
            let rmsnorm_id = PhaseId(format!("layer_{}_residual_rmsnorm", layer));
            edges.push(EmittedEdgeV2 {
                from_phase: attn_id.clone(),
                to_phase: rmsnorm_id.clone(),
                semantic_kind: EdgeSemanticKind::TensorData,
                label: Some("hidden".into()),
                metadata: HashMap::new(),
            });
            phases.push(EmittedPhaseV2 {
                id: rmsnorm_id.clone(),
                kind: EmittedPhaseKind::AccelerateBlock,
                layer_index: Some(layer),
                lane_binding: LaneBinding {
                    primary_lane: "accelerate".into(),
                    fallback_lanes: vec!["mlx".into()],
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

            // MLP phase
            let mlp_id = PhaseId(format!("layer_{}_mlp", layer));
            edges.push(EmittedEdgeV2 {
                from_phase: rmsnorm_id.clone(),
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
            // Build metadata with model dimensions for probe-dependent runners.
            let mut meta = HashMap::new();
            if let Some(li) = pv2.layer_index {
                meta.insert("layer_index".to_string(), li.to_string());
        }
            if self.hidden_size > 0 {
                meta.insert("hidden_size".to_string(), self.hidden_size.to_string());
        }
            if self.num_heads > 0 {
                meta.insert("n_heads".to_string(), self.num_heads.to_string());
        }
            // Assume num_kv_heads == num_heads unless overridden.
            meta.insert("n_kv_heads".to_string(), self.num_heads.to_string());
            if self.head_dim > 0 {
                meta.insert("head_dim".to_string(), self.head_dim.to_string());
        }
            phases.push(EmittedPhase {
                phase_id: pv2.id.0.clone(),
                kind,
                lane: ComputeLane::Metal,
                ops: pv2.operations.iter().map(|o| o.0.clone()).collect(),
                arena_slots: vec![],
                tensor_reads: pv2.tensor_reads.iter().map(|t| t.0.clone()).collect(),
                tensor_writes: pv2.tensor_writes.iter().map(|t| t.0.clone()).collect(),
                estimated_ops: 100,
                metadata: meta,
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
        EmittedPhaseKind::Prologue => PhaseKind::LegacyMlxPrologue,
        EmittedPhaseKind::LayerAttention => PhaseKind::MlxDecode,
        EmittedPhaseKind::LayerMlp => PhaseKind::MlxDecode,
        EmittedPhaseKind::Epilogue => PhaseKind::LegacyMlxEpilogue,
        EmittedPhaseKind::Sampling => PhaseKind::Sampling,
        EmittedPhaseKind::ArenaAlloc => PhaseKind::ArenaAlloc,
        EmittedPhaseKind::MemoryPlanApply => PhaseKind::ArenaAlloc,
        EmittedPhaseKind::WeightResidency => PhaseKind::WeightResidency,
        EmittedPhaseKind::ExplicitMaterialization => PhaseKind::Transfer,
        EmittedPhaseKind::Synchronization => PhaseKind::SyncBarrier,
        EmittedPhaseKind::FusedMetalKernel => PhaseKind::MetalFusedKernel,
        EmittedPhaseKind::CoreMlSubgraph => PhaseKind::CoreMlGraph,
        EmittedPhaseKind::AccelerateBlock => PhaseKind::ResidualRmsNorm,
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
        // Expected: arena_alloc + weight_residency + prologue
        //   + (layer_0_attn + layer_0_residual_rmsnorm + layer_0_mlp)
        //   + (layer_1_attn + layer_1_residual_rmsnorm + layer_1_mlp)
        //   + epilogue + sampling = 11 phases
        assert_eq!(graph.phases.len(), 11);
        // Edges: 10 connections (arena->wr, wr->prologue, prologue->attn0,
        //   attn0->rmsnorm0, rmsnorm0->mlp0, mlp0->attn1, attn1->rmsnorm1,
        //   rmsnorm1->mlp1, mlp1->epilogue, epilogue->sampling)
        assert_eq!(graph.edges.len(), 10);
        // Verify weight_residency phase exists
        assert!(graph.phases.iter().any(|p| p.id.0 == "weight_residency"));
        // Verify residual_rmsnorm phases exist
        assert!(graph.phases.iter().any(|p| p.id.0 == "layer_0_residual_rmsnorm"));
        assert!(graph.phases.iter().any(|p| p.id.0 == "layer_1_residual_rmsnorm"));

    }

    #[test]
    fn test_build_single_layer() {
        let builder = PhaseGraphBuilder::new(1);
        let graph = builder.build_v2();
        // arena + wr + prologue + (attn + rmsnorm + mlp) + epilogue + sampling = 8
        assert_eq!(graph.phases.len(), 8);
    }

    #[test]
    fn test_zero_layer_no_weight_residency() {
        let builder = PhaseGraphBuilder::new(0);
        let graph = builder.build_v2();
        // No weight_residency phase when num_layers == 0
        assert!(!graph.phases.iter().any(|p| p.id.0 == "weight_residency"));
        // arena_alloc + (no prologue) + (no epilogue) + sampling = 2
        assert_eq!(graph.phases.len(), 2);
    }

    #[test]
    fn test_v1_conversion() {
        let builder = PhaseGraphBuilder::new(2);
        let v1 = builder.build_v1();
        assert_eq!(v1.phases.len(), 11);
        assert_eq!(v1.edges.len(), 10);
    }
}
