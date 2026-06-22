use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── New types ─────────────────────────────────────────────────

/// Enriched phase identifier (human-readable).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhaseId(pub String);

/// Enriched edge semantic representing all dependency types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeSemanticKind {
    TensorData,
    ArenaOwnership,
    KvGeneration,
    CoreMlStateEpoch,
    WeightResidency,
    ProducerCompletion,
    RequestOrdering,
    ExplicitMaterialization,
    FallbackActivation,
}

/// Kinds of operations within a phase.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CanonicalOpId(pub String);

/// Tensor identifier within a phase graph.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorId(pub String);

/// State resource identifier (KV cache, Core ML state, etc.).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateResourceId(pub String);

/// Weight residency set identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightResidencySetId(pub String);

/// Artifact binding identifier (fused kernel, Core ML model).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBindingId(pub String);

/// Tensor layout contract describing expected tensor properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLayoutContract {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Option<Vec<usize>>,
    pub alignment: u64,
}

/// Lane binding for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneBinding {
    pub primary_lane: String,
    pub fallback_lanes: Vec<String>,
}

/// Emitted phase kind.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum EmittedPhaseKind {
    Prologue,
    LayerAttention,
    LayerMlp,
    Epilogue,
    Sampling,
    ArenaAlloc,
    MemoryPlanApply,
    WeightResidency,
    ExplicitMaterialization,
    Synchronization,
    FusedMetalKernel,
    CoreMlSubgraph,
    AccelerateBlock,
    LegacyMlxLayer,
}

/// Cancellation semantics for a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancellationClass {
    Preemptible,
    NonPreemptible,
    Barrier,
}

/// Execution class indicating performance / correctness criticality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionClass {
    Required,
    Optional,
    Diagnostic,
}

/// Declared fallback decomposition for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclaredFallback {
    pub reason: String,
    pub decomposed_phase_ids: Vec<PhaseId>,
    pub semantic_kind: EdgeSemanticKind,
}

/// The new enriched EmittedPhase with layer granularity and full metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedPhaseV2 {
    pub id: PhaseId,
    pub kind: EmittedPhaseKind,
    pub layer_index: Option<usize>,
    pub lane_binding: LaneBinding,
    pub operations: Vec<CanonicalOpId>,
    pub tensor_reads: Vec<TensorId>,
    pub tensor_writes: Vec<TensorId>,
    pub state_reads: Vec<StateResourceId>,
    pub state_writes: Vec<StateResourceId>,
    pub required_weights: Option<WeightResidencySetId>,
    pub input_contracts: Vec<TensorLayoutContract>,
    pub output_contracts: Vec<TensorLayoutContract>,
    pub artifact_binding: Option<ArtifactBindingId>,
    pub fallback: Option<DeclaredFallback>,
    pub cancellation_class: CancellationClass,
    pub execution_class: ExecutionClass,
}

/// Enriched edge with explicit semantic kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedEdgeV2 {
    pub from_phase: PhaseId,
    pub to_phase: PhaseId,
    pub semantic_kind: EdgeSemanticKind,
    pub label: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Resolved phase binding — artifact and launch params selected for this phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedPhaseBinding {
    pub phase_id: PhaseId,
    pub artifact_binding: Option<ArtifactBindingId>,
    pub launch_contract: Option<String>,
    pub expected_dtype: String,
    pub expected_shape: Vec<usize>,
}

/// The enriched emitted phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedPhaseGraphV2 {
    pub phases: Vec<EmittedPhaseV2>,
    pub edges: Vec<EmittedEdgeV2>,
    pub compiler_version: String,
}

impl Default for EmittedPhaseGraphV2 {
    fn default() -> Self {
        Self {
            phases: Vec::new(),
            edges: Vec::new(),
            compiler_version: "tribunus-phase-graph-v2".to_string(),
        }
    }
}
