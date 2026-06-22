//! SerializedPhaseProgram — compiled, shape-specialized execution program
//! for the SealedComputeImageExecutable.
//!
//! Contains all phases (units of schedulable work), dependency edges,
//! arena and residency plan references, artifact selection decisions,
//! and declared fallback chains.

use serde::{Deserialize, Serialize};
use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::integration::ContentHash;

pub type ProgramId = String;
pub type ArenaPlanId = String;
pub type ResidencyPlanId = String;
pub type ReceiptId = String;
pub type StateDomainId = String;
pub type PhaseId = String;

/// A compiled, shape-specialized execution program.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializedPhaseProgram {
    pub program_id: ProgramId,
    pub program_hash: ContentHash,
    pub shape_class: ExecutionShapeClass,
    pub execution_kind: ExecutionKind,
    pub phases: Vec<SerializedPhase>,
    pub edges: Vec<SerializedPhaseEdge>,
    pub arena_plan_id: ArenaPlanId,
    pub residency_plan_id: ResidencyPlanId,
    pub default_artifact_selection: ProgramArtifactSelection,
    pub fallback_chains: Vec<DeclaredFallbackChain>,
    pub proof_receipt_ids: Vec<ReceiptId>,
    /// Raw compiled program payload bytes.
    #[serde(default)]
    pub program_bytes: Vec<u8>,
}
impl SerializedPhaseProgram {
    /// Create a new `SerializedPhaseProgram` with the given identity and bytes.
    ///
    /// Other fields (phases, edges, plans, etc.) are set to empty defaults.
    pub fn new(
        _schema_version: u32,
        program_id: String,
        shape_class: ExecutionShapeClass,
        program_bytes: Vec<u8>,
    ) -> Self {
        Self {
            program_id,
            program_hash: ContentHash(0),
            shape_class,
            execution_kind: ExecutionKind::Decode,
            phases: vec![],
            edges: vec![],
            arena_plan_id: String::new(),
            residency_plan_id: String::new(),
            default_artifact_selection: ProgramArtifactSelection {
                artifact_ids: vec![],
            },
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionKind {
    Decode,
    Prefill,
    MixedBatch,
    DiffusionForward,
}

/// A single scheduled phase within a serialized program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializedPhase {
    pub phase_id: PhaseId,
    pub semantic_operation: SemanticOperation,
    pub lane: ExecutionLane,
    pub artifact_identity: CanonicalArtifactIdentity,
    pub input_bindings: Vec<ProgramBinding>,
    pub output_bindings: Vec<ProgramBinding>,
    pub dependency_contract: PhaseDependencyContract,
    pub completion_contract: PhaseCompletionContract,
    pub resource_reservation: PhaseResourceReservation,
    pub state_domain: Option<StateDomainId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticOperation {
    RmsNorm,
    ResidualAdd,
    GateProj,
    UpProj,
    DownProj,
    QProj,
    KProj,
    VProj,
    Silu,
    Mul,
    RoPE,
    Softmax,
    CoreMlGraph { graph_id: String },
    SharedActivation,
    FusedMlpSwiGlu,
    FusedQkvEpilogue,
    FusedResidualRmsNorm,
    DecodeFlashAttention,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionLane {
    Metal,
    CoreMl,
    Accelerate,
    ControlPlaneCpu,
    FusionOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalArtifactIdentity {
    pub artifact_id: String,
    pub artifact_hash: ContentHash,
    pub artifact_kind: PhaseArtifactKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhaseArtifactKind {
    FullLayer,
    Elementwise,
    Projection,
    Attention,
    MlpBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramBinding {
    pub binding_id: String,
    pub binding_kind: BindingKind,
    pub content_object_id: Option<String>,
    pub arena_region_id: Option<String>,
    pub kv_cache_region: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BindingKind {
    ContentStoreObject,
    ActivationArenaRegion,
    KvCacheRegion,
    CoreMlModelHandle,
    CoreMlStateHandle,
    MetalBufferHandle,
    MaterializationRegion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseDependencyContract {
    pub dependencies_satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseCompletionContract {
    pub must_emit_receipt: bool,
    pub must_release_regions: bool,
    pub must_advance_epoch: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseResourceReservation {
    pub threadgroup_memory: u64,
    pub register_count: u32,
}

/// A directed dependency edge between two phases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializedPhaseEdge {
    pub producer: PhaseId,
    pub consumer: PhaseId,
    pub binding_id: String,
    pub dependency_kind: DependencyKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyKind {
    Data,
    TokenOrder,
    Epoch,
}

/// The set of artifact ids selected for this program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub struct ProgramArtifactSelection {
    pub artifact_ids: Vec<String>,
}

/// A declared fallback chain for a compatible variant switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredFallbackChain {
    pub primary: Vec<CanonicalArtifactIdentity>,
    pub fallbacks: Vec<FallbackStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FallbackStep {
    pub artifact: CanonicalArtifactIdentity,
    pub condition: String,
}

impl Default for ContentHash {
    fn default() -> Self {
        ContentHash(0)
    }
}
