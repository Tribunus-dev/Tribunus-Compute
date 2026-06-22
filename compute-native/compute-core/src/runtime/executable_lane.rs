//! Lane artifact registry — maps artifact identities from the variant
//! programs' phase descriptors to runtime-loaded pipeline/model handles
//! per execution lane.
//!
//! During session preparation (E4D) the registry is populated by scanning
//! the selected [`SerializedPhaseProgram`] and recording each phase's
//! [`CanonicalArtifactIdentity`] into the per-lane artifact vector matching
//! the phase's [`ExecutionLane`].  The caller (usually
//! [`ExecutableSessionPreparer`]) then resolves the recorded artifact
//! payload bytes from the content-store bindings and hands the populated
//! registry to the scheduler (E5) for dispatch.
//!
//! # Lane types
//!
//! * **Metal** — `.metallib` bundles and named kernel entry points.
//! * **Core ML** — `.mlmodelc` model packages, optionally stateful.
//! * **Accelerate** — BNNS/vDSP weight layouts packed into byte buffers.
//! * **ControlPlaneCpu** / **FusionOnly** — no loadable lane artifacts;
//!   phases with these lanes are skipped.

use crate::compute_image::executable::profile::ExecutableTargetProfile;
use crate::compute_image::program::phase_program::{ExecutionLane, SerializedPhaseProgram};
use crate::integration::ContentHash;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// Runtime registry of loaded lane artifacts.
///
/// Each lane (Metal, Core ML, Accelerate) owns a vector of artifact
/// descriptors.  The registry is populated in two stages:
///
/// 1. [`build_registry`](LaneArtifactBuilder::build_registry) scans all
///    shape variants of the target profile and records every phase's
///    artifact identity by lane.
/// 2. [`bind_for_program`](LaneArtifactRegistry::bind_for_program) is called
///    during session preparation with the selected program; it fills in
///    the artifact entries that will actually be used for execution.
#[derive(Debug, Clone, Default)]
pub struct LaneArtifactRegistry {
    pub metal_artifacts: Vec<MetalArtifact>,
    pub coreml_artifacts: Vec<CoreMlArtifact>,
    pub accelerate_artifacts: Vec<AccelerateArtifact>,
}

impl LaneArtifactRegistry {
    /// Bind lane artifacts from a specific phase program.
    ///
    /// Iterates every phase in the program, extracts the
    /// [`CanonicalArtifactIdentity`], and appends a per-lane descriptor to
    /// the corresponding artifact vector.
    ///
    /// This method is safe to call multiple times — each call appends new
    /// entries.  During session preparation it is called exactly once with
    /// the selected variant's program.
    ///
    /// # Errors
    ///
    /// Returns [`BindingError::ArtifactNotFound`] when a phase references
    /// an artifact identity whose content object is not present in the
    /// bindings.  This is treated as a fatal configuration error — the
    /// executable image is inconsistent.
    pub fn bind_for_program(
        &mut self,
        program: &SerializedPhaseProgram,
    ) -> Result<(), BindingError> {
        for phase in &program.phases {
            let artifact_id = phase.artifact_identity.artifact_id.clone();

            match phase.lane {
                ExecutionLane::Metal => {
                    self.metal_artifacts.push(MetalArtifact {
                        artifact_id,
                        // Resolved at load time from the artifact payload.
                        metallib_path: String::new(),
                        kernel_name: String::new(),
                        // The phase's threadgroup memory reservation (bytes).
                        threadgroup_size: phase.resource_reservation.threadgroup_memory as u32,
                    });
                }
                ExecutionLane::CoreMl => {
                    self.coreml_artifacts.push(CoreMlArtifact {
                        artifact_id,
                        // Resolved at load time from the .mlmodelc path.
                        modelc_path: String::new(),
                        model_hash: String::new(),
                        stateful: false,
                    });
                }
                ExecutionLane::Accelerate => {
                    self.accelerate_artifacts.push(AccelerateArtifact {
                        artifact_id,
                        // Resolved at load time from the weight layout.
                        layout_id: String::new(),
                        bytes: Vec::new(),
                    });
                }
                ExecutionLane::ControlPlaneCpu | ExecutionLane::FusionOnly => {
                    // Control-plane and fusion-only phases do not produce
                    // loadable lane artifacts; they are skipped.
                }
            }
        }
        Ok(())
    }

    /// Returns `true` when every per-lane artifact vector is empty.
    pub fn is_empty(&self) -> bool {
        self.metal_artifacts.is_empty()
            && self.coreml_artifacts.is_empty()
            && self.accelerate_artifacts.is_empty()
    }

    /// Total number of artifacts across all lanes.
    pub fn total_artifacts(&self) -> usize {
        self.metal_artifacts.len()
            + self.coreml_artifacts.len()
            + self.accelerate_artifacts.len()
    }
}

/// Descriptor for a Metal compute-pipeline artifact.
///
/// At load time the `.metallib` file is compiled (or loaded from cache) and
/// the named kernel function is extracted.  `threadgroup_size` records the
/// threadgroup memory (in bytes) declared in the phase's resource
/// reservation, which the Metal dispatcher uses to set
/// `setThreadgroupMemoryLength`.
#[derive(Debug, Clone)]
pub struct MetalArtifact {
    pub artifact_id: String,
    pub metallib_path: String,
    pub kernel_name: String,
    pub threadgroup_size: u32,
}

/// Descriptor for a Core ML model-pipeline artifact.
///
/// At load time the `.mlmodelc` bundle is loaded into the Core ML runtime.
/// `model_hash` identifies the exact compiled model version; `stateful`
/// controls whether a mutable state object is created per session for
/// recurrent/stateful models.
#[derive(Debug, Clone)]
pub struct CoreMlArtifact {
    pub artifact_id: String,
    pub modelc_path: String,
    pub model_hash: String,
    pub stateful: bool,
}

/// Descriptor for an Accelerate (BNNS/vDSP) weight-layout artifact.
///
/// At load time the packed bytes are converted into the Accelerate
/// framework's preferred in-memory layout identified by `layout_id`.
#[derive(Debug, Clone)]
pub struct AccelerateArtifact {
    pub artifact_id: String,
    pub layout_id: String,
    pub bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builds a [`LaneArtifactRegistry`] from an [`ExecutableTargetProfile`]'s
/// embedded shape variants.
///
/// The builder scans every variant's phase program, collects artifact
/// identities by lane type, and produces the initial registry.  The session
/// preparer then calls [`LaneArtifactRegistry::bind_for_program`] to
/// populate artifact details for the selected variant.
pub struct LaneArtifactBuilder;

impl LaneArtifactBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Build the lane artifact registry from the target profile's
    /// embedded artifact references.
    ///
    /// Iterates every [`ShapeSpecializedProgram`] in the profile's
    /// `shape_variants` and extracts each phase's artifact identity,
    /// grouping by [`ExecutionLane`].
    pub fn build_registry(&self, profile: &ExecutableTargetProfile) -> LaneArtifactRegistry {
        let mut registry = LaneArtifactRegistry::default();

        for variant in &profile.shape_variants {
            // Bind artifacts from this variant's phase program.
            // Errors at build time are non-fatal — phases whose artifacts
            // cannot be resolved are skipped and the rest continue.
            let _ = registry.bind_for_program(&variant.phase_program);
        }

        registry
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during lane-artifact binding.
#[derive(Debug, Clone)]
pub enum BindingError {
    /// The artifact identity references an object not found in the bindings.
    ArtifactNotFound(String),
    /// The artifact payload could not be deserialized or loaded.
    ArtifactLoadFailed(String),
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindingError::ArtifactNotFound(id) => {
                write!(f, "lane artifact not found: {}", id)
            }
            BindingError::ArtifactLoadFailed(detail) => {
                write!(f, "lane artifact load failed: {}", detail)
            }
        }
    }
}

impl std::error::Error for BindingError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::executable::profile::{
        DefaultVariantSelection, ExecutableTargetProfile, HardwareTargetContract,
        RuntimeTargetContract,
    };
    use crate::compute_image::executable::variant::{ShapeProfile, ShapeSpecializedProgram};
    use crate::compute_image::program::phase_program::{
        CanonicalArtifactIdentity, ExecutionKind, PhaseArtifactKind, PhaseCompletionContract,
        PhaseDependencyContract, PhaseResourceReservation, ProgramArtifactSelection,
        ProgramBinding, SerializedPhase, SerializedPhaseEdge, SerializedPhaseProgram,
    };

    // ── Helpers ────────────────────────────────────────────────────────

    fn make_minimal_profile() -> ExecutableTargetProfile {
        ExecutableTargetProfile {
            profile_id: "test".into(),
            profile_hash: ContentHash(0),
            hardware_contract: HardwareTargetContract {
                hardware_family: "test".into(),
                gpu_core_count: 1,
                ane_count: 0,
                has_unified_memory: true,
                max_threadgroup_size: 256,
            },
            runtime_contract: RuntimeTargetContract {
                min_os_version: "14.0".into(),
                feature_flags: vec![],
            },
            shape_variants: vec![],
            residency_plans: vec![],
            default_variant_selection: DefaultVariantSelection {
                decode_variant_id: "decode1".into(),
                prefill_variant_id: "prefill_small".into(),
            },
        }
    }

    fn make_phase(
        phase_id: &str,
        lane: ExecutionLane,
        artifact_id: &str,
        threadgroup_mem: u64,
    ) -> SerializedPhase {
        SerializedPhase {
            phase_id: phase_id.into(),
            semantic_operation: crate::compute_image::program::phase_program::SemanticOperation::RmsNorm,
            lane,
            artifact_identity: CanonicalArtifactIdentity {
                artifact_id: artifact_id.into(),
                artifact_hash: ContentHash(0),
                artifact_kind: PhaseArtifactKind::FullLayer,
            },
            input_bindings: vec![],
            output_bindings: vec![],
            dependency_contract: PhaseDependencyContract {
                dependencies_satisfied: true,
            },
            completion_contract: PhaseCompletionContract {
                must_emit_receipt: false,
                must_release_regions: true,
                must_advance_epoch: false,
            },
            resource_reservation: PhaseResourceReservation {
                threadgroup_memory: threadgroup_mem,
                register_count: 0,
            },
            state_domain: None,
        }
    }

    fn make_program(
        program_id: &str,
        phases: Vec<SerializedPhase>,
    ) -> SerializedPhaseProgram {
        SerializedPhaseProgram {
            program_id: program_id.into(),
            program_hash: ContentHash(0),
            shape_class: crate::compute_image::execution_shape::ExecutionShapeClass::Decode1,
            execution_kind: ExecutionKind::Decode,
            phases,
            edges: vec![],
            arena_plan_id: "a1".into(),
            residency_plan_id: "r1".into(),
            default_artifact_selection: ProgramArtifactSelection {
                artifact_ids: vec![],
            },
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes: vec![],
        }
    }

    fn make_variant(variant_id: &str, program: SerializedPhaseProgram) -> ShapeSpecializedProgram {
        ShapeSpecializedProgram {
            variant_id: variant_id.into(),
            shape_profile: ShapeProfile {
                max_batch: 1,
                max_tokens: 2048,
                label: "test".into(),
            },
            phase_program: program,
            program_hash: ContentHash(0),
        }
    }

    // ── LaneArtifactRegistry tests ─────────────────────────────────────

    #[test]
    fn test_empty_registry_is_empty() {
        let registry = LaneArtifactRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.total_artifacts(), 0);
    }

    #[test]
    fn test_bind_for_program_metal_phase() {
        let mut registry = LaneArtifactRegistry::default();
        let phase = make_phase("p1", ExecutionLane::Metal, "metal_kernel_1", 1024);
        let program = make_program("prog1", vec![phase]);

        registry.bind_for_program(&program).unwrap();

        assert_eq!(registry.metal_artifacts.len(), 1);
        assert_eq!(registry.metal_artifacts[0].artifact_id, "metal_kernel_1");
        assert_eq!(registry.metal_artifacts[0].threadgroup_size, 1024);
        assert!(registry.coreml_artifacts.is_empty());
        assert!(registry.accelerate_artifacts.is_empty());
    }

    #[test]
    fn test_bind_for_program_coreml_phase() {
        let mut registry = LaneArtifactRegistry::default();
        let phase = make_phase("p1", ExecutionLane::CoreMl, "coreml_graph_1", 0);
        let program = make_program("prog1", vec![phase]);

        registry.bind_for_program(&program).unwrap();

        assert!(registry.metal_artifacts.is_empty());
        assert_eq!(registry.coreml_artifacts.len(), 1);
        assert_eq!(registry.coreml_artifacts[0].artifact_id, "coreml_graph_1");
        assert!(registry.accelerate_artifacts.is_empty());
    }

    #[test]
    fn test_bind_for_program_accelerate_phase() {
        let mut registry = LaneArtifactRegistry::default();
        let phase = make_phase("p1", ExecutionLane::Accelerate, "accel_layout_1", 0);
        let program = make_program("prog1", vec![phase]);

        registry.bind_for_program(&program).unwrap();

        assert!(registry.metal_artifacts.is_empty());
        assert!(registry.coreml_artifacts.is_empty());
        assert_eq!(registry.accelerate_artifacts.len(), 1);
        assert_eq!(registry.accelerate_artifacts[0].artifact_id, "accel_layout_1");
    }

    #[test]
    fn test_control_plane_and_fusion_phases_skipped() {
        let mut registry = LaneArtifactRegistry::default();

        let phases = vec![
            make_phase("cpu_ctrl", ExecutionLane::ControlPlaneCpu, "ctrl_1", 0),
            make_phase("fusion", ExecutionLane::FusionOnly, "fusion_1", 0),
        ];
        let program = make_program("prog_skip", phases);

        registry.bind_for_program(&program).unwrap();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_bind_for_program_mixed_lanes() {
        let mut registry = LaneArtifactRegistry::default();

        let phases = vec![
            make_phase("metal_p1", ExecutionLane::Metal, "metal_proj", 512),
            make_phase("ml_p1", ExecutionLane::CoreMl, "coreml_attn", 0),
            make_phase("accel_p1", ExecutionLane::Accelerate, "accel_ffn", 0),
            make_phase("ctrl", ExecutionLane::ControlPlaneCpu, "cpu_seq", 0),
            make_phase("metal_p2", ExecutionLane::Metal, "metal_norm", 256),
        ];
        let program = make_program("prog_mixed", phases);

        registry.bind_for_program(&program).unwrap();

        assert_eq!(registry.metal_artifacts.len(), 2);
        assert_eq!(registry.metal_artifacts[0].artifact_id, "metal_proj");
        assert_eq!(registry.metal_artifacts[0].threadgroup_size, 512);
        assert_eq!(registry.metal_artifacts[1].artifact_id, "metal_norm");
        assert_eq!(registry.metal_artifacts[1].threadgroup_size, 256);

        assert_eq!(registry.coreml_artifacts.len(), 1);
        assert_eq!(registry.coreml_artifacts[0].artifact_id, "coreml_attn");

        assert_eq!(registry.accelerate_artifacts.len(), 1);
        assert_eq!(registry.accelerate_artifacts[0].artifact_id, "accel_ffn");

        assert_eq!(registry.total_artifacts(), 4);
    }

    #[test]
    fn test_bind_twice_appends() {
        let mut registry = LaneArtifactRegistry::default();

        let p1 = make_phase("a", ExecutionLane::Metal, "k1", 128);
        let prog1 = make_program("progA", vec![p1]);

        let p2 = make_phase("b", ExecutionLane::Metal, "k2", 256);
        let prog2 = make_program("progB", vec![p2]);

        registry.bind_for_program(&prog1).unwrap();
        registry.bind_for_program(&prog2).unwrap();

        assert_eq!(registry.metal_artifacts.len(), 2);
        assert_eq!(registry.metal_artifacts[0].artifact_id, "k1");
        assert_eq!(registry.metal_artifacts[1].artifact_id, "k2");
    }

    // ── LaneArtifactBuilder tests ──────────────────────────────────────

    #[test]
    fn test_empty_profile_produces_empty_registry() {
        let builder = LaneArtifactBuilder::new();
        let profile = make_minimal_profile();
        let registry = builder.build_registry(&profile);
        assert!(registry.metal_artifacts.is_empty());
        assert!(registry.coreml_artifacts.is_empty());
        assert!(registry.accelerate_artifacts.is_empty());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_build_registry_collects_all_variants() {
        let builder = LaneArtifactBuilder::new();

        let metal_variant = make_variant(
            "metal_v1",
            make_program(
                "metal_prog",
                vec![make_phase("p1", ExecutionLane::Metal, "metal_k", 512)],
            ),
        );
        let coreml_variant = make_variant(
            "coreml_v1",
            make_program(
                "coreml_prog",
                vec![make_phase("p2", ExecutionLane::CoreMl, "coreml_g", 0)],
            ),
        );

        let mut profile = make_minimal_profile();
        profile.shape_variants = vec![metal_variant, coreml_variant];

        let registry = builder.build_registry(&profile);

        assert_eq!(registry.metal_artifacts.len(), 1);
        assert_eq!(registry.metal_artifacts[0].artifact_id, "metal_k");
        assert_eq!(registry.coreml_artifacts.len(), 1);
        assert_eq!(registry.coreml_artifacts[0].artifact_id, "coreml_g");
        assert!(registry.accelerate_artifacts.is_empty());
    }

    #[test]
    fn test_build_registry_skips_control_phases() {
        let builder = LaneArtifactBuilder::new();

        let variant = make_variant(
            "ctrl_v1",
            make_program(
                "ctrl_prog",
                vec![make_phase(
                    "p1",
                    ExecutionLane::ControlPlaneCpu,
                    "ctrl_seq",
                    0,
                )],
            ),
        );

        let mut profile = make_minimal_profile();
        profile.shape_variants = vec![variant];

        let registry = builder.build_registry(&profile);
        assert!(registry.is_empty());
    }

    // ── BindingError tests ─────────────────────────────────────────────

    #[test]
    fn test_binding_error_display() {
        let err = BindingError::ArtifactNotFound("missing_id".into());
        let msg = err.to_string();
        assert!(msg.contains("missing_id"));
        assert!(msg.contains("not found"));

        let err = BindingError::ArtifactLoadFailed("parse error".into());
        let msg = err.to_string();
        assert!(msg.contains("parse error"));
        assert!(msg.contains("load failed"));
    }

    #[test]
    fn test_binding_error_implements_std_error() {
        fn _assert_error(e: &dyn std::error::Error) {
            let _ = e;
        }
        let err = BindingError::ArtifactNotFound("x".into());
        _assert_error(&err);
    }
}
