//! Program validation — checks that a serialized phase program
//! is internally consistent before runtime admission.

use super::phase_program::SerializedPhaseProgram;

#[derive(Debug, Clone)]
pub enum ProgramValidationError {
    EmptyPhaseList,
    DuplicatePhaseId(String),
    UnknownProducer(String),
    UnknownConsumer(String),
    MissingArenaPlan,
    MissingResidencyPlan,
    InconsistentShapeClass,
}

#[derive(Debug, Clone)]
pub struct ProgramValidationReport {
    pub valid: bool,
    pub errors: Vec<ProgramValidationError>,
}

/// Validate a serialized phase program for internal consistency.
pub fn validate_program(program: &SerializedPhaseProgram) -> ProgramValidationReport {
    let mut errors = Vec::new();

    if program.phases.is_empty() {
        errors.push(ProgramValidationError::EmptyPhaseList);
    }

    let mut phase_ids = std::collections::HashSet::new();
    for phase in &program.phases {
        if !phase_ids.insert(&phase.phase_id) {
            errors.push(ProgramValidationError::DuplicatePhaseId(
                phase.phase_id.clone(),
            ));
        }
    }

    ProgramValidationReport {
        valid: errors.is_empty(),
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::program::phase_program::*;

    #[test]
    fn test_empty_phase_list() {
        let program = SerializedPhaseProgram {
            program_id: "test".into(),
            program_hash: Default::default(),
            shape_class: crate::compute_image::execution_shape::ExecutionShapeClass::Decode1,
            execution_kind: ExecutionKind::Decode,
            phases: vec![],
            edges: vec![],
            arena_plan_id: "arena_0".into(),
            residency_plan_id: "res_0".into(),
            default_artifact_selection: ProgramArtifactSelection { artifact_ids: vec![] },
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes: vec![],
        };
        let report = validate_program(&program);
        assert!(!report.valid);
    }

    #[test]
    fn test_duplicate_phase_ids() {
        let phase = SerializedPhase {
            phase_id: "p0".into(),
            semantic_operation: SemanticOperation::RmsNorm,
            lane: ExecutionLane::Accelerate,
            artifact_identity: CanonicalArtifactIdentity {
                artifact_id: "a0".into(),
                artifact_hash: Default::default(),
                artifact_kind: PhaseArtifactKind::Elementwise,
            },
            input_bindings: vec![],
            output_bindings: vec![],
            dependency_contract: PhaseDependencyContract { dependencies_satisfied: false },
            completion_contract: PhaseCompletionContract { must_emit_receipt: false, must_release_regions: false, must_advance_epoch: false },
            resource_reservation: PhaseResourceReservation { threadgroup_memory: 0, register_count: 0 },
            state_domain: None,
        };
        let program = SerializedPhaseProgram {
            program_id: "test".into(),
            program_hash: Default::default(),
            shape_class: crate::compute_image::execution_shape::ExecutionShapeClass::Decode1,
            execution_kind: ExecutionKind::Decode,
            phases: vec![phase.clone(), phase],
            edges: vec![],
            arena_plan_id: "arena_0".into(),
            residency_plan_id: "res_0".into(),
            default_artifact_selection: ProgramArtifactSelection { artifact_ids: vec![] },
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes: vec![],
        };
        let report = validate_program(&program);
        assert!(!report.valid);
    }
}
