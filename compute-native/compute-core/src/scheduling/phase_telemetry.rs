//! Phase telemetry and work-item metadata.
//!
//! Provides [`PhaseWorkItemMetadata`] for tracking per-dispatch context,
//! [`StepTelemetrySnapshot`] for aggregating phase receipts into a step-level
//! picture, and [`SessionTelemetryState`] for session-wide cumulative counters.

use crate::compute_image::phase_graph::{
    ArtifactBindingId, PhaseId, WeightResidencySetId,
};
use crate::scheduling::receipts::PhaseReceipt;
use std::time::Instant;

/// Metadata for a work item created by the PhaseEngine for each dispatched
/// phase.
#[derive(Debug, Clone)]
pub struct PhaseWorkItemMetadata {
    pub phase_id: PhaseId,
    pub request_id: u64,
    pub lane: String,
    pub layer_index: Option<usize>,
    pub artifact_id: Option<ArtifactBindingId>,
    pub required_weight_set: Option<WeightResidencySetId>,
    pub deadline: Option<Instant>,
}

impl PhaseWorkItemMetadata {
    pub fn new(phase_id: PhaseId, request_id: u64) -> Self {
        Self {
            phase_id,
            request_id,
            lane: String::new(),
            layer_index: None,
            artifact_id: None,
            required_weight_set: None,
            deadline: None,
        }
    }
}

/// Telemetry snapshot for a single step execution.
#[derive(Debug, Clone)]
pub struct StepTelemetrySnapshot {
    pub request_id: u64,
    pub execution_id: u64,
    pub mode: String,
    pub phase_count: usize,
    pub fused_kernel_count: usize,
    pub fallback_count: usize,
    pub total_duration_us: u64,
    pub average_phase_duration_us: f64,
}

impl From<&[PhaseReceipt]> for StepTelemetrySnapshot {
    fn from(receipts: &[PhaseReceipt]) -> Self {
        let phase_count = receipts.len();
        let fused_kernel_count = receipts
            .iter()
            .filter(|r| r.fused_evidence.is_some())
            .count();
        let fallback_count = receipts
            .iter()
            .filter(|r| {
                matches!(
                    r.status,
                    crate::compute_image::phase_dag::PhaseCompletionStatus::FallbackUsed(_)
                )
            })
            .count();
        let total_duration_us: u64 = receipts.iter().map(|r| r.duration_us).sum();
        let average_phase_duration_us = if phase_count > 0 {
            total_duration_us as f64 / phase_count as f64
        } else {
            0.0
        };

        StepTelemetrySnapshot {
            request_id: 0,
            execution_id: 0,
            mode: String::new(),
            phase_count,
            fused_kernel_count,
            fallback_count,
            total_duration_us,
            average_phase_duration_us,
        }
    }
}

/// Session-level telemetry state.
#[derive(Debug, Clone)]
pub struct SessionTelemetryState {
    pub total_steps: u64,
    pub total_phases_dispatched: u64,
    pub total_fused_kernel_dispatches: u64,
    pub total_fallback_dispatches: u64,
    pub total_duration_us: u64,
}

impl SessionTelemetryState {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            total_phases_dispatched: 0,
            total_fused_kernel_dispatches: 0,
            total_fallback_dispatches: 0,
            total_duration_us: 0,
        }
    }

    pub fn record_step(&mut self, snapshot: &StepTelemetrySnapshot) {
        self.total_steps += 1;
        self.total_phases_dispatched += snapshot.phase_count as u64;
        self.total_fused_kernel_dispatches += snapshot.fused_kernel_count as u64;
        self.total_fallback_dispatches += snapshot.fallback_count as u64;
        self.total_duration_us += snapshot.total_duration_us;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::fusion_abi::{
        ArtifactHash, MetalFusionFamily, MetalLaunchContract,
        SealedMetalFusionArtifact,
    };
    use crate::compute_image::fusion_receipts::FusedMetalExecutionEvidence;
    use crate::compute_image::phase_dag::PhaseCompletionStatus;
    use crate::scheduling::receipts::PhaseReceipt;
    use std::collections::HashMap;

    fn make_receipt(
        phase_id: &str,
        duration_us: u64,
        is_fused: bool,
        is_fallback: bool,
    ) -> PhaseReceipt {
        let status = if is_fallback {
            PhaseCompletionStatus::FallbackUsed("decomposed".into())
        } else {
            PhaseCompletionStatus::Complete
        };

        let fused_evidence = if is_fused {
            let launch = MetalLaunchContract {
                entry_point: "fused_kernel".into(),
                threads_per_threadgroup: [32, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                buffer_bindings: HashMap::new(),
            };
            let hash = ArtifactHash {
                sha256: "abc123".into(),
                byte_length: 4096,
            };
            let artifact = SealedMetalFusionArtifact::new(
                "region_0",
                MetalFusionFamily::QkvProj,
                hash,
                launch,
                None,
            );
            let mut evidence = FusedMetalExecutionEvidence::from_artifact(&artifact, duration_us);
            evidence.duration_us = duration_us;
            Some(evidence)
        } else {
            None
        };

        PhaseReceipt {
            phase_id: phase_id.to_string(),
            status,
            duration_us,
            fused_evidence,
        }
    }

    #[test]
    fn test_telemetry_from_receipts() {
        let receipts = vec![
            make_receipt("prologue", 100, false, false),
            make_receipt("layer_0_attn", 500, true, false),
            make_receipt("layer_0_mlp", 400, false, false),
            make_receipt("layer_1_attn", 600, false, true),
            make_receipt("epilogue", 200, false, false),
        ];
        let snapshot = StepTelemetrySnapshot::from(receipts.as_slice());
        assert_eq!(snapshot.phase_count, 5);
        assert_eq!(snapshot.fused_kernel_count, 1);
        assert_eq!(snapshot.fallback_count, 1);
        assert_eq!(snapshot.total_duration_us, 1800);
    }

    #[test]
    fn test_session_telemetry() {
        let mut session = SessionTelemetryState::new();
        assert_eq!(session.total_steps, 0);

        let receipts = vec![make_receipt("phase1", 100, false, false)];
        let snapshot = StepTelemetrySnapshot::from(receipts.as_slice());
        session.record_step(&snapshot);
        assert_eq!(session.total_steps, 1);
        assert_eq!(session.total_phases_dispatched, 1);
    }
}
