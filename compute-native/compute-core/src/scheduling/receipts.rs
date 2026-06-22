//! Phase receipts — evidence that a phase was executed.
//!
//! [`PhaseReceipt`] is produced by the phase engine after every dispatched
//! phase.  It records the phase ID, completion status, duration, and
//! optional fused-kernel evidence.

use crate::compute_image::phase_dag::PhaseCompletionStatus;
use serde::{Deserialize, Serialize};
use crate::compute_image::fusion_receipts::FusedMetalExecutionEvidence;

/// Receipt for an executed phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReceipt {
    pub phase_id: String,
    pub status: PhaseCompletionStatus,
    pub duration_us: u64,
    /// Fused-kernel execution evidence (present when a fused kernel ran).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_evidence: Option<FusedMetalExecutionEvidence>,
}

impl PhaseReceipt {
    pub fn completed(phase_id: &str, duration_us: u64) -> Self {
        Self {
            phase_id: phase_id.into(),
            status: PhaseCompletionStatus::Complete,
            duration_us,
            fused_evidence: None,
        }
    }

    pub fn failed(phase_id: &str, reason: &str) -> Self {
        Self {
            phase_id: phase_id.into(),
            status: PhaseCompletionStatus::Failed(reason.into()),
            duration_us: 0,
            fused_evidence: None,
        }
    }
}
