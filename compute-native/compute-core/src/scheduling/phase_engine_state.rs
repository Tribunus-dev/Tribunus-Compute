use crate::compute_image::phase_graph::PhaseId;
use std::time::Instant;

/// Phase lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseLifecycleState {
    Dormant,
    Ready,
    ResidencyPending,
    LeasePending,
    Admitted,
    Dispatched,
    AwaitingCompletion,
    Validating,
    Publishing,
    Complete,
    // Failure states
    Rejected,
    Cancelled,
    TimedOut,
    FailedBeforePublication,
    FailedAfterTentativeState,
    RolledBack,
    FallbackPending,
    FallbackComplete,
    Quarantined,
}

impl PhaseLifecycleState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            PhaseLifecycleState::Complete
                | PhaseLifecycleState::Rejected
                | PhaseLifecycleState::Cancelled
                | PhaseLifecycleState::TimedOut
                | PhaseLifecycleState::FailedBeforePublication
                | PhaseLifecycleState::FailedAfterTentativeState
                | PhaseLifecycleState::RolledBack
                | PhaseLifecycleState::FallbackComplete
                | PhaseLifecycleState::Quarantined
        )
    }

    pub fn is_success(&self) -> bool {
        matches!(self, PhaseLifecycleState::Complete | PhaseLifecycleState::FallbackComplete)
    }

    pub fn can_proceed(&self) -> bool {
        matches!(
            self,
            PhaseLifecycleState::Complete | PhaseLifecycleState::FallbackComplete
        )
    }
}

/// A handle for a RuntimeWorkItem created by the engine for a phase.
#[derive(Debug, Clone)]
pub struct RuntimeWorkItemHandle {
    pub phase_id: PhaseId,
    pub request_id: u64,
    pub lane: String,
    pub layer_index: Option<usize>,
    pub artifact_id: Option<String>,
    pub required_weight_set: Option<String>,
    pub deadline: Option<Instant>,
}

impl RuntimeWorkItemHandle {
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

/// Phase lifecycle tracker — maps phase IDs to their lifecycle states.
#[derive(Debug, Clone)]
pub struct PhaseLifecycleTracker {
    states: std::collections::HashMap<String, PhaseLifecycleState>,
    activation_generations: std::collections::HashMap<String, u64>,
}

impl PhaseLifecycleTracker {
    pub fn new() -> Self {
        Self {
            states: std::collections::HashMap::new(),
            activation_generations: std::collections::HashMap::new(),
        }
    }

    pub fn register(&mut self, phase_id: &str) {
        self.states.entry(phase_id.to_string()).or_insert(PhaseLifecycleState::Dormant);
    }

    pub fn transition(&mut self, phase_id: &str, to: PhaseLifecycleState) -> Result<(), String> {
        let current = self.states.get(phase_id).copied().unwrap_or(PhaseLifecycleState::Dormant);
        // Allow transition from any state if the target is a terminal failure
        if to.is_terminal() {
            self.states.insert(phase_id.to_string(), to);
            return Ok(());
        }
        match (current, to) {
            (PhaseLifecycleState::Dormant, PhaseLifecycleState::Ready)
            | (PhaseLifecycleState::Ready, PhaseLifecycleState::ResidencyPending)
            | (PhaseLifecycleState::Ready, PhaseLifecycleState::Admitted)
            | (PhaseLifecycleState::ResidencyPending, PhaseLifecycleState::LeasePending)
            | (PhaseLifecycleState::LeasePending, PhaseLifecycleState::Admitted)
            | (PhaseLifecycleState::Admitted, PhaseLifecycleState::Dispatched)
            | (PhaseLifecycleState::Dispatched, PhaseLifecycleState::AwaitingCompletion)
            | (PhaseLifecycleState::AwaitingCompletion, PhaseLifecycleState::Validating)
            | (PhaseLifecycleState::Validating, PhaseLifecycleState::Publishing)
            | (PhaseLifecycleState::Publishing, PhaseLifecycleState::Complete)
            | (PhaseLifecycleState::FallbackPending, PhaseLifecycleState::FallbackComplete)
            => {
                self.states.insert(phase_id.to_string(), to);
                Ok(())
            }
            _ => {
                Err(format!(
                    "invalid phase lifecycle transition: {:?} -> {:?}",
                    current, to
                ))
            }
        }
    }

    pub fn state(&self, phase_id: &str) -> PhaseLifecycleState {
        self.states.get(phase_id).copied().unwrap_or(PhaseLifecycleState::Dormant)
    }

    pub fn all_complete(&self) -> bool {
        self.states.values().all(|s| s.is_terminal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        let mut tracker = PhaseLifecycleTracker::new();
        tracker.register("p1");
        assert!(tracker.transition("p1", PhaseLifecycleState::Ready).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::Admitted).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::Dispatched).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::AwaitingCompletion).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::Validating).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::Publishing).is_ok());
        assert!(tracker.transition("p1", PhaseLifecycleState::Complete).is_ok());
    }

    #[test]
    fn test_invalid_transition() {
        let mut tracker = PhaseLifecycleTracker::new();
        tracker.register("p1");
        assert!(tracker.transition("p1", PhaseLifecycleState::Complete).is_err());
    }

    #[test]
    fn test_terminal_override() {
        let mut tracker = PhaseLifecycleTracker::new();
        tracker.register("p1");
        assert!(tracker.transition("p1", PhaseLifecycleState::Cancelled).is_ok());
        assert_eq!(tracker.state("p1"), PhaseLifecycleState::Cancelled);
    }

    #[test]
    fn test_all_complete() {
        let mut tracker = PhaseLifecycleTracker::new();
        tracker.register("p1");
        tracker.register("p2");
        assert!(!tracker.all_complete());
        let _ = tracker.transition("p1", PhaseLifecycleState::Cancelled);
        let _ = tracker.transition("p2", PhaseLifecycleState::Complete);
        assert!(tracker.all_complete());
    }
}
