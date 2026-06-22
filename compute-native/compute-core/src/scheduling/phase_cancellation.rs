use crate::compute_image::phase_graph::{CancellationClass, PhaseId};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Evidence recorded about cancellation state during phase execution.
#[derive(Debug, Clone)]
pub struct CancellationEvidence {
    pub cancelled_at_dispatch: bool,
    pub cancelled_at_completion: bool,
    pub cancelled_at_publication: bool,
    pub deadline_expired: bool,
}

impl CancellationEvidence {
    pub fn new() -> Self {
        Self {
            cancelled_at_dispatch: false,
            cancelled_at_completion: false,
            cancelled_at_publication: false,
            deadline_expired: false,
        }
    }

    pub fn any_cancelled(&self) -> bool {
        self.cancelled_at_dispatch
            || self.cancelled_at_completion
            || self.cancelled_at_publication
            || self.deadline_expired
    }
}

/// Cancellation checker — checks the shared cancellation flag and deadline.
pub struct CancellationChecker {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancellationChecker {
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            deadline: None,
        }
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Check if the deadline has expired.
    pub fn is_expired(&self) -> bool {
        self.deadline
            .map(|d| Instant::now() >= d)
            .unwrap_or(false)
    }

    /// Combined check — returns true if execution should stop.
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.is_expired()
    }

    /// Check three-point cancellation evidence for a phase.
    pub fn check_phase(
        &self,
        _phase_id: &PhaseId,
        _cancellation_class: CancellationClass,
    ) -> CancellationEvidence {
        CancellationEvidence {
            cancelled_at_dispatch: self.is_cancelled(),
            cancelled_at_completion: self.is_cancelled(),
            cancelled_at_publication: self.is_cancelled(),
            deadline_expired: self.is_expired(),
        }
    }
}

/// Cancellation policy for different phase classes.
#[derive(Debug, Clone)]
pub struct CancellationPolicy {
    /// Whether preemptible phases can be interrupted mid-execution.
    pub allow_preemptible_interrupt: bool,
    /// Whether non-preemptible phases are waited on before discarding.
    pub wait_for_non_preemptible: bool,
    /// Maximum grace period for non-preemptible completion after cancellation.
    pub non_preemptible_grace_ms: u64,
}

impl Default for CancellationPolicy {
    fn default() -> Self {
        Self {
            allow_preemptible_interrupt: true,
            wait_for_non_preemptible: true,
            non_preemptible_grace_ms: 5000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_cancellation_detection() {
        let flag = Arc::new(AtomicBool::new(false));
        let checker = CancellationChecker::new(flag.clone());
        assert!(!checker.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(checker.is_cancelled());
    }

    #[test]
    fn test_deadline_expiry() {
        let flag = Arc::new(AtomicBool::new(false));
        let deadline = Instant::now() + Duration::from_millis(1);
        let checker = CancellationChecker::new(flag).with_deadline(deadline);
        // The deadline may not have expired yet, but should_stop should be false
        // because the deadline is in the future.
        // Wait a tiny bit to let the deadline expire.
        std::thread::sleep(Duration::from_millis(2));
        assert!(checker.is_expired());
        assert!(checker.should_stop());
    }

    #[test]
    fn test_cancellation_evidence() {
        let flag = Arc::new(AtomicBool::new(true));
        let checker = CancellationChecker::new(flag);
        let evidence = checker.check_phase(
            &PhaseId("test".to_string()),
            CancellationClass::Preemptible,
        );
        assert!(evidence.any_cancelled());
        assert!(evidence.cancelled_at_dispatch);
    }
}
