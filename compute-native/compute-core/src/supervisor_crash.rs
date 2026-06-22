//! Worker crash recovery — extracted from worker_supervisor.rs.
//!
//! Contains [`CrashRecoveryState`] for tracking crash retries and buffering
//! in-flight request payloads, plus [`WorkerSupervisor::recover`] for
//! re-spawning the worker, re-loading the model, and re-queuing requests.

use crate::engine_error::{EngineError, EngineErrorCode};
use crate::engine_policy::ExecutionPolicy;
use crate::worker_crash_ledger::WorkerCrashLedger;
use crate::worker_protocol::HostCommand;
use crate::worker_supervisor::{ActiveRequestRegistry, WorkerSupervisor};
use parking_lot::Mutex;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

// ── CrashRecoveryState ──────────────────────────────────────────────────────

/// Shared state for tracking crash recovery across threads.
///
/// The watchdog thread writes to this before marking the runtime faulted;
/// the supervisor reads from it when attempting recovery.
pub struct CrashRecoveryState {
    /// Serialized request payloads captured on crash, to be re-queued.
    pub pending_payloads: Mutex<Vec<Value>>,
    /// When the most recent crash was detected (for backoff computation).
    pub last_crash_time: Mutex<Option<Instant>>,
    /// Number of consecutive recovery attempts made.
    pub retry_count: AtomicU32,
}

impl CrashRecoveryState {
    pub fn new() -> Self {
        Self {
            pending_payloads: Mutex::new(Vec::new()),
            last_crash_time: Mutex::new(None),
            retry_count: AtomicU32::new(0),
        }
    }

    /// Maximum number of automatic recovery attempts before giving up.
    pub const MAX_RETRIES: u32 = 3;

    /// Exponential backoff in seconds for the nth retry (0-indexed).
    /// 1st retry: 1s, 2nd: 2s, 3rd: 4s
    pub fn backoff_seconds(&self) -> u64 {
        1u64 << self.retry_count.load(Ordering::SeqCst)
    }

    /// Capture active request payloads from the registry into pending state.
    pub fn capture_payloads(&self, registry: &ActiveRequestRegistry) {
        *self.pending_payloads.lock() = registry.snapshot_payloads();
    }

    /// Drain and return the captured payloads.
    pub fn drain_payloads(&self) -> Vec<Value> {
        self.pending_payloads.lock().drain(..).collect()
    }
}

// ── Recovery (impl WorkerSupervisor) ───────────────────────────────────────

impl WorkerSupervisor {
    /// Attempt to recover from a worker crash by re-spawning the worker,
    /// re-loading the model, and re-queuing any pending requests.
    ///
    /// Uses exponential backoff between retries (1s, 2s, 4s).
    /// After [`CrashRecoveryState::MAX_RETRIES`] failed attempts, returns
    /// `WorkerCrashed`.
    ///
    /// Recover from a worker crash by re-spawning, re-loading, and
    /// re-queuing previously captured requests.
    ///
    /// This is a convenience that delegates to [`Self::launch_and_handshake`],
    /// then re-queues payloads captured by [`CrashRecoveryState`] from the
    /// watchdog's crash detection. Callers should check
    /// [`CrashRecoveryState::retry_count`] for the limit.
    ///
    /// Returns the new [`WorkerSupervisor`] and the number of requests
    /// re-queued.
    pub fn recover(
        policy: ExecutionPolicy,
        worker_binary: &Path,
        image_dir: &Path,
        image_hash: &str,
        worker_id: &str,
        recovery_state: &CrashRecoveryState,
    ) -> Result<(Self, usize), EngineError> {
        let retry_count = recovery_state
            .retry_count
            .load(std::sync::atomic::Ordering::SeqCst);
        if retry_count >= CrashRecoveryState::MAX_RETRIES {
            return Err(EngineError::new(
                EngineErrorCode::WorkerCrashed,
                format!(
                    "worker crash recovery limit ({}) exceeded after {} retries",
                    CrashRecoveryState::MAX_RETRIES,
                    retry_count,
                ),
            ));
        }

        // Exponential backoff: 2^retry_count seconds (1, 2, 4).
        let backoff_secs = 1u64 << retry_count;
        std::thread::sleep(Duration::from_secs(backoff_secs));

        recovery_state
            .retry_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Drains captured payloads for re-queue.
        let original_payloads = recovery_state.drain_payloads();

        // Spawn a fresh worker via the existing launch method.
        let new_supervisor =
            Self::launch_and_handshake(policy, worker_binary, image_dir, image_hash, worker_id)?;

        // Load the model on the fresh worker.
        new_supervisor.load_model(image_hash)?;

        // Re-queue previously captured requests.
        let re_queued = original_payloads.len();
        for payload in &original_payloads {
            let request_id = payload
                .get("request_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !request_id.is_empty() {
                if let Err(e) = new_supervisor.cmd_writer.send_command_with_request(
                    HostCommand::StartGeneration,
                    request_id,
                    payload.clone(),
                ) {
                    log_error!("failed to re-queue request {request_id}: {e}");
                }
            }
        }

        // Mark recovery in the ledger.
        WorkerCrashLedger::record(new_supervisor.process_ctrl.pid(), 0, None, None, None);

        Ok((new_supervisor, re_queued))
    }
}
