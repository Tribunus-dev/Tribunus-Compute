use crate::kv_cache::KvCache;
use crate::kv_cache::LiveKvCache;
use crate::inference::inference_step_state::StepReceiptLedger;
use crate::scheduling::receipts::PhaseReceipt;
use crate::executor::SinkState;
use crate::profiled_executor::WorkingSetManager;
use crate::backend::accelerate_lane::AccelerateLane;
use crate::backend::coreml_lane::CoreMlLane;
use crate::runtime::executable_session::RuntimeBackends;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Unique identifier for an inference session.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InferenceSessionId(pub String);

/// Mutable per-session state owned by the PhaseEngine.
///
/// Contains KV caches, sink states, the active working set for weight
/// streaming, lane registries, and cancellation sources.
pub struct InferenceSessionState {
    pub session_id: InferenceSessionId,
    pub kv_caches: Vec<LiveKvCache>,
    pub sink_states: Vec<SinkState>,
    pub working_set: Option<WorkingSetManager>,
    pub coreml_models: CoreMlModelRegistryStub,
    pub lane_registry: LaneRegistryStub,
    pub cancellation: Arc<AtomicBool>,
    pub session_epoch: AtomicU64,
    pub receipt_ledger: StepReceiptLedger,
}

/// Stub for the Core ML model registry.
/// In a full implementation this loads artifacts once at session creation time.
pub struct CoreMlModelRegistryStub;

/// Stub for the lane registry.
pub struct LaneRegistryStub;

impl InferenceSessionState {
    pub fn new(session_id: String, kv_caches: Vec<KvCache>, sink_states: Vec<SinkState>) -> Self {
        let live_caches = kv_caches.into_iter().map(LiveKvCache::Fp16).collect();
        Self {
            session_id: InferenceSessionId(session_id),
            kv_caches: live_caches,
            sink_states,
            working_set: None,
            coreml_models: CoreMlModelRegistryStub,
            lane_registry: LaneRegistryStub,
            cancellation: Arc::new(AtomicBool::new(false)),
            session_epoch: AtomicU64::new(0),
            receipt_ledger: StepReceiptLedger::new(),
        }
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Relaxed)
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.cancellation.store(true, Ordering::Relaxed);
    }

    /// Increment and return the session epoch.
    pub fn next_epoch(&self) -> u64 {
        self.session_epoch.fetch_add(1, Ordering::Relaxed)
    }

    /// Record a phase receipt in the session receipt ledger.
    pub fn push_receipt(&mut self, receipt: PhaseReceipt) {
        self.receipt_ledger.push(receipt);
    }
}
