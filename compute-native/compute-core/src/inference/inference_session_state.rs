use crate::kv_cache::KvCache;
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
    pub kv_caches: Vec<KvCache>,
    pub sink_states: Vec<SinkState>,
    pub working_set: Option<WorkingSetManager>,
    pub coreml_models: CoreMlModelRegistryStub,
    pub lane_registry: LaneRegistryStub,
    pub cancellation: Arc<AtomicBool>,
    pub session_epoch: AtomicU64,
}

/// Stub for the Core ML model registry.
/// In a full implementation this loads artifacts once at session creation time.
pub struct CoreMlModelRegistryStub;

/// Stub for the lane registry.
pub struct LaneRegistryStub;

impl InferenceSessionState {
    pub fn new(session_id: String, kv_caches: Vec<KvCache>, sink_states: Vec<SinkState>) -> Self {
        Self {
            session_id: InferenceSessionId(session_id),
            kv_caches,
            sink_states,
            working_set: None,
            coreml_models: CoreMlModelRegistryStub,
            lane_registry: LaneRegistryStub,
            cancellation: Arc::new(AtomicBool::new(false)),
            session_epoch: AtomicU64::new(0),
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
}
