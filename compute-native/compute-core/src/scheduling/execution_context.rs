//! Execution context — runtime state bundled for phase execution.

use std::any::Any;
use mlx_rs::Array;
use std::sync::Arc;

/// Runtime state passed to every phase runner.
pub struct ExecutionContext {
    /// Unique request ID being processed.
    pub request_id: u64,
    /// Current token position in decode.
    pub token_position: usize,
    /// Whether this is a prefill or decode pass.
    pub is_prefill: bool,
    /// Input token IDs for this step (set by caller before dispatch).
    pub token_ids: Vec<i32>,
    /// Current hidden state activation flowing through the DAG.
    /// Populated by the caller before phase dispatch; updated by runners.
    pub hidden_state: Option<Array>,
    /// Per-layer KV caches for the active sequence.
    pub kv_caches: Vec<crate::kv_cache::LiveKvCache>,
    /// Model weights indexed by layer index.
    pub layer_weights: Arc<Vec<crate::profiled_model::LayerWeights>>,
    /// Opaque backend context.  Concrete runners downcast this to access
    /// the MLX executor, Metal device, or Core ML state belonging to
    /// the current inference session.
    pub backend: Option<Box<dyn Any + Send>>,
}

impl ExecutionContext {
    /// Create an empty/default context for testing.
    pub fn new_empty() -> Self {
        Self {
            request_id: 0,
            token_position: 0,
            is_prefill: true,
            token_ids: Vec::new(),
            hidden_state: None,
            kv_caches: Vec::new(),
            layer_weights: Arc::new(Vec::new()),
            backend: None,
        }
    }
}
