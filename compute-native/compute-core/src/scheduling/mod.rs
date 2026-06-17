//! Continuous batching scheduler ported from omlx.
//!
//! Reference: `ref/omlx/scheduler.py`, design: `docs/omlx-scheduler.md`
//!
//! Manages request queuing, prefill/decode phase scheduling, batch construction,
//! and token budget allocation across concurrent requests.

pub mod batch;
pub mod request;
pub mod scheduler;
pub mod slot;

pub use scheduler::Scheduler;

/// Request lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Paused,
    Completed,
    Cancelled,
}

/// A single inference request
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub priority: u8,
    pub state: RequestState,
    pub created_at: std::time::Instant,
    pub slot: Option<usize>,
}

/// A batch of slots for model execution
#[derive(Debug, Clone)]
pub struct Batch {
    pub slots: Vec<Slot>,
    pub batch_size: usize,
    pub max_batch_size: usize,
}

/// A slot in the batch (one model execution unit)
#[derive(Debug, Clone)]
pub struct Slot {
    pub id: usize,
    pub request_id: Option<u64>,
    pub tokens_generated: usize,
    pub kv_cache_start: usize,
    pub kv_cache_length: usize,
    /// Target execution backend for this slot.
    /// 0=MLX, 1=Accelerate, 2=CoreML, 3=ANE/Orion
    pub backend_id: u32,
    /// Page IDs allocated from the paged allocator for this slot's KV cache.
    pub kv_cache_pages: Vec<usize>,
}

/// Continuous batching scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_total_tokens: usize,
    pub max_prefill_batch: usize,
    pub prefill_many_ratio: f64,
    pub pause_threshold: usize,
    /// Default backend_id for new slots (0=MLX).
    pub default_backend_id: u32,
    /// KV cache length per slot in tokens.
    pub kv_cache_length: usize,
    /// Maximum KV cache memory pool in bytes (0 = unlimited).
    pub kv_cache_pool_bytes: u64,
    /// Number of KV cache pages to pre-allocate per slot.
    /// Default 64 (64 x 512 bytes = 32 KB per slot).
    pub kv_cache_pages_per_slot: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_total_tokens: 4096,
            max_prefill_batch: 8,
            prefill_many_ratio: 0.5,
            pause_threshold: 2048,
            default_backend_id: 0,
            kv_cache_length: 4096,
            kv_cache_pool_bytes: 256 * 1024 * 1024,
            kv_cache_pages_per_slot: 64,
        }
    }
}
