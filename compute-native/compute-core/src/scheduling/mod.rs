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
pub mod saved_request;
pub mod token_budget;
pub use token_budget::*;

pub use scheduler::Scheduler;
pub use saved_request::SavedRequest;
pub use saved_request::{
    PRIORITY_HIGHEST,
    PRIORITY_DEFAULT,
    PRIORITY_LOWEST,
    MAX_PREEMPTIONS_BEFORE_BOOST,
    STARVATION_PRIORITY_BOOST,
};

use std::sync::{Arc, LazyLock};

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

// ---------------------------------------------------------------------------
// HardwareConfig — auto-detect hardware and configure for max throughput
// ---------------------------------------------------------------------------

/// Auto-detect hardware and return optimal configuration.
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub total_ram_gb: u32,
    pub gpu_cores: u32,
    pub ane_cores: u32,
    pub cpu_cores: u32,
    pub memory_bw_gb_s: u32,
    pub is_memory_rich: bool,
    pub recommended_batch_size: u32,
    pub recommended_spec_length: u32,
    pub enable_weight_streaming: bool,
    pub enable_kv_disk_eviction: bool,
    pub max_concurrent_sequences: u32,
}

impl HardwareConfig {
    /// Detect hardware characteristics and return optimal configuration.
    pub fn detect() -> Self {
        let total_ram_mb = crate::gpu_memory::total_physical_ram_mb();
        let total_ram_gb = (total_ram_mb / 1024) as u32;
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        // M3 Ultra has 32 CPU cores, M4 Pro/Max approximately 14-16, base Mx approximately 8-10.
        // Heuristic: >= 24 CPU cores strongly indicates a Max/Ultra die.
        let is_ultra = cpu_count >= 24;

        Self {
            total_ram_gb,
            gpu_cores: if is_ultra { 80 } else { 8 },
            ane_cores: if is_ultra { 32 } else { 16 },
            cpu_cores: cpu_count,
            memory_bw_gb_s: if is_ultra { 800 } else { 100 },
            is_memory_rich: total_ram_gb > 64,
            recommended_batch_size: if is_ultra { 32 } else { 4 },
            recommended_spec_length: if is_ultra { 32 } else { 8 },
            enable_weight_streaming: total_ram_gb <= 32,
            enable_kv_disk_eviction: total_ram_gb <= 32,
            max_concurrent_sequences: if is_ultra { 64 } else { 8 },
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceTelemetry — observable metrics feeding the EXO autoscaler
// ---------------------------------------------------------------------------

/// Telemetry snapshot used by the EXO cluster autoscaler.
///
/// Tracks the scheduler's current queue depth and a moving average of
/// inference latency.
#[derive(Debug, Clone)]
pub struct InferenceTelemetrySnapshot {
    pub queue_depth: usize,
    pub avg_latency_us: f64,
}

/// Thread-safe inference telemetry collector.
///
/// Updated by the scheduler and read by the EXO autoscaler.
/// Provides a `global()` singleton for convenience.
#[derive(Clone)]
pub struct InferenceTelemetry {
    inner: Arc<std::sync::Mutex<InferenceTelemetryInner>>,
}

static GLOBAL_INFERENCE_TELEMETRY: LazyLock<InferenceTelemetry> =
    LazyLock::new(|| InferenceTelemetry::new());

#[derive(Debug, Clone)]
struct InferenceTelemetryInner {
    queue_depth: usize,
    /// Sliding window of recent latencies in microseconds.
    latencies: Vec<f64>,
    max_latency_samples: usize,
}

impl InferenceTelemetry {
    /// Return the global singleton telemetry collector.
    pub fn global() -> Self {
        GLOBAL_INFERENCE_TELEMETRY.clone()
    }

    fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(InferenceTelemetryInner {
                queue_depth: 0,
                latencies: Vec::with_capacity(128),
                max_latency_samples: 128,
            })),
        }
    }

    /// Record the current queue depth (called by the scheduler).
    pub fn set_queue_depth(&self, depth: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.queue_depth = depth;
        }
    }

    /// Record a single inference latency in microseconds.
    pub fn record_latency(&self, latency_us: f64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.latencies.push(latency_us);
            if inner.latencies.len() > inner.max_latency_samples {
                inner.latencies.remove(0);
            }
        }
    }

    /// Atomically snapshot the current telemetry values.
    pub fn snapshot(&self) -> InferenceTelemetrySnapshot {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let avg_latency_us = if inner.latencies.is_empty() {
            0.0
        } else {
            let sum: f64 = inner.latencies.iter().sum();
            sum / inner.latencies.len() as f64
        };
        InferenceTelemetrySnapshot {
            queue_depth: inner.queue_depth,
            avg_latency_us,
        }
    }
}
