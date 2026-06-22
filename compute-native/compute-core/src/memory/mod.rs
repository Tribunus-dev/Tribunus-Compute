//! Unified memory island for Tribunus Compute.
//!
//! See `docs/compute-image-memory-architecture.md` and
//! `docs/unified-memory-island.md`.

pub mod allocator;
pub mod ane_program_cache;
pub mod ane_weight_dict;
pub mod candle_bridge;
pub mod compute_image_bridge;
pub mod coreml_warmup;
pub mod enforcer;
pub mod iosurface_storage;
pub mod monitor;
pub mod orion_bridge;
pub mod plan;
pub mod pool;
pub mod telemetry;

pub use allocator::BlockHandle;
pub use allocator::IosurfaceAllocator;
pub use allocator::KvCacheBlockAllocator;
pub use allocator::PagedIosurfaceAllocator;
pub use enforcer::MemoryEnforcer;
pub use monitor::MemoryMonitor;
pub use pool::EnginePool;

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Severe = 3,
    Oom = 4,
}
