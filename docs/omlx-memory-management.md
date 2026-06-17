# Proactive Memory Management — Port Plan

Sources:
- `omlx/omlx/memory_monitor.py` (844 lines)
- `omlx/omlx/process_memory_enforcer.py` (1.4K lines)
- `omlx/omlx/engine_pool.py` (1.4K lines)
Status: Reference copied to `ref/omlx/memory_monitor.py`, `ref/omlx/process_memory_enforcer.py`

## What it is

A proactive memory management system for Apple Silicon that prevents OOM
(out-of-memory) during LLM inference. Key features:

1. **Real-time memory monitoring** — tracks RSS, VM, swap usage via psutil
2. **Pressure-based eviction** — model swaps, cache eviction, KV cache compression
3. **Engine pool management** — load/unload models on demand based on memory pressure
4. **Graceful degradation** — reduce context length, disable features under pressure
5. **Swap file management** — controlled swap usage on low-RAM systems

## Memory Pressure Levels

```
Level 0 (Normal):  < 70% RAM used — no action
Level 1 (Warning): 70-80% RAM used — reduce token budget, compress KV cache
Level 2 (Critical): 80-90% RAM used — evict idle models, flush caches
Level 3 (Severe):   > 90% RAM used — suspend non-critical engines, force GC
Level 4 (OOM):      Swap > 50% — emergency: free all non-essential memory
```

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/memory/`

#### `mod.rs`
```rust
pub mod monitor;
pub mod enforcer;
pub mod pool;

pub use monitor::MemoryMonitor;
pub use enforcer::MemoryEnforcer;
pub use pool::EnginePool;
```

#### `monitor.rs` — Memory monitoring
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Normal = 0,
    Warning = 1,    // > 70% RAM used
    Critical = 2,   // > 80% RAM used
    Severe = 3,     // > 90% RAM used
    Oom = 4,        // > 50% swap used
}

/// Memory statistics snapshot
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub rss_bytes: u64,
    pub total_ram_bytes: u64,
    pub vm_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub pressure: MemoryPressure,
}

impl MemoryStats {
    /// Compute memory pressure level
    pub fn pressure(&self) -> MemoryPressure {
        let ratio = self.rss_bytes as f64 / self.total_ram_bytes as f64;
        if ratio > 0.90 { MemoryPressure::Severe }
        else if ratio > 0.80 { MemoryPressure::Critical }
        else if ratio > 0.70 { MemoryPressure::Warning }
        else { MemoryPressure::Normal }
    }
}

/// Real-time memory monitor (polling-based)
pub struct MemoryMonitor {
    stats: MemoryStats,
    last_update: Instant,
    poll_interval: Duration,
}

impl MemoryMonitor {
    pub fn new(poll_interval: Duration) -> Self { /* ... */ }

    /// Poll system memory stats
    pub fn poll(&mut self) -> MemoryStats {
        // On macOS: read from libc via mach_host_statistics
        // Or use mach_vm_info for more granular data
    }

    /// Register a memory pressure callback
    pub fn on_pressure<F: Fn(MemoryPressure)>(&mut self, callback: F) { /* ... */ }
}
```

#### `enforcer.rs` — Proactive memory enforcement
```rust
/// Actions the enforcer can take under pressure
#[derive(Debug, Clone)]
pub enum MemoryAction {
    CompressKvCache,        // Apply KV cache quantization
    EvictPrefixCache,       // Clear prefix cache entries
    ReduceContextLength,    // Truncate oldest tokens in context
    SwapModelToDisk,        // Offload an idle model
    SuspendEngine,          // Pause a running engine
    ForceGarbageCollection, // Explicit memory reclamation
    FreePagedCache,         // Evict SSD-paged cache blocks
    FreeRotatingCache,      // Clear rotating KV caches
}

/// Proactive memory enforcer
pub struct MemoryEnforcer {
    monitor: MemoryMonitor,
    actions: Vec<MemoryAction>,
    current_pressure: MemoryPressure,
}

impl MemoryEnforcer {
    /// Run one enforcement cycle
    pub fn enforce(&mut self) -> Vec<MemoryAction> {
        let stats = self.monitor.poll();
        let pressure = stats.pressure();

        if pressure > self.current_pressure {
            // Pressure increasing — take action
            let actions = self.escalate(pressure);
            self.current_pressure = pressure;
            actions
        } else if pressure < self.current_pressure {
            // Pressure decreasing — can de-escalate
            self.deescalate(pressure);
            vec![]
        } else {
            vec![]  // Stable
        }
    }

    /// Get actions to take at a pressure level
    fn escalate(&self, pressure: MemoryPressure) -> Vec<MemoryAction> {
        match pressure {
            MemoryPressure::Warning => vec![MemoryAction::CompressKvCache],
            MemoryPressure::Critical => vec![
                MemoryAction::CompressKvCache,
                MemoryAction::EvictPrefixCache,
            ],
            MemoryPressure::Severe => vec![
                MemoryAction::ReduceContextLength,
                MemoryAction::SwapModelToDisk,
            ],
            MemoryPressure::Oom => vec![
                MemoryAction::SuspendEngine,
                MemoryAction::ForceGarbageCollection,
            ],
            _ => vec![],
        }
    }
}
```

#### `pool.rs` — Engine pool with memory-aware lifecycle
```rust
/// Engine lifecycle state
#[derive(Debug, Clone, Copy)]
pub enum EngineLifecycle {
    Loading,
    Active,
    Idle,
    Unloading,
    Swapped,
}

/// Engine pool manages model lifecycles based on memory pressure
pub struct EnginePool {
    engines: HashMap<String, EngineEntry>,
    max_concurrent: usize,
    memory_enforcer: MemoryEnforcer,
}

pub struct EngineEntry {
    pub id: String,
    pub state: EngineLifecycle,
    pub last_access: Instant,
    pub memory_estimate: u64,  // bytes
}

impl EnginePool {
    /// Check if we can load a new model
    pub fn can_load(&self, estimated_bytes: u64) -> bool {
        let stats = MemoryMonitor::poll();
        let would_use = stats.rss_bytes + estimated_bytes;
        would_use < (stats.total_ram_bytes as f64 * 0.80) as u64
    }

    /// Evict the least recently used idle engine
    pub fn evict_idle(&mut self) -> Option<String> {
        self.engines.iter()
            .filter(|(_, e)| matches!(e.state, EngineLifecycle::Idle))
            .min_by_key(|(_, e)| e.last_access)
            .map(|(id, _)| id.clone())
    }
}
```

### Integration

- `MemoryMonitor` runs as a background task in the session
- `MemoryEnforcer` actions are dispatched to `CacheManager` and `EnginePool`
- Hooks into session lifecycle for graceful degradation
- Uses macOS-specific APIs (mach_vm_info, proc_info) for accurate stats
