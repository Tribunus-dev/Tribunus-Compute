//! Memory-aware engine pool — manages model lifecycles
//! based on memory pressure.
//!
//! Reference: `ref/omlx/process_memory_enforcer.py` + `ref/omlx/engine_pool.py`
//! Design: `docs/omlx-memory-management.md`

use std::collections::HashMap;
use std::time::Instant;

/// Engine lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineLifecycle {
    Loading,
    Active,
    Idle,
    Unloading,
    Swapped,
}

/// Entry in the engine pool
#[derive(Debug, Clone)]
pub struct EngineEntry {
    pub id: String,
    pub state: EngineLifecycle,
    pub last_access: Instant,
    pub memory_estimate: u64,
}

/// Memory-aware engine pool
///
/// Decides whether new models can be loaded based on available memory,
/// and evicts idle models when memory pressure is high.
#[allow(dead_code)]
pub struct EnginePool {
    engines: HashMap<String, EngineEntry>,
    max_concurrent: usize,
}

impl EnginePool {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            engines: HashMap::new(),
            max_concurrent,
        }
    }

    /// Check if we can load a new model with estimated memory usage
    pub fn can_load(&self, estimated_bytes: u64) -> bool {
        let total_ram = 0; // TODO: get from system
        let current_usage = self
            .engines
            .values()
            .filter(|e| matches!(e.state, EngineLifecycle::Active | EngineLifecycle::Loading))
            .map(|e| e.memory_estimate)
            .sum::<u64>();
        (current_usage + estimated_bytes) < (total_ram as f64 * 0.80) as u64
    }

    /// Register a new engine
    pub fn register(&mut self, id: String, memory_estimate: u64) {
        self.engines.insert(
            id.clone(),
            EngineEntry {
                id,
                state: EngineLifecycle::Loading,
                last_access: Instant::now(),
                memory_estimate,
            },
        );
    }

    /// Mark an engine as idle (candidate for eviction)
    pub fn mark_idle(&mut self, id: &str) {
        if let Some(entry) = self.engines.get_mut(id) {
            entry.state = EngineLifecycle::Idle;
            entry.last_access = Instant::now();
        }
    }

    /// Evict the least recently used idle engine
    pub fn evict_idle(&mut self) -> Option<String> {
        self.engines
            .iter()
            .filter(|(_, e)| e.state == EngineLifecycle::Idle)
            .min_by_key(|(_, e)| e.last_access)
            .map(|(id, _)| id.clone())
    }

    /// Number of active engines
    pub fn active_count(&self) -> usize {
        self.engines
            .values()
            .filter(|e| e.state == EngineLifecycle::Active)
            .count()
    }
}
