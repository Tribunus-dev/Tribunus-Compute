//! Proactive memory enforcement — prevents OOM by taking action
//! based on memory pressure level.
//!
//! Reference: `ref/omlx/process_memory_enforcer.py`
//! Design: `docs/omlx-memory-management.md`

use super::monitor::MemoryMonitor;
use super::MemoryPressure;

/// Actions the enforcer can take under pressure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAction {
    CompressKvCache,
    EvictPrefixCache,
    ReduceContextLength,
    SwapModelToDisk,
    SuspendEngine,
    ForceGarbageCollection,
    FreePagedCache,
}

/// Proactive memory enforcer
///
/// Monitors memory pressure and dispatches appropriate actions:
/// - Warning:     compress KV cache
/// - Critical:    evict prefix cache + compress KV cache
/// - Severe:      reduce context length, swap idle models
/// - OOM:         suspend engines, force garbage collection
pub struct MemoryEnforcer {
    monitor: MemoryMonitor,
    current_pressure: MemoryPressure,
}

impl MemoryEnforcer {
    pub fn new(mut monitor: MemoryMonitor) -> Self {
        let current_pressure = monitor.poll().pressure();
        Self {
            monitor,
            current_pressure,
        }
    }

    /// Run one enforcement cycle — returns actions to take
    pub fn enforce(&mut self) -> Vec<MemoryAction> {
        let stats = self.monitor.poll();
        let pressure = stats.pressure();

        if pressure > self.current_pressure {
            self.current_pressure = pressure;
            self.escalate(pressure)
        } else if pressure < self.current_pressure {
            self.current_pressure = pressure;
            vec![] // De-escalating, no actions needed
        } else {
            vec![] // Stable
        }
    }

    fn escalate(&self, pressure: MemoryPressure) -> Vec<MemoryAction> {
        match pressure {
            MemoryPressure::Warning => {
                vec![MemoryAction::CompressKvCache]
            }
            MemoryPressure::Critical => {
                vec![
                    MemoryAction::CompressKvCache,
                    MemoryAction::EvictPrefixCache,
                ]
            }
            MemoryPressure::Severe => {
                vec![
                    MemoryAction::ReduceContextLength,
                    MemoryAction::SwapModelToDisk,
                ]
            }
            MemoryPressure::Oom => {
                vec![
                    MemoryAction::SuspendEngine,
                    MemoryAction::ForceGarbageCollection,
                ]
            }
            _ => vec![],
        }
    }
}
