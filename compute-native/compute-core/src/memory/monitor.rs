//! Real-time memory monitoring via macOS system APIs.
//!
//! Reference: `ref/omlx/memory_monitor.py`
//! Uses mach_vm_info and proc_info for accurate per-process stats on Apple Silicon.

use std::time::{Duration, Instant};

use super::MemoryPressure;

/// Memory statistics snapshot
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub rss_bytes: u64,
    pub total_ram_bytes: u64,
    pub vm_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
}

impl MemoryStats {
    /// Compute memory pressure level from current stats
    pub fn pressure(&self) -> MemoryPressure {
        let ratio = self.rss_bytes as f64 / self.total_ram_bytes.max(1) as f64;
        if self.swap_used_bytes > 0
            && self.swap_total_bytes > 0
            && self.swap_used_bytes as f64 / self.swap_total_bytes as f64 > 0.5
        {
            MemoryPressure::Oom
        } else if ratio > 0.90 {
            MemoryPressure::Severe
        } else if ratio > 0.80 {
            MemoryPressure::Critical
        } else if ratio > 0.70 {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }
}

/// Real-time memory monitor
///
/// Polls system memory stats at configurable intervals and
/// triggers callbacks when pressure levels change.
#[allow(dead_code)]
pub struct MemoryMonitor {
    stats: MemoryStats,
    last_update: Instant,
    poll_interval: Duration,
    last_pressure: MemoryPressure,
}

impl MemoryMonitor {
    pub fn new(poll_interval: Duration) -> Self {
        Self {
            stats: MemoryStats {
                rss_bytes: 0,
                total_ram_bytes: 0,
                vm_bytes: 0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
            },
            last_update: Instant::now(),
            poll_interval,
            last_pressure: MemoryPressure::Normal,
        }
    }

    /// Poll current memory stats from the system
    pub fn poll(&mut self) -> MemoryStats {
        // TODO: implement macOS-specific memory polling
        // - host_statistics64/mach_vm_info for RSS
        // - sysctl for total RAM
        // - proc_info for swap
        self.last_update = Instant::now();
        self.stats.clone()
    }

    /// Get last known pressure level
    pub fn pressure(&self) -> MemoryPressure {
        self.last_pressure
    }
}
