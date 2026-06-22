//! Tensix-specific profiling for Tenstorrent device operations.
//!
//! Provides [`TensixProfileEvent`] for per-operation metrics (core count,
//! tile throughput, math fidelity, NOC utilization, circular buffer occupancy,
//! DRAM traffic) and [`TensixProfileCollector`] for accumulating events from
//! the device at runtime.
//!
//! Both types are behind `#[cfg(feature = "tensix")]`.

use std::time::Instant;

/// Per-operation profiling for Tensix kernel execution.
///
/// Records raw device counters and host-side overhead so that a downstream
/// profiler or scheduler can compute wall-time estimates, detect bottlenecks,
/// and tune tile / core allocation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensixProfileEvent {
    pub op_kind: String,         // "matmul", "sdpa", "rms_norm", "rope", "silu"
    pub core_count: u32,         // Tensix cores used
    pub workload_tiles: u32,     // Total tiles processed
    pub math_fidelity: String,   // "LoFi", "HiFi2", "HiFi3", "HiFi4"
    pub data_format: String,     // "Float16", "BFloat16", "Float32"
    pub kernel_cycles: u64,      // Raw cycle count (from device)
    pub sync_ns: u64,            // Host-device sync wait
    pub dram_bytes_read: u64,    // DRAM input bytes
    pub dram_bytes_written: u64, // DRAM output bytes
    pub cb_occupancy: f32,       // Circular buffer utilization 0..1
    pub noc_utilization: f32,    // NOC bandwidth utilization 0..1
    pub host_queue_ns: u64,      // Time spent in host command queue
}

impl TensixProfileEvent {
    pub fn new(op_kind: &str) -> Self {
        TensixProfileEvent {
            op_kind: op_kind.to_string(),
            core_count: 1,
            workload_tiles: 0,
            math_fidelity: "HiFi3".into(),
            data_format: "BFloat16".into(),
            kernel_cycles: 0,
            sync_ns: 0,
            dram_bytes_read: 0,
            dram_bytes_written: 0,
            cb_occupancy: 0.5,
            noc_utilization: 0.0,
            host_queue_ns: 0,
        }
    }

    /// Estimate total wall time (kernel + sync + queue overhead).
    pub fn total_ns(&self) -> u64 {
        self.kernel_cycles + self.sync_ns + self.host_queue_ns
    }
}

/// Collects Tensix profiling events from the device.
///
/// Accumulates [`TensixProfileEvent`] samples and tracks elapsed wall time.
/// In stub mode (no real device) `sync_from_device` is a no-op.
pub struct TensixProfileCollector {
    pub events: Vec<TensixProfileEvent>,
    start: Instant,
}

impl TensixProfileCollector {
    pub fn new() -> Self {
        TensixProfileCollector {
            events: Vec::new(),
            start: Instant::now(),
        }
    }

    /// Record a completed op.
    pub fn record(&mut self, event: TensixProfileEvent) {
        self.events.push(event);
    }

    /// Read accumulated profiling data from device.
    ///
    /// In stub mode, this is a no-op. A real implementation would call
    /// `tensix_read_profiler()` via FFI.
    pub fn sync_from_device(&mut self) {
        // In real mode, calls tensix_read_profiler() from FFI
        // In stub mode, nothing to do
    }

    /// Total elapsed wall time.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    /// Clear accumulated events and reset the wall clock.
    pub fn reset(&mut self) {
        self.events.clear();
        self.start = Instant::now();
    }
}

impl Default for TensixProfileCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensix_event_new_defaults() {
        let ev = TensixProfileEvent::new("matmul");
        assert_eq!(ev.op_kind, "matmul");
        assert_eq!(ev.core_count, 1);
        assert_eq!(ev.workload_tiles, 0);
        assert_eq!(ev.math_fidelity, "HiFi3");
        assert_eq!(ev.data_format, "BFloat16");
        assert_eq!(ev.total_ns(), 0);
    }

    #[test]
    fn tensix_event_total_ns() {
        let mut ev = TensixProfileEvent::new("sdpa");
        ev.kernel_cycles = 1000;
        ev.sync_ns = 200;
        ev.host_queue_ns = 50;
        assert_eq!(ev.total_ns(), 1250);
    }

    #[test]
    fn tensix_collector_record_and_elapsed() {
        let mut coll = TensixProfileCollector::new();
        assert!(coll.events.is_empty());
        coll.record(TensixProfileEvent::new("rms_norm"));
        coll.record(TensixProfileEvent::new("rope"));
        assert_eq!(coll.events.len(), 2);
    }

    #[test]
    fn tensix_collector_reset() {
        let mut coll = TensixProfileCollector::new();
        coll.record(TensixProfileEvent::new("silu"));
        assert_eq!(coll.events.len(), 1);
        coll.reset();
        assert!(coll.events.is_empty());
    }

    #[test]
    fn tensix_collector_default() {
        let coll = TensixProfileCollector::default();
        assert!(coll.events.is_empty());
    }

    #[test]
    fn tensix_collector_sync_is_noop() {
        let mut coll = TensixProfileCollector::new();
        coll.sync_from_device(); // must not panic
        assert!(coll.events.is_empty());
    }
}
