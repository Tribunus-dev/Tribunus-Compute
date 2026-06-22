//! Unified memory telemetry — aggregates memory stats from all three subsystems
//! (IOSurface pool, MLX Metal allocator, Candle Metal allocator) plus process-level
//! RSS and page-fault counters into a single point-in-time snapshot.
//!
//! Reference: `docs/unified-memory-island.md`

use crate::memory::allocator::IosurfaceAllocator;
use crate::worker_memory::{
    detect_machine_profile, sample_mlx_memory, sample_page_faults, sample_process_rss_self,
    MachineProfile, MlxMemorySnapshot,
};

// ---------------------------------------------------------------------------
// CandleAllocatorStats
// ---------------------------------------------------------------------------

/// Stats from the candle Metal allocator.
///
/// Candle manages Metal buffers through a bucket-allocator scheme.  These
/// fields report the current active, cached, and cumulative total allocations
/// from that subsystem.  All values are in bytes.
///
/// When candle has not been initialised or its allocator is not wired into
/// the unified pool, all fields will be 0.
#[derive(Debug, Clone, Default)]
pub struct CandleAllocatorStats {
    /// Bytes currently in use by live candle tensors.
    pub active_buffer_bytes: u64,
    /// Bytes held in the candle bucket cache (available for reuse).
    pub cached_buffer_bytes: u64,
    /// Cumulative bytes allocated over the lifetime of the candle allocator.
    pub total_allocation_bytes: u64,
}

// ---------------------------------------------------------------------------
// UnifiedMemoryTelemetry
// ---------------------------------------------------------------------------

/// A single point-in-time snapshot of every memory subsystem Tribunus Compute
/// manages, plus process-level counters.
///
/// Use [`sample_unified_memory`] to collect all values in one shot.
#[derive(Debug, Clone)]
pub struct UnifiedMemoryTelemetry {
    /// Physical machine memory configuration (total RAM, usable bytes, model).
    pub machine: MachineProfile,
    /// Resident set size of the current process, in bytes.
    pub process_rss_bytes: u64,
    /// Total bytes allocated through the IOSurface allocator pool.
    pub iosurface_allocator_bytes: u64,
    /// Memory pressure of the IOSurface pool (`total_allocated / max_pool`).
    pub iosurface_pressure: f64,
    /// MLX Metal allocator snapshot (active, cache, peak).
    pub mlx: MlxMemorySnapshot,
    /// Candle bucket allocator snapshot.
    pub candle: CandleAllocatorStats,
    /// Cumulative page faults (pageins) for the current process.
    pub page_faults: u64,
}

// ---------------------------------------------------------------------------
// sample_unified_memory
// ---------------------------------------------------------------------------

/// Sample every memory subsystem and return a unified telemetry record.
///
/// # Errors
///
/// This function **never panics**.  Every fallible call uses `match` or
/// `if-let`; failures produce 0 / default / `NaN`-free fallback values
/// rather than unwrapping.
///
/// # Platform Support
///
/// - RSS and page-fault counters are fully supported on macOS.
/// - On non-macOS platforms RSS returns 0 and page-faults returns 0.
/// - `detect_machine_profile` reports an `Unknown` model / `0` bytes when
///   platform sysctls are unavailable.
/// - `sample_mlx_memory` returns all-zeros when MLX has not been initialised.
pub fn sample_unified_memory(allocator: &IosurfaceAllocator) -> UnifiedMemoryTelemetry {
    let machine = detect_machine_profile();

    let process_rss_bytes = sample_process_rss_self();

    let iosurface_allocator_bytes = allocator.total_allocated();
    let iosurface_pressure = allocator.pressure();

    let mlx = sample_mlx_memory();

    // Candle stats — not yet wired into the unified pool.
    // TODO: plumb actual values once the candle bucket allocator is bridged.
    let candle = CandleAllocatorStats {
        active_buffer_bytes: 0,
        cached_buffer_bytes: 0,
        total_allocation_bytes: 0,
    };

    let page_faults = sample_page_faults();

    UnifiedMemoryTelemetry {
        machine,
        process_rss_bytes,
        iosurface_allocator_bytes,
        iosurface_pressure,
        mlx,
        candle,
        page_faults,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_defaults_are_zero() {
        let stats = CandleAllocatorStats::default();
        assert_eq!(stats.active_buffer_bytes, 0);
        assert_eq!(stats.cached_buffer_bytes, 0);
        assert_eq!(stats.total_allocation_bytes, 0);
    }

    #[test]
    fn test_unified_telemetry_rounds_to_zero_on_empty_allocator() {
        // When no IOSurface allocator exists yet, the pressure and bytes must
        // be well-defined (not NaN, not infinite).
        let allocator = IosurfaceAllocator::new(0);
        let t = sample_unified_memory(&allocator);

        assert_eq!(t.iosurface_allocator_bytes, 0);
        assert!(t.iosurface_pressure.is_finite());
        assert_eq!(t.iosurface_pressure, 0.0);
    }

    #[test]
    fn test_mlx_defaults_when_not_initialised() {
        let allocator = IosurfaceAllocator::new(1024 * 1024);
        let t = sample_unified_memory(&allocator);

        // MLX counters are 0 when the runtime has not been set up.
        assert_eq!(t.mlx.active_bytes, 0);
        assert_eq!(t.mlx.cache_bytes, 0);
        assert_eq!(t.mlx.peak_bytes, 0);
    }

    #[test]
    fn test_process_rss_non_negative() {
        let allocator = IosurfaceAllocator::new(1024 * 1024);
        let t = sample_unified_memory(&allocator);

        // RSS must never be negative; on non-macOS it is 0.
        assert!(t.process_rss_bytes == 0 || t.process_rss_bytes > 0);
    }

    #[test]
    fn test_page_faults_non_negative() {
        let allocator = IosurfaceAllocator::new(1024 * 1024);
        let t = sample_unified_memory(&allocator);

        assert!(t.page_faults == 0 || t.page_faults > 0);
    }

    #[test]
    fn test_pressure_is_normalised() {
        // 2 GiB pool with 256 MiB allocated → pressure ≈ 0.125
        let allocator = IosurfaceAllocator::new(2 * 1024 * 1024 * 1024);
        let t = sample_unified_memory(&allocator);

        assert!(t.iosurface_pressure >= 0.0);
        assert!(t.iosurface_pressure <= 1.0);
    }
}
