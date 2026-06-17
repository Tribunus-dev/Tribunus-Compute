//! Memory plan bridge — pre-computed IOSurface allocation plans for the
//! Metal allocator.
//!
//! The memory plan tells MLX's Metal allocator exactly which IOSurface slice
//! to use for each allocation in sequence.  This replaces the runtime
//! allocation + copy + JIT compile pattern with a compile-time planned
//! sequence of zero-copy IOSurface-backed allocations.
//!
//! # Architecture
//!
//! 1. The compiler produces a [`MemoryPlan`] with an ordered list of
//!    (iosurface_ptr, size) pairs — one per allocation the model needs
//!    during a forward pass.
//! 2. Before executing a planned region, the executor calls
//!    [`set_memory_plan`] which passes the plan to the Metal allocator
//!    via the C FFI bridge.
//! 3. For each `malloc(size)` call during execution, the Metal allocator
//!    checks the next plan slot.  If it exists and sizes match, it wraps
//!    the IOSurface pointer as an `MTLBuffer` instead of allocating new
//!    GPU memory.
//! 4. After the region completes, the executor calls [`clear_memory_plan`].

use std::ffi::c_void;

// ── C-compatible types ────────────────────────────────────────────────────

/// Memory plan slot — matches `mlx_memory_plan_slot` in mlx/c/memory.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemoryPlanSlot {
    /// Base address of the pre-assigned IOSurface slice.
    pub iosurface_ptr: *mut c_void,
    /// Expected allocation size in bytes.
    pub size: usize,
}

// Safety: the type is POD (plain old data) with no references or Drop.
unsafe impl Send for MemoryPlanSlot {}
unsafe impl Sync for MemoryPlanSlot {}

// ── FFI declarations ──────────────────────────────────────────────────────

extern "C" {
    /// Set the memory plan for the Metal allocator.
    ///
    /// `slots` must remain valid for the duration of the planned region.
    /// The allocator copies the plan entries internally.
    fn mlx_set_memory_plan(num_slots: usize, slots: *const MemoryPlanSlot) -> i32;

    /// Clear the memory plan without consuming remaining slots.
    fn mlx_clear_memory_plan() -> i32;
}

// ── Safe wrapper ──────────────────────────────────────────────────────────

/// A pre-computed memory plan: an ordered list of (ptr, size) pairs that
/// the Metal allocator will use instead of allocating new GPU memory.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Ordered allocation slots.
    pub slots: Vec<MemoryPlanSlot>,
}

impl MemoryPlan {
    /// Create a new empty memory plan.
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Add a slot: an IOSurface pointer and its expected allocation size.
    pub fn add_slot(&mut self, ptr: *mut c_void, size: usize) {
        self.slots.push(MemoryPlanSlot {
            iosurface_ptr: ptr,
            size,
        });
    }

    /// Number of slots in the plan.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True if the plan has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Send this plan to the Metal allocator via the C FFI bridge.
    ///
    /// The allocator will use the pre-assigned IOSurface slices for its
    /// next `len()` allocations, in order.
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - All `iosurface_ptr` values point to valid, mapped IOSurface memory
    /// - The IOSurface memory remains valid until `clear_memory_plan()` is
    ///   called or all plan slots are consumed
    /// - The actual allocations made by MLX during the planned region match
    ///   the plan slots in both count and size
    pub unsafe fn apply(&self) -> Result<(), String> {
        if self.slots.is_empty() {
            return Ok(());
        }
        let ret = mlx_set_memory_plan(self.slots.len(), self.slots.as_ptr());
        if ret != 0 {
            return Err(format!("mlx_set_memory_plan returned {}", ret));
        }
        Ok(())
    }
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Clear the active memory plan from the Metal allocator.
///
/// After calling this, subsequent `malloc`s will use normal Metal buffer
/// allocation (heap/cache) instead of the plan.
///
/// Safe to call when no plan is active (no-op).
pub fn clear_memory_plan() -> Result<(), String> {
    let ret = unsafe { mlx_clear_memory_plan() };
    if ret != 0 {
        return Err(format!("mlx_clear_memory_plan returned {}", ret));
    }
    Ok(())
}

// ── Integration with the compiler ─────────────────────────────────────────

/// Generate a memory plan from the compiler's `ScheduledModule`.
///
/// Walks the scheduled regions' [`MemoryPlan`] and produces an ordered
/// list of [`MemoryPlanSlot`] entries that the executor passes to the
/// Metal allocator before running the module.
///
/// When `compression_ratio` is `Some(r)`, the estimated KV cache byte sizes
/// are scaled down by `r` (e.g. 4.57 for TurboQuant3: 16 bits / 3.5 bits).
/// Pass `None` for uncompressed FP16 mode (no scaling).
///
/// Returns `None` if the scheduled module has no material allocations
/// that need planning (empty or all in-place).
pub fn plan_from_scheduled_module(
    scheduled: &crate::compiler::scheduled::ScheduledModule,
    arena: &crate::arena::Arena,
    compression_ratio: Option<f64>,
) -> Option<MemoryPlan> {
    let mut plan = MemoryPlan::new();
    let ratio = compression_ratio.unwrap_or(1.0);

    for region in &scheduled.regions {
        let scaled = (region.temp_memory_bytes as f64 / ratio) as u64;
        if scaled == 0 {
            continue;
        }

        // The arena pre-allocates a contiguous block. We carve out slices
        // for each planned allocation within it.
        let slice_base = unsafe { arena.base_ptr() as *mut c_void };
        let offset = plan.slots.len() as u64 * 4096; // page-aligned offset
        let ptr = unsafe { (slice_base as *mut u8).add(offset as usize) as *mut c_void };
        let size = scaled as usize;

        plan.add_slot(ptr, size);
    }

    if plan.is_empty() {
        None
    } else {
        Some(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_plan() {
        let plan = MemoryPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_add_slot() {
        let mut plan = MemoryPlan::new();
        let dummy = 0xdeadbeef as *mut c_void;
        plan.add_slot(dummy, 4096);
        assert_eq!(plan.len(), 1);
        assert!(!plan.is_empty());
        assert_eq!(plan.slots[0].iosurface_ptr, dummy);
        assert_eq!(plan.slots[0].size, 4096);
    }

    #[test]
    fn test_default_is_empty() {
        let plan: MemoryPlan = Default::default();
        assert!(plan.is_empty());
    }
}
