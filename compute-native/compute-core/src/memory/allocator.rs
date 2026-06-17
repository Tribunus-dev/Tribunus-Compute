//! IosurfaceAllocator — unified Metal-backed memory allocator.
//!
//! All subsystems (mlx-rs, candle, Core ML) draw from this allocator
//! through the unified memory island architecture.
//!
//! All allocated memory is IOSurface-backed and zero-copy shareable
//! across the MLX, candle, and Core ML backends.
//!
//! Reference: Arena for IOSurface allocation lifecycle;
//! ExternalArray + new_external_array() for the MLX bridge.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use std::sync::{Arc, Weak};

use crate::arena::Arena;
use mlx_rs::Dtype;

/// Unique identifier for an allocated arena within the `IosurfaceAllocator`.
pub type ArenaId = u64;

/// A unified IOSurface-backed allocator that all subsystems draw from.
///
/// Allocates IOSurface-backed memory via [`Arena`] and exposes it
/// to mlx-rs, candle, and Core ML without copies.
///
/// # Lifecycle
///
/// 1. `allocate` — creates a new IOSurface-backed arena, tracks its byte
///    consumption, and returns a unique `ArenaId`.
/// 2. `get_arena` — transfers ownership of the arena out of the allocator.
///    The caller is responsible for dropping it (which frees the IOSurface).
/// 3. `free` — removes the arena from tracking and drops it (IOSurface teardown).
///
/// # Pool limits
///
/// `max_pool_bytes` caps total IOSurface allocations. When set to `0` the
/// pool is unlimited. [`pressure()`](Self::pressure) reports the fraction
/// of the pool that is currently allocated.
pub struct IosurfaceAllocator {
    /// Next arena ID (monotonically increasing).
    next_id: AtomicU64,
    /// Active arenas tracked by ID.
    active_arenas: Mutex<HashMap<ArenaId, Arena>>,
    /// Total bytes currently allocated across all tracked arenas.
    total_allocated_bytes: AtomicU64,
    /// Maximum pool size in bytes (0 = unlimited).
    max_pool_bytes: u64,
}

impl IosurfaceAllocator {
    /// Create a new `IosurfaceAllocator`.
    ///
    /// `max_pool_bytes` limits the total IOSurface memory. Pass `0` for
    /// no limit.
    pub fn new(max_pool_bytes: u64) -> Self {
        Self {
            next_id: AtomicU64::new(0),
            active_arenas: Mutex::new(HashMap::new()),
            total_allocated_bytes: AtomicU64::new(0),
            max_pool_bytes,
        }
    }

    /// Allocate a new IOSurface-backed arena.
    ///
    /// Returns a unique [`ArenaId`] on success. The allocation is checked
    /// against the pool limit (`max_pool_bytes`) before creating the arena.
    ///
    /// # Errors
    ///
    /// - Returns an error if `dtype` is not `Float16` (the only dtype
    ///   currently supported by [`Arena::new`]).
    /// - Returns an error if allocating would exceed `max_pool_bytes`.
    /// - Returns an error if the underlying IOSurface allocation fails.
    pub fn allocate(
        &self,
        logical_dim0: u32,
        logical_dim1: u32,
        dtype: Dtype,
    ) -> Result<ArenaId, String> {
        // 1. Estimate byte cost before allocating.
        let estimated_bytes = (logical_dim0 as u64)
            .saturating_mul(logical_dim1 as u64)
            .saturating_mul(bytes_per_element(dtype));

        let current = self.total_allocated();
        if self.max_pool_bytes > 0 && current.saturating_add(estimated_bytes) > self.max_pool_bytes
        {
            return Err(format!(
                "IosurfaceAllocator: allocation would exceed pool limit: \
                 {} + {} > {}",
                current, estimated_bytes, self.max_pool_bytes,
            ));
        }

        // 2. Create the arena through the IOSurface bridge.
        let arena = Arena::new(logical_dim0, logical_dim1, dtype)?;

        // 3. Get the actual byte size (may differ from estimate due to
        //    IOSurface row-stride alignment).
        let actual_bytes = arena.byte_len() as u64;

        // 4. Re-check pool limit with actual size (defensive — the estimate
        //    should always be >= actual for IOSurface, but alignment padding
        //    on M-series can increase the physical allocation).
        if self.max_pool_bytes > 0 && current.saturating_add(actual_bytes) > self.max_pool_bytes {
            // Drop the arena (frees the IOSurface), then return an error.
            drop(arena);
            return Err(format!(
                "IosurfaceAllocator: actual allocation {} exceeds pool limit {} \
                 (current: {})",
                actual_bytes, self.max_pool_bytes, current,
            ));
        }

        // 5. Assign an id and track the arena.
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.total_allocated_bytes
            .fetch_add(actual_bytes, Ordering::Relaxed);

        let mut arenas = self.active_arenas.lock();
        if let Some(_prev) = arenas.insert(id, arena) {
            // This should never happen with monotonically increasing ids.
            // Defensive: decrement the counter (we already added it) and bail.
            self.total_allocated_bytes
                .fetch_sub(actual_bytes, Ordering::Relaxed);
            return Err(format!("IosurfaceAllocator: id collision on {}", id));
        }

        Ok(id)
    }

    /// Transfer ownership of an allocated arena out of the allocator.
    ///
    /// The arena is removed from internal tracking. The caller becomes
    /// responsible for dropping it (which triggers IOSurface teardown).
    ///
    /// Returns `None` if `id` is not tracked.
    ///
    /// # Note on byte accounting
    ///
    /// Because the arena is transferred to the caller, [`total_allocated`]
    /// is **not** decremented. Use [`free`](Self::free) when you want the
    /// allocator to manage the full lifecycle (including byte accounting).
    pub fn get_arena(&self, id: ArenaId) -> Option<Arena> {
        let mut arenas = self.active_arenas.lock();
        arenas.remove(&id)
    }

    /// Free an arena and reclaim its IOSurface memory.
    ///
    /// The arena is removed from internal tracking, its bytes are deducted
    /// from [`total_allocated`](Self::total_allocated), and the arena is
    /// dropped (triggering IOSurface teardown).
    ///
    /// Returns an error if `id` is not tracked.
    pub fn free(&self, id: ArenaId) -> Result<(), String> {
        let mut arenas = self.active_arenas.lock();
        let arena = arenas.remove(&id);

        match arena {
            Some(a) => {
                let byte_len = a.byte_len() as u64;
                self.total_allocated_bytes
                    .fetch_sub(byte_len, Ordering::Relaxed);
                // Arena drops here — frees the IOSurface.
                Ok(())
            }
            None => Err(format!("IosurfaceAllocator: arena {} not found", id)),
        }
    }

    /// Current total IOSurface allocation in bytes.
    ///
    /// This is the sum of all tracked arenas' `byte_len` values.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated_bytes.load(Ordering::Relaxed)
    }

    /// Memory pressure as a fraction of `max_pool_bytes`.
    ///
    /// Returns `0.0` when `max_pool_bytes` is `0` (unlimited pool).
    /// Returns `1.0` or greater when total allocation meets or exceeds
    /// the pool limit.
    pub fn pressure(&self) -> f64 {
        if self.max_pool_bytes == 0 {
            return 0.0;
        }
        self.total_allocated() as f64 / self.max_pool_bytes as f64
    }
}

/// Compute the byte size of a single element for the given dtype.
///
/// This is used for pre-allocation pool-limit checks. Actual physical
/// allocation may differ due to IOSurface row-stride alignment.
fn bytes_per_element(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Float16 => 2,
        Dtype::Float32 | Dtype::Bfloat16 => 4,
        Dtype::Int8 | Dtype::Uint8 => 1,
        Dtype::Int16 | Dtype::Uint16 => 2,
        Dtype::Int32 | Dtype::Uint32 => 4,
        Dtype::Int64 | Dtype::Uint64 => 8,
        // Default fallback — Float32-sized.
        _ => 4,
    }
}

/// Paged sub-allocator within a single large IOSurface arena.
///
/// Pages are allocated from a free bitmap. All backends (MLX, Accelerate,
/// Core ML) share the same physical pages via the single IOSurface.
pub struct PagedIosurfaceAllocator {
    /// The single large IOSurface arena backing all pages.
    arena: Arena,
    /// Total number of pages.
    num_pages: usize,
    /// Page size in bytes.
    page_size: usize,
    /// Free page bitmap (1 = free, 0 = allocated).
    free_bitmap: Vec<u64>,
}

impl PagedIosurfaceAllocator {
    /// Create a new paged allocator over a single IOSurface arena.
    /// Total memory = num_pages * page_size.
    pub fn new(arena: Arena, num_pages: usize, page_size: usize) -> Self {
        let bitmap_words = (num_pages + 63) / 64;
        // All bits start as 1 (free).
        let free_bitmap = vec![!0u64; bitmap_words];
        Self {
            arena,
            num_pages,
            page_size,
            free_bitmap,
        }
    }

    /// Allocate a contiguous run of `count` pages.
    /// Returns None if insufficient contiguous free pages.
    pub fn allocate_pages(&mut self, count: usize) -> Option<Vec<usize>> {
        let bits = self.num_pages;
        let mut start = 0;
        while start < bits {
            let word_idx = start / 64;
            let bit_off = start % 64;
            if word_idx >= self.free_bitmap.len() {
                break;
            }
            let word = self.free_bitmap[word_idx];
            let masked = word & (!0u64 << bit_off);
            if masked == 0 {
                start = (word_idx + 1) * 64;
                continue;
            }
            let first_free = word_idx * 64 + masked.trailing_zeros() as usize;
            if first_free >= bits {
                break;
            }
            let mut ok = true;
            for i in 0..count {
                let pg = first_free + i;
                if pg >= bits {
                    ok = false;
                    break;
                }
                let w = pg / 64;
                let b = pg % 64;
                if (self.free_bitmap[w] & (1u64 << b)) == 0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                for i in 0..count {
                    let pg = first_free + i;
                    let w = pg / 64;
                    let b = 1u64 << (pg % 64);
                    self.free_bitmap[w] &= !b;
                }
                return Some((0..count).map(|i| first_free + i).collect());
            }
            start = first_free + 1;
        }
        None
    }

    /// Free a previously allocated page.
    pub fn free_page(&mut self, page_id: usize) {
        if page_id >= self.num_pages {
            return;
        }
        let w = page_id / 64;
        let b = 1u64 << (page_id % 64);
        self.free_bitmap[w] |= b;
    }

    /// Get the device pointer for a page (base + page_id * page_size).
    pub fn page_address(&self, page_id: usize) -> *const std::ffi::c_void {
        let offset = page_id * self.page_size;
        unsafe { (self.arena.base_ptr() as *mut u8).add(offset) as *const std::ffi::c_void }
    }

    /// Get the IOSurface base pointer.
    pub fn base_ptr(&self) -> *const std::ffi::c_void {
        // SAFETY: arena is valid (IOSurface is locked)
        unsafe { self.arena.base_ptr() as *const std::ffi::c_void }
    }

    /// Return the page_size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Return number of free pages.
    pub fn free_pages(&self) -> usize {
        let mut count = 0usize;
        for w in 0..self.free_bitmap.len() {
            let word = self.free_bitmap[w];
            count += word.count_ones() as usize;
        }
        count.min(self.num_pages)
    }

    /// Return the total number of pages (capacity).
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }
}

/// High-level KV cache block allocator on top of `PagedIosurfaceAllocator`.
///
/// Adds generation-counter safety and block-level tracking. Blocks are
/// individual pages from the underlying bitmap page allocator. Each block
/// carries a generation counter that increments when the page is recycled,
/// preventing use-after-free via stale `BlockHandle`s.
///
/// Default block (page) size: 512 bytes = 1 KV head x 128 head_dim x FP16.
/// For Gemma 4 with n_kv_heads=4, each layer needs 4 blocks per token position.
pub struct KvCacheBlockAllocator {
    /// Underlying page-level allocator.
    pager: PagedIosurfaceAllocator,
    /// Per-page generation counters, incremented on each free.
    generations: Vec<u32>,
}

impl KvCacheBlockAllocator {
    /// Create a new KV cache block allocator wrapping an existing pager.
    ///
    /// The pager must already be initialized with the desired IOSurface arena
    /// and page configuration. Generation counters start at 0 for every page.
    pub fn new(pager: PagedIosurfaceAllocator) -> Self {
        let num_pages = pager.num_pages();
        Self {
            pager,
            generations: vec![0u32; num_pages],
        }
    }

    /// Allocate a single block (page).
    ///
    /// Returns `(page_index, generation)` on success, or an error if no pages
    /// remain. The returned generation must be checked against the allocator's
    /// current generation at access time to prevent use-after-free.
    pub fn alloc_block(&mut self) -> Result<(usize, u32), String> {
        let pages = self
            .pager
            .allocate_pages(1)
            .ok_or_else(|| "KvCacheBlockAllocator: no free blocks available".to_string())?;
        let idx = pages[0];
        let gen = self.generations[idx];
        Ok((idx, gen))
    }

    /// Free a block by page index.
    ///
    /// The page is returned to the underlying allocator's free pool and its
    /// generation counter is incremented. Any `BlockHandle` still holding the
    /// old generation will be stale.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_pages`.
    pub fn free_block(&mut self, index: usize) {
        assert!(
            index < self.generations.len(),
            "KvCacheBlockAllocator: block index {} out of range (max {})",
            index,
            self.generations.len()
        );
        self.pager.free_page(index);
        self.generations[index] = self.generations[index].wrapping_add(1);
    }

    /// Translate a block index to a raw pointer into the IOSurface.
    ///
    /// Returns `None` when the index is out of range.
    pub fn block_ptr(&self, index: usize) -> Option<*const u8> {
        if index >= self.pager.num_pages() {
            return None;
        }
        Some(self.pager.page_address(index) as *const u8)
    }

    /// Total number of blocks currently allocated.
    pub fn allocated_blocks(&self) -> usize {
        self.pager.num_pages() - self.pager.free_pages()
    }

    /// Number of free blocks available.
    pub fn free_blocks(&self) -> usize {
        self.pager.free_pages()
    }

    /// Total pool capacity in blocks.
    pub fn capacity_blocks(&self) -> usize {
        self.pager.num_pages()
    }

    /// True if the pool has no more free blocks.
    pub fn is_full(&self) -> bool {
        self.pager.free_pages() == 0
    }

    /// Page size in bytes (size of each block).
    pub fn page_size(&self) -> usize {
        self.pager.page_size()
    }
}

/// A guard that auto-frees a KV cache block when dropped.
///
/// Each handle holds a page index and the generation that was current at
/// allocation time. On `Drop`, if the handle has not been explicitly freed,
/// it returns the block to the allocator. The generation counter prevents
/// use-after-free: any attempt to access a block through a stale handle
/// will detect the mismatch.
pub struct BlockHandle {
    /// Weak reference to the allocator, so handles do not prevent the
    /// allocator from being dropped.
    allocator: Weak<Mutex<KvCacheBlockAllocator>>,
    /// Page index within the IOSurface.
    page_index: usize,
    /// Generation at allocation time.
    generation: u32,
    /// True if this handle has been explicitly freed or marked.
    freed: bool,
}

impl BlockHandle {
    /// Create a new block handle.
    ///
    /// `allocator` must be the same `Arc<Mutex<KvCacheBlockAllocator>>` that
    /// allocated the page. `index` and `generation` come from
    /// [`KvCacheBlockAllocator::alloc_block`].
    pub fn new(
        allocator: &Arc<Mutex<KvCacheBlockAllocator>>,
        index: usize,
        generation: u32,
    ) -> Self {
        Self {
            allocator: Arc::downgrade(allocator),
            page_index: index,
            generation,
            freed: false,
        }
    }

    /// The page index this handle refers to.
    pub fn index(&self) -> usize {
        self.page_index
    }

    /// The generation this handle holds (snapshot from allocation time).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Mark this handle as freed without returning the block to the allocator.
    ///
    /// Use this when the block was already freed externally (e.g. through
    /// the allocator directly) to prevent a double-free on drop.
    pub fn mark_freed(&mut self) {
        self.freed = true;
    }

    /// Check whether this handle is still valid against the allocator's
    /// current generation for this page index.
    ///
    /// Returns `None` when the allocator has been dropped (no way to verify).
    /// Returns `Some(true)` when the generation still matches.
    /// Returns `Some(false)` when the generation is stale (use-after-free).
    pub fn is_valid(&self) -> Option<bool> {
        let alloc = self.allocator.upgrade()?;
        let guard = alloc.lock();
        if self.page_index >= guard.generations.len() {
            return Some(false);
        }
        Some(guard.generations[self.page_index] == self.generation)
    }
}

impl Drop for BlockHandle {
    fn drop(&mut self) {
        if self.freed {
            return;
        }
        if let Some(alloc) = self.allocator.upgrade() {
            let mut guard = alloc.lock();
            guard.free_block(self.page_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_allocator_zero_max() {
        let alloc = IosurfaceAllocator::new(0);
        assert_eq!(alloc.total_allocated(), 0);
        assert_eq!(alloc.pressure(), 0.0);
    }

    #[test]
    fn test_allocate_and_free() {
        let alloc = IosurfaceAllocator::new(1024 * 1024);
        let id = alloc
            .allocate(1, 4, Dtype::Float16)
            .expect("allocate should succeed");
        assert!(alloc.total_allocated() > 0);
        assert_eq!(alloc.free(id), Ok(()));
        assert_eq!(alloc.total_allocated(), 0);
    }

    #[test]
    fn test_get_arena_transfers_ownership() {
        let alloc = IosurfaceAllocator::new(0);
        let id = alloc.allocate(1, 4, Dtype::Float16).expect("allocate");
        let arena = alloc.get_arena(id).expect("get_arena should find id");
        assert_eq!(arena.element_count(), 4);
        // Second get should be None.
        assert!(alloc.get_arena(id).is_none());
        // Arena is dropped here — IOSurface teardown.
        // total_allocated is NOT decremented (caller owns it now).
    }

    #[test]
    fn test_free_unknown_id() {
        let alloc = IosurfaceAllocator::new(0);
        let result = alloc.free(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_pressure() {
        let alloc = IosurfaceAllocator::new(0);
        assert_eq!(alloc.pressure(), 0.0);

        let bounded = IosurfaceAllocator::new(200);
        let _id = bounded.allocate(10, 10, Dtype::Float16);
        // 10 x 10 x 2 = 200 bytes → pressure should be ~1.0
        // Actual may differ with IOSurface row-stride padding.
        let p = bounded.pressure();
        assert!(
            p >= 0.99 && p <= 2.0,
            "pressure {} out of expected range [0.99, 2.0] for 200-byte pool",
            p
        );
    }

    #[test]
    fn test_allocate_exceeds_pool() {
        let alloc = IosurfaceAllocator::new(2); // 2-byte pool
        let result = alloc.allocate(1, 4, Dtype::Float16); // 1 x 4 x 2 = 8 bytes
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceed"));
        assert_eq!(alloc.total_allocated(), 0);
    }

    #[test]
    fn test_dtype_not_supported() {
        let alloc = IosurfaceAllocator::new(0);
        // Arena::new only supports Float16.
        let result = alloc.allocate(1, 4, Dtype::Float32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("FP16"));
    }

    #[test]
    fn test_monotonic_ids() {
        let alloc = IosurfaceAllocator::new(0);
        let id1 = alloc.allocate(1, 1, Dtype::Float16).expect("allocate 1");
        let id2 = alloc.allocate(1, 1, Dtype::Float16).expect("allocate 2");
        assert!(id2 > id1);
    }

    #[test]
    fn test_total_allocated_after_free() {
        let alloc = IosurfaceAllocator::new(0);
        let id = alloc.allocate(1, 4, Dtype::Float16).expect("allocate");
        let before = alloc.total_allocated();
        alloc.free(id).expect("free");
        assert_eq!(
            alloc.total_allocated(),
            before - alloc.get_arena(id).map_or(0, |a| a.byte_len() as u64)
        );
    }

    #[test]
    fn test_multiple_arenas() {
        let alloc = IosurfaceAllocator::new(0);
        let id1 = alloc.allocate(1, 4, Dtype::Float16).expect("allocate 1");
        let id2 = alloc.allocate(2, 4, Dtype::Float16).expect("allocate 2");
        assert_ne!(id1, id2);

        let _ = alloc.free(id1);
        assert!(alloc.total_allocated() > 0);

        let _ = alloc.free(id2);
        assert_eq!(alloc.total_allocated(), 0);
    }
}
