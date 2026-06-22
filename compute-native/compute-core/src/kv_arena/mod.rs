//! Paged KV cache arena: physical blocks, COW refcounting, prefix caching,
//! backend residency mapping, and admission control with eviction.

pub mod backend;
pub mod block;
pub mod prefix;
pub mod refcount;

use crate::compute_image::kv_plan::KvCachePlan;
use backend::ResidencyTable;
use block::{tokens_to_blocks, BackendAffinity, PhysicalBlock, PhysicalBlockId};
use prefix::{PrefixCacheIndex, PrefixHash};

/// Globally-unique sequence identifier, assigned monotonically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

impl SequenceId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        SequenceId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for SequenceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Receipt returned after a successful sequence admission.
#[derive(Clone, Debug)]
pub struct AdmissionReceipt {
    pub sequence_id: SequenceId,
    pub allocated_blocks: Vec<PhysicalBlockId>,
    pub total_tokens: u64,
    pub total_blocks: usize,
    pub bytes_allocated: u64,
    pub prefix_hits: usize,
    pub new_blocks: usize,
}

/// Per-sequence logical-to-physical block translation table.
pub struct LogicalBlockTable {
    pub sequence_id: SequenceId,
    pub logical_to_physical: Vec<PhysicalBlockId>,
    pub token_count: u64,
    pub max_blocks: usize,
}

impl LogicalBlockTable {
    pub fn new(seq: SequenceId, max_blocks: usize) -> Self {
        Self {
            sequence_id: seq,
            logical_to_physical: Vec::new(),
            token_count: 0,
            max_blocks,
        }
    }

    /// Translate a logical block index to its physical block id.
    pub fn translate(&self, logical: u32) -> Option<PhysicalBlockId> {
        self.logical_to_physical.get(logical as usize).copied()
    }

    /// Append a physical block id to the end of the table.
    pub fn append(&mut self, physical: PhysicalBlockId) {
        self.logical_to_physical.push(physical);
    }

    /// Return the last physical block id, if any.
    pub fn last_physical(&self) -> Option<PhysicalBlockId> {
        self.logical_to_physical.last().copied()
    }
}

/// Errors that can occur during block allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArenaError {
    CapacityExceeded {
        requested: usize,
        available: usize,
        total: usize,
    },
    BackendMismatch {
        expected: BackendAffinity,
        actual: BackendAffinity,
    },
    OutOfMemory {
        attempted: usize,
        available: usize,
    },
}

use std::fmt;
impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::CapacityExceeded {
                requested,
                available,
                total,
            } => {
                write!(
                    f,
                    "KV arena capacity exceeded: requested {requested} blocks, \
                     {available} available of {total} total"
                )
            }
            ArenaError::BackendMismatch { expected, actual } => {
                write!(f, "backend mismatch: expected {expected:?}, got {actual:?}")
            }
            ArenaError::OutOfMemory {
                attempted,
                available,
            } => {
                write!(
                    f,
                    "KV arena out of memory: attempted {attempted} bytes, \
                     {available} available"
                )
            }
        }
    }
}
impl std::error::Error for ArenaError {}

/// Arena of physical KV cache blocks, with prefix caching, backend residency tracking,
/// admission control, and block eviction when capacity is exceeded.
pub struct KvBlockArena {
    pub blocks: Vec<PhysicalBlock>,
    pub free_list: Vec<usize>,
    pub block_size: usize,
    pub capacity: usize,
    pub backend: BackendAffinity,
    pub prefix_cache: PrefixCacheIndex,
    pub residency: ResidencyTable,
    pub logical_tables: Vec<LogicalBlockTable>,
    pub eviction_policy: String,
    pub cow_policy: String,
}

impl KvBlockArena {
    /// Create a new arena with the given block size, capacity, and backend affinity.
    pub fn new(block_size: usize, capacity: usize, backend: BackendAffinity) -> Self {
        let plan = KvCachePlan {
            block_tokens: block_size as u32,
            max_blocks: capacity as u32,
            ..KvCachePlan::default()
        };
        Self::from_plan(&plan, backend)
    }

    /// Create a new arena with prefix caching enabled (same as `new` — caching is always on).
    pub fn new_with_cache(block_size: usize, capacity: usize, backend: BackendAffinity) -> Self {
        Self::new(block_size, capacity, backend)
    }

    /// Create a new arena from a compiled KvCachePlan.
    pub fn from_plan(plan: &KvCachePlan, backend: BackendAffinity) -> Self {
        KvBlockArena {
            blocks: Vec::with_capacity(plan.max_blocks as usize),
            free_list: Vec::new(),
            block_size: plan.block_tokens as usize,
            capacity: plan.max_blocks as usize,
            backend,
            prefix_cache: PrefixCacheIndex::new(),
            residency: ResidencyTable::new(),
            logical_tables: Vec::new(),
            eviction_policy: plan.eviction_policy.clone(),
            cow_policy: plan.cow_policy.clone(),
        }
    }

    /// Try to allocate a physical block.
    ///
    /// Returns `Err(ArenaError::CapacityExceeded)` when the arena is full and no
    /// block can be evicted.
    pub fn try_allocate(&mut self) -> Result<PhysicalBlockId, ArenaError> {
        // Recycle from free list first
        if let Some(idx) = self.free_list.pop() {
            let id = PhysicalBlockId(idx as u32);
            self.blocks[idx] = PhysicalBlock::new(id, self.block_size, self.backend);
            return Ok(id);
        }

        // Try eviction when at capacity
        if self.blocks.len() >= self.capacity {
            if self.evict_one().is_some() {
                // evict_one pushed the slot to free_list; pop and reinit
                let idx = self.free_list.pop().unwrap();
                let id = PhysicalBlockId(idx as u32);
                self.blocks[idx] = PhysicalBlock::new(id, self.block_size, self.backend);
                return Ok(id);
            }
            return Err(ArenaError::CapacityExceeded {
                requested: 1,
                available: 0,
                total: self.capacity,
            });
        }

        // Allocate a fresh block
        let id = PhysicalBlockId(self.blocks.len() as u32);
        self.blocks
            .push(PhysicalBlock::new(id, self.block_size, self.backend));
        Ok(id)
    }

    /// Release a physical block, decrementing its refcount.
    /// When the refcount reaches zero the block is recycled.
    pub fn release(&mut self, id: PhysicalBlockId) {
        let idx = id.0 as usize;
        if idx >= self.blocks.len() {
            return;
        }
        self.blocks[idx].dec_ref();
        if self.blocks[idx].is_completely_free() {
            self.free_list.push(idx);
        }
    }

    /// Allocate a block, checking the prefix cache first.
    /// If a cached block with matching content hash exists, its refcount
    /// is incremented and it is returned — no new allocation is made.
    pub fn allocate_prefixed(&mut self, hash: &PrefixHash) -> PhysicalBlockId {
        if let Some(cached_id) = self.prefix_cache.lookup(hash) {
            // Found a cached block — inc refcount so it stays live
            if let Some(block) = self.blocks.iter_mut().find(|b| b.id.0 == cached_id) {
                block.inc_ref();
            }
            return PhysicalBlockId(cached_id);
        }
        // Cache miss — allocate new block and register in prefix cache
        let id = self.try_allocate().unwrap_or_else(|_| {
            // Fallback: evict and retry
            self.evict_one();
            self.try_allocate()
                .expect("KV arena: out of blocks even with eviction")
        });
        self.prefix_cache.insert(*hash, id.0);
        id
    }

    /// Admit a new sequence, allocating enough physical blocks to hold
    /// `token_count` tokens. Returns an `AdmissionReceipt` on success,
    /// or an `ArenaError` if insufficient capacity exists after eviction.
    pub fn admit_sequence(&mut self, token_count: u32) -> Result<AdmissionReceipt, ArenaError> {
        let blocks_needed = tokens_to_blocks(token_count as usize, self.block_size);
        let mut allocated = Vec::with_capacity(blocks_needed);

        for _ in 0..blocks_needed {
            match self.try_allocate() {
                Ok(id) => allocated.push(id),
                Err(e) => return Err(e),
            }
        }

        let seq = SequenceId::new();
        let mut table = LogicalBlockTable::new(seq, self.capacity);
        for &pid in &allocated {
            table.append(pid);
        }
        table.token_count = token_count as u64;
        self.logical_tables.push(table);

        // Rough byte estimate: each token stores 2 fp32 values (key + value)
        // per layer — at minimum one block worth of token storage.
        let bytes_per_block = (self.block_size * std::mem::size_of::<f32>() * 2) as u64;
        let bytes_allocated = allocated.len() as u64 * bytes_per_block;

        Ok(AdmissionReceipt {
            sequence_id: seq,
            allocated_blocks: allocated,
            total_tokens: token_count as u64,
            total_blocks: blocks_needed,
            bytes_allocated,
            prefix_hits: 0,
            new_blocks: blocks_needed,
        })
    }

    /// Evict the least recently used block with refcount == 0.
    ///
    /// Removes the block's entry from the prefix cache and pushes its
    /// slot index onto the free list so the next `try_allocate` can
    /// recycle it.
    ///
    /// Returns `Some(PhysicalBlockId)` of the evicted block, or `None`
    /// if no block is eligible for eviction.
    pub fn evict_one(&mut self) -> Option<PhysicalBlockId> {
        // (Blocks in the free list are already recycled and don't need eviction.)
        let eligible = || {
            self.blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.is_completely_free())
                .filter(|(i, _)| !self.free_list.contains(i))
        };

        let idx = match self.eviction_policy.as_str() {
            "fifo" => eligible().min_by_key(|(i, _)| *i).map(|(i, _)| i)?,
            "lru_refcount" => eligible()
                .min_by_key(|(_, b)| {
                    let rc = b.refcount.load(std::sync::atomic::Ordering::Acquire);
                    let ts = b.last_access_ns.load(std::sync::atomic::Ordering::Acquire);
                    (rc, ts)
                })
                .map(|(i, _)| i)?,
            // "lru" is the default
            _ => eligible()
                .min_by_key(|(_, b)| b.last_access_ns.load(std::sync::atomic::Ordering::Acquire))
                .map(|(i, _)| i)?,
        };

        let block = &self.blocks[idx];
        self.prefix_cache.remove(&PrefixHash(block.id.0 as u64));
        self.free_list.push(idx);
        Some(block.id)
    }

    /// Number of blocks available for allocation without exceeding capacity.
    pub fn available_blocks(&self) -> usize {
        self.capacity.saturating_sub(self.blocks.len()) + self.free_list.len()
    }

    /// Look up a logical block table by sequence id.
    pub fn get_logical_table(&self, seq: SequenceId) -> Option<&LogicalBlockTable> {
        self.logical_tables.iter().find(|t| t.sequence_id == seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_arena::block::DEFAULT_BLOCK_SIZE;

    #[test]
    fn test_admit_sequence() {
        let mut arena = KvBlockArena::new(DEFAULT_BLOCK_SIZE, 64, BackendAffinity::HostPinned);
        let receipt = arena.admit_sequence(128).expect("admission should succeed");

        assert_eq!(receipt.total_tokens, 128);
        assert!(receipt.total_blocks > 0);
        assert_eq!(receipt.allocated_blocks.len(), receipt.total_blocks);

        // Every allocated block must carry a valid id
        for &pid in &receipt.allocated_blocks {
            assert_ne!(pid, PhysicalBlockId::INVALID);
        }

        // The logical block table was stored
        let table = arena
            .get_logical_table(receipt.sequence_id)
            .expect("logical table should exist after admit");
        assert_eq!(table.logical_to_physical.len(), receipt.total_blocks);
        assert_eq!(table.token_count, 128);
        assert_eq!(table.sequence_id, receipt.sequence_id);
    }

    #[test]
    fn test_capacity_exceeded() {
        let mut arena = KvBlockArena::new(DEFAULT_BLOCK_SIZE, 1, BackendAffinity::HostPinned);

        // Fill the only slot
        let _ = arena.try_allocate().expect("first alloc should succeed");

        // Second alloc must fail — no evictable blocks (refcount == 1)
        let err = arena.try_allocate().unwrap_err();
        assert!(
            matches!(err, ArenaError::CapacityExceeded { .. }),
            "expected CapacityExceeded, got {err:?}"
        );
    }

    #[test]
    fn test_prefix_cache_hit() {
        let mut arena = KvBlockArena::new(DEFAULT_BLOCK_SIZE, 64, BackendAffinity::HostPinned);
        let hash = PrefixHash(42);

        // First call: cache miss → new block allocated and registered
        let id1 = arena.allocate_prefixed(&hash);
        assert_ne!(id1, PhysicalBlockId::INVALID);

        // Second call with same hash: cache hit → same block returned
        let id2 = arena.allocate_prefixed(&hash);
        assert_eq!(id1, id2, "prefix cache should reuse the same block id");
    }

    #[test]
    fn test_eviction() {
        let mut arena = KvBlockArena::new(DEFAULT_BLOCK_SIZE, 2, BackendAffinity::HostPinned);

        // Fill both slots
        let b0 = arena.try_allocate().expect("alloc 0");
        let _b1 = arena.try_allocate().expect("alloc 1");

        // Release b0 to make it evictable (refcount -> 0, pushed to free_list)
        arena.release(b0);

        // try_allocate pops from free_list — b0 is recycled
        let recycled = arena.try_allocate().expect("recycle via free list");
        assert_eq!(recycled, b0);

        // All blocks have refcount == 1 and no block is in the free list,
        // so evict_one returns None
        assert!(arena.evict_one().is_none());

        // Release one block, then pop it from free_list so it's no longer
        // in the free list but still has refcount == 0.
        arena.release(b0);
        arena.free_list.retain(|&i| i != b0.0 as usize);

        // Now evict_one should find it (refcount == 0, not in free_list)
        let evicted = arena.evict_one();
        assert!(evicted.is_some(), "should evict a refcount-zero block");
        assert_eq!(evicted.unwrap(), b0);
    }

    #[test]
    fn test_logical_block_table() {
        let seq = SequenceId::new();
        let mut table = LogicalBlockTable::new(seq, 10);

        let p0 = PhysicalBlockId(0);
        let p1 = PhysicalBlockId(1);
        let p2 = PhysicalBlockId(2);

        table.append(p0);
        table.append(p1);
        table.append(p2);

        // Forward translations
        assert_eq!(table.translate(0), Some(p0));
        assert_eq!(table.translate(1), Some(p1));
        assert_eq!(table.translate(2), Some(p2));

        // Out-of-bounds
        assert_eq!(table.translate(3), None);

        // Last physical
        assert_eq!(table.last_physical(), Some(p2));

        // Empty table
        let empty = LogicalBlockTable::new(SequenceId::new(), 10);
        assert_eq!(empty.translate(0), None);
        assert_eq!(empty.last_physical(), None);
    }
}
