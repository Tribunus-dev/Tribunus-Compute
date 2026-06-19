//! Paged KV cache arena: physical blocks, COW refcounting, prefix caching,
//! and backend residency mapping.

pub mod refcount;
pub mod backend;
pub mod block;
pub mod prefix;

use block::{BackendAffinity, PhysicalBlock, PhysicalBlockId};
use backend::ResidencyTable;
use prefix::{PrefixCacheIndex, PrefixHash};

/// Arena of physical KV cache blocks, with prefix caching and backend residency tracking.
pub struct KvBlockArena {
    pub blocks: Vec<PhysicalBlock>,
    pub free_list: Vec<usize>,
    pub block_size: usize,
    pub capacity: usize,
    pub backend: BackendAffinity,
    pub prefix_cache: PrefixCacheIndex,
    pub residency: ResidencyTable,
}

impl KvBlockArena {
    /// Create a new arena with the given block size, capacity, and backend affinity.
    pub fn new(block_size: usize, capacity: usize, backend: BackendAffinity) -> Self {
        Self {
            blocks: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            block_size,
            capacity,
            backend,
            prefix_cache: PrefixCacheIndex::new(),
            residency: ResidencyTable::new(),
        }
    }

    /// Create a new arena with prefix caching enabled (same as `new` — caching is always on).
    pub fn new_with_cache(block_size: usize, capacity: usize, backend: BackendAffinity) -> Self {
        Self::new(block_size, capacity, backend)
    }

    /// Allocate a new physical block, recycling from the free list if available.
    pub fn allocate(&mut self) -> PhysicalBlockId {
        if let Some(idx) = self.free_list.pop() {
            let id = PhysicalBlockId(idx as u32);
            // Re-initialize the block at this recycled index
            self.blocks[idx] = PhysicalBlock::new(id, self.block_size, self.backend);
            id
        } else {
            let id = PhysicalBlockId(self.blocks.len() as u32);
            self.blocks.push(PhysicalBlock::new(id, self.block_size, self.backend));
            id
        }
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
        let id = self.allocate();
        self.prefix_cache.insert(*hash, id.0);
        id
    }
}
