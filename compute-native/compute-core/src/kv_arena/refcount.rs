//! Copy-on-write refcounting and LRU eviction for KV cache blocks.

use crate::kv_arena::block::PhysicalBlock;
use std::collections::VecDeque;

/// Eviction policy selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used (default)
    Lru,
    /// First In First Out
    Fifo,
}

/// COW (Copy-on-Write) refcount manager.
pub struct CowManager {
    policy: EvictionPolicy,
}

impl CowManager {
    pub fn new(policy: EvictionPolicy) -> Self {
        CowManager { policy }
    }

    /// Check if a block is shareable (refcount > 1 from different requests).
    pub fn is_shareable(refcount: u64) -> bool {
        refcount > 1
    }

    /// When a request needs to write to a shared block, trigger COW:
    /// decrement the shared block's refcount and return true indicating
    /// the caller must allocate a fresh block for writing.
    pub fn prepare_write(block: &PhysicalBlock) -> bool {
        if block.refcount.load(std::sync::atomic::Ordering::Acquire) > 1 {
            block.dec_ref(); // release our shared reference
            true // caller must COW
        } else {
            false // exclusive ownership, safe to write
        }
    }
}

/// LRU tracker for eviction decisions.
pub struct LruTracker {
    queue: VecDeque<u32>, // physical block IDs in LRU order
    capacity: usize,
}

impl LruTracker {
    pub fn new(capacity: usize) -> Self {
        LruTracker {
            queue: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn touch(&mut self, block_id: u32) {
        // Move to back (most recently used)
        if let Some(pos) = self.queue.iter().position(|&id| id == block_id) {
            self.queue.remove(pos);
        }
        self.queue.push_back(block_id);
        // Trim if over capacity
        while self.queue.len() > self.capacity {
            self.queue.pop_front();
        }
    }

    /// Get the least recently used block ID for eviction.
    pub fn evict_candidate(&self) -> Option<u32> {
        self.queue.front().copied()
    }

    pub fn remove(&mut self, block_id: u32) {
        if let Some(pos) = self.queue.iter().position(|&id| id == block_id) {
            self.queue.remove(pos);
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
}
