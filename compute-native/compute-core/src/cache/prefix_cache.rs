//! Block-aware prefix cache for KV cache reuse.
//!
//! Reference: `ref/omlx/cache/prefix_cache.py`, design: `docs/omlx-prefix-cache.md`
//!
//! Detects common prefixes across requests, stores KV cache blocks indexed
//! by token hash, and reuses cached blocks to avoid redundant computation.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

/// Fixed number of tokens per prefix block
pub const PREFIX_BLOCK_SIZE: usize = 64;

/// Maximum entries in the tip lineage map
pub const TIP_LINEAGE_MAX_ENTRIES: usize = 4096;

/// Hash of a prefix block (token IDs within the block)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

/// A prefix cache block entry
#[derive(Debug, Clone)]
pub struct BlockCacheEntry {
    pub block_hash: BlockHash,
    pub block_index: usize,
    pub last_access: Instant,
}

/// Block table for a sequence (ordered list of block indices)
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    pub blocks: Vec<usize>,
    pub block_hashes: Vec<BlockHash>,
}

impl BlockTable {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Stats for prefix cache performance
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub cached_blocks: usize,
    pub evicted_blocks: usize,
    pub avg_prefix_length: f64,
}

/// Block-aware prefix cache manager
///
/// Implements the prefix cache from ref/omlx/cache/prefix_cache.py:
/// - Block hashing for prefix matching
/// - LRU eviction
/// Tip lineage tracking for conversation chains
#[allow(dead_code)]
pub struct BlockAwarePrefixCache {
    cache: HashMap<BlockHash, BlockCacheEntry>,
    lru_order: VecDeque<BlockHash>,
    max_blocks: usize,
    tip_lineage: HashMap<BlockHash, BlockHash>,
    stats: PrefixCacheStats,
}

impl BlockAwarePrefixCache {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_blocks),
            lru_order: VecDeque::with_capacity(max_blocks),
            max_blocks,
            tip_lineage: HashMap::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    /// Compute block hash from a slice of token IDs
    pub fn compute_block_hash(_tokens: &[u32]) -> BlockHash {
        // TODO: implement block hashing (SipHash or SHA-256)
        todo!("block hash computation not yet implemented")
    }

    /// Find longest matching prefix in the cache
    ///
    /// Returns (matched block indices, remaining token slice start)
    pub fn find_prefix(&self, _tokens: &[u32]) -> (Vec<&BlockCacheEntry>, usize) {
        // TODO: walk the prefix tree to find longest match
        todo!("prefix lookup not yet implemented")
    }

    /// Insert new blocks into the cache
    pub fn insert(&mut self, _tokens: &[u32]) {
        // TODO: chunk tokens, hash, store in cache, update LRU
    }

    /// Evict least recently used blocks
    #[allow(dead_code)]
    fn evict_lru(&mut self) {
        while self.cache.len() > self.max_blocks {
            if let Some(hash) = self.lru_order.pop_front() {
                self.cache.remove(&hash);
                self.stats.evicted_blocks += 1;
            }
        }
    }

    /// Get cache stats
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_hash_creation() {
        // Just tests the type compiles and default behavior
        let _cache = BlockAwarePrefixCache::new(1024);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::default();
        assert!(table.is_empty());
        table.blocks.push(0);
        assert_eq!(table.len(), 1);
    }
}
