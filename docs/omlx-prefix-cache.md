# Block-Aware Prefix Cache — Port Plan

Source: `omlx/omlx/cache/prefix_cache.py` (3K lines)
Depends on: `paged_cache.py`, `paged_ssd_cache.py`, `hybrid_cache.py`
Status: Reference copied to `ref/omlx/cache/prefix_cache.py`

## What it is

An automatic prefix caching system for LLM KV cache that:
1. Detects common prefixes across requests (same system prompt, chat history prefix)
2. Stores KV cache blocks indexed by token hash
3. Reuses cached blocks across requests to avoid redundant computation
4. Evicts stale blocks via LRU
5. Supports SSD persistence for long-term reuse

## Architecture

```
Request tokens → [hash blocks] → hash → BlockCacheEntry
                                      ↕
                              BlockTable (list of block indices)
                                      ↕
                              PagedCacheManager (in-memory)
                                      ↕
                              PagedSSDCacheManager (SSD storage)
```

## Key concepts

- **Block**: Fixed-size chunk of KV cache (e.g., 64 tokens)
- **Block hash**: Hash of token IDs in the block (used for matching)
- **Block table**: Ordered list of block indices for a sequence
- **Prefix tree**: Hash trie for efficient longest-prefix matching
- **Tip lineage**: Chain of latest block entries per conversation

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/cache/prefix_cache.rs`

```rust
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

const PREFIX_BLOCK_SIZE: usize = 64;
const TIP_LINEAGE_MAX_ENTRIES: usize = 4096;

/// Hash of a prefix block (token IDs within the block)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

/// A prefix cache block entry
pub struct BlockCacheEntry {
    pub block_hash: BlockHash,
    pub block_index: usize,
    pub last_access: std::time::Instant,
}

/// Block table for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub blocks: Vec<usize>,       // indices into the cache
    pub block_hashes: Vec<BlockHash>,
}

/// Block-aware prefix cache manager
pub struct BlockAwarePrefixCache {
    cache: HashMap<BlockHash, BlockCacheEntry>,
    lru_order: VecDeque<BlockHash>,
    max_blocks: usize,
    tip_lineage: HashMap<BlockHash, BlockHash>,
    stats: PrefixCacheStats,
}

impl BlockAwarePrefixCache {
    /// Compute block hash from a slice of token IDs
    pub fn compute_block_hash(tokens: &[u32]) -> BlockHash { /* ... */ }

    /// Find longest matching prefix in the cache
    pub fn find_prefix(&self, tokens: &[u32]) -> (Vec<usize>, BlockTable) {
        // 1. Chunk tokens into blocks
        // 2. Hash each block
        // 3. Walk the prefix tree to find longest match
        // 4. Return matched blocks + unmatched suffix
    }

    /// Insert new blocks into the cache
    pub fn insert(&mut self, tokens: &[u32], block_table: BlockTable) { /* ... */ }

    /// Evict least recently used blocks
    fn evict_lru(&mut self) { /* ... */ }
}

/// Stats for prefix cache performance
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub cached_blocks: usize,
    pub evicted_blocks: usize,
    pub avg_prefix_length: f64,
}
```

### Integration

- Cache lifecycle managed by `compute-core/src/session.rs`
- Queried during prefill to skip already-computed prefix
- Updated after generation to cache new blocks
- Compatible with existing `KVCache` in `mlx-rs-core`
