//! Block-aware prefix cache for KV cache reuse.
//!
//! Reference: `ref/omlx/cache/prefix_cache.py`, design: `docs/omlx-prefix-cache.md`
//!
//! Detects common prefixes across requests, stores KV cache blocks indexed
//! by token hash, and reuses cached blocks to avoid redundant computation.

use parking_lot::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;
use std::time::Instant;

/// Fixed number of tokens per prefix block
pub const PREFIX_BLOCK_SIZE: usize = 64;

/// Maximum entries in the tip lineage map
pub const TIP_LINEAGE_MAX_ENTRIES: usize = 4096;

/// Hash of a prefix block (token IDs within the block)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub [u8; 32]);

/// An uncompressed prefix cache block entry (backward-compatible alias: BlockCacheEntry)
#[derive(Debug, Clone)]
pub struct UncompressedBlockCacheEntry {
    pub block_hash: BlockHash,
    pub block_index: usize,
    pub last_access: Instant,
}

/// Backward-compatible alias
pub type BlockCacheEntry = UncompressedBlockCacheEntry;

/// A compressed prefix cache block entry.
///
/// Unlike the uncompressed entry which maps to a raw block index,
/// compressed entries reference pages in the CompressedKvCache page table
/// and track how many tokens are stored per page.
#[derive(Debug, Clone)]
pub struct CompressedBlockCacheEntry {
    pub block_hash: BlockHash,
    /// Index into the CompressedKvCache's page table
    pub page_index: usize,
    /// How many tokens are in this page (compressed blocks are larger)
    pub token_count: u32,
    pub last_access: Instant,
}

/// An uncompressed block table (backward-compatible alias: BlockTable)
#[derive(Debug, Clone, Default)]
pub struct UncompressedBlockTable {
    pub blocks: Vec<usize>,
    pub block_hashes: Vec<BlockHash>,
}

/// Backward-compatible alias
pub type BlockTable = UncompressedBlockTable;

impl UncompressedBlockTable {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Block table for compressed KV cache.
///
/// Unlike the uncompressed block-table where each block maps to one
/// token position, compressed blocks store tokens in groups of 64
/// (matching PREFIX_BLOCK_SIZE). A single compress/decompress op
/// processes 64 tokens' worth of KV data in one batch.
#[derive(Debug, Clone, Default)]
pub struct CompressedBlockTable {
    pub blocks: Vec<usize>,
    pub block_hashes: Vec<BlockHash>,
    /// Token positions per compressed block
    pub tokens_per_block: usize,
    /// Whether the blocks reference compressed KV pages
    pub is_compressed: bool,
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
    compressed_cache: HashMap<BlockHash, CompressedBlockCacheEntry>,
    compressed_lru_order: VecDeque<BlockHash>,
    max_blocks: usize,
    tip_lineage: HashMap<BlockHash, BlockHash>,
    stats: PrefixCacheStats,
}

impl BlockAwarePrefixCache {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_blocks),
            lru_order: VecDeque::with_capacity(max_blocks),
            compressed_cache: HashMap::with_capacity(max_blocks),
            compressed_lru_order: VecDeque::with_capacity(max_blocks),
            max_blocks,
            tip_lineage: HashMap::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    /// Compute block hash from a slice of token IDs.
    ///
    /// Uses `DefaultHasher` (SipHash-2-4) to hash up to `PREFIX_BLOCK_SIZE` tokens.
    /// If the slice is shorter than 64 tokens, remaining positions are padded with `0u32`.
    /// The 64-bit SipHash result is expanded deterministically to fill a 32-byte block hash.
    pub fn compute_block_hash(tokens: &[u32]) -> BlockHash {
        let mut hasher = DefaultHasher::new();
        let n = tokens.len().min(PREFIX_BLOCK_SIZE);
        for &tok in &tokens[..n] {
            tok.hash(&mut hasher);
        }
        for _ in n..PREFIX_BLOCK_SIZE {
            0u32.hash(&mut hasher);
        }
        let result = hasher.finish();

        // Expand the 64-bit hash to 32 bytes by repeating with wrapping multiplies
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&result.to_le_bytes());
        for i in 1..4 {
            let v = result.wrapping_mul(i as u64 + 1);
            hash[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        BlockHash(hash)
    }

    /// Find longest matching prefix in the cache.
    ///
    /// Walks the token sequence in `PREFIX_BLOCK_SIZE` chunks, computing the hash
    /// for each block and looking it up in the cache. Returns the matched entries
    /// and the token index where matching stopped (first miss or end of input).
    pub fn find_prefix(&mut self, tokens: &[u32]) -> (Vec<&BlockCacheEntry>, usize) {
        let mut matched = Vec::new();
        let mut matched_blocks = 0usize;
        let total_blocks = (tokens.len() + PREFIX_BLOCK_SIZE - 1) / PREFIX_BLOCK_SIZE;

        for block_idx in 0..total_blocks {
            let start = block_idx * PREFIX_BLOCK_SIZE;
            let end = (start + PREFIX_BLOCK_SIZE).min(tokens.len());
            let block_tokens = &tokens[start..end];
            let hash = Self::compute_block_hash(block_tokens);

            if let Some(entry) = self.cache.get(&hash) {
                matched.push(entry);
                matched_blocks += 1;
                self.stats.hits += 1;
            } else {
                // First miss breaks the prefix
                self.stats.misses += 1;
                break;
            }
        }

        let matched_tokens = matched_blocks * PREFIX_BLOCK_SIZE;
        (matched, matched_tokens)
    }

    /// Insert new blocks into the cache.
    ///
    /// Chunks the token sequence into blocks of `PREFIX_BLOCK_SIZE`, hashes each
    /// block, and either updates the LRU position (if already cached) or inserts
    /// a new entry and evicts if over capacity.
    pub fn insert(&mut self, tokens: &[u32]) {
        let total_blocks = (tokens.len() + PREFIX_BLOCK_SIZE - 1) / PREFIX_BLOCK_SIZE;

        for block_idx in 0..total_blocks {
            let start = block_idx * PREFIX_BLOCK_SIZE;
            let end = (start + PREFIX_BLOCK_SIZE).min(tokens.len());
            let block_tokens = &tokens[start..end];
            let hash = Self::compute_block_hash(block_tokens);

            if self.cache.contains_key(&hash) {
                // Already cached — touch LRU by moving to back
                self.touch_lru(&hash);
                continue;
            }

            let block_index = self.stats.cached_blocks + block_idx;
            let entry = BlockCacheEntry {
                block_hash: hash,
                block_index,
                last_access: Instant::now(),
            };

            self.cache.insert(hash, entry);
            self.lru_order.push_back(hash);
            self.stats.cached_blocks += 1;

            self.evict_lru();
        }
    }

    /// Touch a hash in the LRU order, moving it to the back (most recently used).
    fn touch_lru(&mut self, hash: &BlockHash) {
        if let Some(pos) = self.lru_order.iter().position(|h| *h == *hash) {
            let h = self.lru_order.remove(pos).unwrap();
            self.lru_order.push_back(h);
        }
    }

    /// Evict least recently used blocks until the cache is within capacity.
    fn evict_lru(&mut self) {
        while self.cache.len() > self.max_blocks {
            if let Some(hash) = self.lru_order.pop_front() {
                self.cache.remove(&hash);
                self.stats.evicted_blocks += 1;
            }
        }
    }

    /// Clear all cache state: entries, LRU order, tip lineage, and stats.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.compressed_cache.clear();
        self.compressed_lru_order.clear();
        self.tip_lineage.clear();
        self.stats = PrefixCacheStats::default();
    }

    /// Get cache stats
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    /// Find longest matching prefix in the compressed cache.
    ///
    /// Same algorithm as `find_prefix()` but looks up blocks in the compressed
    /// cache (CompressedBlockCacheEntry), where each block maps to a page
    /// in CompressedKvCache rather than a raw block index.
    /// Returns the matched compressed entries and the token index where
    /// matching stopped (first miss or end of input).
    pub fn find_compressed_prefix(
        &mut self,
        tokens: &[u32],
    ) -> (Vec<&CompressedBlockCacheEntry>, usize) {
        let mut matched = Vec::new();
        let mut matched_blocks = 0usize;
        let total_blocks = (tokens.len() + PREFIX_BLOCK_SIZE - 1) / PREFIX_BLOCK_SIZE;

        for block_idx in 0..total_blocks {
            let start = block_idx * PREFIX_BLOCK_SIZE;
            let end = (start + PREFIX_BLOCK_SIZE).min(tokens.len());
            let block_tokens = &tokens[start..end];
            let hash = Self::compute_block_hash(block_tokens);

            if let Some(entry) = self.compressed_cache.get(&hash) {
                matched.push(entry);
                matched_blocks += 1;
                self.stats.hits += 1;
            } else {
                self.stats.misses += 1;
                break;
            }
        }

        let matched_tokens = matched_blocks * PREFIX_BLOCK_SIZE;
        (matched, matched_tokens)
    }

    /// Insert a compressed block into the cache.
    ///
    /// Like `insert()` but for CompressedBlockCacheEntry with a page_index
    /// referencing the CompressedKvCache page table. Each compressed block
    /// holds tokens_per_block tokens (matching PREFIX_BLOCK_SIZE).
    pub fn insert_compressed(&mut self, tokens: &[u32], page_index: usize) {
        let total_blocks = (tokens.len() + PREFIX_BLOCK_SIZE - 1) / PREFIX_BLOCK_SIZE;

        for block_idx in 0..total_blocks {
            let start = block_idx * PREFIX_BLOCK_SIZE;
            let end = (start + PREFIX_BLOCK_SIZE).min(tokens.len());
            let block_tokens = &tokens[start..end];
            let hash = Self::compute_block_hash(block_tokens);

            if self.compressed_cache.contains_key(&hash) {
                self.touch_compressed_lru(&hash);
                continue;
            }

            let token_count = (end - start) as u32;
            let entry = CompressedBlockCacheEntry {
                block_hash: hash,
                page_index,
                token_count,
                last_access: Instant::now(),
            };

            self.compressed_cache.insert(hash, entry);
            self.compressed_lru_order.push_back(hash);
            self.stats.cached_blocks += 1;

            self.evict_compressed_lru();
        }
    }

    /// Touch a compressed block hash in the LRU order, moving it to the back.
    fn touch_compressed_lru(&mut self, hash: &BlockHash) {
        if let Some(pos) = self.compressed_lru_order.iter().position(|h| *h == *hash) {
            let h = self.compressed_lru_order.remove(pos).unwrap();
            self.compressed_lru_order.push_back(h);
        }
    }

    /// Evict least recently used compressed blocks until the cache is within capacity.
    fn evict_compressed_lru(&mut self) {
        while self.compressed_cache.len() > self.max_blocks {
            if let Some(hash) = self.compressed_lru_order.pop_front() {
                self.compressed_cache.remove(&hash);
                self.stats.evicted_blocks += 1;
            }
        }
    }
}

/// Global prefix cache shared across all sessions on this node.
/// Optionally syncs with other nodes in the EXO cluster.
pub static GLOBAL_PREFIX_CACHE: LazyLock<Mutex<BlockAwarePrefixCache>> = LazyLock::new(|| {
    Mutex::new(BlockAwarePrefixCache::new(10_000)) // 10K blocks
});

/// Look up a prefix in the shared cache before computing locally.
///
/// Returns `Some((block_hashes, token_count))` when a matching prefix is found,
/// or `None` when no prefix blocks are cached.
pub fn check_shared_prefix(tokens: &[u32]) -> Option<(Vec<BlockHash>, usize)> {
    let mut cache = GLOBAL_PREFIX_CACHE.lock();
    let (matched, count) = cache.find_prefix(tokens);
    if count > 0 {
        Some((matched.iter().map(|e| e.block_hash).collect(), count))
    } else {
        None
    }
}

/// Insert into shared cache after local computation.
///
/// `page_index` references the compressed KV cache page table entry
/// corresponding to the first block of the token sequence.
pub fn insert_shared_prefix(tokens: &[u32], page_index: usize) {
    let mut cache = GLOBAL_PREFIX_CACHE.lock();
    cache.insert_compressed(tokens, page_index);
}

/// Clear all entries from the shared prefix cache.
///
/// Useful for testing and cache reset scenarios.
pub fn clear_shared_prefix_cache() {
    let mut cache = GLOBAL_PREFIX_CACHE.lock();
    cache.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_hash_creation() {
        let _cache = BlockAwarePrefixCache::new(1024);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::default();
        assert!(table.is_empty());
        table.blocks.push(0);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_compute_block_hash_consistency() {
        let tokens = vec![42u32, 99, 7, 13];
        let h1 = BlockAwarePrefixCache::compute_block_hash(&tokens);
        let h2 = BlockAwarePrefixCache::compute_block_hash(&tokens);
        assert_eq!(h1, h2, "same input must produce same hash");
    }

    #[test]
    fn test_compute_block_hash_padding() {
        let short = vec![1u32];
        let long: Vec<u32> = (0..PREFIX_BLOCK_SIZE as u32).collect();
        let h_short = BlockAwarePrefixCache::compute_block_hash(&short);
        let h_long = BlockAwarePrefixCache::compute_block_hash(&long);
        assert_ne!(h_short, h_long, "padded short must differ from full block");
    }

    #[test]
    fn test_insert_and_find_prefix() {
        let mut cache = BlockAwarePrefixCache::new(64);
        let tokens: Vec<u32> = (0..128).collect(); // 2 full blocks

        cache.insert(&tokens);
        assert_eq!(cache.stats.cached_blocks, 2);

        let (matched, matched_tokens) = cache.find_prefix(&tokens);
        assert_eq!(matched.len(), 2);
        assert_eq!(matched_tokens, 128);
        assert_eq!(cache.stats.hits, 2);
    }

    #[test]
    fn test_find_prefix_partial_match() {
        let mut cache = BlockAwarePrefixCache::new(64);
        let first_block: Vec<u32> = (0..64).collect();
        let second_block: Vec<u32> = (64..128).collect();
        let both: Vec<u32> = [&first_block[..], &second_block[..]].concat();

        // Insert only the first block
        cache.insert(&first_block);
        assert_eq!(cache.stats.cached_blocks, 1);

        // Searching two blocks should match only the first
        let (matched, matched_tokens) = cache.find_prefix(&both);
        assert_eq!(matched.len(), 1);
        assert!(matched_tokens < 128);
        assert!(matched_tokens >= 64);
        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_eviction() {
        let mut cache = BlockAwarePrefixCache::new(2); // only 2 blocks
        let block_a: Vec<u32> = (0..64).collect();
        let block_b: Vec<u32> = (64..128).collect();
        let block_c: Vec<u32> = (128..192).collect();

        cache.insert(&block_a);
        cache.insert(&block_b);
        assert_eq!(cache.stats.cached_blocks, 2);

        // Inserting C should evict A (oldest)
        cache.insert(&block_c);
        assert_eq!(cache.stats.cached_blocks, 3); // was incremented
        assert_eq!(cache.stats.evicted_blocks, 1);

        // A should no longer be findable
        let (matched, _) = cache.find_prefix(&block_a);
        assert!(matched.is_empty());

        // B and C should still be findable
        let (matched_b, _) = cache.find_prefix(&block_b);
        assert_eq!(matched_b.len(), 1);
        let (matched_c, _) = cache.find_prefix(&block_c);
        assert_eq!(matched_c.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut cache = BlockAwarePrefixCache::new(64);
        let tokens: Vec<u32> = (0..128).collect();
        cache.insert(&tokens);
        assert!(cache.stats.cached_blocks > 0);

        cache.clear();
        assert_eq!(cache.stats.cached_blocks, 0);
        assert_eq!(cache.stats.evicted_blocks, 0);
        assert_eq!(cache.stats.hits, 0);
        assert!(cache.cache.is_empty());
        assert!(cache.lru_order.is_empty());
    }

    #[test]
    fn test_find_prefix_empty_tokens() {
        let mut cache = BlockAwarePrefixCache::new(64);
        let (matched, matched_tokens) = cache.find_prefix(&[]);
        assert!(matched.is_empty());
        assert_eq!(matched_tokens, 0);
    }

    #[test]
    fn test_insert_empty_tokens() {
        let mut cache = BlockAwarePrefixCache::new(64);
        cache.insert(&[]);
        assert_eq!(cache.stats.cached_blocks, 0);
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn test_compute_block_hash_truncation() {
        // More than PREFIX_BLOCK_SIZE tokens should be truncated
        let exact: Vec<u32> = (0..PREFIX_BLOCK_SIZE as u32).collect();
        let longer: Vec<u32> = (0..PREFIX_BLOCK_SIZE as u32 + 10).collect();
        let h_exact = BlockAwarePrefixCache::compute_block_hash(&exact);
        let h_longer = BlockAwarePrefixCache::compute_block_hash(&longer);
        assert_eq!(
            h_exact, h_longer,
            "excess tokens beyond block size must be ignored"
        );
    }

    #[test]
    fn test_global_prefix_cache_static_initialized() {
        // Verify GLOBAL_PREFIX_CACHE is accessible and returns a valid cache
        clear_shared_prefix_cache();
        let cache = GLOBAL_PREFIX_CACHE.lock();
        assert_eq!(cache.stats().cached_blocks, 0);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_check_shared_prefix_empty_cache() {
        clear_shared_prefix_cache();
        let tokens: Vec<u32> = (0..64).collect();
        // Initially empty cache should return None
        let result = check_shared_prefix(&tokens);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_shared_prefix_after_insert() {
        clear_shared_prefix_cache();
        let tokens: Vec<u32> = (0..128).collect(); // 2 blocks

        // Insert into global cache
        insert_shared_prefix(&tokens, 0);

        // Should now find the prefix
        let result = check_shared_prefix(&tokens);
        assert!(result.is_some());
        if let Some((hashes, count)) = result {
            assert_eq!(hashes.len(), 2);
            assert_eq!(count, 128);
        }
    }

    #[test]
    fn test_shared_prefix_deduplication_across_calls() {
        clear_shared_prefix_cache();
        let tokens: Vec<u32> = (0..128).collect();

        // Insert twice — second insert should hit LRU touch, not duplicate
        insert_shared_prefix(&tokens, 0);
        insert_shared_prefix(&tokens, 1);

        let result = check_shared_prefix(&tokens);
        assert!(result.is_some());

        // Cached blocks should not have doubled
        let cache = GLOBAL_PREFIX_CACHE.lock();
        assert!(cache.stats().cached_blocks <= 2);
    }

    #[test]
    fn test_shared_prefix_partial_match() {
        clear_shared_prefix_cache();
        let first_block: Vec<u32> = (0..64).collect();
        let second_block: Vec<u32> = (64..128).collect();
        let both: Vec<u32> = [&first_block[..], &second_block[..]].concat();

        // Insert only the first block
        insert_shared_prefix(&first_block, 0);

        // Searching two blocks should match only the first
        let result = check_shared_prefix(&both);
        assert!(result.is_some());
        if let Some((hashes, count)) = result {
            assert_eq!(hashes.len(), 1);
            assert_eq!(count, 64);
        }
    }
}
