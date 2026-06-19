//! Prefix hashing and content-based KV cache block dedup.
//! Ported concept from vLLM's automatic prefix caching.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A content hash for a KV block, used for prefix matching.
/// Two blocks with the same hash contain identical KV data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixHash(pub u64);

impl PrefixHash {
    /// Compute a prefix hash from token IDs and model metadata.
    /// Includes: model ID, tokenizer version, prompt tokens, layer id, head id.
    pub fn compute(
        model_id: u64,
        tokenizer_version: u64,
        tokens: &[u32],
        layer_id: u32,
        block_offset: usize,
        block_size: usize,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model_id.hash(&mut hasher);
        tokenizer_version.hash(&mut hasher);
        layer_id.hash(&mut hasher);
        block_offset.hash(&mut hasher);
        block_size.hash(&mut hasher);
        for &t in tokens.iter().take(block_size) {
            t.hash(&mut hasher);
        }
        PrefixHash(hasher.finish())
    }

    /// Compute a hash from just the token slice (for quick lookup).
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &t in tokens {
            t.hash(&mut hasher);
        }
        PrefixHash(hasher.finish())
    }
}

/// Global prefix hash index — maps content hashes to physical block IDs.
pub struct PrefixCacheIndex {
    /// hash -> physical block id
    index: HashMap<PrefixHash, u32>,
    /// hash -> hit count (for cache policy decisions)
    hits: HashMap<PrefixHash, u64>,
}

impl PrefixCacheIndex {
    pub fn new() -> Self {
        PrefixCacheIndex {
            index: HashMap::new(),
            hits: HashMap::new(),
        }
    }

    /// Look up a prefix hash. Returns the physical block id if found.
    pub fn lookup(&mut self, hash: &PrefixHash) -> Option<u32> {
        if let Some(&block_id) = self.index.get(hash) {
            *self.hits.entry(*hash).or_insert(0) += 1;
            Some(block_id)
        } else {
            None
        }
    }

    /// Register a new prefix hash -> physical block mapping.
    pub fn insert(&mut self, hash: PrefixHash, block_id: u32) {
        self.index.insert(hash, block_id);
        self.hits.entry(hash).or_insert(0);
    }

    /// Remove a prefix hash (e.g., when block is evicted).
    pub fn remove(&mut self, hash: &PrefixHash) {
        self.index.remove(hash);
        self.hits.remove(hash);
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total: u64 = self.hits.values().sum();
        let hits: u64 = self.hits.values().filter(|&&v| v > 0).count() as u64;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    pub fn clear(&mut self) {
        self.index.clear();
        self.hits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_hash_compute() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = PrefixHash::compute(0, 1, &tokens, 0, 0, 16);
        let h2 = PrefixHash::compute(0, 1, &tokens, 0, 0, 16);
        assert_eq!(h1, h2, "same inputs should produce same hash");

        let h3 = PrefixHash::compute(0, 1, &tokens, 1, 0, 16);
        assert_ne!(h1, h3, "different layer should produce different hash");
    }

    #[test]
    fn test_from_tokens() {
        let tokens = vec![100u32, 200, 300];
        let h1 = PrefixHash::from_tokens(&tokens);
        let h2 = PrefixHash::from_tokens(&tokens);
        assert_eq!(h1, h2);

        let h3 = PrefixHash::from_tokens(&[100u32, 200]);
        assert_ne!(h1, h3, "different token slice should differ");
    }

    #[test]
    fn test_prefix_cache_index() {
        let mut idx = PrefixCacheIndex::new();
        let hash = PrefixHash(42);
        assert!(idx.lookup(&hash).is_none());

        idx.insert(hash, 7);
        assert_eq!(idx.lookup(&hash), Some(7));
        assert_eq!(idx.len(), 1);
        assert!(idx.hit_rate() > 0.0);
    }

    #[test]
    fn test_remove_and_clear() {
        let mut idx = PrefixCacheIndex::new();
        idx.insert(PrefixHash(10), 1);
        idx.insert(PrefixHash(20), 2);
        assert_eq!(idx.len(), 2);

        idx.remove(&PrefixHash(10));
        assert_eq!(idx.len(), 1);
        assert!(idx.lookup(&PrefixHash(10)).is_none());

        idx.clear();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_hit_rate_zero_when_empty() {
        let idx = PrefixCacheIndex::new();
        assert_eq!(idx.hit_rate(), 0.0);
    }
}
