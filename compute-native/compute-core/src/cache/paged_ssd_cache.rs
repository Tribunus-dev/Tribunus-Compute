//! SSD-backed paged KV cache with safetensors serialization.
//!
//! Reference: `ref/omlx/cache/paged_ssd_cache.py`, design: `docs/omlx-ssd-cache.md`
//!
//! Enables larger effective cache sizes than available RAM by backing
//! paged KV cache blocks to SSD with hash-based subdirectory lookup.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

/// Default relative cache directory name
pub const DEFAULT_SSD_CACHE_DIR: &str = "paged_ssd_cache";

/// Number of subdirectory levels for hash-based block storage
pub const HASH_SUBDIR_DEPTH: usize = 2;

/// Characters per subdirectory level
pub const HASH_CHARS_PER_LEVEL: usize = 2;

/// Target fill ratio for LRU eviction (evict when 90% full)
pub const EVICTION_TARGET_RATIO: f64 = 0.9;

/// Metadata for a cached block on SSD
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    pub block_hash: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub last_access: Instant,
    pub num_tokens: usize,
}

/// Configuration for the SSD cache
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    pub cache_dir: PathBuf,
    pub max_ssd_size_bytes: u64,
    pub max_block_size: usize,
    pub startup_scan: bool,
    pub io_threads: usize,
    pub eviction_target_ratio: f64,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(DEFAULT_SSD_CACHE_DIR),
            max_ssd_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            max_block_size: 1024 * 1024,                 // 1 MB per block
            startup_scan: false,
            io_threads: 4,
            eviction_target_ratio: EVICTION_TARGET_RATIO,
        }
    }
}

/// Stats for the SSD cache
#[derive(Debug, Default)]
pub struct SsdCacheStats {
    pub ssd_reads: AtomicU64,
    pub ssd_writes: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub evictions: AtomicU64,
    pub current_ssd_bytes: AtomicU64,
}

/// SSD-backed paged cache manager
///
/// Implements the SSD cache from ref/omlx/cache/paged_ssd_cache.py:
/// - Block-level safetensors serialization
/// - Hash-based subdirectory structure
/// - LRU-based eviction
/// - Startup scan for cache reuse
#[allow(dead_code)]
pub struct PagedSSDCacheManager {
    config: SsdCacheConfig,
    index: HashMap<String, BlockMetadata>,
    lru: VecDeque<String>,
    current_size_bytes: u64,
    stats: SsdCacheStats,
}

/// Errors during SSD cache operations
#[derive(Debug, thiserror::Error)]
pub enum SsdCacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Block not found: {0}")]
    BlockNotFound(String),
    #[error("Cache directory creation failed: {0}")]
    DirCreationFailed(String),
}

impl PagedSSDCacheManager {
    pub fn new(config: SsdCacheConfig) -> Self {
        Self {
            config,
            index: HashMap::new(),
            lru: VecDeque::new(),
            current_size_bytes: 0,
            stats: SsdCacheStats::default(),
        }
    }

    /// Compute the SSD path for a block hash
    #[allow(dead_code)]
    fn block_path(&self, hash: &str) -> PathBuf {
        let mut path = self.config.cache_dir.clone();
        for level in 0..HASH_SUBDIR_DEPTH {
            let start = level * HASH_CHARS_PER_LEVEL;
            path.push(&hash[start..start + HASH_CHARS_PER_LEVEL]);
        }
        path.push(format!("{}.safetensors", hash));
        path
    }

    /// Store a block to SSD
    pub fn store_block(&mut self, _hash: &str, _kv_data: &[u8]) -> Result<(), SsdCacheError> {
        // TODO: implement per paged_ssd_cache.py reference
        // - Serialize to safetensors format
        // - Write to hash-based path
        // - Update LRU + index
        // - Evict if over limit
        Ok(())
    }

    /// Load a block from SSD
    pub fn load_block(&self, _hash: &str) -> Result<Vec<u8>, SsdCacheError> {
        // TODO: implement per paged_ssd_cache.py reference
        todo!("SSD block loading not yet implemented")
    }

    /// Evict oldest blocks until under target ratio
    #[allow(dead_code)]
    fn evict_lru(&mut self) -> usize {
        let target =
            (self.config.max_ssd_size_bytes as f64 * self.config.eviction_target_ratio) as u64;
        let mut evicted = 0;
        while self.current_size_bytes > target {
            if let Some(hash) = self.lru.pop_front() {
                if let Some(meta) = self.index.remove(&hash) {
                    let _ = std::fs::remove_file(&meta.path);
                    self.current_size_bytes -= meta.size_bytes;
                    evicted += 1;
                }
            } else {
                break;
            }
        }
        evicted
    }

    /// Scan existing cache on startup
    pub fn startup_scan(&mut self) -> Result<(), SsdCacheError> {
        // TODO: walk cache directory, parse safetensors metadata, build index
        Ok(())
    }

    /// Get cache stats
    pub fn stats(&self) -> &SsdCacheStats {
        &self.stats
    }
}
