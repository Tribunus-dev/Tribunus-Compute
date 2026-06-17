# Paged SSD Cache — Port Plan

Source: `omlx/omlx/cache/paged_ssd_cache.py` (3.5K lines)
Depends on: `paged_cache.py`, safetensors serialization
Status: Reference copied to `ref/omlx/cache/paged_ssd_cache.py`

## What it is

SSD-backed storage for paged KV cache blocks, enabling larger effective cache
sizes than available RAM. Key features:

1. Block-level safetensors serialization (compatible with mlx-lm format)
2. Hash-based subdirectory structure for O(1) lookup
3. LRU-based SSD cache size management with configurable eviction
4. Startup scan to reuse cache files from previous runs
5. Thread-safe concurrent IO via ThreadPoolExecutor

## Architecture

```
PagedSSDCacheManager
├── Cache directory: ~/Library/Caches/omlx/paged_ssd_cache/
│   └── xx/xx/<block_hash>.safetensors  (hash-based 2-level subdirs)
├── LRU eviction queue (ordered by last_access)
├── ThreadPoolExecutor (serialization/deserialization)
└── Stats: hits, misses, ssd_reads, ssd_writes, evictions
```

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/cache/paged_ssd_cache.rs`

```rust
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DEFAULT_SSD_CACHE_DIR: &str = "paged_ssd_cache";
const HASH_SUBDIR_DEPTH: usize = 2;
const HASH_CHARS_PER_LEVEL: usize = 2;

/// Configuration for the SSD cache
pub struct SsdCacheConfig {
    pub cache_dir: PathBuf,
    pub max_ssd_size_bytes: u64,
    pub max_block_size: usize,
    pub startup_scan: bool,
    pub io_threads: usize,
    /// Target fill ratio for LRU eviction (e.g. 0.9 = evict when 90% full)
    pub eviction_target_ratio: f64,
}

/// SSD-backed paged cache manager
pub struct PagedSSDCacheManager {
    config: SsdCacheConfig,
    /// In-memory index: block_hash -> BlockMetadata
    index: HashMap<String, BlockMetadata>,
    /// LRU order for eviction
    lru: VecDeque<String>,
    /// Current SSD usage in bytes
    current_size_bytes: u64,
    stats: SsdCacheStats,
}

/// Metadata for a cached block on SSD
pub struct BlockMetadata {
    pub block_hash: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub last_access: Instant,
    pub num_tokens: usize,
}

impl PagedSSDCacheManager {
    /// Compute the SSD path for a block hash
    fn block_path(&self, hash: &str) -> PathBuf {
        // e.g., hash "abcdef..." -> "ab/cd/abcdef...safetensors"
    }

    /// Store a block to SSD
    pub fn store_block(
        &mut self,
        hash: &str,
        kv_data: &[u8],
    ) -> Result<(), SsdCacheError> {
        // 1. Serialize to safetensors format
        // 2. Write to hash-based path
        // 3. Update LRU + index
        // 4. Evict if over limit
    }

    /// Load a block from SSD
    pub fn load_block(
        &self,
        hash: &str,
    ) -> Result<Vec<u8>, SsdCacheError> {
        // 1. Look up in index
        // 2. Read from SSD
        // 3. Update last_access
    }

    /// Evict blocks until under eviction_target_ratio
    fn evict_lru(&mut self) -> usize {
        // 1. Walk LRU from tail
        // 2. Delete files
        // 3. Remove from index
    }

    /// Scan existing cache on startup
    fn startup_scan(&mut self) -> Result<(), SsdCacheError> {
        // 1. Walk cache directory
        // 2. Parse safetensors metadata
        // 3. Build index
    }
}

/// Thread-safe wrapper for concurrent access
pub struct ConcurrentSsdCache {
    inner: Arc<Mutex<PagedSSDCacheManager>>,
}

/// Stats
pub struct SsdCacheStats {
    pub ssd_reads: AtomicU64,
    pub ssd_writes: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub evictions: AtomicU64,
    pub current_ssd_bytes: AtomicU64,
}
```

### Integration

- Used by `BlockAwarePrefixCache` as backing store
- Configurable via `SsdCacheConfig` in session config
- Complements in-memory `PagedCache` for multi-tier caching
