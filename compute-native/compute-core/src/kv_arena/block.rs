//! Physical and logical KV block management.
//! Each block stores KV data for a fixed number of tokens.

use std::sync::atomic::{AtomicU64, Ordering};

/// Default number of tokens per block (vLLM uses 16; 32 is better for unified memory).
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// Unique identifier for a physical block in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub u32);

impl PhysicalBlockId {
    pub const INVALID: PhysicalBlockId = PhysicalBlockId(u32::MAX);
}

/// Logical block index within a request's block table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalBlockIdx(pub u32);

/// A physical block of KV cache memory.
pub struct PhysicalBlock {
    pub id: PhysicalBlockId,
    pub block_size: usize,         // tokens per block
    pub token_count: AtomicU64,    // how many tokens currently stored
    pub refcount: AtomicU64,       // number of active references (COW)
    pub last_access_ns: AtomicU64, // LRU timestamp
    pub backend: BackendAffinity,  // which backend owns this block's memory
}

/// Which backend's memory region this block lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendAffinity {
    MlxMetal,
    CandleCpu,
    Tensix,
    IntelLevelZero,
    HostPinned,
    Ane,
}

impl PhysicalBlock {
    pub fn new(id: PhysicalBlockId, block_size: usize, backend: BackendAffinity) -> Self {
        PhysicalBlock {
            id,
            block_size,
            token_count: AtomicU64::new(0),
            refcount: AtomicU64::new(1),
            last_access_ns: AtomicU64::new(0),
            backend,
        }
    }

    pub fn touch(&self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.last_access_ns.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    pub fn is_full(&self) -> bool {
        self.token_count.load(Ordering::Relaxed) >= self.block_size as u64
    }

    pub fn add_tokens(&self, count: u64) {
        self.token_count.fetch_add(count, Ordering::Relaxed);
    }

    pub fn dec_ref(&self) -> u64 {
        self.refcount
            .fetch_sub(1, Ordering::AcqRel)
            .saturating_sub(1)
    }

    pub fn inc_ref(&self) -> u64 {
        self.refcount.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn is_completely_free(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 0
    }
}

/// Entry in a request's block table: logical -> physical mapping.
#[derive(Clone, Debug)]
pub struct BlockTableEntry {
    pub logical: LogicalBlockIdx,
    pub physical: PhysicalBlockId,
    pub token_count: u64, // number of valid tokens in this block
    pub prefix_hash: u64, // content hash for prefix matching
    pub is_dirty: bool,   // whether this block was modified since last sync
}

/// A request's complete block table (virtual address space for KV cache).
#[derive(Clone, Debug, Default)]
pub struct BlockTable {
    entries: Vec<BlockTableEntry>,
}

impl BlockTable {
    pub fn new() -> Self {
        BlockTable {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: BlockTableEntry) {
        self.entries.push(entry);
    }

    pub fn get(&self, logical: LogicalBlockIdx) -> Option<&BlockTableEntry> {
        self.entries.iter().find(|e| e.logical == logical)
    }

    pub fn get_mut(&mut self, logical: LogicalBlockIdx) -> Option<&mut BlockTableEntry> {
        self.entries.iter_mut().find(|e| e.logical == logical)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn total_physical_blocks(&self) -> usize {
        self.entries.len()
    }

    pub fn total_tokens(&self) -> u64 {
        self.entries.iter().map(|e| e.token_count).sum()
    }

    pub fn last_entry(&self) -> Option<&BlockTableEntry> {
        self.entries.last()
    }

    pub fn entries(&self) -> &[BlockTableEntry] {
        &self.entries
    }
}

/// Convert tokens to number of blocks needed (ceiling division).
pub fn tokens_to_blocks(num_tokens: usize, block_size: usize) -> usize {
    (num_tokens + block_size - 1) / block_size
}
