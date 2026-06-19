//! Per-backend block residency mapping.
//! Each backend (MLX, Candle, Tensix) can store KV blocks in its own
//! memory domain. The arena tracks which backend owns which block.

use crate::backend::residency::MemoryDomain;
use crate::kv_arena::block::{BackendAffinity, PhysicalBlockId};

/// Maps a physical block to its backend residency info.
#[derive(Clone, Debug)]
pub struct BlockResidency {
    pub block: PhysicalBlockId,
    pub backend: BackendAffinity,
    pub memory_domain: MemoryDomain,
    pub device_ptr: Option<u64>,
    pub byte_size: u64,
}

impl BlockResidency {
    pub fn new(block: PhysicalBlockId, backend: BackendAffinity, byte_size: u64) -> Self {
        let memory_domain = match backend {
            BackendAffinity::MlxMetal => MemoryDomain::SharedUnified,
            BackendAffinity::CandleCpu => MemoryDomain::HostPageable,
            BackendAffinity::Tensix => MemoryDomain::DeviceLocal,
            BackendAffinity::IntelLevelZero => MemoryDomain::SharedUnified,
            BackendAffinity::HostPinned => MemoryDomain::HostPinned,
            BackendAffinity::Ane => MemoryDomain::MappedExternal,
        };
        BlockResidency {
            block,
            backend,
            memory_domain,
            device_ptr: None,
            byte_size,
        }
    }
}

/// Residency table for all active KV blocks across all backends.
pub struct ResidencyTable {
    entries: Vec<BlockResidency>,
}

impl ResidencyTable {
    pub fn new() -> Self {
        ResidencyTable {
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, residency: BlockResidency) {
        self.entries.push(residency);
    }

    pub fn get(&self, block: PhysicalBlockId) -> Option<&BlockResidency> {
        self.entries.iter().find(|r| r.block == block)
    }

    pub fn get_mut(&mut self, block: PhysicalBlockId) -> Option<&mut BlockResidency> {
        self.entries.iter_mut().find(|r| r.block == block)
    }

    pub fn remove(&mut self, block: PhysicalBlockId) {
        self.entries.retain(|r| r.block != block);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn total_memory_bytes(&self) -> u64 {
        self.entries.iter().map(|r| r.byte_size).sum()
    }
}
