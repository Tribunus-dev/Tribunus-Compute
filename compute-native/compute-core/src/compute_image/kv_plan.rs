//! KvCachePlan -- compiled KV cache layout contract.
//! The ComputeImage declares the KV cache plan so the runtime can
//! instantiate KvBlockArena from it without runtime configuration decisions.

use serde::{Deserialize, Serialize};

/// Compiled KV cache layout for a ComputeImage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvCachePlan {
    /// Block size in tokens (16 for CUDA paged, 32 for Apple unified, 64 for Tensix).
    pub block_tokens: u32,
    /// Maximum number of blocks before eviction.
    pub max_blocks: u32,
    /// KV cache element dtype.
    pub kv_dtype: KVDtype,
    /// Physical layout strategy.
    pub layout: KvLayout,
    /// Which layers/heads partition this plan.
    pub layer_partition: Vec<u32>,
    /// Memory domain for backend residency.
    pub residency_domain: String, // "SharedUnified", "DeviceLocal", "MappedExternal"
    /// Compatibility key for prefix cache sharing.
    pub prefix_key: PrefixCompatibilityKey,
    /// Eviction policy.
    pub eviction_policy: String, // "lru", "fifo", "lru_refcount"
    /// COW policy.
    pub cow_policy: String, // "copy_on_write", "share_full"
}

/// KV cache element dtype with FP8 support.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KVDtype {
    Bf16,
    F16,
    F32,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
}

/// Physical KV layout strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayout {
    /// Fixed-size blocks with logical-to-physical block tables (vLLM PagedAttention).
    PagedBlocks,
    /// Virtual-contiguous logical view backed by non-contiguous physical pages.
    VirtualContiguousPagedPhysical,
}

/// Prefix cache compatibility key.
/// Two ComputeImages with the same prefix key can share KV cache blocks.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixCompatibilityKey {
    /// SHA-256 of the model architecture config.
    pub model_digest: String,
    /// SHA-256 of the tokenizer config.
    pub tokenizer_digest: String,
    /// SHA-256 of the ComputeImage compile parameters.
    pub compile_digest: String,
}

impl PrefixCompatibilityKey {
    pub fn new(model_digest: &str, tokenizer_digest: &str, compile_digest: &str) -> Self {
        PrefixCompatibilityKey {
            model_digest: model_digest.to_string(),
            tokenizer_digest: tokenizer_digest.to_string(),
            compile_digest: compile_digest.to_string(),
        }
    }

    /// Combined key for index lookup.
    pub fn composite_key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.model_digest, self.tokenizer_digest, self.compile_digest
        )
    }
}

impl Default for KvCachePlan {
    fn default() -> Self {
        KvCachePlan {
            block_tokens: 32,
            max_blocks: 4096,
            kv_dtype: KVDtype::Bf16,
            layout: KvLayout::PagedBlocks,
            layer_partition: Vec::new(),
            residency_domain: "SharedUnified".into(),
            prefix_key: PrefixCompatibilityKey::new("", "", ""),
            eviction_policy: "lru".into(),
            cow_policy: "share_full".into(),
        }
    }
}

impl KvCachePlan {
    pub fn for_apple_unified_memory() -> Self {
        KvCachePlan {
            block_tokens: 32, // larger blocks for unified memory (no fragmentation penalty)
            max_blocks: 8192,
            kv_dtype: KVDtype::Bf16,
            layout: KvLayout::VirtualContiguousPagedPhysical,
            residency_domain: "SharedUnified".into(),
            eviction_policy: "lru".into(),
            cow_policy: "share_full".into(),
            ..Default::default()
        }
    }

    pub fn for_tensix() -> Self {
        KvCachePlan {
            block_tokens: 64, // larger blocks for Tensix SRAM efficiency
            max_blocks: 2048,
            kv_dtype: KVDtype::Bf16,
            layout: KvLayout::PagedBlocks,
            residency_domain: "DeviceLocal".into(),
            eviction_policy: "lru_refcount".into(),
            cow_policy: "copy_on_write".into(),
            ..Default::default()
        }
    }

    pub fn for_cpu() -> Self {
        KvCachePlan {
            block_tokens: 16,
            max_blocks: 16384,
            kv_dtype: KVDtype::F32,
            layout: KvLayout::PagedBlocks,
            residency_domain: "HostPageable".into(),
            ..Default::default()
        }
    }
}
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvState {
    Unallocated,
    Allocated,
    Primed,
    Decoding,
    Synchronized,
    Invalidated,
    Released,
}

pub struct RuntimePage {
    pub state: KvState,
    pub counter: AtomicU64,
}

impl RuntimePage {
    pub fn new() -> Self {
        Self {
            state: KvState::Unallocated,
            counter: AtomicU64::new(0),
        }
    }

    pub fn allocate(&mut self) {
        assert_eq!(self.state, KvState::Unallocated);
        self.state = KvState::Allocated;
    }

    pub fn prime(&mut self) {
        assert_eq!(self.state, KvState::Allocated);
        self.state = KvState::Primed;
    }

    pub fn validate_then_prepare(&mut self) -> bool {
        if self.state == KvState::Primed || self.state == KvState::Synchronized {
            self.state = KvState::Decoding;
            self.counter.fetch_add(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn append(&mut self) {
        assert_eq!(self.state, KvState::Decoding);
        self.state = KvState::Synchronized;
    }

    pub fn read(&self) -> KvState {
        self.state
    }

    pub fn rollback(&mut self) {
        if self.state == KvState::Decoding || self.state == KvState::Synchronized {
            self.state = KvState::Primed;
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
    }

    pub fn release(&mut self) {
        self.state = KvState::Released;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_runtime_state_machine() {
        let mut page = RuntimePage::new();
        assert_eq!(page.read(), KvState::Unallocated);

        page.allocate();
        assert_eq!(page.read(), KvState::Allocated);

        page.prime();
        assert_eq!(page.read(), KvState::Primed);

        assert!(page.validate_then_prepare());
        assert_eq!(page.read(), KvState::Decoding);
        assert_eq!(page.counter.load(Ordering::SeqCst), 1);

        page.append();
        assert_eq!(page.read(), KvState::Synchronized);

        page.rollback();
        assert_eq!(page.read(), KvState::Primed);
        assert_eq!(page.counter.load(Ordering::SeqCst), 0);

        page.release();
        assert_eq!(page.read(), KvState::Released);
    }
}
