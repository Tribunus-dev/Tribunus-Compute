//! TensorResidency — auditable contract for where a tensor lives.
//!
//! Model weights are loaded once into host memory and are NOT owned by any
//! backend. Operations between ANE/GPU/CPU pass through **ring buffers** in
//! the IOSurface island — Apple's shared memory mechanism.
//!
//! Every tensor in the system has a known residency. The scheduler uses this
//! to determine when transfers (page-table operations on an IOSurface mapping)
//! are needed vs when zero-copy is safe. The residency contract is the audit
//! trail for every buffer reference transition.

use std::collections::HashMap;

// ── Backend identity ───────────────────────────────────────────────────────

/// Which backend currently references this tensor.
///
/// Multiple backends may reference the same IOSurface simultaneously.
/// This ID tracks the **current reference**, not ownership — there is no
/// single owner when ring buffers are involved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendId {
    /// Apple MLX on Metal GPU
    MlxMetal,
    /// Candle CPU backend
    CandleCpu,
    /// Tensix (Tenstorrent Wormhole)
    TensixTensix,
    /// Intel Level Zero (iGPU)
    IntelLevelZero,
    /// Intel OpenCL (iGPU fallback)
    IntelOpenCl,
    /// Accelerate framework (CPU BLAS)
    Accelerate,
    /// CoreML (ANE via IOSurface ring buffer)
    CoreMl,
    /// Apple Neural Engine (direct ANE, separate from CoreML)
    Ane,
    /// Host CPU (pageable memory, weights)
    HostCpu,
    Unknown,
}

// ── Memory domain ──────────────────────────────────────────────────────────

/// Physical memory domain of a tensor.
///
/// * **MappedExternal** is the PRIMARY domain for IOSurface-backed buffers.
///   ANE, GPU, and CPU all access the same IOSurface through a ring buffer
///   — no data copies, only page-table operations.
/// * **HostPageable** covers weight tensors, which are loaded once into host
///   memory and mapped into each backend on demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Pageable host CPU memory (weights, host-level scratch)
    HostPageable,
    /// Pinned host CPU memory (DMA-able)
    HostPinned,
    /// Unified shared memory (CPU+GPU accessible, zero-copy)
    SharedUnified,
    /// Device-local memory (GPU DRAM / Tensix SRAM)
    DeviceLocal,
    /// Mapped external buffer (IOSurface ring buffer — primary domain)
    MappedExternal,
    /// Borrowed reference (no ownership, transient)
    Borrowed,
}

// ── Coherency state ────────────────────────────────────────────────────────

/// Coherency state between host and device.
///
/// IOSurface-backed buffers are **always** SharedCoherent — Apple's hardware
/// coherency layer handles cache synchronization transparently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoherencyState {
    /// Host and device are in sync (no copy needed)
    Coherent,
    /// Host has latest copy, device is stale
    HostDirty,
    /// Device has latest copy, host is stale
    DeviceDirty,
    /// Shared memory — always coherent (USM, IOSurface ring buffer)
    SharedCoherent,
}

// ── Transfer decision (ʼstatic) ──────────────────────────────────────────

/// Decision from the residency checker.
///
/// On Apple Silicon, "transfers" between ANE/GPU/CPU over an IOSurface ring
/// buffer are page-table operations, not data copies. The decision classifies
/// the *kind* of operation the scheduler must perform.
///
/// The `'static` lifetime enables `check_transfer` in the trait to return
/// decisions without borrowing local state — all string references are
/// compile-time literals.
#[derive(Clone, Debug)]
pub enum TransferDecision {
    /// No operation needed — same backend
    NoCopy,
    /// Zero-copy via shared memory / IOSurface ring buffer — no bytes moved
    ZeroCopy(&'static str),
    /// Ring-buffer dispatch — map the IOSurface into the target backend's
    /// address space (page-table operation, not a data copy)
    RingBufferDispatch(BackendId, BackendId),
    /// Host-to-device transfer needed (weights)
    HostToDevice,
    /// Device-to-host transfer needed (readback)
    DeviceToHost,
}

impl std::fmt::Display for TransferDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferDecision::NoCopy => write!(f, "no-copy"),
            TransferDecision::ZeroCopy(reason) => write!(f, "zero-copy ({reason})"),
            TransferDecision::RingBufferDispatch(a, b) => {
                write!(f, "ring-buffer: {a:?} -> {b:?}")
            }
            TransferDecision::HostToDevice => write!(f, "host->device"),
            TransferDecision::DeviceToHost => write!(f, "device->host"),
        }
    }
}

// ── TensorResidency ────────────────────────────────────────────────────────

/// Complete residency record for a tensor.
///
/// Tracks the current backend reference, memory domain, and coherence state.
/// On Apple Silicon, `MappedExternal` + `SharedCoherent` means the tensor
/// lives in an IOSurface ring buffer accessible by ANE, GPU, and CPU without
/// copies.
#[derive(Clone, Debug)]
pub struct TensorResidency {
    /// Logical name for diagnostics
    pub tensor_name: String,
    /// Backend that currently references this tensor
    pub backend: BackendId,
    /// Memory domain (MappedExternal for IOSurface ring buffers)
    pub memory: MemoryDomain,
    /// Coherency state (SharedCoherent for IOSurface)
    pub coherency: CoherencyState,
    /// Size in bytes
    pub byte_size: u64,
    /// Allocation / IOSurface ID
    pub allocation_id: u64,
    /// Last backend to write this tensor
    pub last_writer: BackendId,
    /// Last backend to read this tensor
    pub last_reader: BackendId,
    /// How many times this tensor's mapping was changed (transferred between
    /// backends via ring-buffer dispatch)
    pub transfer_count: u64,
    /// Total bytes transferred (zero for ring-buffer dispatch — only counts
    /// actual host<->device copies for weight tensors)
    pub total_transfer_bytes: u64,
}

impl TensorResidency {
    /// Create a new residency record.
    ///
    /// `backend` is the backend that created or first referenced this tensor.
    /// For IOSurface-backed tensors, use `MemoryDomain::MappedExternal`; the
    /// coherency state is always `SharedCoherent` for the ring-buffer island.
    pub fn new(name: &str, backend: BackendId, memory: MemoryDomain, size: u64) -> Self {
        let coherency = match memory {
            MemoryDomain::MappedExternal | MemoryDomain::SharedUnified => {
                CoherencyState::SharedCoherent
            }
            _ => CoherencyState::Coherent,
        };
        TensorResidency {
            tensor_name: name.to_string(),
            backend,
            memory,
            coherency,
            byte_size: size,
            allocation_id: 0,
            last_writer: backend,
            last_reader: backend,
            transfer_count: 0,
            total_transfer_bytes: 0,
        }
    }

    /// Check whether reading from `target_backend` requires a transfer.
    ///
    /// Returns a [`TransferDecision`] classifying the operation.
    pub fn requires_transfer(&self, target_backend: BackendId) -> TransferDecision {
        // Same backend — always no-op.
        if self.backend == target_backend {
            return TransferDecision::NoCopy;
        }

        match self.memory {
            // IOSurface ring buffer — page-table dispatch, not a copy.
            MemoryDomain::MappedExternal => {
                TransferDecision::RingBufferDispatch(self.backend, target_backend)
            }
            // Unified shared memory — always zero-copy.
            MemoryDomain::SharedUnified => TransferDecision::ZeroCopy("shared unified"),

            // Host-pageable/pinned: weights or host buffers.
            MemoryDomain::HostPageable | MemoryDomain::HostPinned => {
                if self.backend == BackendId::HostCpu {
                    TransferDecision::HostToDevice
                } else {
                    TransferDecision::DeviceToHost
                }
            }
            // Device-local: explicit copy or ring-buffer dispatch.
            MemoryDomain::DeviceLocal => {
                if self.backend == BackendId::HostCpu {
                    TransferDecision::HostToDevice
                } else if target_backend == BackendId::HostCpu {
                    TransferDecision::DeviceToHost
                } else {
                    // Both are devices — ring-buffer dispatch via IOSurface.
                    TransferDecision::RingBufferDispatch(self.backend, target_backend)
                }
            }
            // Borrowed: transient reference, no transfer.
            MemoryDomain::Borrowed => TransferDecision::NoCopy,
        }
    }

    /// Record a mapping/transfer event.
    pub fn record_transfer(&mut self, bytes: u64) {
        self.transfer_count += 1;
        self.total_transfer_bytes += bytes;
    }
}

// ── ResidencyLedger ────────────────────────────────────────────────────────

/// Ledger of all residency records for a single inference request.
///
/// Tracks tensors by name and accumulates transfer statistics for the
/// scheduler to report.
pub struct ResidencyLedger {
    records: HashMap<String, TensorResidency>,
    /// Number of actual data-copy transfers (host<->device)
    total_copies: u64,
    /// Total bytes copied
    total_copy_bytes: u64,
    /// Number of zero-copy or ring-buffer dispatch operations
    zero_copy_ops: u64,
    /// Number of ring-buffer page-table operations
    ring_buffer_dispatches: u64,
}

impl ResidencyLedger {
    pub fn new() -> Self {
        ResidencyLedger {
            records: HashMap::new(),
            total_copies: 0,
            total_copy_bytes: 0,
            zero_copy_ops: 0,
            ring_buffer_dispatches: 0,
        }
    }

    /// Record a new tensor residency.
    pub fn record(&mut self, name: &str, backend: BackendId, domain: MemoryDomain, size: u64) {
        let residency = TensorResidency::new(name, backend, domain, size);
        self.records.insert(name.to_string(), residency);
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TensorResidency> {
        self.records.get(name)
    }

    /// Number of actual data-copy transfers.
    pub fn transfer_count(&self) -> u64 {
        self.total_copies
    }

    /// Number of zero-copy operations (shared memory hits).
    pub fn zero_copy_count(&self) -> u64 {
        self.zero_copy_ops
    }

    /// Number of ring-buffer dispatches (page-table operations).
    pub fn ring_buffer_dispatch_count(&self) -> u64 {
        self.ring_buffer_dispatches
    }

    /// A one-line summary for the scheduler log.
    pub fn summary(&self) -> String {
        format!(
            "ResidencyLedger: {} tensors, {} copies ({} bytes), {} zero-copy, {} ring-buffer dispatches",
            self.records.len(),
            self.total_copies,
            self.total_copy_bytes,
            self.zero_copy_ops,
            self.ring_buffer_dispatches,
        )
    }
}


use std::collections::VecDeque;

// ── Weight Cache ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightCacheKey {
    pub tensor_identity: String,
    pub capability: String,
    pub topology_hash: String,
    pub layout_version: u32,
    pub data_format: String,
}

#[derive(Clone, Debug)]
pub struct WeightCacheEntry {
    pub residency: TensorResidency,
    pub session_id: Option<String>,
}

pub struct WeightCache {
    pub max_dram_bytes: u64,
    pub current_dram_bytes: u64,
    pub entries: HashMap<WeightCacheKey, WeightCacheEntry>,
    pub lru_order: VecDeque<WeightCacheKey>,
    
    pub hits: u64,
    pub misses: u64,
    pub upload_avoidance_bytes: u64,
}

impl WeightCache {
    pub fn new(max_dram_bytes: u64) -> Self {
        Self {
            max_dram_bytes,
            current_dram_bytes: 0,
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            hits: 0,
            misses: 0,
            upload_avoidance_bytes: 0,
        }
    }

    pub fn get(&mut self, key: &WeightCacheKey) -> Option<&TensorResidency> {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            let k = self.lru_order.remove(pos).unwrap();
            self.lru_order.push_back(k);
            
            let entry = self.entries.get(key).unwrap();
            self.hits += 1;
            self.upload_avoidance_bytes += entry.residency.byte_size;
            Some(&entry.residency)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: WeightCacheKey, residency: TensorResidency, session_id: Option<String>) {
        let size = residency.byte_size;
        
        // Remove existing key if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_dram_bytes -= old_entry.residency.byte_size;
            if let Some(pos) = self.lru_order.iter().position(|k| *k == key) {
                self.lru_order.remove(pos);
            }
        }

        while self.current_dram_bytes + size > self.max_dram_bytes && !self.lru_order.is_empty() {
            // Find victim: unpinned first, else oldest pinned
            let mut victim_idx = None;
            for (i, k) in self.lru_order.iter().enumerate() {
                if let Some(entry) = self.entries.get(k) {
                    if entry.session_id.is_none() {
                        victim_idx = Some(i);
                        break;
                    }
                }
            }

            let idx = victim_idx.unwrap_or(0);
            let victim_key = self.lru_order.remove(idx).unwrap();
            if let Some(victim_entry) = self.entries.remove(&victim_key) {
                self.current_dram_bytes -= victim_entry.residency.byte_size;
            }
        }

        self.entries.insert(key.clone(), WeightCacheEntry { residency, session_id });
        self.lru_order.push_back(key);
        self.current_dram_bytes += size;
    }

    pub fn invalidate_reset(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.current_dram_bytes = 0;
    }
}
