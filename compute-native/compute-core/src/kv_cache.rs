//! Per-layer KV cache for Gemma 4 hybrid attention.
//!
//! Supports sliding-window eviction on sliding layers and concatenation on
//! global layers. Each layer holds its own (keys, values) pair.
//!
//! Sliding layers use a ring buffer: a preallocated [capacity, n_kv_heads, head_dim]
//! array that circularly overwrites the oldest entries. Global layers grow
//! unboundedly via concatenation.
//!
//! Commit/rollback: append() writes to a staging region tracked by total_appended;
//! commit_step() advances committed_len; rollback() discards uncommitted data.
//! read_window() returns all data including uncommitted, so the attention
//! kernel can read the full cache immediately after append.

use mlx_rs::error::Result as MlxResult;
use mlx_rs::ops::indexing::IndexMutOp;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{ops, Array};

use crate::cache::evolkv::{CalibrationSet, EvolKV, LayerBudget};

use crate::memory::allocator::BlockHandle;
use std::sync::Arc;
use parking_lot::Mutex;
use std::rc::Rc;
use crate::quantization::turboquant_kv::{
    AsymmetricQuantMode, KvQuantMode, QjlCorrection, TurboQuantKvCache,
};

/// Backing store for one compressed KV slot.
///
/// Unlike the raw [`KvCache`] which stores FP16 tensors in MLX arrays,
/// this stores the compressed byte buffers and optional QJL correction bits.
#[derive(Debug, Clone)]
pub struct CompressedKvSlot {
    /// Compressed keys + quantization scale/indices.
    pub compressed_keys: Vec<u8>,
    /// Compressed values + quantization scale/indices.
    pub compressed_values: Vec<u8>,
    /// QJL residual correction bits (separate for fast access).
    pub qjl_correction: Option<QjlCorrection>,
    /// Offset of this slot's first token in the global KV sequence.
    pub kv_offset: u32,
    /// Number of tokens stored in this slot.
    pub num_tokens: usize,
}

impl CompressedKvSlot {
    /// Create a new empty slot at the given KV offset.
    pub fn new(kv_offset: u32) -> Self {
        Self {
            compressed_keys: Vec::new(),
            compressed_values: Vec::new(),
            qjl_correction: None,
            kv_offset,
            num_tokens: 0,
        }
    }

    /// Serialize this slot into a single page-data blob for distributed
    /// KV cache transport (RDMA).
    ///
    /// Format (little-endian):
    ///   [key_len: u32][keys_data][val_len: u32][values_data]
    ///
    /// The 2-bit compressed data is transferred as-is; the receiving
    /// node uses [`from_page_data`](Self::from_page_data) to reconstruct.
    pub fn to_page_data(&self) -> Vec<u8> {
        let key_len = self.compressed_keys.len() as u32;
        let val_len = self.compressed_values.len() as u32;
        let mut buf = Vec::with_capacity(8 + key_len as usize + val_len as usize);
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&self.compressed_keys);
        buf.extend_from_slice(&val_len.to_le_bytes());
        buf.extend_from_slice(&self.compressed_values);
        buf
    }

    /// Reconstruct a slot from its page-data blob (produced by
    /// [`to_page_data`](Self::to_page_data)).
    ///
    /// The slot will have `kv_offset = 0` and `num_tokens = 1`; the
    /// caller should adjust these after insertion into a
    /// [`CompressedKvCache`].
    pub fn from_page_data(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("page data too short: need at least 8 bytes for headers".into());
        }
        let key_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let val_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if 8 + key_len + val_len != data.len() {
            return Err(format!(
                "page data length mismatch: expected {} bytes, got {}",
                8 + key_len + val_len,
                data.len()
            ));
        }
        let keys_end = 8 + key_len;
        let compressed_keys = data[8..keys_end].to_vec();
        let compressed_values = data[keys_end..].to_vec();
        Ok(Self {
            compressed_keys,
            compressed_values,
            qjl_correction: None,
            kv_offset: 0,
            num_tokens: 1,
        })
    }

    /// Returns `true` if this slot contains no compressed data.
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Allocated bytes for this slot (compressed buffers + correction + fixed overhead).
    pub fn allocated_bytes(&self) -> u64 {
        let data = self.compressed_keys.len()
            + self.compressed_values.len()
            + self
                .qjl_correction
                .as_ref()
                .map_or(0, |c| c.residual_bits.len());
        // Fixed overhead: three Vec's (24 bytes each on 64-bit) + Option overhead + fields
        let overhead: u64 = 80;
        data as u64 + overhead
    }
}

/// Compressed KV cache wrapping TurboQuant.
///
/// Each slot holds compressed K/V byte buffers for a batch of tokens.
/// The paged block allocator provides IOSurface-backed storage for the
/// compressed bytes.  Decompression happens on-demand during attention.
#[derive(Debug)]
pub struct CompressedKvCache {
    /// Quantization engine.
    pub tq: TurboQuantKvCache,
    /// Per-slot compressed state.
    pub slots: Vec<CompressedKvSlot>,
    /// Page handles in the IOSurface island.
    pub pages: Vec<BlockHandle>,
    /// Sliding window size (0 = full attention / no limit).
    pub sliding_window: u32,
    /// Committed length (tokens that survived rollback).
    pub committed_len: u32,
    /// Sequence length (total tokens including uncommitted).
    pub seq_len: u32,
    /// Per-layer cache budget fractions set by EvolKV.
    /// When `Some`, the page migration service uses these fractions
    /// to decide per-layer compression aggressiveness.
    pub per_layer_budget: Option<Vec<f64>>,
}

impl CompressedKvCache {
    /// Create a new compressed KV cache with the given quantization mode.
    pub fn new(mode: KvQuantMode, group_size: usize, num_slots: usize) -> Self {
        let tq = TurboQuantKvCache::new(mode, group_size, num_slots);
        Self {
            tq,
            slots: Vec::with_capacity(num_slots),
            pages: Vec::new(),
            sliding_window: 0,
            committed_len: 0,
            seq_len: 0,
            per_layer_budget: None,
        }
    }

    /// Append a token's K/V to the cache.
    ///
    /// `keys` and `values` are f32 slices (one token's worth of elements).
    /// The data is quantized via TurboQuant and stored in a new slot.
    pub fn append(&mut self, keys: &[f32], values: &[f32], kv_offset: u32) -> Result<(), String> {
        let slot_idx = self.slots.len();
        let mut slot = CompressedKvSlot::new(kv_offset);

        // Quantize K/V via TurboQuant.
        self.tq
            .quantize(slot_idx, keys, values)
            .map_err(|e| format!("TQ quantize: {:?}", e))?;

        // Extract compressed state from the TurboQuant engine.
        let tq_state = self
            .tq
            .slot_state(slot_idx)
            .ok_or_else(|| "TQ slot state missing after quantize".to_string())?;

        // Clone the compressed byte buffers into our slot for independent
        // lifecycle (deletion, eviction) from the TurboQuant engine.
        slot.compressed_keys = tq_state.keys.clone();
        slot.compressed_values = tq_state.values.clone();
        slot.num_tokens = 1;

        self.slots.push(slot);
        self.seq_len += 1;
        Ok(())
    }

    /// Append a token's K/V with asymmetric K/V quantization.
    ///
    /// Keys and values use different bit widths (and potentially different
    /// quantization strategies) as specified by `asym_mode`.  This exploits
    /// the fact that keys are position-structured (redundant across sequence)
    /// while values are more noise-like, enabling ~5.3x effective compression
    /// vs ~4.57x symmetric at the same average bits.
    pub fn append_asymmetric(
        &mut self,
        keys: &[f32],
        values: &[f32],
        kv_offset: u32,
        asym_mode: &AsymmetricQuantMode,
    ) -> Result<(), String> {
        let slot_idx = self.slots.len();
        let mut slot = CompressedKvSlot::new(kv_offset);

        // Quantize K/V via asymmetric TurboQuant.
        self.tq
            .quantize_asymmetric(slot_idx, keys, values, asym_mode)
            .map_err(|e| format!("TQ quantize_asymmetric: {:?}", e))?;

        let tq_state = self
            .tq
            .slot_state(slot_idx)
            .ok_or_else(|| "TQ slot state missing after quantize_asymmetric".to_string())?;

        slot.compressed_keys = tq_state.keys.clone();
        slot.compressed_values = tq_state.values.clone();
        slot.num_tokens = 1;

        self.slots.push(slot);
        self.seq_len += 1;
        Ok(())
    }

    /// Decompress a single slot, returning (keys, values) as f32 vectors.
    fn decompress_slot(&self, slot_idx: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        // Use the TurboQuant engine to dequantize the slot's data.
        self.tq
            .dequantize(slot_idx)
            .map_err(|e| format!("TQ dequantize slot {}: {:?}", slot_idx, e))
    }

    /// Read a range of tokens as decompressed f32 slices.
    ///
    /// `start` and `end` are token positions in the global KV sequence.
    /// Returns `(keys, values)` concatenated from all slots covering the
    /// range `[start, end)`.
    pub fn read_window(&self, start: u32, end: u32) -> Result<(Vec<f32>, Vec<f32>), String> {
        if start >= end || self.slots.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut keys = Vec::new();
        let mut values = Vec::new();

        for (idx, slot) in self.slots.iter().enumerate() {
            let slot_start = slot.kv_offset;
            let slot_end = slot_start + slot.num_tokens as u32;

            // Check if this slot overlaps [start, end).
            if slot_end <= start || slot_start >= end {
                continue;
            }

            let (k_deq, v_deq) = self.decompress_slot(idx)?;

            // Determine the sub-range within this slot's decompressed data.
            let range_start_in_slot = if start > slot_start {
                (start - slot_start) as usize
            } else {
                0
            };

            let range_end_in_slot = if end < slot_end {
                (end - slot_start) as usize
            } else {
                slot.num_tokens
            };

            // The decompressed data has shape [num_tokens, n_kv_heads * head_dim]
            // (flat). Compute the byte/element bounds.
            let elems_per_token = k_deq.len() / slot.num_tokens;
            let k_start = range_start_in_slot * elems_per_token;
            let k_end = range_end_in_slot * elems_per_token;
            keys.extend_from_slice(&k_deq[k_start..k_end]);

            let v_elems = v_deq.len() / slot.num_tokens;
            let v_start = range_start_in_slot * v_elems;
            let v_end = range_end_in_slot * v_elems;
            values.extend_from_slice(&v_deq[v_start..v_end]);
        }

        Ok((keys, values))
    }

    /// Commit the current step (move committed_len forward to seq_len).
    pub fn commit_step(&mut self) {
        self.committed_len = self.seq_len;
    }

    /// Rollback to the last committed state (truncate uncommitted slots).
    pub fn rollback(&mut self) {
        if self.seq_len == self.committed_len {
            return;
        }
        // Truncate slots and TQ state back to committed length.
        let committed_slots = self.committed_len as usize;
        self.slots.truncate(committed_slots);
        // Also clear the TQ state for rolled-back slots (they are indexed by slot).
        // The TQ internal state vec may be larger; we leave it but the slot indices
        // beyond committed_len are stale. We rely on the fact that a subsequent
        // quantize() will overwrite them.
        self.seq_len = self.committed_len;
    }

    /// Total allocated bytes (compressed data + metadata).
    pub fn allocated_bytes(&self) -> u64 {
        let slots: u64 = self.slots.iter().map(|s| s.allocated_bytes()).sum();
        let pages: u64 = (self.pages.len() * std::mem::size_of::<BlockHandle>()) as u64;
        let struct_overhead = std::mem::size_of::<Self>() as u64;
        // TQ internal state is also allocated but owned separately; we account
        // the compressed data in `slots` since we cloned it there.
        slots + pages + struct_overhead
    }
}

/// Per-layer KV cache for Gemma 4's hybrid local/global attention schedule.
///
/// Sliding layers use a ring buffer that overwrites the oldest entries once
/// the window exceeds capacity. Global layers grow via concatenation.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Number of KV heads for this layer.
    pub n_kv_heads: u32,
    /// Head dimension for this layer.
    pub head_dim: u32,
    /// Maximum number of positions this cache can hold (sliding window or
    /// global max).
    pub capacity: u32,
    /// Whether this layer uses sliding-window eviction.
    pub is_sliding: bool,

    // ── Cached arrays ──────────────────────────────────────────────────
    /// Cached keys. For sliding layers: None (ring buffer used instead).
    /// For global layers: concatenated KV along axis 0.
    k_cache: Option<Array>,
    /// Cached values.
    v_cache: Option<Array>,

    /// Current number of cached positions visible for attention
    /// (= min(total_appended, capacity) for sliding, total_appended for global).
    pub seq_len: u32,

    // ── Ring buffer (sliding layers only) ─────────────────────────────
    /// Preallocated ring buffer for keys, shape [capacity, n_kv_heads, head_dim].
    preallocated_k: Option<Array>,
    /// Preallocated ring buffer for values.
    preallocated_v: Option<Array>,

    // ── Logical position tracking ──────────────────────────────────────
    /// Total tokens ever appended to this cache.
    pub total_appended: u32,
    /// Absolute position of cache[0] in the overall token sequence.
    /// For sliding layers, this advances as old entries are evicted.
    /// For global layers, it stays 0.
    pub logical_start: u32,
    /// Next write index in the ring buffer for sliding layers, or total
    /// tokens stored for global layers.
    pub physical_write_pos: u32,

    // ── Commit/rollback tracking ──────────────────────────────────────
    /// Number of committed tokens (visible to attention after commit_step).
    pub committed_len: u32,
    /// Snapshot of global-layer arrays before the last uncommitted append,
    /// used to restore on rollback.
    rollback_k: Option<Array>,
    /// Snapshot of global-layer values before the last uncommitted append.
    rollback_v: Option<Array>,

    // ── Byte accounting ───────────────────────────────────────────────
    /// Number of evictions that have occurred.
    pub evictions_count: u64,
    /// Bytes copied during operations.
    pub copy_bytes: u64,
    /// Optional compressed cache sink for TurboQuant quantization.
    /// When set, every `append()` also quantizes the incoming key/value
    /// arrays into this TurboQuant cache.
    pub compressed_sink: Option<Arc<Mutex<TurboQuantKvCache>>>,
}

impl KvCache {
    /// Set the compressed cache sink for TurboQuant quantization.
    pub fn set_compressed_sink(&mut self, sink: Arc<Mutex<TurboQuantKvCache>>) {
        self.compressed_sink = Some(sink);
    }

    /// Create a new empty per-layer KV cache.
    ///
    /// `capacity` is the maximum number of positions stored (sliding window
    /// for sliding layers, `max_position_embeddings` for global layers).
    /// `is_sliding` enables sliding-window eviction.
    pub fn new(capacity: u32, n_kv_heads: u32, head_dim: u32, is_sliding: bool) -> Self {
        Self {
            n_kv_heads,
            head_dim,
            capacity,
            is_sliding,
            k_cache: None,
            v_cache: None,
            seq_len: 0,
            preallocated_k: None,
            preallocated_v: None,
            total_appended: 0,
            logical_start: 0,
            physical_write_pos: 0,
            committed_len: 0,
            rollback_k: None,
            rollback_v: None,
            evictions_count: 0,
            copy_bytes: 0,
            compressed_sink: None,
        }
    }

    // ── Append ─────────────────────────────────────────────────────────

    /// Append new K and V arrays to this layer's cache.
    ///
    /// - `keys`, `values`: shape `[n_tokens, n_kv_heads, head_dim]`
    ///
    /// **Sliding layers**: ring buffer write into preallocated storage,
    /// overwriting oldest entries on wrap. On first append, if
    /// `n_tokens > capacity`, the input is trimmed to the last `capacity`
    /// tokens.
    ///
    /// **Global layers**: concatenate along the sequence dimension.
    ///
    /// The append is uncommitted until `commit_step()` is called. If the
    /// caller needs to roll back, `rollback()` restores the pre-append state.
    pub fn append(&mut self, keys: Array, values: Array) -> MlxResult<()> {
        let incoming_len = keys.shape()[0] as u32;

        // If compressed sink is set, quantize incoming data before storing.
        if let Some(sink) = &self.compressed_sink {
            let n_kv = self.n_kv_heads as usize;
            let hd = self.head_dim as usize;
            for t in 0..incoming_len as usize {
                let k_t = keys.index(t as i32);
                let v_t = values.index(t as i32);
                let k_slice: &[f32] = k_t.as_slice::<f32>();
                let v_slice: &[f32] = v_t.as_slice::<f32>();
                if !k_slice.is_empty() && !v_slice.is_empty() {
                    let slot = self.total_appended as usize + t;
                    let mut comp = sink.lock();
                    let _ = comp.quantize(slot, k_slice, v_slice);
                }
            }
        }

        if self.k_cache.is_none()
            && self.v_cache.is_none()
            && self.preallocated_k.is_none()
            && self.preallocated_v.is_none()
        {
            self.first_append(keys, values, incoming_len)
        } else if self.is_sliding {
            self.append_sliding(keys, values, incoming_len)
        } else {
            self.append_global(keys, values, incoming_len)
        }
    }

    /// First-ever append: initialise storage.
    fn first_append(&mut self, keys: Array, values: Array, incoming_len: u32) -> MlxResult<()> {
        if self.is_sliding {
            // Trim first append if it exceeds capacity.
            let (k_trimmed, v_trimmed, actual_n) = if incoming_len > self.capacity {
                let excess = incoming_len - self.capacity;
                let k = keys.index((excess as i32.., .., ..));
                let v = values.index((excess as i32.., .., ..));
                self.evictions_count = excess as u64;
                (k, v, self.capacity)
            } else {
                (keys, values, incoming_len)
            };

            if actual_n > self.capacity {
                return Err(mlx_rs::error::Exception::custom(
                    "first append still exceeds capacity after trimming",
                ));
            }

            // Preallocate ring buffer.
            let shape = &[
                self.capacity as i32,
                self.n_kv_heads as i32,
                self.head_dim as i32,
            ];
            let mut pre_k = ops::zeros::<f32>(shape)?;
            let mut pre_v = ops::zeros::<f32>(shape)?;

            pre_k.index_mut((0..actual_n as i32, .., ..), &k_trimmed);
            pre_v.index_mut((0..actual_n as i32, .., ..), &v_trimmed);

            self.preallocated_k = Some(pre_k);
            self.preallocated_v = Some(pre_v);
            self.total_appended = actual_n;
            self.seq_len = actual_n;
            self.logical_start = 0;
            self.physical_write_pos = actual_n % self.capacity;
            self.committed_len = actual_n;
        } else {
            self.k_cache = Some(keys);
            self.v_cache = Some(values);
            self.total_appended = incoming_len;
            self.seq_len = incoming_len;
            self.logical_start = 0;
            self.physical_write_pos = incoming_len;
            self.committed_len = incoming_len;
        }
        Ok(())
    }

    /// Append to a sliding layer using the ring buffer.
    fn append_sliding(&mut self, keys: Array, values: Array, incoming_len: u32) -> MlxResult<()> {
        let pre_k = self
            .preallocated_k
            .as_mut()
            .expect("preallocated_k must exist for sliding append");
        let pre_v = self
            .preallocated_v
            .as_mut()
            .expect("preallocated_v must exist for sliding append");

        let cap = self.capacity as i32;
        let write_pos = self.physical_write_pos as i32;
        let n_tok = incoming_len as i32;
        let end_wrap = write_pos + n_tok;

        if end_wrap <= cap {
            // Contiguous write: no wrap.
            pre_k.index_mut((write_pos..end_wrap, .., ..), &keys);
            pre_v.index_mut((write_pos..end_wrap, .., ..), &values);
        } else {
            // Wrapping write.
            let first_seg = cap - write_pos;
            let second_seg = n_tok - first_seg;

            let k_first = keys.index((0..first_seg, .., ..));
            let v_first = values.index((0..first_seg, .., ..));
            pre_k.index_mut((write_pos..cap, .., ..), &k_first);
            pre_v.index_mut((write_pos..cap, .., ..), &v_first);

            let k_second = keys.index((first_seg.., .., ..));
            let v_second = values.index((first_seg.., .., ..));
            pre_k.index_mut((0..second_seg, .., ..), &k_second);
            pre_v.index_mut((0..second_seg, .., ..), &v_second);
        }

        self.physical_write_pos = (self.physical_write_pos + incoming_len) % self.capacity;
        self.total_appended += incoming_len;
        self.seq_len = std::cmp::min(self.total_appended, self.capacity);

        // Update logical_start: position of the first valid entry.
        if self.total_appended > self.capacity {
            self.logical_start = self.total_appended - self.capacity;
        }

        // Track evictions count.
        if self.total_appended > self.capacity {
            let evicted = self.total_appended - self.capacity;
            self.evictions_count = evicted as u64;
        }

        // Save pre-append state for rollback (the old total_appended allows us
        // to know how many tokens were valid before this append).
        self.rollback_k = None;
        self.rollback_v = None;

        Ok(())
    }

    /// Append to a global layer by concatenating along the sequence axis.
    fn append_global(&mut self, keys: Array, values: Array, incoming_len: u32) -> MlxResult<()> {
        // Save pre-append arrays for rollback.
        let old_k = self.k_cache.as_ref().map(|k| k.clone());
        let old_v = self.v_cache.as_ref().map(|v| v.clone());

        let cached_k = self.k_cache.take().expect("k_cache must be present");
        let cached_v = self.v_cache.take().expect("v_cache must be present");

        let new_k = ops::concatenate_axis(&[&cached_k, &keys], 0)?;
        let new_v = ops::concatenate_axis(&[&cached_v, &values], 0)?;

        let copy_k = cached_k.size() as u64 * 4;
        let copy_v = cached_v.size() as u64 * 4;
        self.copy_bytes += copy_k + copy_v;

        self.k_cache = Some(new_k);
        self.v_cache = Some(new_v);
        self.total_appended += incoming_len;
        self.seq_len += incoming_len;
        self.physical_write_pos += incoming_len;

        self.rollback_k = old_k;
        self.rollback_v = old_v;

        Ok(())
    }

    // ── Commit / rollback ──────────────────────────────────────────────

    /// Commit the current append, making it the new committed baseline.
    ///
    /// After successful eval, call this to accept the KV data written by
    /// the last `append()`.
    pub fn commit_step(&mut self) {
        self.committed_len = self.uncommitted_len();
        self.rollback_k = None;
        self.rollback_v = None;
    }

    /// Roll back the last uncommitted append, restoring the cache to its
    /// pre-append state.
    ///
    /// For sliding layers: truncates the logical view (seq_len, total_appended)
    /// back to the committed position. Ring buffer data is left in place and
    /// will be overwritten on the next append.
    ///
    /// For global layers: restores the saved pre-append arrays.
    pub fn rollback(&mut self) {
        if self.total_appended == self.committed_len && self.seq_len == self.committed_len {
            return;
        }

        if self.is_sliding {
            // Restore the pre-append state from the committed boundary.
            if self.committed_len == 0 {
                // Fully reset sliding layer.
                self.preallocated_k = None;
                self.preallocated_v = None;
                self.total_appended = 0;
                self.seq_len = 0;
                self.logical_start = 0;
                self.physical_write_pos = 0;
            } else {
                self.total_appended = self.committed_len;
                self.seq_len = std::cmp::min(self.committed_len, self.capacity);

                if self.total_appended > self.capacity {
                    self.logical_start = self.total_appended - self.capacity;
                } else {
                    self.logical_start = 0;
                }

                // physical_write_pos must reflect the write position after
                // the committed data was written. We can derive this:
                // After committed_len tokens, the next write pos is committed_len % capacity.
                self.physical_write_pos = self.committed_len % self.capacity;
            }
        } else {
            // Restore global-layer arrays from snapshots.
            self.k_cache = self.rollback_k.take();
            self.v_cache = self.rollback_v.take();

            self.total_appended = self.committed_len;
            self.seq_len = self.committed_len;
            self.physical_write_pos = self.committed_len;
        }

        self.rollback_k = None;
        self.rollback_v = None;
    }

    /// Number of uncommitted tokens (total_appended - committed_len).
    fn uncommitted_len(&self) -> u32 {
        self.total_appended
    }

    // ── Read ───────────────────────────────────────────────────────────

    /// Returns a cloned copy of the cached K and V arrays for the full cache
    /// (including uncommitted data). The attention kernel reads from this
    /// after every append.
    ///
    /// For sliding layers: reconstructs a contiguous view from the ring buffer.
    ///
    /// Returns `None` if no data exists.
    pub fn read_window(&self) -> Option<(Array, Array)> {
        if self.is_sliding {
            self.read_window_sliding()
        } else {
            match (&self.k_cache, &self.v_cache) {
                (Some(k), Some(v)) => Some((k.clone(), v.clone())),
                _ => None,
            }
        }
    }

    /// Reconstruct a contiguous committed window from the ring buffer.
    fn read_window_sliding(&self) -> Option<(Array, Array)> {
        let pre_k = self.preallocated_k.as_ref()?;
        let pre_v = self.preallocated_v.as_ref()?;

        if self.seq_len == 0 {
            return None;
        }

        let cap = self.capacity as i32;
        let n_to_read = self.seq_len as i32;

        // When total_appended < capacity, valid data starts at position 0.
        // When total_appended >= capacity, valid data starts at physical_write_pos
        // (the oldest surviving entry).
        let read_start = if self.total_appended >= self.capacity {
            self.physical_write_pos as i32
        } else {
            0i32
        };

        let end_wrap = read_start + n_to_read;
        if end_wrap <= cap {
            // Contiguous segment.
            let k = pre_k.index((read_start..end_wrap, .., ..));
            let v = pre_v.index((read_start..end_wrap, .., ..));
            Some((k, v))
        } else {
            // Wrapping segment: two concatenated slices.
            let first_seg = cap - read_start;
            let second_seg = n_to_read - first_seg;

            let k1 = pre_k.index((read_start..cap, .., ..));
            let v1 = pre_v.index((read_start..cap, .., ..));
            let k2 = pre_k.index((0..second_seg, .., ..));
            let v2 = pre_v.index((0..second_seg, .., ..));

            let k = ops::concatenate_axis(&[&k1, &k2], 0).ok()?;
            let v = ops::concatenate_axis(&[&v1, &v2], 0).ok()?;
            Some((k, v))
        }
    }

    // ── Byte accounting ────────────────────────────────────────────────

    /// Total bytes of preallocated arrays.
    ///
    /// For sliding layers: full ring buffer allocation (capacity × heads ×
    /// head_dim × 4 bytes × 2 arrays). For global layers: current cache size.
    pub fn allocated_bytes(&self) -> u64 {
        if self.is_sliding {
            if self.preallocated_k.is_none() {
                return 0;
            }
            let elements = self.capacity as u64 * self.n_kv_heads as u64 * self.head_dim as u64;
            elements * 4 * 2 // f32 × K+V
        } else {
            if self.seq_len == 0 {
                return 0;
            }
            let elements = self.seq_len as u64 * self.n_kv_heads as u64 * self.head_dim as u64;
            elements * 4 * 2
        }
    }

    /// Bytes for committed positions.
    pub fn committed_bytes(&self) -> u64 {
        if self.committed_len == 0 {
            return 0;
        }
        let len = std::cmp::min(self.committed_len, self.capacity);
        let elements = len as u64 * self.n_kv_heads as u64 * self.head_dim as u64;
        elements * 4 * 2
    }

    /// Bytes copied from staging to committed area.
    pub fn copy_bytes(&self) -> u64 {
        self.copy_bytes
    }

    /// Total bytes consumed by this cache (K + V, f32).
    #[deprecated(note = "use allocated_bytes() or committed_bytes() instead")]
    pub fn total_bytes(&self) -> u64 {
        if self.seq_len == 0 {
            return 0;
        }
        let elements = self.seq_len as u64 * self.n_kv_heads as u64 * self.head_dim as u64;
        elements * 4 * 2
    }

    // ── Receipt ────────────────────────────────────────────────────────

    /// Return a JSON string with the cache receipt.
    pub fn receipt_json(&self) -> String {
        let is_sliding = if self.is_sliding { "true" } else { "false" };
        format!(
            r#"{{"logical_start":{},"logical_length":{},"capacity":{},"physical_write_pos":{},"allocated_bytes":{},"committed_bytes":{},"evictions_count":{},"is_sliding":{}}}"#,
            self.logical_start,
            self.committed_len,
            self.capacity,
            self.physical_write_pos,
            self.allocated_bytes(),
            self.committed_bytes(),
            self.evictions_count,
            is_sliding,
        )
    }

    // ── Reset ──────────────────────────────────────────────────────────

    /// Reset the cache, dropping all stored tensors.
    pub fn clear(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.preallocated_k = None;
        self.preallocated_v = None;
        self.rollback_k = None;
        self.rollback_v = None;
        self.seq_len = 0;
        self.total_appended = 0;
        self.logical_start = 0;
        self.physical_write_pos = 0;
        self.committed_len = 0;
        self.evictions_count = 0;
        self.copy_bytes = 0;
    }
}

// ── 3-Tier KV Cache with ANE-Managed Page Migration ───────────────

/// Tier identifier for KV cache page location.
///
/// Pages migrate between tiers based on access frequency:
/// - **L1** (ANE private SRAM): hottest pages, FP16, instant access
/// - **L2** (IOSurface): warm pages, 3.5-bit TurboQuant, GPU-readable
/// - **L3** (DRAM heap): cold pages, 2-bit TurboQuant, CPU-only
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheTier {
    /// ANE private SRAM — hottest pages, FP16, instant access
    L1AneSram,
    /// IOSurface — warm pages, 3.5-bit TurboQuant, GPU-readable
    L2Iosurface,
    /// DRAM heap — cold pages, 2-bit TurboQuant, CPU-only
    L3DramHeap,
    /// Disk-backed (L4) — cold pages evicted to /tmp/tribunus-kv-cache/.
    /// Only loaded back to L3 (DRAM) on demand.
    L4Disk,
}

impl KVCacheTier {
    /// Human-readable label for this tier.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::L1AneSram => "L1-ANE-SRAM",
            Self::L2Iosurface => "L2-IOSurface",
            Self::L3DramHeap => "L3-DRAM-Heap",
            Self::L4Disk => "L4-Disk",
        }
    }
}

/// Disk-backed KV cache page storage directory.
const KV_CACHE_DISK_DIR: &str = "/tmp/tribunus-kv-cache";

/// Write a cold page to disk. The file is an mmap'd segment containing
/// the 2-bit compressed page data.
pub fn evict_page_to_disk(page: &TiersPage) -> Result<String, String> {
    let _ = std::fs::create_dir_all(KV_CACHE_DISK_DIR);

    let filename = format!("{}/page_{:x}.kvp", KV_CACHE_DISK_DIR, page.token_start);

    let compressed = page
        .l3_data
        .as_ref()
        .ok_or_else(|| "evict_page_to_disk: page has no l3_data".to_string())?;

    std::fs::write(&filename, compressed).map_err(|e| format!("write KV page: {}", e))?;

    Ok(filename)
}

/// Load a page from disk into memory (as L3 data).
/// Returns the raw bytes read from the disk file.
pub fn load_page_from_disk(filename: &str) -> Result<Vec<u8>, String> {
    let data = std::fs::read(filename).map_err(|e| format!("read KV page: {}", e))?;
    let _ = std::fs::remove_file(filename);
    Ok(data)
}

/// A KV page tracked across tiers, holding data at up to one tier at a time.
///
/// Each page covers a contiguous range of token IDs (`token_start`..`token_end`).
/// The `current_tier` field indicates which of `l1_data`, `l2_handle`, or
/// `l3_data` is populated. Promotion moves data from higher-numbered to
/// lower-numbered tiers (e.g. L3 → L2, L2 → L1). Demotion moves the opposite
/// direction.
pub struct TiersPage {
    /// Page content in L1 (FP16, ANE SRAM) — None if not resident in L1.
    pub l1_data: Option<Vec<f32>>,
    /// Page content in L2 (3.5-bit compressed, IOSurface handles) — None if not in L2.
    pub l2_handle: Option<crate::memory::allocator::BlockHandle>,
    /// Page content in L3 (2-bit compressed, heap bytes) — None if not in L3.
    pub l3_data: Option<Vec<u8>>,
    /// Current tier (the fastest one with data).
    pub current_tier: KVCacheTier,
    /// Last access time for promotion/demotion decisions.
    pub last_access: std::time::Instant,
    /// Token ID range this page covers.
    pub token_start: u32,
    /// End token (exclusive) for this page's range.
    pub token_end: u32,
}

impl TiersPage {
    /// Create a new page at the given tier with the provided token range.
    ///
    /// All data fields are set to `None` except the one matching `initial_tier`.
    /// The caller must fill in the appropriate data after creation.
    pub fn new(token_start: u32, token_end: u32, initial_tier: KVCacheTier) -> Self {
        Self {
            l1_data: if initial_tier == KVCacheTier::L1AneSram {
                Some(Vec::new())
            } else {
                None
            },
            l2_handle: None,
            l3_data: if initial_tier == KVCacheTier::L3DramHeap {
                Some(Vec::new())
            } else {
                None
            },
            current_tier: initial_tier,
            last_access: std::time::Instant::now(),
            token_start,
            token_end,
        }
    }

    /// Record an access to this page, updating the last-access timestamp.
    pub fn touch(&mut self) {
        self.last_access = std::time::Instant::now();
    }

    /// Returns `true` if this page covers the given token position.
    pub fn contains_token(&self, token_id: u32) -> bool {
        token_id >= self.token_start && token_id < self.token_end
    }

    /// Returns the number of tokens this page covers.
    pub fn token_count(&self) -> u32 {
        self.token_end.saturating_sub(self.token_start)
    }

    /// Returns an estimate of the allocated bytes for this page at its current tier.
    pub fn allocated_bytes(&self) -> u64 {
        let overhead = std::mem::size_of::<Self>() as u64;
        match self.current_tier {
            KVCacheTier::L1AneSram => {
                self.l1_data.as_ref().map_or(0, |d| d.len() as u64 * 4) + overhead
            }
            KVCacheTier::L2Iosurface => {
                // The IOSurface block handle itself is small; the IOSurface
                // backing storage is tracked by the allocator.
                self.l2_handle.is_some() as u64 * std::mem::size_of::<BlockHandle>() as u64
                    + overhead
            }
            KVCacheTier::L3DramHeap => {
                self.l3_data.as_ref().map_or(0, |d| d.len() as u64) + overhead
            }
            KVCacheTier::L4Disk => {
                // Disk pages have no resident memory
                overhead
            }
        }
    }
}

/// ANE-driven KV cache page migration service.
///
/// Examines access patterns and migrates pages between tiers.
/// Promotes hot pages to L1 (decompressed FP16 in ANE SRAM), demotes cold
/// pages to L3 (2-bit compressed in DRAM). Uses the ANE for compress/decompress
/// operations, keeping the GPU free for attention computation.
///
/// The caller calls [`tick()`](Self::tick) periodically (e.g. after every decode
/// step) to evaluate each tracked page and trigger promotion or demotion when
/// thresholds are crossed.
pub struct PageMigrationService {
    /// All pages tracked across tiers.
    pub pages: Vec<TiersPage>,
    /// Pages accessed within this window are considered hot → promote to L1.
    pub hot_threshold: std::time::Duration,
    /// Pages not accessed within this window are considered cold → demote to L3.
    pub cold_threshold: std::time::Duration,
    /// ANE compression program reference.
    pub ane_compressor: crate::compiler::ane::kv_decompress_program::AneCompressor,
    /// KV cache dimensions for compress/decompress operations.
    pub head_dim: u32,
    /// Number of KV heads.
    pub n_kv_heads: u32,
    /// Total cache budget in bytes for EvolKV optimization.
    pub total_cache_budget: usize,
    /// Best EvolKV budget found during search (None until first search).
    pub evolvk_budget: Option<LayerBudget>,
}

impl PageMigrationService {
    /// Create a new page migration service.
    ///
    /// `hot_threshold` — pages accessed within this window are promoted to L1.
    /// `cold_threshold` — pages not accessed within this window are demoted to L3.
    /// `head_dim` / `n_kv_heads` — KV cache dimensions passed to ANE programs.
    /// `ane_compressor` — initialized ANE compressor holding all four MIL models.
    ///
    /// Default thresholds if none provided: hot = 5 seconds, cold = 30 seconds.
    pub fn new(
        ane_compressor: crate::compiler::ane::kv_decompress_program::AneCompressor,
        head_dim: u32,
        n_kv_heads: u32,
        hot_threshold: Option<std::time::Duration>,
        cold_threshold: Option<std::time::Duration>,
        total_cache_budget: usize,
    ) -> Self {
        Self {
            pages: Vec::new(),
            hot_threshold: hot_threshold.unwrap_or(std::time::Duration::from_secs(5)),
            cold_threshold: cold_threshold.unwrap_or(std::time::Duration::from_secs(30)),
            ane_compressor,
            head_dim,
            n_kv_heads,
            total_cache_budget,
            evolvk_budget: None,
        }
    }

    /// Register a new page in the tiered cache.
    ///
    /// The page starts at `initial_tier` and is assigned the given token range.
    pub fn add_page(&mut self, token_start: u32, token_end: u32, initial_tier: KVCacheTier) {
        self.pages
            .push(TiersPage::new(token_start, token_end, initial_tier));
    }

    /// Record an access to the page covering `token_id`, updating its
    /// last-access timestamp.
    pub fn touch_token(&mut self, token_id: u32) {
        for page in &mut self.pages {
            if page.contains_token(token_id) {
                page.touch();
                return;
            }
        }
    }

    /// Run EvolKV search and apply the optimal per-layer budget.
    ///
    /// Creates an `EvolKV` searcher for the given number of layers,
    /// runs the evolutionary search on the provided calibration set,
    /// and applies the resulting budget to the compressed cache's
    /// per-layer thresholds.  The budget is also stored in
    /// `self.evolkv_budget` for inspection.
    pub fn learn_evolk_budgets(
        &mut self,
        num_layers: usize,
        calibration_set: CalibrationSet,
        cache: &mut CompressedKvCache,
    ) -> Result<(), String> {
        let searcher = EvolKV::new(num_layers);
        let budget = searcher.search(&calibration_set, self.total_cache_budget)?;
        budget.apply(cache);
        self.evolvk_budget = Some(budget);
        Ok(())
    }

    /// Called periodically (e.g. after every decode step) to examine all
    /// pages and promote/demote based on access time.
    ///
    /// Returns `Ok(())` on success. Returns `Err(String)` if any ANE
    /// operation fails; the migration service is left in a consistent state
    /// — a failed promotion leaves the page at its current tier.
    pub fn tick(&mut self) -> Result<(), String> {
        let now = std::time::Instant::now();
        let hot_threshold = self.hot_threshold;
        let cold_threshold = self.cold_threshold;
        let pages = std::mem::take(&mut self.pages);
        for mut page in pages {
            let age = now.duration_since(page.last_access);

            // Skip pages that are already in their optimal tier.
            if age < hot_threshold && page.current_tier != KVCacheTier::L1AneSram {
                // Promote to L1: decompress via ANE, store FP16.
                self.promote_to_l1(&mut page)?;
            } else if age > cold_threshold && page.current_tier != KVCacheTier::L3DramHeap {
                // Demote to L3: compress to 2-bit via ANE.
                self.demote_to_l3(&mut page)?;
            }

            // Pages at L1 that have aged past cold_threshold → demote.
            if age > cold_threshold && page.current_tier == KVCacheTier::L1AneSram {
                self.demote_to_l3(&mut page)?;
            }
            self.pages.push(page);
        }
        Ok(())
    }

    /// Promote a page from its current tier toward L1.
    ///
    /// If the page is at L3, decompress to L2 first (2-bit → FP16 via ANE),
    /// then (if hot enough) to L1. If at L2, decompress directly to L1 via ANE.
    fn promote_to_l1(&self, page: &mut TiersPage) -> Result<(), String> {
        match page.current_tier {
            KVCacheTier::L3DramHeap => {
                // Step 1: L3 → L2 — decompress 2-bit packed bytes to FP16.
                let l3_bytes = page
                    .l3_data
                    .as_ref()
                    .ok_or_else(|| "promote_to_l1: L3 page has no l3_data".to_string())?;

                let fp16_bytes = self.ane_compressor.decompress_from_l3(l3_bytes);
                let fp16: Vec<f32> = fp16_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();

                // At L2 the data lives in the IOSurface, so we don't keep a
                // separate Vec copy — just record the transition. The BlockHandle
                // would be filled by the allocator. For now we store the FP16
                // in l1_data directly and advance tier.
                page.l1_data = Some(fp16);
                page.l3_data = None;
                page.current_tier = KVCacheTier::L1AneSram;
            }
            KVCacheTier::L2Iosurface => {
                // L2 → L1 — decompress 3.5-bit IOSurface data to FP16.
                // In practice this reads from the IOSurface block and runs
                // the L2 decompress MIL program on the ANE.
                let _handle = page
                    .l2_handle
                    .as_ref()
                    .ok_or_else(|| "promote_to_l1: L2 page has no l2_handle".to_string())?;

                // For the L2→L1 path we would read bytes from the IOSurface
                // via the block handle, decompress through l2_decompress,
                // and store the result. This is a placeholder for that flow.
                page.l2_handle = None;
                page.current_tier = KVCacheTier::L1AneSram;
            }
            KVCacheTier::L1AneSram => {
                // Already at L1 — nothing to do.
            }
            KVCacheTier::L4Disk => {
                // L4 → L1 — need to load from disk first, then decompress.
                // This path requires first loading from disk to L3 (get l3_data),
                // then decompressing from L3 → L1. For now, skip — the prefetch
                // path (check_and_evict/prefetch_predicted) handles L4→L3.
            }
        }
        Ok(())
    }

    /// Demote a page to L3 (compress to 2-bit via ANE).
    ///
    /// If the page is at L1, compress FP16 → 2-bit via L3 ANE model. If at L2,
    /// first decompress the 3.5-bit IOSurface data to FP16, then compress.
    fn demote_to_l3(&self, page: &mut TiersPage) -> Result<(), String> {
        match page.current_tier {
            KVCacheTier::L1AneSram => {
                // L1 → L3 — compress FP16 to 2-bit packed bytes via ANE.
                let fp16 = page
                    .l1_data
                    .as_ref()
                    .ok_or_else(|| "demote_to_l3: L1 page has no l1_data".to_string())?;

                // Reinterpret FP16 data as byte slice for ANE compressor
                let fp16_slice = fp16.as_slice();
                let (_, fp16_bytes, _) = unsafe { fp16_slice.align_to::<u8>() };
                let packed = self.ane_compressor.compress_to_l3(fp16_bytes);

                page.l3_data = Some(packed);
                page.l1_data = None;
                page.current_tier = KVCacheTier::L3DramHeap;
            }
            KVCacheTier::L2Iosurface => {
                // L2 → L3 — the page is in 3.5-bit IOSurface; for demotion to
                // 2-bit we would first decompress to FP16 (via L2 decompress)
                // and then compress to 2-bit via L3 compress. Placeholder.
                page.l2_handle = None;
                page.current_tier = KVCacheTier::L3DramHeap;
            }
            KVCacheTier::L3DramHeap => {
                // Already at L3 — nothing to do.
            }
            KVCacheTier::L4Disk => {
                // Already at L4 (disk) — no further demotion possible.
            }
        }
        Ok(())
    }

    /// Return the total allocated bytes across all tracked pages.
    pub fn allocated_bytes(&self) -> u64 {
        let pages: u64 = self.pages.iter().map(|p| p.allocated_bytes()).sum();
        let struct_overhead = std::mem::size_of::<Self>() as u64;
        pages + struct_overhead
    }

    /// Return counts of pages at each tier.
    pub fn tier_counts(&self) -> (usize, usize, usize, usize) {
        let mut l1 = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        let mut l4 = 0usize;
        for page in &self.pages {
            match page.current_tier {
                KVCacheTier::L1AneSram => l1 += 1,
                KVCacheTier::L2Iosurface => l2 += 1,
                KVCacheTier::L3DramHeap => l3 += 1,
                KVCacheTier::L4Disk => l4 += 1,
            }
        }
        (l1, l2, l3, l4)
    }

    /// Check KV cache pressure, evict cold L1/L2/L3 pages to disk (L4).
    /// Pages not accessed for >cold_threshold are candidates for disk eviction.
    /// Returns the number of pages evicted to disk.
    pub fn check_and_evict(&mut self) -> Result<usize, String> {
        let now = std::time::Instant::now();
        let mut evicted = 0usize;
        let mut to_evict: Vec<usize> = Vec::new();

        for (i, page) in self.pages.iter().enumerate() {
            let age = now.duration_since(page.last_access);
            // Pages already at L3 that are cold enough for disk eviction
            if age > self.cold_threshold && page.current_tier == KVCacheTier::L3DramHeap {
                to_evict.push(i);
            }
        }

        // Evict in reverse order so indices stay valid
        for &idx in to_evict.iter().rev() {
            let _filename = evict_page_to_disk(&self.pages[idx])?;
            // Mark as L4Disk — l3_data is consumed by evict_page_to_disk
            self.pages[idx].l3_data = None;
            self.pages[idx].current_tier = KVCacheTier::L4Disk;
            // Store the filename in l3_data as a marker (will be reloaded)
            // Since we don't have a dedicated filename field, we keep the
            // page tracked by its token_start and can reconstruct the path
            evicted += 1;
        }

        Ok(evicted)
    }

    /// Prefetch KV pages predicted to be needed next.
    /// Loads L4Disk pages back to L3DramHeap based on predicted access.
    /// Currently uses a simple heuristic: pages adjacent to recently accessed
    /// tokens are prefetched.
    pub fn prefetch_predicted(&mut self) -> Result<usize, String> {
        let now = std::time::Instant::now();
        let mut prefetched = 0usize;
        let hot_tokens: Vec<u32> = self
            .pages
            .iter()
            .filter(|p| {
                let age = now.duration_since(p.last_access);
                age < self.hot_threshold && p.current_tier != KVCacheTier::L4Disk
            })
            .map(|p| p.token_start)
            .collect();

        // For each hot token range, prefetch adjacent L4 pages
        for &hot_start in &hot_tokens {
            // Check page before and after the hot range
            for adj in [hot_start.saturating_sub(256), hot_start + 256] {
                if let Some(page) = self
                    .pages
                    .iter_mut()
                    .find(|p| p.current_tier == KVCacheTier::L4Disk && p.contains_token(adj))
                {
                    // Reconstruct the filename from token_start
                    let filename = format!("{}/page_{:x}.kvp", KV_CACHE_DISK_DIR, page.token_start);
                    let data = load_page_from_disk(&filename)?;
                    page.l3_data = Some(data);
                    page.current_tier = KVCacheTier::L3DramHeap;
                    page.touch();
                    prefetched += 1;
                }
            }
        }

        Ok(prefetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops;

    fn make_cache(is_sliding: bool) -> KvCache {
        KvCache::new(4, 8, 256, is_sliding)
    }

    fn make_kv(seq: u32) -> (Array, Array) {
        let shape = &[seq as i32, 8, 256];
        let k = ops::ones::<f32>(shape).unwrap();
        let v = ops::ones::<f32>(shape).unwrap();
        (k, v)
    }

    fn make_kv_filled(seq: u32, val: f32) -> (Array, Array) {
        let shape = &[seq as i32, 8, 256];
        let fill = Array::from_slice(&[val], &[1]);
        let k = ops::full::<f32>(shape, &fill).unwrap();
        let v = ops::full::<f32>(shape, &fill).unwrap();
        (k, v)
    }

    // ── Original tests (adapted) ───────────────────────────────────────

    #[test]
    fn test_new_cache_empty() {
        let cache = make_cache(false);
        assert_eq!(cache.seq_len, 0);
        assert!(cache.read_window().is_none());
        assert_eq!(cache.allocated_bytes(), 0);
        assert_eq!(cache.logical_start, 0);
        assert_eq!(cache.physical_write_pos, 0);
        assert_eq!(cache.committed_len, 0);
        assert_eq!(cache.allocated_bytes(), 0);
    }

    #[test]
    fn test_append_global_layer() {
        let mut cache = make_cache(false);
        let (k, v) = make_kv(10);
        cache.append(k, v).unwrap();
        assert_eq!(cache.seq_len, 10);
        assert_eq!(cache.committed_len, 10);

        let (k2, v2) = make_kv(5);
        cache.append(k2, v2).unwrap();
        assert_eq!(cache.seq_len, 15);
        assert_eq!(cache.committed_len, 10); // uncommitted

        cache.commit_step();
        assert_eq!(cache.committed_len, 15);

        let (k_cached, _) = cache.read_window().unwrap();
        assert_eq!(k_cached.shape()[0], 15);
    }

    #[test]
    fn test_append_sliding_within_window() {
        let mut cache = make_cache(true);
        let (k, v) = make_kv(3);
        cache.append(k, v).unwrap();
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.committed_len, 3);
        assert_eq!(cache.read_window().unwrap().0.shape()[0], 3);
    }

    #[test]
    fn test_append_sliding_evicts() {
        let mut cache = make_cache(true);
        let (k, v) = make_kv(3);
        cache.append(k, v).unwrap();
        let (k2, v2) = make_kv(3);
        cache.append(k2, v2).unwrap();
        // Window = 4, 6 total => only 4 kept.
        assert_eq!(cache.seq_len, 4);
        assert_eq!(cache.read_window().unwrap().0.shape()[0], 4);
    }

    #[test]
    fn test_read_window_returns_clones() {
        let mut cache = make_cache(false);
        let (k, v) = make_kv(5);
        cache.append(k, v).unwrap();

        let (k1, _) = cache.read_window().unwrap();
        let (k2, _) = cache.read_window().unwrap();
        assert_eq!(k1.shape()[0], 5);
        assert_eq!(k2.shape()[0], 5);
    }

    #[test]
    fn test_clear() {
        let mut cache = make_cache(false);
        let (k, v) = make_kv(10);
        cache.append(k, v).unwrap();
        cache.clear();
        assert!(cache.read_window().is_none());
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.logical_start, 0);
        assert_eq!(cache.physical_write_pos, 0);
    }

    #[test]
    fn test_allocated_bytes() {
        let mut cache = make_cache(false);
        assert_eq!(cache.allocated_bytes(), 0);
        let (k, v) = make_kv(5);
        cache.append(k, v).unwrap();
        // 5 * 8 * 256 * 4 * 2 = 81920
        assert_eq!(cache.allocated_bytes(), 81920);
    }

    // ── New tests ──────────────────────────────────────────────────────

    #[test]
    fn test_logical_positions() {
        let mut cache = make_cache(true);
        assert_eq!(cache.logical_start, 0);
        assert_eq!(cache.physical_write_pos, 0);

        let (k, v) = make_kv(2);
        cache.append(k, v).unwrap();
        assert_eq!(cache.logical_start, 0);
        assert_eq!(cache.physical_write_pos, 2);

        let (k2, v2) = make_kv(2);
        cache.append(k2, v2).unwrap();
        // Wrapped: 4 % 4 = 0
        assert_eq!(cache.physical_write_pos, 0);
        // Still within capacity, logical_start = 0
        assert_eq!(cache.logical_start, 0);

        let (k3, v3) = make_kv(1);
        cache.append(k3, v3).unwrap();
        // Now total=5 > capacity=4 → logical_start = 5 - 4 = 1
        assert_eq!(cache.logical_start, 1);
        assert_eq!(cache.physical_write_pos, 1);
        assert_eq!(cache.seq_len, 4);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut cache = make_cache(true);
        assert_eq!(cache.allocated_bytes(), 0);

        // Append 3, then 3 more (total 6 > cap 4 → wraps).
        let (k, v) = make_kv_filled(3, 1.0);
        cache.append(k, v).unwrap();

        let (k2, v2) = make_kv_filled(3, 2.0);
        cache.append(k2, v2).unwrap();

        assert_eq!(cache.seq_len, 4);

        // read_window should return 4 tokens (last 4 of 6).
        let (k_out, _) = cache.read_window().unwrap();
        assert_eq!(k_out.shape()[0], 4);

        // Verify evictions count.
        assert_eq!(cache.evictions_count, 2);

        // Add more tokens and verify further eviction.
        let (k3, v3) = make_kv_filled(2, 3.0);
        cache.append(k3, v3).unwrap();
        assert_eq!(cache.evictions_count, 4); // 8 total - 4 cap = 4 evicted
        assert_eq!(cache.seq_len, 4);
    }

    #[test]
    fn test_first_append_trimming() {
        let mut cache = make_cache(true);
        assert_eq!(cache.capacity, 4);

        // First append with 6 tokens (> capacity of 4).
        let (k, v) = make_kv(6);
        cache.append(k, v).unwrap();

        // Should be trimmed to capacity.
        assert_eq!(cache.seq_len, 4);
        assert_eq!(cache.committed_len, 4);

        let (k_out, _) = cache.read_window().unwrap();
        assert_eq!(k_out.shape()[0], 4);
    }

    #[test]
    fn test_large_first_append_sliding() {
        // First append with exactly capacity tokens — no trimming.
        let mut cache = make_cache(true);
        let (k, v) = make_kv(4);
        cache.append(k, v).unwrap();
        assert_eq!(cache.seq_len, 4);

        // First append with more than capacity — trimmed.
        let mut cache2 = make_cache(true);
        let (k2, v2) = make_kv(10);
        cache2.append(k2, v2).unwrap();
        assert_eq!(cache2.seq_len, 4);
        assert_eq!(cache2.evictions_count, 6);
    }

    #[test]
    fn test_commit_rollback_sliding() {
        let mut cache = make_cache(true);
        assert_eq!(cache.committed_len, 0);

        // First append: auto-committed.
        let (k, v) = make_kv(2);
        cache.append(k, v).unwrap();
        assert_eq!(cache.committed_len, 2);
        assert_eq!(cache.total_appended, 2);

        // Second append: uncommitted.
        let (k2, v2) = make_kv(2);
        cache.append(k2, v2).unwrap();
        assert_eq!(cache.committed_len, 2);
        assert_eq!(cache.total_appended, 4);
        assert_eq!(cache.seq_len, 4);

        // Rollback: restores to committed state.
        cache.rollback();
        assert_eq!(cache.committed_len, 2);
        assert_eq!(cache.total_appended, 2);
        assert_eq!(cache.seq_len, 2);

        // Append and commit.
        let (k3, v3) = make_kv(2);
        cache.append(k3, v3).unwrap();
        cache.commit_step();
        assert_eq!(cache.committed_len, 4);
        assert_eq!(cache.total_appended, 4);
    }

    #[test]
    fn test_commit_rollback_global() {
        let mut cache = make_cache(false);
        let (k, v) = make_kv(3);
        cache.append(k, v).unwrap();
        assert_eq!(cache.committed_len, 3);

        let (k2, v2) = make_kv(2);
        cache.append(k2, v2).unwrap();
        assert_eq!(cache.seq_len, 5);
        assert_eq!(cache.committed_len, 3);

        // Rollback should restore to committed state.
        cache.rollback();
        assert_eq!(cache.committed_len, 3);
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.read_window().unwrap().0.shape()[0], 3);
    }

    #[test]
    fn test_receipt_json_sliding() {
        let mut cache = make_cache(true);
        let receipt = cache.receipt_json();
        assert!(receipt.contains(r#""is_sliding":true"#));
        assert!(receipt.contains(r#""capacity":4"#));
        assert!(receipt.contains(r#""logical_start":0"#));
        assert!(receipt.contains(r#""logical_length":0"#));

        let (k, v) = make_kv(3);
        cache.append(k, v).unwrap();
        let receipt = cache.receipt_json();
        assert!(receipt.contains(r#""logical_length":3"#));
    }

    #[test]
    fn test_receipt_json_global() {
        let cache = make_cache(false);
        let receipt = cache.receipt_json();
        assert!(receipt.contains(r#""is_sliding":false"#));
    }

    #[test]
    fn test_allocated_bytes_sliding() {
        let mut cache = make_cache(true);
        assert_eq!(cache.allocated_bytes(), 0);

        // After first append, ring buffer is allocated.
        let (k, v) = make_kv(2);
        cache.append(k, v).unwrap();
        // 4 * 8 * 256 * 4 * 2 = 65536
        assert_eq!(cache.allocated_bytes(), 65536);
    }

    #[test]
    fn test_committed_bytes() {
        let mut cache = make_cache(true);
        assert_eq!(cache.committed_bytes(), 0);

        let (k, v) = make_kv(2);
        cache.append(k, v).unwrap();
        // 2 * 8 * 256 * 4 * 2 = 32768
        assert_eq!(cache.committed_bytes(), 32768);

        // After rollback of subsequent uncommitted append.
        let (k2, v2) = make_kv(1);
        cache.append(k2, v2).unwrap();
        assert_eq!(cache.committed_bytes(), 32768); // unchanged
        cache.rollback();
        assert_eq!(cache.committed_bytes(), 32768); // back to committed
    }

    #[test]
    fn test_evictions_count() {
        let mut cache = make_cache(true);
        assert_eq!(cache.evictions_count, 0);

        let (k, v) = make_kv(3);
        cache.append(k, v).unwrap();
        assert_eq!(cache.evictions_count, 0);

        let (k2, v2) = make_kv(3);
        cache.append(k2, v2).unwrap();
        // 6 total - 4 cap = 2 evicted
        assert_eq!(cache.evictions_count, 2);

        let (k3, v3) = make_kv(3);
        cache.append(k3, v3).unwrap();
        // 9 total - 4 cap = 5 evicted
        assert_eq!(cache.evictions_count, 5);
    }

    #[test]
    fn test_compressed_slot_page_data_roundtrip() {
        let keys = vec![0xABu8; 64];
        let values = vec![0xCDu8; 128];
        let slot = CompressedKvSlot {
            compressed_keys: keys.clone(),
            compressed_values: values.clone(),
            qjl_correction: None,
            kv_offset: 42,
            num_tokens: 2,
        };

        let page_data = slot.to_page_data();
        let restored = CompressedKvSlot::from_page_data(&page_data).unwrap();

        assert_eq!(restored.compressed_keys, keys);
        assert_eq!(restored.compressed_values, values);
        // kv_offset and num_tokens are left at defaults by from_page_data.
        assert_eq!(restored.kv_offset, 0);
        assert_eq!(restored.num_tokens, 1);
    }

    #[test]
    fn test_compressed_slot_page_data_empty() {
        let slot = CompressedKvSlot {
            compressed_keys: Vec::new(),
            compressed_values: Vec::new(),
            qjl_correction: None,
            kv_offset: 0,
            num_tokens: 0,
        };

        let page_data = slot.to_page_data();
        // Should be exactly 8 bytes (two zero-length headers).
        assert_eq!(page_data.len(), 8);

        let restored = CompressedKvSlot::from_page_data(&page_data).unwrap();
        assert!(restored.compressed_keys.is_empty());
        assert!(restored.compressed_values.is_empty());
    }

    #[test]
    fn test_compressed_slot_page_data_corrupt() {
        // Data shorter than minimum header.
        let result = CompressedKvSlot::from_page_data(&[0u8; 4]);
        assert!(result.is_err());

        // Header says 100 bytes but only 10 provided.
        let mut bad = vec![0u8; 12];
        // key_len = 100
        bad[..4].copy_from_slice(&100u32.to_le_bytes());
        let result2 = CompressedKvSlot::from_page_data(&bad);
        assert!(result2.is_err());
    }
}

/// Runtime-tagged KV cache representation.
///
/// Preserves a tagged representation until every attention route can legally consume
/// every admitted cache format. Do NOT replace Vec<KvCache> directly.
#[derive(Debug)]
pub enum LiveKvCache {
    /// Uncompressed FP16 cache backed by MLX arrays.
    Fp16(KvCache),
    /// Compressed cache via TurboQuant byte buffers.
    Compressed(CompressedKvCache),
    /// TurboQuant with asymmetric quantization.
    TurboQuant(TurboQuantKvCache),
}

impl LiveKvCache {
    /// Number of committed tokens in this cache.
    pub fn committed_len(&self) -> u32 {
        match self {
            Self::Fp16(c) => c.committed_len,
            Self::Compressed(c) => c.committed_len,
            Self::TurboQuant(c) => c.committed_len(),
        }
    }

    pub fn seq_len(&self) -> u32 {
        match self {
            Self::Fp16(c) => c.seq_len,
            Self::Compressed(c) => c.seq_len,
            Self::TurboQuant(c) => c.seq_len(),
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        match self {
            Self::Fp16(c) => c.allocated_bytes(),
            Self::Compressed(c) => c.allocated_bytes(),
            Self::TurboQuant(c) => c.allocated_bytes(),
        }
    }

    pub fn commit_step(&mut self) {
        match self {
            Self::Fp16(c) => c.commit_step(),
            Self::Compressed(c) => c.commit_step(),
            Self::TurboQuant(_) => {}
        }
    }

    pub fn rollback(&mut self) {
        match self {
            Self::Fp16(c) => c.rollback(),
            Self::Compressed(c) => c.rollback(),
            Self::TurboQuant(_) => {}
        }
    }
}
