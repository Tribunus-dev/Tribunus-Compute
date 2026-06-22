//! CPU Attention Scheduler — L2-cache-aware work-partition + work-stealing.
//!
//! Ported from vLLM's `cpu_attn_impl.hpp` [`AttentionScheduler`] and
//! [`AttentionMainLoop`] patterns.  The scheduler partitions QKV attention
//! work into tiles that fit in L2 cache; each work item is stolen atomically
//! by a Rayon thread.
//!
//! # Feature gate
//!
//! `#[cfg(feature = "candle-cpu")]` – compiled only when the candle-cpu
//! backend is active.

// ── Constants ──────────────────────────────────────────────────────────────

/// Maximum Q-head tile iterations within one work item group.
const MAX_Q_TILE_ITER_NUM: usize = 128;

/// Default block-size alignment (token count multiples).  Matches vLLM's
/// `blocksize_alignment` for the VEC ISA path.
const BLOCK_SIZE_ALIGNMENT: usize = 16;

// ── L2 cache detection ─────────────────────────────────────────────────────

/// Returns half of the L2 cache size (the portion the scheduler is allowed
/// to use for tile buffers), falling back to 256 KiB if undetectable.
///
/// Uses platform-specific system calls:
/// - macOS / Apple Silicon: `sysctl "hw.l2cachesize"`
/// - Linux / x86-64       : `sysconf _SC_LEVEL2_CACHE_SIZE`
fn get_available_l2_size() -> usize {
    #[cfg(target_os = "macos")]
    {
        let mut size: usize = 0;
        let mut mib: [libc::c_int; 2] = [libc::CTL_HW, libc::HW_L2CACHESIZE];
        let mut len = std::mem::size_of::<usize>();
        // Safe: sysctl writes into a valid `size` and doesn't touch memory
        // beyond `len`.  We pass pointers to stack variables.
        let rc = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                &mut size as *mut _ as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && size > 0 {
            return size >> 1; // use 50 %
        }
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL2_CACHE_SIZE) };
        if size > 0 {
            return (size as usize) >> 1;
        }
    }

    // Fallback – 512 KiB L2 → 256 KiB usable.
    256 * 1024
}

// ── Data types ─────────────────────────────────────────────────────────────

/// A single unit of CPU-attention work that can be stolen by a parallel
/// thread.
#[derive(Debug, Clone)]
pub struct AttentionWorkItem {
    /// Index of the request / batch entry.
    pub request_id: usize,
    /// Range of Q heads to process: `[start, end)`.
    pub q_head_range: (usize, usize),
    /// Total KV sequence length (tokens) this item covers.
    pub kv_seq_len: usize,
    /// Start offset into the KV cache.
    pub kv_start: usize,
    /// End offset (exclusive) into the KV cache.
    pub kv_end: usize,
}

/// The complete schedule produced by [`CpuAttentionScheduler`].
#[derive(Debug, Clone)]
pub struct AttentionScheduleResult {
    /// All work items, to be stolen by the thread pool.
    pub work_items: Vec<AttentionWorkItem>,
    /// Partition of `work_items` index ranges per thread for static
    /// fallback scheduling.
    pub work_item_ranges: Vec<(usize, usize)>,
    /// Number of effective threads (threads that received at least one
    /// work item).
    pub effective_threads: usize,
    /// L2 cache size estimate used during tiling (bytes).
    pub l2_cache_estimate: usize,
    /// Default tile token count for this schedule.
    pub default_tile_tokens: usize,
    /// Max Q heads per iteration (constrained by register pressure).
    pub max_q_heads_per_iter: usize,
}

// ── L2-cache-aware scheduler ───────────────────────────────────────────────

/// CPU Attention Scheduler — L2-cache-aware work partition for CPU attention.
///
/// Ported from vLLM's `cpu_attn_impl.hpp` [`AttentionScheduler`] pattern.
///
/// The scheduler:
/// 1. Splits Q heads into groups sized so that Q, K, V tiles and logits
///    intermediate buffers all fit inside L2 cache (50 % of available L2).
/// 2. Creates [`AttentionWorkItem`] values that each represent one tile.
/// 3. The work items are designed to be stolen atomically via
///    [`AtomicUsize`] counter by Rayon parallel iterators.
#[derive(Debug, Clone)]
pub struct CpuAttentionScheduler {
    batch_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    l2_cache_size: usize,
    max_q_heads_per_iter: usize,
    enable_kv_split: bool,
}

impl CpuAttentionScheduler {
    /// Create a new scheduler.
    ///
    /// * `batch_size`   — number of requests in the batch.
    /// * `num_q_heads`  — total query heads (per-token).
    /// * `num_kv_heads` — total key/value heads.
    /// * `head_dim`     — dimension of each head.
    /// * `l2_cache`     — optional override for the L2 cache size (bytes).
    ///   When `None` the scheduler probes the system.
    /// * `max_q_heads_per_iter` — max Q heads that can be held in registers
    ///   per iteration (defaults to `num_q_heads` / `num_kv_heads`, clamped
    ///   to 8).
    /// * `enable_kv_split` — when `true`, the scheduler may split KV
    ///   sequences across threads for very long contexts.
    pub fn new(
        batch_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        l2_cache: Option<usize>,
        max_q_heads_per_iter: Option<usize>,
        enable_kv_split: bool,
    ) -> Self {
        let l2_cache_size = l2_cache.unwrap_or_else(get_available_l2_size);
        // vLLM limits `max_num_q_per_iter` based on register pressure; the
        // common default is `q_heads_per_kv` (for GQA) clamped to at most 8.
        let q_per_kv = num_q_heads / num_kv_heads;
        let max_q_heads_per_iter = max_q_heads_per_iter.unwrap_or(q_per_kv.clamp(1, 8));

        Self {
            batch_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            l2_cache_size,
            max_q_heads_per_iter,
            enable_kv_split,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /// Calculate the default tile token count (== Q tile == K tile tokens
    /// that together fit in L2 cache).
    ///
    /// The cache must hold:
    ///   - Q tile:     `tile_size × head_dim × 4` (f32 Q buffer)
    ///   - K cache:    `tile_size × head_dim × elem_size`
    ///   - V cache:    `tile_size × head_dim × elem_size`
    ///   - Q@K^T:      `max_q_heads_per_iter × tile_size × 4` (f32 logits)
    ///   - Partial O:   `tile_size × head_dim × 4`
    ///
    /// By default we assume f32 (`elem_size = 4`).
    fn calc_default_tile_tokens(&self) -> usize {
        let cache = self.l2_cache_size;
        let hd = self.head_dim;
        let elem = 4; // f32
        let q_buf_elem = 4; // f32 Q buffer
        let logits_elem = 4; // f32 logits
        let out_elem = 4; // f32 partial output

        // denominator: Q_buf + K + V + output (per head-dim)
        let denom =
            hd * (q_buf_elem + 2 * elem + out_elem) + self.max_q_heads_per_iter * logits_elem;

        if denom == 0 {
            return BLOCK_SIZE_ALIGNMENT;
        }

        let tile = cache / denom;
        let tile = tile.min(MAX_Q_TILE_ITER_NUM * self.max_q_heads_per_iter);
        // Round down to alignment
        let tile = (tile / BLOCK_SIZE_ALIGNMENT) * BLOCK_SIZE_ALIGNMENT;
        tile.max(BLOCK_SIZE_ALIGNMENT)
    }

    /// Calculate KV tile tokens given a known Q tile token count.
    fn calc_kv_tile_tokens(&self, q_tile_tokens: usize, one_round: bool) -> usize {
        let cache = self.l2_cache_size;
        let hd = self.head_dim;
        let elem = 4; // f32

        let q_tile_cost = q_tile_tokens * hd * (4 /* Q buf */ + 4/* output */);

        let remaining = if cache > q_tile_cost {
            cache - q_tile_cost
        } else {
            // Degenerate: use a minimal tile.
            return BLOCK_SIZE_ALIGNMENT;
        };

        let denom = if one_round {
            self.max_q_heads_per_iter * 4 /* logits elem */
        } else {
            self.max_q_heads_per_iter * 4 /* logits elem */ + 2 * hd * elem
        };

        if denom == 0 {
            return BLOCK_SIZE_ALIGNMENT;
        }

        let tile = remaining / denom;
        let tile = (tile / BLOCK_SIZE_ALIGNMENT) * BLOCK_SIZE_ALIGNMENT;
        tile.max(BLOCK_SIZE_ALIGNMENT)
    }

    /// Clamp KV position bounds according to causal/sliding-window rules.
    fn calc_kv_tile_bounds(
        kv_start: usize,
        kv_end: usize,
        q_start: usize,
        q_end: usize,
        window_size: Option<usize>,
        causal: bool,
    ) -> (usize, usize) {
        let mut left = kv_start;
        let mut right = kv_end;

        if let Some(ws) = window_size {
            let left_limit = if q_start > ws { q_start - ws } else { 0 };
            if left < left_limit {
                left = left_limit;
            }
            if causal {
                // causal: right bound is current Q position
                if right > q_end {
                    right = q_end;
                }
            } else if let Some(ws) = window_size {
                let right_limit = q_end + ws;
                if right > right_limit {
                    right = right_limit;
                }
            }
        } else if causal {
            if right > q_end {
                right = q_end;
            }
        }

        (left, right)
    }

    /// Align a KV position down to the block-alignment boundary.
    fn align_kv_down(pos: usize) -> usize {
        (pos / BLOCK_SIZE_ALIGNMENT) * BLOCK_SIZE_ALIGNMENT
    }

    /// Align a KV position up to the block-alignment boundary.
    fn align_kv_up(pos: usize) -> usize {
        ((pos + BLOCK_SIZE_ALIGNMENT - 1) / BLOCK_SIZE_ALIGNMENT) * BLOCK_SIZE_ALIGNMENT
    }

    // ── Main scheduling ──────────────────────────────────────────────

    /// Produce a schedule of [`AttentionWorkItem`] values for the given
    /// batch.
    ///
    /// * `seq_lens`       — `batch_size` element slice of each request's
    ///   sequence length (including all KV tokens).
    /// * `query_lengths`  — `batch_size` element slice of each request's
    ///   Q-token count (usually 1 for decode, >1 for prefill).
    /// * `causal`         — whether the batch uses causal masking.
    ///
    /// Returns a schedule that the caller feeds to [`execute_attention`].
    pub fn schedule(
        &self,
        seq_lens: &[usize],
        query_lengths: &[usize],
        causal: bool,
    ) -> AttentionScheduleResult {
        assert_eq!(seq_lens.len(), self.batch_size);
        assert_eq!(query_lengths.len(), self.batch_size);

        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let default_tile_tokens = self.calc_default_tile_tokens();
        let q_per_kv = self.num_q_heads / self.num_kv_heads;
        let max_q_tokens_per_iter = self.max_q_heads_per_iter / q_per_kv;
        let min_split_kv_len = ((self.max_q_heads_per_iter * 4 + BLOCK_SIZE_ALIGNMENT - 1)
            / BLOCK_SIZE_ALIGNMENT)
            * BLOCK_SIZE_ALIGNMENT;

        // Total KV length across all requests (used for load balancing).
        let total_kv_len: usize = seq_lens.iter().sum();

        // KV length per thread target.
        let kv_per_thread = {
            let per = (total_kv_len / thread_count).max(BLOCK_SIZE_ALIGNMENT);
            Self::align_kv_up(per)
        };

        let mut work_items: Vec<AttentionWorkItem> = Vec::with_capacity(1024);
        // Work-item count per thread for the static fallback.
        let mut counts_per_thread: Vec<usize> = vec![0; thread_count];

        // Current thread being filled.
        let mut thread_idx = 0usize;
        let mut remaining_kv = kv_per_thread as i64;

        for req_idx in 0..self.batch_size {
            let seq_len = seq_lens[req_idx];
            let q_tokens = query_lengths[req_idx];
            let q_start = seq_len - q_tokens;

            // Iterate Q tiles.
            let mut q_tile_offset = 0usize;
            while q_tile_offset < q_tokens {
                let q_tile_tokens = (q_tokens - q_tile_offset).min(max_q_tokens_per_iter);
                let q_tile_start = q_start + q_tile_offset;
                let q_tile_end = q_tile_start + q_tile_tokens;

                // Determine KV range for this Q tile (accounting for causal
                // window).
                let (kv_tile_left, kv_tile_right) = Self::calc_kv_tile_bounds(
                    0,
                    seq_len,
                    q_tile_start,
                    q_tile_end,
                    None, // sliding window
                    causal,
                );
                let kv_aligned =
                    Self::align_kv_up(kv_tile_right) - Self::align_kv_down(kv_tile_left);
                let mut kv_remaining = kv_aligned;
                let mut kv_pos = Self::align_kv_down(kv_tile_left);

                while kv_remaining > 0 {
                    // Check if remaining KV fits in the current thread's
                    // budget.
                    let fits = (kv_remaining as i64) <= remaining_kv + min_split_kv_len as i64;
                    let is_last_thread = thread_idx == thread_count - 1;

                    if fits || is_last_thread {
                        // Allocate all remaining KV to one work item.
                        work_items.push(AttentionWorkItem {
                            request_id: req_idx,
                            q_head_range: (0, self.num_q_heads),
                            kv_seq_len: seq_len,
                            kv_start: kv_pos,
                            kv_end: kv_pos + kv_remaining,
                        });
                        counts_per_thread[thread_idx] += 1;
                        remaining_kv -= kv_remaining as i64;

                        // If we're in the middle of a thread and a new Q tile
                        // is coming, let the next thread pick it up.
                        if remaining_kv < -(min_split_kv_len as i64) {
                            remaining_kv = 0;
                        }

                        kv_remaining = 0;
                        break;
                    }

                    // If remaining_kv is too small and this thread already
                    // has work, switch threads.
                    if remaining_kv < min_split_kv_len as i64 && counts_per_thread[thread_idx] > 0 {
                        thread_idx = (thread_idx + 1).min(thread_count - 1);
                        remaining_kv = kv_per_thread as i64;
                        continue;
                    }

                    // Try to split KV if the threshold allows and we're
                    // in a split-eligible region.
                    if self.enable_kv_split
                        && q_tile_tokens == 1
                        && q_tile_offset + max_q_tokens_per_iter >= q_tokens
                    {
                        // Split KV: take what fits in this thread.
                        let split_size = remaining_kv.max(min_split_kv_len as i64) as usize;
                        let split_size = split_size.min(kv_remaining);

                        work_items.push(AttentionWorkItem {
                            request_id: req_idx,
                            q_head_range: (0, self.num_q_heads),
                            kv_seq_len: seq_len,
                            kv_start: kv_pos,
                            kv_end: kv_pos + split_size,
                        });
                        counts_per_thread[thread_idx] += 1;

                        kv_pos += split_size;
                        kv_remaining -= split_size;
                        remaining_kv = kv_per_thread as i64;
                        thread_idx = (thread_idx + 1).min(thread_count - 1);
                    } else {
                        // Just fill the current thread and switch.
                        let fill = remaining_kv.max(0) as usize;
                        if fill > 0 {
                            work_items.push(AttentionWorkItem {
                                request_id: req_idx,
                                q_head_range: (0, self.num_q_heads),
                                kv_seq_len: seq_len,
                                kv_start: kv_pos,
                                kv_end: kv_pos + fill.min(kv_remaining),
                            });
                            counts_per_thread[thread_idx] += 1;
                            kv_pos += fill.min(kv_remaining);
                            kv_remaining -= fill.min(kv_remaining);
                        }

                        remaining_kv = kv_per_thread as i64;
                        thread_idx = (thread_idx + 1).min(thread_count - 1);
                    }
                }

                q_tile_offset += max_q_tokens_per_iter;
            }
        }

        // Build prefix-sum ranges per thread.
        let mut ranges = Vec::with_capacity(thread_count + 1);
        let mut acc = 0usize;
        for c in &counts_per_thread {
            ranges.push(acc);
            acc += c;
        }
        ranges.push(work_items.len());

        let effective = counts_per_thread
            .iter()
            .take_while(|&&c| c > 0)
            .count()
            .max(1);

        AttentionScheduleResult {
            work_items,
            work_item_ranges: (0..effective).map(|i| (ranges[i], ranges[i + 1])).collect(),
            effective_threads: effective,
            l2_cache_estimate: self.l2_cache_size,
            default_tile_tokens: default_tile_tokens,
            max_q_heads_per_iter: self.max_q_heads_per_iter,
        }
    }

    /// Execute the scheduled attention work items using work-stealing
    /// parallelism via Rayon.
    ///
    /// The caller provides buffers for Q, K, V cache and output, plus a
    /// closure that processes one work item.  The closure receives:
    ///
    ///   `(work_item: &AttentionWorkItem, thread_id: usize)`
    ///
    /// and writes partial attention results into the output buffer.
    ///
    /// This is the work-stealing main loop analogous to vLLM's
    /// `#pragma omp parallel for schedule(static, 1)` + atomic counter.
    pub fn execute_attention<F>(schedule: &AttentionScheduleResult, work_fn: F)
    where
        F: Fn(&AttentionWorkItem, usize) + Send + Sync,
    {
        let work_items = &schedule.work_items;

        if work_items.is_empty() {
            return;
        }

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Atomic counter for work-stealing (ported from vLLM's
        // `omp parallel for schedule(static, 1)` + atomic counter pattern).
        let counter = AtomicUsize::new(0);
        let counter_ref: &AtomicUsize = &counter;
        let work_fn = Arc::new(work_fn);

        rayon::scope(|s| {
            let num_threads = rayon::current_num_threads();
            for _ in 0..num_threads {
                let wf = Arc::clone(&work_fn);
                // Only `wf` (Arc) is moved; `counter_ref` and `work_items`
                // are shared references that implement `Copy`.
                s.spawn(move |_| loop {
                    let idx = counter_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= work_items.len() {
                        break;
                    }
                    wf(&work_items[idx], idx);
                });
            }
        });
    }

    /// Compute a default tile token count for external use, based on the
    /// configured head / cache parameters.
    pub fn default_tile_tokens(&self) -> usize {
        self.calc_default_tile_tokens()
    }

    /// Return the configured L2 cache size estimate.
    pub fn l2_cache_size(&self) -> usize {
        self.l2_cache_size
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_cache_detection() {
        let size = get_available_l2_size();
        // Should be at least 16 KiB on any real system.
        assert!(size >= 16 * 1024, "L2 estimate suspiciously small: {size}");
    }

    #[test]
    fn test_schedule_basic_prefill() {
        let scheduler = CpuAttentionScheduler::new(
            1,                // batch size
            32,               // num_q_heads
            8,                // num_kv_heads
            128,              // head_dim
            Some(512 * 1024), // fake 512 KiB L2
            None,             // default max_q_heads_per_iter
            false,            // no kv split
        );

        let schedule = scheduler.schedule(
            &[4096], // seq_lens
            &[4096], // query_lengths (full prefill)
            true,    // causal
        );

        assert!(!schedule.work_items.is_empty());
        assert!(schedule.effective_threads >= 1);
        assert!(schedule.default_tile_tokens >= BLOCK_SIZE_ALIGNMENT);
    }

    #[test]
    fn test_schedule_decode_batch() {
        let scheduler = CpuAttentionScheduler::new(
            4, // 4 requests
            32,
            8,
            128,
            Some(512 * 1024),
            None,
            false,
        );

        // Decode: each request has seq_len >> query_lengths (1 token each).
        let schedule = scheduler.schedule(&[1024, 2048, 4096, 8192], &[1, 1, 1, 1], true);

        assert!(!schedule.work_items.is_empty());
        assert_eq!(schedule.effective_threads, schedule.work_item_ranges.len());
    }

    #[test]
    fn test_execute_attention_work_stealing() {
        let scheduler = CpuAttentionScheduler::new(2, 8, 4, 64, Some(256 * 1024), None, false);

        let schedule = scheduler.schedule(&[128, 256], &[128, 256], true);

        // Count how many times each work item is visited.
        let visited = std::sync::atomic::AtomicUsize::new(0);

        CpuAttentionScheduler::execute_attention(&schedule, |_item, _tid| {
            visited.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });

        assert_eq!(
            visited.load(std::sync::atomic::Ordering::Relaxed),
            schedule.work_items.len()
        );
    }

    #[test]
    fn test_default_tile_tokens_nonzero() {
        let scheduler = CpuAttentionScheduler::new(1, 32, 8, 128, None, None, false);
        assert!(scheduler.default_tile_tokens() > 0);
    }

    #[test]
    fn test_kv_tile_bounds_causal() {
        let (l, r) = CpuAttentionScheduler::calc_kv_tile_bounds(
            0, 100, // kv range
            50, 60,   // q range
            None, // no sliding window
            true, // causal
        );
        // KV right bound should be clamped to Q end (causal masking).
        assert_eq!(l, 0);
        assert_eq!(r, 60);
    }

    #[test]
    fn test_kv_tile_bounds_sliding_window() {
        let (l, r) = CpuAttentionScheduler::calc_kv_tile_bounds(
            0,
            100,
            50,
            60,
            Some(32), // 32-token window
            true,     // causal
        );
        // Left bound should be max(0, 50 - 32) = 18
        assert_eq!(l, 18);
        // Right bound is Q end (causal)
        assert_eq!(r, 60);
    }
}
