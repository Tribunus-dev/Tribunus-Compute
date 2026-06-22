use super::{
    saved_request::{SavedRequest, MAX_PREEMPTIONS_BEFORE_BOOST, STARVATION_PRIORITY_BOOST},
    Batch, HardwareConfig, Request, SchedulerConfig, Slot,
};
use crate::backend::routing::ComputeRouteProfile;
use crate::kv_cache::CompressedKvSlot;
use crate::memory::allocator::{IosurfaceAllocator, PagedIosurfaceAllocator};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Continuous batching scheduler
///
/// Implements the scheduling loop from ref/omlx/scheduler.py:
/// 1. Poll for new requests
/// 2. Build prefill batch
/// 3. Build decode batch
/// 4. Process results
/// 5. Check memory
#[allow(dead_code)]
pub struct Scheduler {
    queue: Vec<Request>,
    active: Vec<Request>,
    slots: Vec<Slot>,
    config: SchedulerConfig,
    /// Maximum concurrent in-flight sequences (batch x decode iterations).
    max_concurrent: usize,
    /// Maximum sequence length (context window) in tokens.
    max_seq_len: usize,
    /// Whether to batch multiple prefill prompts into one batched forward pass.
    batch_prefills: bool,
    /// Whether to batch multiple decode steps into one forward pass.
    batch_decodes: bool,
    /// Maximum number of prompts to merge in a single prefill batch.
    max_prefill_batch: usize,
    route_profile: Option<ComputeRouteProfile>,
    kv_cache_allocator: Option<Arc<Mutex<IosurfaceAllocator>>>,
    kv_cache_pager: Option<PagedIosurfaceAllocator>,
    /// Preempted requests awaiting resume.
    preempted: Vec<SavedRequest>,
    /// Tracks how many times each request has been preempted.
    starvation_counters: HashMap<u64, usize>,
    /// Whether preemption is enabled. Defaults to true.
    preemption_enabled: bool,
}

impl Scheduler {
    /// Create a new scheduler with the given config.
    pub fn new(config: SchedulerConfig) -> Self {
        let config_for_slots = config.clone();
        let slots = (0..config_for_slots.max_batch_size)
            .map(|id| Slot {
                id,
                request_id: None,
                tokens_generated: 0,
                kv_cache_start: 0,
                kv_cache_length: 0,
                backend_id: config_for_slots.default_backend_id,
                kv_cache_pages: vec![],
            })
            .collect();
        Self {
            queue: Vec::new(),
            active: Vec::new(),
            slots,
            max_prefill_batch: config.max_prefill_batch,
            max_concurrent: config.max_batch_size,
            config,
            max_seq_len: 4096,
            batch_prefills: false,
            batch_decodes: false,
            route_profile: None,
            kv_cache_allocator: None,
            kv_cache_pager: None,
            preempted: Vec::new(),
            starvation_counters: HashMap::new(),
            preemption_enabled: true,
        }
    }

    /// Configure scheduling parameters for the detected hardware.
    ///
    /// On M3 Ultra (memory-rich): enables prefill batching, decode batching,
    /// 256K context, 32-batch prefill, and 64 concurrent sequences.
    pub fn configure_for_hardware(&mut self, hw: &HardwareConfig) {
        self.config.max_batch_size = hw.recommended_batch_size as usize;
        self.max_concurrent = hw.max_concurrent_sequences as usize;
        self.max_seq_len = 262_144; // 256K context window

        if hw.is_memory_rich {
            // Prefill batching: concatenate multiple prefill prompts into
            // one batched forward pass. MLX handles this naturally by
            // stacking sequences along the batch dimension.
            self.batch_prefills = true;
            self.max_prefill_batch = hw.recommended_batch_size as usize;

            // Decode batching: run multiple decode steps in a single
            // forward pass by batching across sequences.
            self.batch_decodes = true;
        }
    }

    /// Enqueue a new request into the scheduler.
    pub fn enqueue(&mut self, request: Request) {
        self.queue.push(request);
    }

    /// Set the compute route profile for deterministic backend routing.
    pub fn set_route_profile(&mut self, profile: ComputeRouteProfile) {
        self.route_profile = Some(profile);
    }

    /// Return a mutable reference to the scheduler configuration.
    pub(crate) fn config_mut(&mut self) -> &mut SchedulerConfig {
        &mut self.config
    }

    /// Set the KV cache allocator for IOSurface-backed arena allocation.
    pub fn set_kv_cache_allocator(&mut self, allocator: Arc<Mutex<IosurfaceAllocator>>) {
        self.kv_cache_allocator = Some(allocator);
    }

    /// Set the paged KV cache allocator for IOSurface-backed page allocation.
    pub fn set_kv_cache_pager(&mut self, pager: PagedIosurfaceAllocator) {
        self.kv_cache_pager = Some(pager);
    }

    /// Build the next batch to execute.
    ///
    /// Polls queued requests into the active set (respecting max_batch_size and
    /// max_prefill_batch limits), then either:
    /// - Assigns free slots for prefill (new requests with no slot), or
    /// - Extends KV cache lengths by 1 for decode (all active slots already assigned).
    pub fn next_batch(&mut self) -> Batch {
        // Sort queued requests by priority ascending so pop() yields highest priority
        self.queue.sort_by(|a, b| a.priority.cmp(&b.priority));

        // Poll: move queued requests to active within batch and prefill limits
        while self.active.len() < self.config.max_batch_size
            && self.active.len() < self.config.max_prefill_batch
        {
            if let Some(req) = self.queue.pop() {
                self.active.push(req);
            } else {
                break;
            }
        }

        // Count active requests without a slot assigned — these need prefill
        let prefill_count = self.active.iter().filter(|r| r.slot.is_none()).count();

        if prefill_count > 0 {
            // Ensure enough free slots exist before assigning
            let free_count = self.slots.iter().filter(|s| s.is_free()).count();
            if free_count < prefill_count {
                self.add_slots(prefill_count - free_count);
            }

            // Prefill: find free slots and assign them to active requests
            for req in self.active.iter_mut() {
                if req.slot.is_none() {
                    if let Some(slot) = self.slots.iter_mut().find(|s| s.is_free()) {
                        let prompt_len = req.prompt.len();
                        slot.request_id = Some(req.id);
                        slot.kv_cache_length = prompt_len;
                        slot.tokens_generated = 0;
                        slot.kv_cache_start = 0;
                        // Determine backend from route profile or fall back to default
                        slot.backend_id = match self.route_profile.as_ref() {
                            Some(profile) => profile
                                .operations
                                .iter()
                                .find(|op| op.operation_id.0 == req.id)
                                .map(|op| op.backend.0)
                                .unwrap_or(self.config.default_backend_id),
                            None => self.config.default_backend_id,
                        };
                        req.slot = Some(slot.id);

                        // Allocate KV cache pages via paged allocator
                        if let Some(pager) = &mut self.kv_cache_pager {
                            if let Some(page_ids) =
                                pager.allocate_pages(self.config.kv_cache_pages_per_slot)
                            {
                                slot.kv_cache_pages = page_ids;
                            }
                        }
                    }
                }
            }
        } else {
            // Decode: extend KV cache length by 1 for every active slot
            for slot in self.slots.iter_mut() {
                if slot.request_id.is_some() {
                    slot.kv_cache_length += 1;
                }
            }
        }

        // Collect all active slots into the batch
        let batch_slots: Vec<Slot> = self
            .slots
            .iter()
            .filter(|s| s.request_id.is_some())
            .cloned()
            .collect();
        let batch_size = batch_slots.len();

        Batch {
            slots: batch_slots,
            batch_size,
            max_batch_size: self.config.max_batch_size,
        }
    }

    /// Process completed batch results.
    ///
    /// Increments `tokens_generated` for every active slot. When a request reaches
    /// its `max_tokens`, the slot is freed and the request is removed from `active`.
    pub fn process_results(&mut self, batch: &Batch) {
        for batch_slot in &batch.slots {
            if let Some(request_id) = batch_slot.request_id {
                // Update the internal slot state
                if let Some(slot) = self.slots.iter_mut().find(|s| s.id == batch_slot.id) {
                    slot.tokens_generated += 1;

                    // Check if the request has reached its max_tokens
                    if let Some(req) = self.active.iter().find(|r| r.id == request_id) {
                        if slot.tokens_generated >= req.max_tokens {
                            // Free KV cache pages back to the pager
                            if let Some(pager) = &mut self.kv_cache_pager {
                                for &page_id in &slot.kv_cache_pages {
                                    pager.free_page(page_id);
                                }
                            }
                            slot.kv_cache_pages.clear();
                            slot.request_id = None;
                            slot.tokens_generated = 0;
                            slot.kv_cache_length = 0;
                            slot.kv_cache_start = 0;
                        }
                    }
                }
            }
        }

        // Remove completed requests from active (those whose slots were freed)
        self.active
            .retain(|req| self.slots.iter().any(|s| s.request_id == Some(req.id)));
    }

    /// Add more slots when the initial pool is exhausted.
    fn add_slots(&mut self, count: usize) {
        let start_id = self.slots.len();
        for i in 0..count {
            self.slots.push(Slot::new(start_id + i));
        }
    }

    // ── Preemption API ─────────────────────────────────────────────────

    /// Enqueue a request with a specific priority level.
    ///
    /// `priority` ranges from 0 (highest — never preempted) to 255 (lowest —
    /// first to be swapped out). The default priority is 128.
    ///
    /// When a higher-priority request is enqueued while the scheduler is full,
    /// the caller should call [`preempt_lowest`](Self::preempt_lowest) before
    /// the next [`next_batch`](Self::next_batch) call to make room.
    pub fn enqueue_with_priority(&mut self, req: Request, priority: u8) {
        let mut req = req;
        req.priority = priority;
        self.queue.push(req);
    }

    /// Preempt the lowest-priority running request, save its KV cache state,
    /// and return a [`SavedRequest`] that can be used to resume it later.
    ///
    /// Returns `None` when:
    /// - Preemption is disabled.
    /// - There are fewer active requests than `max_batch_size` (no need to preempt).
    /// - All active requests are at the highest priority (0) — none can be evicted.
    ///
    /// # Starvation protection
    ///
    /// After [`MAX_PREEMPTIONS_BEFORE_BOOST`] preemptions, a request gets a
    /// permanent priority boost that protects it from further preemption.
    /// The boost degrades by [`STARVATION_PRIORITY_BOOST`] per preemption cycle.
    pub fn preempt_lowest(&mut self) -> Option<SavedRequest> {
        if !self.preemption_enabled {
            return None;
        }

        // Only preempt when we are actually at capacity.
        if self.active.len() < self.config.max_batch_size {
            return None;
        }

        // Apply starvation boosts before selecting the victim.
        self._apply_starvation_boosts();

        // Find the active request with the highest priority value (lowest priority).
        // Ties are broken by creation time (older requests preserved).
        let victim_idx = self
            .active
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| (r.priority, r.created_at))
            .map(|(i, _)| i);

        let victim_idx = victim_idx?;
        let victim = &self.active[victim_idx];

        // Don't preempt requests at the highest priority.
        if victim.priority == 0 {
            return None;
        }

        let request_id = victim.id;
        let prompt = victim.prompt.clone();
        let max_tokens = victim.max_tokens;
        let priority = victim.priority;

        // Get the slot assigned to this request.
        let slot_id = victim.slot?;
        let slot_idx = self.slots.iter().position(|s| s.id == slot_id)?;
        let slot = &self.slots[slot_idx];

        let tokens_generated = slot.tokens_generated;
        let kv_cache_length = slot.kv_cache_length;
        let kv_cache_start = slot.kv_cache_start;
        let kv_cache_pages = slot.kv_cache_pages.clone();

        // Increment starvation counter.
        let preemption_count = self
            .starvation_counters
            .get(&request_id)
            .copied()
            .unwrap_or(0);
        self.starvation_counters
            .insert(request_id, preemption_count + 1);

        // Build CompressedKvSlot snapshot from the page metadata.
        // Each page corresponds to one compressed slot.
        let kv_cache_snapshot: Vec<CompressedKvSlot> = kv_cache_pages
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let tokens_per_page = 64; // PREFIX_BLOCK_SIZE
                CompressedKvSlot {
                    compressed_keys: Vec::new(),
                    compressed_values: Vec::new(),
                    qjl_correction: None,
                    kv_offset: (kv_cache_start + i * tokens_per_page) as u32,
                    num_tokens: tokens_per_page
                        .min(kv_cache_length.saturating_sub(i * tokens_per_page)),
                }
            })
            .collect();

        // Remove the request from active set.
        self.active.remove(victim_idx);

        // Clear the slot but DO NOT free the pages — they stay allocated
        // for the SavedRequest to re-attach on resume.
        if let Some(slot) = self.slots.get_mut(slot_idx) {
            slot.request_id = None;
            slot.tokens_generated = 0;
            slot.kv_cache_length = 0;
            slot.kv_cache_start = 0;
            // Clear the page list so the slot doesn't double-own them.
            slot.kv_cache_pages.clear();
        }

        let saved = SavedRequest {
            request_id,
            kv_cache_snapshot,
            prompt,
            max_tokens,
            tokens_generated,
            kv_cache_length,
            kv_cache_start,
            priority,
            kv_cache_pages,
            preemption_count: preemption_count + 1,
        };

        self.preempted.push(saved.clone());
        Some(saved)
    }

    /// Resume a preempted request from its saved state.
    ///
    /// Re-assigns the saved KV cache pages to a free slot and adds the
    /// request back into the active set. The request continues decoding
    /// from where it left off.
    ///
    /// If all slots are occupied, the resumed request is enqueued at its
    /// original priority (which may already have a starvation boost) and
    /// will be activated on the next [`next_batch`](Self::next_batch) call.
    pub fn resume(&mut self, saved: SavedRequest) {
        let request_id = saved.request_id;
        let tokens_generated = saved.tokens_generated;
        let kv_cache_length = saved.kv_cache_length;
        let kv_cache_start = saved.kv_cache_start;
        let kv_cache_pages = saved.kv_cache_pages.clone();
        let priority = saved.priority;
        let max_tokens = saved.max_tokens;
        let prompt = saved.prompt.clone();

        // Remove from the preempted list.
        self.preempted.retain(|s| s.request_id != request_id);

        // Find a free slot.
        if let Some(slot) = self.slots.iter_mut().find(|s| s.is_free()) {
            slot.request_id = Some(request_id);
            slot.tokens_generated = tokens_generated;
            slot.kv_cache_length = kv_cache_length;
            slot.kv_cache_start = kv_cache_start;
            slot.kv_cache_pages = kv_cache_pages;
            slot.backend_id = self.config.default_backend_id;

            // Reconstruct the request and add to active.
            let req = Request {
                id: saved.request_id,
                prompt,
                max_tokens,
                priority,
                state: super::RequestState::Decoding,
                created_at: std::time::Instant::now(),
                slot: Some(slot.id),
            };
            self.active.push(req);
        } else {
            // No free slots — push back into the queue at saved priority.
            let req = Request {
                id: saved.request_id,
                prompt,
                max_tokens,
                priority,
                state: super::RequestState::Paused,
                created_at: std::time::Instant::now(),
                slot: None,
            };
            self.queue.push(req);
        }
    }

    /// Enable or disable request preemption.
    pub fn set_preemption_enabled(&mut self, enabled: bool) {
        self.preemption_enabled = enabled;
    }

    /// Returns the number of currently preempted (saved) requests.
    pub fn preempted_count(&self) -> usize {
        self.preempted.len()
    }

    /// Returns a reference to the list of preempted requests.
    pub fn preempted_requests(&self) -> &[SavedRequest] {
        &self.preempted
    }

    /// Apply starvation boosts to preempted requests before they are
    /// re-enqueued. Called automatically by [`preempt_lowest`](Self::preempt_lowest).
    ///
    /// After a request has been preempted [`MAX_PREEMPTIONS_BEFORE_BOOST`] times,
    /// its priority is improved by [`STARVATION_PRIORITY_BOOST`] per extra preemption,
    /// making it harder to preempt again. The boost is capped at 0 (highest priority).
    fn _apply_starvation_boosts(&mut self) {
        // Boost priorities of requests that have been heavily preempted.
        for req in self.active.iter_mut() {
            if let Some(&count) = self.starvation_counters.get(&req.id) {
                if count >= MAX_PREEMPTIONS_BEFORE_BOOST {
                    let cycles = (count - MAX_PREEMPTIONS_BEFORE_BOOST + 1) as u8;
                    let boost = cycles.saturating_mul(STARVATION_PRIORITY_BOOST);
                    // Lower priority value = higher priority.
                    // We subtract the boost (protect from preemption).
                    // The boost is reversible: save the original and cap at 0.
                    let boosted = req.priority.saturating_sub(boost);
                    req.priority = boosted;
                }
            }
        }
    }

    /// Gracefully drain all active, queued, and preempted requests.
    ///
    /// Cancels in-flight work, frees KV cache pages from active slots, and
    /// returns all saved (preempted) requests so their KV cache state can be
    /// persisted to disk before the process exits.
    pub fn drain_all(&mut self) -> Vec<SavedRequest> {
        // Drop all queued requests.
        self.queue.clear();

        // Collect saved (preempted) requests for KV cache persistence.
        let drained = std::mem::take(&mut self.preempted);

        // Cancel all active requests and free their KV cache pages.
        for slot in self.slots.iter_mut() {
            if slot.request_id.is_some() {
                if let Some(pager) = &mut self.kv_cache_pager {
                    for &page_id in &slot.kv_cache_pages {
                        pager.free_page(page_id);
                    }
                }
                slot.kv_cache_pages.clear();
                slot.request_id = None;
                slot.tokens_generated = 0;
                slot.kv_cache_length = 0;
                slot.kv_cache_start = 0;
            }
        }

        self.active.clear();
        self.starvation_counters.clear();

        drained
    }
}
