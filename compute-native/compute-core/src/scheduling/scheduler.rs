use super::{Batch, Request, SchedulerConfig, Slot};
use crate::backend::routing::ComputeRouteProfile;
use crate::memory::allocator::{IosurfaceAllocator, PagedIosurfaceAllocator};
use parking_lot::Mutex;
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
    route_profile: Option<ComputeRouteProfile>,
    kv_cache_allocator: Option<Arc<Mutex<IosurfaceAllocator>>>,
    kv_cache_pager: Option<PagedIosurfaceAllocator>,
}

impl Scheduler {
    /// Create a new scheduler with the given config.
    pub fn new(config: SchedulerConfig) -> Self {
        let slots = (0..config.max_batch_size)
            .map(|id| Slot {
                id,
                request_id: None,
                tokens_generated: 0,
                kv_cache_start: 0,
                kv_cache_length: 0,
                backend_id: config.default_backend_id,
                kv_cache_pages: vec![],
            })
            .collect();
        Self {
            queue: Vec::new(),
            active: Vec::new(),
            slots,
            config,
            route_profile: None,
            kv_cache_allocator: None,
            kv_cache_pager: None,
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
}
