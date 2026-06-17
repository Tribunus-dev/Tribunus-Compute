//! Batch construction utilities for the continuous batching scheduler.
//!
//! Reference: `ref/omlx/scheduler.py`

use super::{Batch, Request, Slot};

/// Build a prefill batch from queued requests
pub fn build_prefill_batch(requests: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = requests
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: 0,
            kv_cache_start: 0,
            kv_cache_length: req.prompt.len(),
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}

/// Build a decode batch from active requests
pub fn build_decode_batch(active: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = active
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: req.max_tokens,
            kv_cache_start: 0,
            kv_cache_length: req.max_tokens,
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}
