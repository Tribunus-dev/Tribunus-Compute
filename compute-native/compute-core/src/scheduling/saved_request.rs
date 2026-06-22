//! Request preemption support for continuous batching.
//!
//! A [`SavedRequest`] captures a preempted request's KV cache state
//! so it can be resumed later. The scheduler stores one or more of these
//! when a higher-priority request forces a running sequence to yield.

use crate::kv_cache::CompressedKvSlot;

/// Highest priority value — requests at this level are never preempted.
pub const PRIORITY_HIGHEST: u8 = 0;

/// Default priority for requests that don't specify otherwise.
pub const PRIORITY_DEFAULT: u8 = 128;

/// Lowest priority value — requests at this level are preempted first.
pub const PRIORITY_LOWEST: u8 = 255;

/// Maximum number of times a request can be preempted before it gets
/// a starvation boost that effectively exempts it from further preemption.
pub const MAX_PREEMPTIONS_BEFORE_BOOST: usize = 3;

/// Priority boost applied to a starved request on each preemption cycle.
/// The boost reduces the priority value (making it higher priority),
/// protecting it from being preempted again.
pub const STARVATION_PRIORITY_BOOST: u8 = 64;

/// A request whose KV cache has been saved to allow preemption.
///
/// The `kv_cache_snapshot` holds the compressed KV page data that was
/// in GPU-accessible memory. On resume, these slots are re-assigned to
/// the request's new slot and the pages are re-attached.
#[derive(Debug, Clone)]
pub struct SavedRequest {
    /// Original request ID.
    pub request_id: u64,
    /// Compressed KV pages — each entry corresponds to one page's worth
    /// of tokens, identified by `kv_offset` and `num_tokens`.
    pub kv_cache_snapshot: Vec<CompressedKvSlot>,
    /// The original prompt tokens (needed to reconstruct the request
    /// on resume when the KV cache fully covers the prompt).
    pub prompt: Vec<u32>,
    /// Maximum tokens for the original request.
    pub max_tokens: usize,
    /// Tokens already generated before preemption.
    pub tokens_generated: usize,
    /// KV cache length at preemption time.
    pub kv_cache_length: usize,
    /// KV cache start position.
    pub kv_cache_start: usize,
    /// Priority at preemption time (may have been boosted by anti-starvation).
    pub priority: u8,
    /// Page IDs from the paged allocator — used to re-attach pages on resume.
    pub kv_cache_pages: Vec<usize>,
    /// Number of times this request has been preempted.
    pub preemption_count: usize,
}
