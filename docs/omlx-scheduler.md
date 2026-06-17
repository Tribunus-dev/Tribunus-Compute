# Continuous Batching Scheduler — Port Plan

Source: `omlx/omlx/scheduler.py` (10K lines), `engine_core.py` (1.2K lines)
Status: Reference copied to `ref/omlx/scheduler.py`

## What it is

A continuous batching scheduler for LLM inference that manages:
1. Request queuing and prioritization (FCFS with priority lanes)
2. Dynamic batch construction from ready requests
3. Prefill/decode phase scheduling
4. Speculative decode scheduling (MTP, draft models)
5. Pause/resume for long-running generations
6. Token budget management across concurrent requests

## Key concepts

- **Slot**: A position in the batch (model-level parallel execution unit)
- **Generation step**: One forward pass through the model
- **Prefill phase**: Compute KV cache for prompt tokens
- **Decode phase**: Generate one token at a time
- **Speculative decode**: Draft model verification + rejection sampling

## Batch scheduling loop

```
while server_running:
    # Receive new requests
    new = poll_requests()
    for req in new:
        prefill_queue.push(req)

    # Build prefill batch (max pending prefills)
    prefill_batch = prefill_queue.drain(max_batch_size)
    for slot in free_slots:
        results = engine.prefill(prefill_batch[slot])

    # Build decode batch
    decode_batch = active_generations[:max_batch_size]
    results = engine.generate(decode_batch)

    # Process results
    for result in results:
        if result.is_done():
            send_response(result)
            free_slots.push(result.slot)

    # Memory check
    enforcer.enforce()
```

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/scheduling/`

```rust
pub mod scheduler;
pub mod batch;
pub mod slot;
pub mod request;

pub use scheduler::Scheduler;
pub use batch::Batch;
pub use slot::Slot;
pub use request::Request;
```

### Key types

```rust
/// Request lifecycle state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Paused,
    Completed,
    Cancelled,
}

/// A single inference request
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub priority: u8,
    pub state: RequestState,
    pub created_at: Instant,
    pub slot: Option<usize>,
}

/// A batch of slots for model execution
pub struct Batch {
    pub slots: Vec<Slot>,
    pub batch_size: usize,
    pub max_batch_size: usize,
}

/// A slot in the batch (one model execution unit)
pub struct Slot {
    pub id: usize,
    pub request_id: Option<u64>,
    pub tokens_generated: usize,
    pub kv_cache_start: usize,
    pub kv_cache_length: usize,
}

/// Continuous batching scheduler
pub struct Scheduler {
    queue: Vec<Request>,
    active: Vec<Request>,
    slots: Vec<Slot>,
    config: SchedulerConfig,
}

pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_total_tokens: usize,
    pub max_prefill_batch: usize,
    pub prefill_many_ratio: f64,
    pub pause_threshold: usize,
}
```
