//! Server inference engine — continuous batching bridge.
//!
//! Owns the scheduling::Scheduler, per-request ProfiledInferenceSession
//! instances, and the loaded model.  Drives the prefill–decode loop
//! and delivers CompletionResult values through an mpsc channel.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use crate::kv_cache::KvCache;
use crate::memory::allocator::PagedIosurfaceAllocator;
use crate::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};
use crate::scheduling::{Request, RequestState, Scheduler, SchedulerConfig};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a completed inference request.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub request_id: u64,
    pub tokens: Vec<u32>,
    /// Placeholder decoded text.  A real tokenizer would replace this.
    pub text: String,
    /// Reason generation finished: `"length"` when `max_tokens` reached.
    pub finish_reason: String,
}

// ---------------------------------------------------------------------------
// Internal tracking
// ---------------------------------------------------------------------------

/// Per-request state kept alongside the scheduler's own bookkeeping.
struct PendingRequest {
    prompt: Vec<u32>,
    max_tokens: usize,
    state: RequestState,
    generated_tokens: Vec<u32>,
}

// ---------------------------------------------------------------------------
// ServerEngine
// ---------------------------------------------------------------------------

/// The inference engine — owns scheduler, sessions, and the loaded model.
pub struct ServerEngine {
    scheduler: Scheduler,
    sessions: HashMap<u64, ProfiledInferenceSession>,
    model: Arc<LoadedProfiledModel>,
    /// KV cache page allocator (shared with the scheduler during setup).
    #[allow(dead_code)]
    kv_cache_pager: Option<PagedIosurfaceAllocator>,
    /// Engine-scoped view of pending (and in-flight) requests.
    pending_requests: HashMap<u64, PendingRequest>,
    /// Channel for delivering completed results to the caller.
    completion_tx: mpsc::Sender<CompletionResult>,
    /// Receiver half, wrapped in a Mutex so that `try_recv_completion`
    /// can accept `&self`.
    completion_rx: Mutex<mpsc::Receiver<CompletionResult>>,
}

impl ServerEngine {
    /// Create a new engine with the given config.
    pub fn new(
        model: Arc<LoadedProfiledModel>,
        scheduler_config: SchedulerConfig,
        kv_cache_pager: Option<PagedIosurfaceAllocator>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(256);
        Self {
            scheduler: Scheduler::new(scheduler_config),
            sessions: HashMap::new(),
            model,
            kv_cache_pager,
            pending_requests: HashMap::new(),
            completion_tx: tx,
            completion_rx: Mutex::new(rx),
        }
    }

    /// Enqueue a new completion request.
    ///
    /// Returns immediately with the request's assigned ID.
    pub async fn enqueue(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        let request = Request::new(prompt_tokens.clone(), max_tokens);
        let request_id = request.id;

        self.pending_requests.insert(
            request_id,
            PendingRequest {
                prompt: prompt_tokens,
                max_tokens,
                state: RequestState::Queued,
                generated_tokens: Vec::with_capacity(max_tokens),
            },
        );
        self.scheduler.enqueue(request);

        request_id
    }

    /// Run one batch iteration: poll scheduler, execute, process results.
    ///
    /// Returns the number of slots processed (may be 0 when the queue is
    /// empty or the scheduler has no room).
    pub async fn step(&mut self) -> Result<usize, String> {
        let batch = self.scheduler.next_batch();
        if batch.slots.is_empty() {
            return Ok(0);
        }

        let n_layers = self.model.reader.manifest.execution_plan.layers.len();
        let plan = &self.model.reader.manifest.execution_plan;

        // Track which requests need their completion sent this step so we
        // can run scheduler.process_results() before removing them.
        let mut completed_ids: Vec<u64> = Vec::new();

        // ── Execute each slot ──────────────────────────────────────────
        for slot in &batch.slots {
            let req_id = match slot.request_id {
                Some(id) => id,
                None => continue,
            };

            // Look up the engine-side request info (prompt, state, etc.).
            let info = match self.pending_requests.get_mut(&req_id) {
                Some(info) => info,
                None => continue,
            };

            // Create a session on first encounter for this request.
            if !self.sessions.contains_key(&req_id) {
                let kv_caches: Vec<KvCache> = (0..n_layers)
                    .map(|l| {
                        let layer = &plan.layers[l];
                        // Sliding layers cap at sliding_window; global layers
                        // use a generous default (consistent with existing
                        // code in profiled_executor / tribunus-compute-worker).
                        let capacity = if layer.attention_kind == "sliding_attention" {
                            layer.sliding_window
                        } else {
                            32_768
                        };
                        KvCache::new(
                            capacity,
                            layer.n_kv_heads,
                            layer.head_dim,
                            layer.attention_kind == "sliding_attention",
                        )
                    })
                    .collect();

                let session = ProfiledInferenceSession::new(format!("req_{}", req_id), kv_caches);
                self.sessions.insert(req_id, session);
            }

            let session = self.sessions.get_mut(&req_id).unwrap();
            let model = &self.model;

            // ── Dispatch phase ─────────────────────────────────────────
            match info.state {
                RequestState::Queued | RequestState::Prefilling => {
                    info.state = RequestState::Decoding;
                    let token = session
                        .prefill(&info.prompt, model)
                        .map_err(|e| format!("prefill req {}: {}", req_id, e))?;
                    info.generated_tokens.push(token);
                }
                RequestState::Decoding => {
                    let last_token = *info.generated_tokens.last().unwrap_or(&0);
                    let token = session
                        .decode_one(last_token, model)
                        .map_err(|e| format!("decode req {}: {}", req_id, e))?;
                    info.generated_tokens.push(token);
                }
                _ => {
                    // Paused / Completed / Cancelled — no inference work.
                }
            }

            // Check for completion (max_tokens reached).
            if info.generated_tokens.len() >= info.max_tokens {
                info.state = RequestState::Completed;
                completed_ids.push(req_id);
            }
        }

        // ── Send completion results ────────────────────────────────────
        for req_id in &completed_ids {
            if let Some(info) = self.pending_requests.get(req_id) {
                let result = CompletionResult {
                    request_id: *req_id,
                    tokens: info.generated_tokens.clone(),
                    text: format!("<{} tokens generated>", info.generated_tokens.len()),
                    finish_reason: "length".into(),
                };
                // Non-blocking send; channel capacity (256) is sufficient
                // for any single step.
                let _ = self.completion_tx.try_send(result);
            }
        }

        // ── Let the scheduler free slots / clean up its own state ──────
        self.scheduler.process_results(&batch);

        // ── Remove completed requests from engine-level maps ───────────
        for req_id in &completed_ids {
            self.sessions.remove(req_id);
            self.pending_requests.remove(req_id);
        }

        Ok(batch.slots.len())
    }

    /// Run the main batch loop, blocking until all currently-enqueued
    /// requests finish.
    ///
    /// Returns every completion that was produced during the loop.
    pub async fn run_batch_loop(&mut self) -> Result<Vec<CompletionResult>, String> {
        let mut results: Vec<CompletionResult> = Vec::new();

        while !self.pending_requests.is_empty() {
            let count = self.step().await?;

            if count == 0 {
                // No slots available — yield to the executor so other
                // tasks can make progress while we wait.
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }

            // Drain whatever completions step() produced.
            loop {
                let result = self.completion_rx.lock().await.try_recv();
                match result {
                    Ok(cr) => results.push(cr),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        return Err("completion channel closed prematurely".into());
                    }
                }
            }
        }

        Ok(results)
    }

    /// Attempt to receive a completed result without blocking.
    ///
    /// Returns `None` if no result is available yet.
    pub fn try_recv_completion(&self) -> Option<CompletionResult> {
        self.completion_rx.blocking_lock().try_recv().ok()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Convert chat messages to a tokenized prompt using a simple template.
///
/// This is a simplified placeholder — real deployments MUST use the model's
/// actual `chat_template` from `tokenizer_config.json` and the real
/// tokenizer to produce correct token IDs.
///
/// The simplified format uses `<|im_start|>` / `<|im_end|>` markers with
/// role-specific token prefixes (token ids `1`, `3`, `5`, `6` as stand-ins).
///
/// # Parameters
/// - `messages` — `(role, content)` pairs, e.g. `("user", "Hello")`.
///
/// # Returns
/// Flat `Vec<u32>` of token IDs that can be passed directly to
/// [`ServerEngine::enqueue`].
pub fn apply_chat_template(messages: &[(String, String)]) -> Vec<u32> {
    let mut tokens = Vec::new();

    for (role, content) in messages {
        match role.as_str() {
            "system" => tokens.extend_from_slice(&[1, 3]), // <|im_start|>system
            "user" => tokens.extend_from_slice(&[1, 5]),   // <|im_start|>user
            "assistant" => tokens.extend_from_slice(&[1, 6]), // <|im_start|>assistant
            _ => tokens.push(1),                           // bare <|im_start|>
        }
        // Placeholder: encode content bytes as token IDs.
        // Real implementations must use the correct tokenizer.
        for byte in content.bytes() {
            tokens.push(byte as u32);
        }
        tokens.push(2); // <|im_end|>
    }

    // Final assistant turn marker
    tokens.push(1); // <|im_start|>
    tokens.push(6); // assistant

    tokens
}
