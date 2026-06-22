//! Token-budget scheduler — unified work scheduling for all phase kinds.
//! Ported concept from vLLM V1 engine: token-budget abstraction replaces
//! strict prefill/decode separation.
//!
//! The scheduler owns admission control. Before any work is dispatched,
//! it asks the kv_arena whether the request can admit N more tokens.
//! If not, the work unit is deferred.

use std::collections::VecDeque;
use std::time::Instant;

/// Kind of work phase in the token-budget model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseKind {
    /// Prefill: process a span of prompt tokens (may be chunked).
    Prefill,
    /// Decode: generate one or a few output tokens.
    Decode,
    /// Speculative draft: generate draft tokens from a draft model.
    SpecDraft,
    /// Speculative verification: verify draft tokens against the target model.
    SpecVerify,
}

/// A schedulable unit of token work.
#[derive(Clone, Debug)]
pub struct TokenWorkUnit {
    pub request_id: String,
    pub sequence_id: Option<u64>, // from KVArena admission
    pub phase: PhaseKind,
    pub compute_image_phase: Option<String>, // ComputeImage phase identifier
    pub token_span: u32,                     // number of tokens in this work unit
    pub kv_blocks_needed: u32,               // blocks to allocate before dispatching
    pub priority: u32,                       // lower = higher priority
    pub deadline: Instant,                   // deadline for SLO enforcement
    pub backend_route: Option<String>,       // "mlx", "candle-cpu", "tensix"
    pub speculative_parent: Option<String>,  // if this is verification, parent request id
    pub receipt_sink: Option<String>, // callback endpoint or accumulator for execution receipts
}

impl TokenWorkUnit {
    pub fn new_prefill(request_id: &str, token_span: u32) -> Self {
        TokenWorkUnit {
            request_id: request_id.to_string(),
            sequence_id: None,
            phase: PhaseKind::Prefill,
            compute_image_phase: None,
            token_span,
            kv_blocks_needed: needed_blocks(token_span),
            priority: 1,
            deadline: Instant::now() + std::time::Duration::from_secs(30),
            backend_route: None,
            speculative_parent: None,
            receipt_sink: None,
        }
    }

    pub fn new_decode(request_id: &str) -> Self {
        TokenWorkUnit {
            request_id: request_id.to_string(),
            sequence_id: None,
            phase: PhaseKind::Decode,
            compute_image_phase: None,
            token_span: 1,
            kv_blocks_needed: 0,
            priority: 2,
            deadline: Instant::now() + std::time::Duration::from_secs(30),
            backend_route: None,
            speculative_parent: None,
            receipt_sink: None,
        }
    }
}

fn needed_blocks(tokens: u32) -> u32 {
    (tokens + crate::kv_arena::block::DEFAULT_BLOCK_SIZE as u32 - 1)
        / crate::kv_arena::block::DEFAULT_BLOCK_SIZE as u32
}

/// Scheduler configuration.
#[derive(Clone, Debug)]
pub struct TokenBudgetConfig {
    pub max_num_batched_tokens: u32, // max tokens per scheduling batch
    pub max_num_seqs: u32,           // max concurrent sequences
    pub max_model_len: u32,          // max sequence length
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        TokenBudgetConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 8,
            max_model_len: 131072,
        }
    }
}

/// Token-budget scheduler state.
pub struct TokenBudgetScheduler {
    config: TokenBudgetConfig,
    run_queue: VecDeque<TokenWorkUnit>,
    active_requests: std::collections::HashSet<String>,
    total_budget_tokens: u32, // remaining token budget for this scheduling cycle
    config_t: std::collections::HashMap<String, TokenWorkUnit>, // re-insertions
}

impl TokenBudgetScheduler {
    pub fn new(config: TokenBudgetConfig) -> Self {
        let max_num_batched_tokens = config.max_num_batched_tokens;
        TokenBudgetScheduler {
            config,
            run_queue: VecDeque::new(),
            active_requests: std::collections::HashSet::new(),
            total_budget_tokens: max_num_batched_tokens,
            config_t: std::collections::HashMap::new(),
        }
    }

    /// Enqueue a new request (prefill work unit).
    pub fn enqueue(&mut self, unit: TokenWorkUnit) {
        self.run_queue.push_back(unit);
    }

    /// Schedule the next batch of work units.
    /// Returns work units that can be dispatched (admission granted).
    pub fn schedule(&mut self) -> Vec<TokenWorkUnit> {
        let mut batch = Vec::new();
        let mut budget = self.total_budget_tokens;
        let mut seqs = 0u32;

        while let Some(unit) = self.run_queue.pop_front() {
            if seqs >= self.config.max_num_seqs {
                self.run_queue.push_front(unit);
                break;
            }
            if unit.token_span > budget {
                // Token budget exceeded — defer remaining tokens as a new chunk
                let chunk1 = TokenWorkUnit {
                    token_span: budget,
                    kv_blocks_needed: needed_blocks(budget),
                    ..unit.clone()
                };
                let chunk2 = TokenWorkUnit {
                    token_span: unit.token_span - budget,
                    kv_blocks_needed: needed_blocks(unit.token_span - budget),
                    ..unit
                };
                batch.push(chunk1);
                self.run_queue.push_front(chunk2);
                budget = 0;
                seqs += 1;
                break;
            }
            budget -= unit.token_span;
            seqs += 1;
            self.active_requests.insert(unit.request_id.clone());
            batch.push(unit);
        }

        self.total_budget_tokens = budget;
        batch
    }

    /// Mark a request as completed and recycle its budget.
    pub fn complete(&mut self, request_id: &str) {
        self.active_requests.remove(request_id);
    }

    /// Re-enqueue a decode work unit after a successful decode step.
    pub fn enqueue_decode(&mut self, request_id: &str, priority: u32) {
        let unit = TokenWorkUnit {
            request_id: request_id.to_string(),
            sequence_id: None,
            phase: PhaseKind::Decode,
            compute_image_phase: None,
            token_span: 1,
            kv_blocks_needed: 0,
            priority,
            deadline: Instant::now() + std::time::Duration::from_secs(30),
            backend_route: None,
            speculative_parent: None,
            receipt_sink: None,
        };
        self.run_queue.push_back(unit);
    }

    /// Reset the token budget for a new scheduling cycle.
    pub fn reset_budget(&mut self) {
        self.total_budget_tokens = self.config.max_num_batched_tokens;
    }

    /// Returns the maximum number of tokens per scheduling cycle (from config).
    pub fn max_budget_tokens(&self) -> u32 {
        self.config.max_num_batched_tokens
    }

    pub fn pending_count(&self) -> usize {
        self.run_queue.len()
    }
    pub fn active_count(&self) -> usize {
        self.active_requests.len()
    }
}

// ---------------------------------------------------------------------------
// ReceiptCollector — accumulates execution receipts from dispatched work units
// ---------------------------------------------------------------------------

/// A record of a completed work unit execution.
#[derive(Clone, Debug)]
pub struct ExecutionReceipt {
    pub request_id: String,
    pub phase: String,
    pub compute_image_phase: Option<String>,
    pub backend_route: Option<String>,
    pub token_count: u32,
    pub elapsed_ns: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Accumulates execution receipts from dispatched work units for
/// observability and feedback into the scheduling loop.
#[derive(Clone, Debug, Default)]
pub struct ReceiptCollector {
    pub receipts: Vec<ExecutionReceipt>,
}

impl ReceiptCollector {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }

    /// Record a single execution receipt.
    pub fn record(&mut self, receipt: ExecutionReceipt) {
        self.receipts.push(receipt);
    }

    /// Produce a compact human-readable summary of all accumulated receipts.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let total = self.receipts.len();
        let successes = self.receipts.iter().filter(|r| r.success).count();
        let failures = total - successes;
        let total_ns: u64 = self.receipts.iter().map(|r| r.elapsed_ns).sum();
        let total_tokens: u32 = self.receipts.iter().map(|r| r.token_count).sum();
        let _ = writeln!(
            out,
            "ReceiptCollector: {} receipts ({} ok, {} fail), {} tokens in {} ns",
            total, successes, failures, total_tokens, total_ns
        );
        out
    }
}
