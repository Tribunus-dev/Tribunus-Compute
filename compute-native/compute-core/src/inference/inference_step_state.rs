//! Mutable per-step state for inference.
//!
//! Created fresh for each prefill chunk or decode step. Tracks activation,
//! phase completion, receipts, sampling, and the final output token.

use crate::scheduling::receipts::PhaseReceipt;
use crate::scheduling::activation_binding::{CurrentActivation, ActivationGeneration};
use crate::compute_image::phase_graph::TensorId;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Unique request identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestId(pub u64);

/// Unique execution identifier for this step.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionId(pub u64);

/// Inference mode for the current step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    Prefill,
    Decode,
}

/// Token input for a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInput {
    pub token_ids: Vec<u32>,
}

/// Status table tracking phase completion for the current step.
#[derive(Debug, Clone)]
pub struct PhaseStatusTable {
    statuses: HashMap<String, PhaseStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseStatus {
    Pending,
    Ready,
    Running,
    Complete,
    Failed,
}

impl PhaseStatusTable {
    pub fn new() -> Self {
        Self {
            statuses: HashMap::new(),
        }
    }

    pub fn set(&mut self, phase_id: &str, status: PhaseStatus) {
        self.statuses.insert(phase_id.to_string(), status);
    }

    pub fn get(&self, phase_id: &str) -> PhaseStatus {
        self.statuses
            .get(phase_id)
            .copied()
            .unwrap_or(PhaseStatus::Pending)
    }
}

impl Default for PhaseStatusTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Ledger of phase receipts for the current step.
#[derive(Debug, Clone)]
pub struct StepReceiptLedger {
    pub receipts: Vec<PhaseReceipt>,
}

impl StepReceiptLedger {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }

    pub fn push(&mut self, receipt: PhaseReceipt) {
        self.receipts.push(receipt);
    }

    pub fn take(&mut self) -> Vec<PhaseReceipt> {
        std::mem::take(&mut self.receipts)
    }
}

impl Default for StepReceiptLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Output of an inference step.
#[derive(Debug, Clone)]
pub struct InferenceStepOutput {
    pub token: Option<u32>,
    pub logits: Option<Array>,
    pub receipts: Vec<PhaseReceipt>,
}

/// Mutable per-step state.
///
/// Created fresh for each prefill chunk or decode step.
pub struct InferenceStepState {
    pub request_id: RequestId,
    pub execution_id: ExecutionId,
    pub mode: InferenceMode,
    pub token_position: usize,
    pub input_tokens: TokenInput,
    pub current_activation: Option<CurrentActivation>,
    pub logits: Option<Array>,
    pub output_activation: Option<CurrentActivation>,
    pub phase_status: PhaseStatusTable,
    pub receipt_ledger: StepReceiptLedger,
    pub deadline: Option<Instant>,
    pub terminal_output: Option<InferenceStepOutput>,
}

impl InferenceStepState {
    pub fn new_prefill(request_id: u64, execution_id: u64, tokens: Vec<u32>) -> Self {
        Self {
            request_id: RequestId(request_id),
            execution_id: ExecutionId(execution_id),
            mode: InferenceMode::Prefill,
            token_position: 0,
            input_tokens: TokenInput { token_ids: tokens },
            current_activation: None,
            logits: None,
            output_activation: None,
            phase_status: PhaseStatusTable::new(),
            receipt_ledger: StepReceiptLedger::new(),
            deadline: None,
            terminal_output: None,
        }
    }

    pub fn new_decode(request_id: u64, execution_id: u64, token: u32, position: usize) -> Self {
        Self {
            request_id: RequestId(request_id),
            execution_id: ExecutionId(execution_id),
            mode: InferenceMode::Decode,
            token_position: position,
            input_tokens: TokenInput {
                token_ids: vec![token],
            },
            current_activation: None,
            logits: None,
            output_activation: None,
            phase_status: PhaseStatusTable::new(),
            receipt_ledger: StepReceiptLedger::new(),
            deadline: None,
            terminal_output: None,
        }
    }
}
