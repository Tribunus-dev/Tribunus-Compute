//! Receipt types for executable compilation and admission.

use serde::{Deserialize, Serialize};

pub type ReceiptId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableCompilationReceipt {
    pub receipt_id: ReceiptId,
    pub model_name: String,
    pub profile_count: u32,
    pub total_weight_bytes: u64,
    pub compilation_duration_ms: u64,
    pub seal_hash: String,
}
