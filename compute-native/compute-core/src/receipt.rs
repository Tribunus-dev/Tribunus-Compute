#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenReceipt {
    pub token_index: u32,
    pub backend: String,
    pub bytes_copied_h2d: u64,
    pub bytes_copied_d2d: u64,
    pub bytes_copied_d2h: u64,
    pub arena_allocations: u32,
    pub arena_failures: u32,
    pub fallback_count: u32,
    pub fallback_by_priority: Vec<u32>,
    pub stage_durations_us: Vec<u64>,
    pub speculative_branches_accepted: u32,
    pub speculative_branches_rejected: u32,
    pub kv_page_faults: u32,
    pub disk_bytes_read: u64,
}

pub fn backend_id_to_label(id: u8) -> String {
    match id {
        0 => "mlx_gpu".into(),
        1 => "ane".into(),
        2 => "accelerate".into(),
        _ => "cpu".into(),
    }
}

pub struct SessionReceipts {
    pub per_token: Vec<TokenReceipt>,
    pub total_tokens: u32,
    pub total_backend_switches: u32,
    pub total_fallbacks: u32,
}
