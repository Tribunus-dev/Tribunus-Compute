//! Packing policy types for content-addressed weight layout.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingPolicy {
    pub alignment: u64,
    pub interleave_config: Option<InterleaveConfig>,
    pub padding_mode: PaddingMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleaveConfig {
    pub block_size: u32,
    pub num_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingMode {
    AlignToPage,
    AlignTo(u64),
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingResult {
    pub packed_bytes: Vec<u8>,
    pub original_bytes: u64,
    pub aligned_bytes: u64,
    pub waste_bytes: u64,
}
