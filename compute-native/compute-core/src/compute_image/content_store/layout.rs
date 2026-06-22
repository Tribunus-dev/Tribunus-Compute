//! Target layout identity and layout specification types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetLayoutIdentity {
    pub layout_id: String,
    pub target_backend: String,
    pub quantization_format: String,
    pub tile_shape: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSpec {
    pub identity: TargetLayoutIdentity,
    pub row_major: bool,
    pub requires_transpose_at_load: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutRegistryEntry {
    pub layout_id: String,
    pub layout_spec: LayoutSpec,
    pub compatible_hardware: Vec<String>,
}
