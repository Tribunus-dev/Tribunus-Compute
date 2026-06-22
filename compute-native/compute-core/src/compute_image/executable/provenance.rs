//! Compiler provenance — who compiled the image and when.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerProvenance {
    pub compiler_name: String,
    pub compiler_version: String,
    pub compilation_timestamp: String,
    pub source_model_hash: String,
    pub target_profile_ids: Vec<String>,
}
