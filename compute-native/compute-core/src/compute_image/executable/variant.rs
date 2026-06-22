//! Shape-specialized program variant.

use crate::integration::ContentHash;
use crate::compute_image::program::phase_program::SerializedPhaseProgram;
use serde::{Deserialize, Serialize};

pub type ShapeSpecializedVariantId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSpecializedProgram {
    pub variant_id: ShapeSpecializedVariantId,
    pub shape_profile: ShapeProfile,
    pub phase_program: SerializedPhaseProgram,
    pub program_hash: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeProfile {
    pub max_batch: u32,
    pub max_tokens: u32,
    pub label: String,
}
