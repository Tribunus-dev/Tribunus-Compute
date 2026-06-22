//! ShapeVariantDefinition — compiled shape-specialized variant metadata.
//!
//! Each variant identifies a compiled program specialized for a
//! particular [`ExecutionShapeClass`] with declared batch/token budgets
//! and optional hardware feature requirements.

use serde::{Deserialize, Serialize};

use crate::compute_image::execution_shape::ExecutionShapeClass;

/// Metadata for a compiled shape-specialized variant.
///
/// A variant is the result of lowering the canonical model graph through
/// a specific execution shape class and target profile.  The runtime uses
/// these definitions to select the correct variant for a given execution
/// request and to validate that the variant is compatible with the current
/// hardware capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVariantDefinition {
    /// Unique variant identifier (e.g. "decode1", "prefill_small").
    pub variant_id: ShapeVariantId,
    /// Execution shape class this variant was compiled for.
    pub shape_class: ExecutionShapeClass,
    /// Human-readable description of the variant's purpose.
    pub description: String,
    /// Maximum batch size this variant supports, if bounded.
    pub max_batch: Option<u32>,
    /// Maximum token count this variant supports, if bounded.
    pub max_tokens: Option<u32>,
    /// Hardware feature names the variant requires (e.g. "ane", "metal",
    /// "fp16", "unified_memory").  Empty implies no hardware constraints.
    #[serde(default)]
    pub required_hardware_features: Vec<String>,

    /// Label of the target profile this variant was compiled for.
    pub target_profile_label: String,

    /// Hash of the compiled program used to detect changes.
    pub program_hash: u64,

    /// Serialized program payload bytes.
    #[serde(default)]
    pub program_data: Vec<u8>,
}

/// String-based identifier for a shape variant.
pub type ShapeVariantId = String;

/// Return the canonical set of required shape-specialized variants.
///
/// The compiler emits at least these variants for every model.  Individual
/// target profiles may produce additional variants.
pub fn required_variants() -> Vec<ShapeVariantDefinition> {
    vec![
        ShapeVariantDefinition {
            variant_id: "decode1".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: "Single-token decode (batch=1)".into(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
        ShapeVariantDefinition {
            variant_id: "decode_batch2".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 2 },
            description: "Batch-2 decode".into(),
            max_batch: Some(2),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
        ShapeVariantDefinition {
            variant_id: "decode_batch4".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 4 },
            description: "Batch-4 decode".into(),
            max_batch: Some(4),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
        ShapeVariantDefinition {
            variant_id: "prefill_small".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 512 },
            description: "Prefill up to 512 tokens".into(),
            max_batch: Some(1),
            max_tokens: Some(512),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
        ShapeVariantDefinition {
            variant_id: "prefill_medium".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            description: "Prefill 513–4096 tokens".into(),
            max_batch: Some(1),
            max_tokens: Some(4096),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
        ShapeVariantDefinition {
            variant_id: "prefill_large".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 32768 },
            description: "Prefill 4097–32768 tokens".into(),
            max_batch: Some(1),
            max_tokens: Some(32768),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        },
    ]
}
