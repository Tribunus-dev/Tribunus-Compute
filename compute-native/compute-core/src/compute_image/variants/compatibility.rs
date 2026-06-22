//! Variant compatibility checking — determine whether a compiled shape
//! variant is compatible with a requested execution shape and runtime
//! hardware capabilities.
//!
//! The compatibility checker collects **all** violations rather than
//! short-circuiting on the first one, so the runtime can report *why* a
//! variant is incompatible, not just *that* it is.

use serde::{Deserialize, Serialize};

use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::compute_image::variants::shape_class::ShapeVariantDefinition;

/// Structured report of compatibility between a variant and a request.
///
/// When `compatible` is `false`, the `violations` vector contains every
/// reason the variant is unsuitable, enabling informative diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCompatibilityReport {
    /// Identifier of the variant that was checked.
    pub variant_id: String,
    /// Whether the variant is fully compatible.
    pub compatible: bool,
    /// All violations found (empty iff `compatible` is true).
    pub violations: Vec<CompatibilityViolation>,
}

/// A single reason a variant is incompatible with the request or runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityViolation {
    /// The variant's execution shape class does not cover the requested
    /// execution shape.
    ShapeMismatch {
        /// Shape the variant was compiled for.
        expected: ExecutionShapeClass,
        /// Shape that was requested.
        actual: ExecutionShapeClass,
    },
    /// The requested batch size exceeds the variant's maximum.
    BatchOverflow {
        /// Maximum batch size the variant supports.
        max_batch: u32,
        /// Batch size that was requested.
        requested: u32,
    },
    /// The requested token count exceeds the variant's maximum.
    TokenOverflow {
        /// Maximum token count the variant supports.
        max_tokens: u32,
        /// Token count that was requested.
        requested: u32,
    },
    /// A hardware feature required by the variant is not available on the
    /// runtime device.
    MissingHardwareFeature(String),
    /// The variant's runtime version requirement is not met by the current
    /// runtime.
    RuntimeVersionMismatch {
        /// Version string required by the variant.
        required: String,
        /// Version string of the current runtime.
        actual: String,
    },
}

/// Snapshot of the runtime's hardware capabilities for compatibility
/// checking.
///
/// Captured once at startup and reused across all variant checks in a
/// session.  Values reflect the actual device, not the maximum possible
/// for the architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCapabilitySnapshot {
    /// Maximum threadgroup size supported by the GPU (Metal / Vulkan).
    pub max_threadgroup_size: u32,
    /// Whether the Apple Neural Engine (ANE) is available.
    pub has_ane: bool,
    /// Number of ANE cores available.
    pub ane_count: u32,
    /// Whether the system has unified memory (CPU + GPU).
    pub has_unified_memory: bool,
    /// Whether FP16 compute is supported on the primary GPU.
    pub supports_fp16: bool,
    /// Whether INT8 compute is supported on the primary GPU.
    pub supports_int8: bool,
    /// Whether Core ML is available for model execution.
    pub coreml_available: bool,
    /// Whether Metal is available for GPU compute.
    pub metal_available: bool,
}

/// Check whether a compiled variant is compatible with a requested
/// execution shape and the current runtime's hardware capabilities.
///
/// Gathers **all** violations into the returned report rather than
/// stopping at the first incompatibility.  The runtime can use the
/// report's `violations` list to produce rich diagnostics.
pub fn check_variant_compatibility(
    variant: &ShapeVariantDefinition,
    request_shape: &ExecutionShapeClass,
    runtime_caps: &RuntimeCapabilitySnapshot,
) -> VariantCompatibilityReport {
    let mut violations: Vec<CompatibilityViolation> = Vec::new();

    // 1. Shape class compatibility.
    if !is_shape_compatible(variant, request_shape) {
        violations.push(CompatibilityViolation::ShapeMismatch {
            expected: variant.shape_class.clone(),
            actual: request_shape.clone(),
        });
    }

    // 2. Batch size limits.
    if let Some(max_batch) = variant.max_batch {
        let requested_batch = extract_batch_size(request_shape);
        if requested_batch > max_batch {
            violations.push(CompatibilityViolation::BatchOverflow {
                max_batch,
                requested: requested_batch,
            });
        }
    }

    // 3. Token count limits.
    if let Some(max_tokens) = variant.max_tokens {
        let requested_tokens = extract_token_count(request_shape);
        if requested_tokens > max_tokens {
            violations.push(CompatibilityViolation::TokenOverflow {
                max_tokens,
                requested: requested_tokens,
            });
        }
    }

    // 4. Hardware feature requirements.
    for feature in &variant.required_hardware_features {
        if !runtime_has_feature(runtime_caps, feature) {
            violations.push(CompatibilityViolation::MissingHardwareFeature(
                feature.clone(),
            ));
        }
    }

    VariantCompatibilityReport {
        variant_id: variant.variant_id.clone(),
        compatible: violations.is_empty(),
        violations,
    }
}

/// Returns `true` if the variant's shape class can service the requested
/// execution shape.
///
/// Compatible pairs:
/// - **Exact match** on the same discriminant.
/// - **Decode1** covers `DecodeBatch { max_batch: 1 }`.
/// - **DecodeBatch** with sufficient capacity covers `Decode1` (batch ≤ max)
///   and smaller `DecodeBatch` requests.
/// - **PrefillBucket** with sufficient capacity covers smaller
///   `PrefillBucket` requests.
/// - **ChunkedPrefill**, **MixedBatch**, and **DiffusionForward** only match
///   the same discriminant (exact match).
pub fn is_shape_compatible(variant: &ShapeVariantDefinition, shape: &ExecutionShapeClass) -> bool {
    match (&variant.shape_class, shape) {
        // Same discriminant and same parameters → trivially compatible.
        (a, b) if a == b => true,

        // Decode1 can service a single-token decode requested via
        // DecodeBatch { max_batch: 1 }.
        (ExecutionShapeClass::Decode1, ExecutionShapeClass::DecodeBatch { max_batch: 1 }) => true,

        // A DecodeBatch variant can service Decode1 (batch-1 request
        // fits within any batch budget).
        (ExecutionShapeClass::DecodeBatch { .. }, ExecutionShapeClass::Decode1) => true,

        // A DecodeBatch variant with >= requested batch can service a
        // smaller DecodeBatch request.
        (
            ExecutionShapeClass::DecodeBatch {
                max_batch: variant_batch,
            },
            ExecutionShapeClass::DecodeBatch {
                max_batch: requested_batch,
            },
        ) => *variant_batch >= *requested_batch,

        // A PrefillBucket variant with >= requested tokens can service
        // a smaller PrefillBucket request.
        (
            ExecutionShapeClass::PrefillBucket {
                tokens: variant_tokens,
            },
            ExecutionShapeClass::PrefillBucket {
                tokens: requested_tokens,
            },
        ) => *variant_tokens >= *requested_tokens,

        // ChunkedPrefill, MixedBatch, DiffusionForward — only exact
        // match (already handled by the `a == b` arm above).
        _ => false,
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Extract the batch size implied by an `ExecutionShapeClass`.
fn extract_batch_size(shape: &ExecutionShapeClass) -> u32 {
    match shape {
        ExecutionShapeClass::Decode1 => 1,
        ExecutionShapeClass::DecodeBatch { max_batch } => *max_batch,
        ExecutionShapeClass::PrefillBucket { .. } => 1,
        ExecutionShapeClass::ChunkedPrefill { .. } => 1,
        ExecutionShapeClass::MixedBatch => 1, // mixed batches report batch=1;
        // the actual concurrency is
        // managed by the scheduler.
        ExecutionShapeClass::DiffusionForward { .. } => 1,
    }
}

/// Extract the peak token count implied by an `ExecutionShapeClass`.
fn extract_token_count(shape: &ExecutionShapeClass) -> u32 {
    match shape {
        ExecutionShapeClass::Decode1 => 1,
        ExecutionShapeClass::DecodeBatch { .. } => 1,
        ExecutionShapeClass::PrefillBucket { tokens } => *tokens,
        ExecutionShapeClass::ChunkedPrefill { chunk_tokens } => *chunk_tokens,
        ExecutionShapeClass::MixedBatch => 1,
        ExecutionShapeClass::DiffusionForward { max_canvas_tokens } => *max_canvas_tokens,
    }
}

/// Check whether the runtime snapshot advertises a named hardware feature.
///
/// Feature names are case-sensitive and should match the canonical set
/// used by the compiler when populating
/// [`ShapeVariantDefinition::required_hardware_features`].
fn runtime_has_feature(caps: &RuntimeCapabilitySnapshot, feature: &str) -> bool {
    match feature {
        "ane" => caps.has_ane,
        "metal" => caps.metal_available,
        "coreml" => caps.coreml_available,
        "unified_memory" => caps.has_unified_memory,
        "fp16" => caps.supports_fp16,
        "int8" => caps.supports_int8,
        _ => {
            // Unknown feature names may be forward-looking or misspelled.
            // Refuse conservatively.
            false
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_shape_compatible ──────────────────────────────────────────────

    #[test]
    fn exact_match_decode1() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode1".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(&variant, &ExecutionShapeClass::Decode1));
    }

    #[test]
    fn exact_match_decode_batch() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode_batch4".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 4 },
            description: String::new(),
            max_batch: Some(4),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 4 }
        ));
    }

    #[test]
    fn decode_batch_fits_smaller_request() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode_batch4".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 4 },
            description: String::new(),
            max_batch: Some(4),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 2 }
        ));
    }

    #[test]
    fn decode_batch_overflow_exact_incompatible() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode_batch2".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 2 },
            description: String::new(),
            max_batch: Some(2),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 4 }
        ));
    }

    #[test]
    fn decode_batch_covers_decode1() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode_batch4".into(),
            shape_class: ExecutionShapeClass::DecodeBatch { max_batch: 4 },
            description: String::new(),
            max_batch: Some(4),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(&variant, &ExecutionShapeClass::Decode1));
    }

    #[test]
    fn decode1_covers_decode_batch_1() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode1".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 1 }
        ));
    }

    #[test]
    fn decode1_does_not_cover_larger_decode_batch() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode1".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 2 }
        ));
    }

    #[test]
    fn prefill_bucket_fits_smaller_request() {
        let variant = ShapeVariantDefinition {
            variant_id: "prefill_large".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 32768 },
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(32768),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 4096 }
        ));
    }

    #[test]
    fn prefill_bucket_does_not_cover_larger_request() {
        let variant = ShapeVariantDefinition {
            variant_id: "prefill_small".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 512 },
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(512),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 4096 }
        ));
    }

    #[test]
    fn chunked_prefill_only_exact() {
        let variant = ShapeVariantDefinition {
            variant_id: "chunked_prefill".into(),
            shape_class: ExecutionShapeClass::ChunkedPrefill { chunk_tokens: 1024 },
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1024),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::ChunkedPrefill { chunk_tokens: 1024 }
        ));
        // Different parameter value is NOT compatible for ChunkedPrefill
        // (semantically different chunk sizes).
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::ChunkedPrefill { chunk_tokens: 2048 }
        ));
    }

    #[test]
    fn mixed_batch_only_exact() {
        let variant = ShapeVariantDefinition {
            variant_id: "mixed".into(),
            shape_class: ExecutionShapeClass::MixedBatch,
            description: String::new(),
            max_batch: None,
            max_tokens: None,
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::MixedBatch
        ));
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::Decode1
        ));
    }

    #[test]
    fn diffusion_forward_only_exact() {
        let variant = ShapeVariantDefinition {
            variant_id: "diffusion".into(),
            shape_class: ExecutionShapeClass::DiffusionForward {
                max_canvas_tokens: 16384,
            },
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(16384),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DiffusionForward {
                max_canvas_tokens: 16384,
            }
        ));
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::DiffusionForward {
                max_canvas_tokens: 8192,
            }
        ));
    }

    #[test]
    fn different_shape_classes_incompatible() {
        let variant = ShapeVariantDefinition {
            variant_id: "decode1".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec![],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 512 }
        ));
        assert!(!is_shape_compatible(
            &variant,
            &ExecutionShapeClass::MixedBatch
        ));
    }

    // ── check_variant_compatibility ──────────────────────────────────────

    fn full_caps() -> RuntimeCapabilitySnapshot {
        RuntimeCapabilitySnapshot {
            max_threadgroup_size: 1024,
            has_ane: true,
            ane_count: 1,
            has_unified_memory: true,
            supports_fp16: true,
            supports_int8: true,
            coreml_available: true,
            metal_available: true,
        }
    }

    fn minimal_caps() -> RuntimeCapabilitySnapshot {
        RuntimeCapabilitySnapshot {
            max_threadgroup_size: 256,
            has_ane: false,
            ane_count: 0,
            has_unified_memory: false,
            supports_fp16: false,
            supports_int8: false,
            coreml_available: false,
            metal_available: false,
        }
    }

    fn sample_variant() -> ShapeVariantDefinition {
        ShapeVariantDefinition {
            variant_id: "prefill_medium".into(),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(4096),
            required_hardware_features: vec!["metal".into(), "fp16".into()],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        }
    }

    #[test]
    fn fully_compatible() {
        let variant = sample_variant();
        let report = check_variant_compatibility(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 2048 },
            &full_caps(),
        );
        assert!(
            report.compatible,
            "expected compatible, got {:?}",
            report.violations
        );
        assert!(report.violations.is_empty());
        assert_eq!(report.variant_id, "prefill_medium");
    }

    #[test]
    fn shape_mismatch_reported() {
        let variant = sample_variant();
        let report =
            check_variant_compatibility(&variant, &ExecutionShapeClass::Decode1, &full_caps());
        assert!(!report.compatible);
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v, CompatibilityViolation::ShapeMismatch { .. })));
    }

    #[test]
    fn batch_overflow_reported() {
        let variant = sample_variant(); // max_batch=1
        let report = check_variant_compatibility(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 4 },
            &full_caps(),
        );
        assert!(!report.compatible);
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v, CompatibilityViolation::BatchOverflow { .. })));
    }

    #[test]
    fn token_overflow_reported() {
        let variant = sample_variant(); // max_tokens=4096
        let report = check_variant_compatibility(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 8192 },
            &full_caps(),
        );
        assert!(!report.compatible);
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v, CompatibilityViolation::TokenOverflow { .. })));
    }

    #[test]
    fn missing_hardware_feature_reported() {
        let variant = sample_variant(); // requires ["metal", "fp16"]
        let report = check_variant_compatibility(
            &variant,
            &ExecutionShapeClass::PrefillBucket { tokens: 2048 },
            &minimal_caps(),
        );
        assert!(!report.compatible);
        // Both metal and fp16 should be missing.
        let missing: Vec<&String> = report
            .violations
            .iter()
            .filter_map(|v| {
                if let CompatibilityViolation::MissingHardwareFeature(f) = v {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            missing.contains(&&"metal".to_string()),
            "expected 'metal' in {missing:?}"
        );
        assert!(
            missing.contains(&&"fp16".to_string()),
            "expected 'fp16' in {missing:?}"
        );
    }

    #[test]
    fn all_violations_collected_not_short_circuited() {
        // A radically incompatible request should produce every violation
        // type the variant's constraints can trigger.
        let variant = sample_variant(); // shape=Prefill{4096}, batch=1, tokens=4096, metal+fp16
        let report = check_variant_compatibility(
            &variant,
            &ExecutionShapeClass::DecodeBatch { max_batch: 8 },
            &minimal_caps(),
        );
        assert!(!report.compatible);
        // Expect: shape mismatch (Prefill vs DecodeBatch), batch overflow (1 < 8),
        // missing metal, missing fp16.  Token overflow is not triggered because
        // DecodeBatch has token count 1, which is <= max_tokens=4096.
        let kinds: Vec<&str> = report
            .violations
            .iter()
            .map(|v| match v {
                CompatibilityViolation::ShapeMismatch { .. } => "shape",
                CompatibilityViolation::BatchOverflow { .. } => "batch",
                CompatibilityViolation::TokenOverflow { .. } => "token",
                CompatibilityViolation::MissingHardwareFeature(_) => "hw",
                CompatibilityViolation::RuntimeVersionMismatch { .. } => "version",
            })
            .collect();
        assert!(
            kinds.contains(&"shape"),
            "expected shape violation: {kinds:?}"
        );
        assert!(
            kinds.contains(&"batch"),
            "expected batch violation: {kinds:?}"
        );
        assert!(
            !kinds.contains(&"token"),
            "unexpected token overflow: {kinds:?}"
        );
        assert_eq!(
            kinds.iter().filter(|k| *k == &"hw").count(),
            2,
            "expected two hw violations: {kinds:?}"
        );
    }

    #[test]
    fn unknown_feature_refused() {
        let variant = ShapeVariantDefinition {
            variant_id: "test".into(),
            shape_class: ExecutionShapeClass::Decode1,
            description: String::new(),
            max_batch: Some(1),
            max_tokens: Some(1),
            required_hardware_features: vec!["nonexistent_feature".into()],
            target_profile_label: "default".into(),
            program_hash: 0,
            program_data: vec![],
        };
        let report =
            check_variant_compatibility(&variant, &ExecutionShapeClass::Decode1, &full_caps());
        assert!(!report.compatible);
        assert!(matches!(
            &report.violations[0],
            CompatibilityViolation::MissingHardwareFeature(f) if f == "nonexistent_feature"
        ));
    }
}
