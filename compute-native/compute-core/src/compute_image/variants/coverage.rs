//! Variant coverage analysis — shape-class coverage, overlaps, and completeness.
//!
//! The compiler produces one or more [`ShapeVariantDefinition`] instances per
//! compile pipeline.  This module analyzes their coverage against the required
//! set of [`ExecutionShapeClass`] values, detecting gaps (missing shape classes)
//! and overlaps (multiple variants targeting the same shape class).
//!
//! Coverage is a prerequisite for admission: an executable with incomplete
//! shape-class coverage may be rejected at admission time for certain
//! execution profiles.

use std::collections::{BTreeSet, HashMap};

use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::compute_image::variants::shape_class::ShapeVariantDefinition;

/// Report detailing how well a set of variants covers the required shape
/// classes.
#[derive(Debug, Clone)]
pub struct VariantCoverageReport {
    /// Total number of variants submitted for analysis.
    pub total_variants: usize,
    /// Shape classes that have at least one variant.
    pub covered_shape_classes: Vec<ExecutionShapeClass>,
    /// Required shape classes with zero variants.
    pub missing_shape_classes: Vec<ExecutionShapeClass>,
    /// Shape classes where two or more distinct variants exist (overlap).
    pub overlapping_variants: Vec<OverlapDescriptor>,
    /// Fraction [0.0, 1.0] of required shape classes that are covered.
    pub coverage_ratio: f64,
}

/// Describes an overlap: one shape class served by multiple variant ids.
#[derive(Debug, Clone)]
pub struct OverlapDescriptor {
    /// The shape class with overlapping variants.
    pub shape_class: ExecutionShapeClass,
    /// The variant ids that all target this shape class.
    pub variant_ids: Vec<String>,
}

/// Analyze a slice of variants and produce a coverage report against the
/// full required shape-class set.
///
/// Deterministic: identical input slices always produce identical reports.
pub fn analyze_coverage(variants: &[ShapeVariantDefinition]) -> VariantCoverageReport {
    let required = required_shape_classes();
    let required_set: BTreeSet<ExecutionShapeClass> = required.iter().cloned().collect();

    // Collect unique variant ids per shape class.
    let mut class_to_ids: HashMap<ExecutionShapeClass, Vec<String>> = HashMap::new();
    for v in variants {
        class_to_ids
            .entry(v.shape_class.clone())
            .or_default()
            .push(v.variant_id.clone());
    }

    let covered_shape_classes: Vec<ExecutionShapeClass> = {
        let mut covered: Vec<ExecutionShapeClass> = class_to_ids.keys().cloned().collect();
        covered.sort_by(|a, b| a.variant_name().cmp(b.variant_name()));
        covered
    };

    let covered_set: BTreeSet<ExecutionShapeClass> = class_to_ids.keys().cloned().collect();

    // Missing: required but absent from covered_set.
    let missing_shape_classes: Vec<ExecutionShapeClass> = {
        let mut missing: Vec<ExecutionShapeClass> =
            required_set.difference(&covered_set).cloned().collect();
        missing.sort_by(|a, b| a.variant_name().cmp(b.variant_name()));
        missing
    };

    // Overlap: shape classes with >1 unique variant id.
    let overlapping_variants: Vec<OverlapDescriptor> = {
        let mut overlaps: Vec<OverlapDescriptor> = class_to_ids
            .into_iter()
            .filter(|(_, ids)| ids.len() > 1)
            .map(|(shape_class, variant_ids)| OverlapDescriptor {
                shape_class,
                variant_ids,
            })
            .collect();
        overlaps.sort_by(|a, b| {
            a.shape_class
                .variant_name()
                .cmp(b.shape_class.variant_name())
        });
        overlaps
    };

    let coverage_ratio = if required_set.is_empty() {
        1.0
    } else {
        covered_set.len() as f64 / required_set.len() as f64
    };

    VariantCoverageReport {
        total_variants: variants.len(),
        covered_shape_classes,
        missing_shape_classes,
        overlapping_variants,
        coverage_ratio,
    }
}

/// Return the canonical set of execution-shape classes that every production
/// executable MUST cover.
///
/// Parametric variants (`DecodeBatch`, `PrefillBucket`, `ChunkedPrefill`,
/// `DiffusionForward`) are enumerated at typical representative values —
/// admission gates MAY require a broader set depending on the target profile.
pub fn required_shape_classes() -> Vec<ExecutionShapeClass> {
    vec![
        ExecutionShapeClass::Decode1,
        ExecutionShapeClass::DecodeBatch { max_batch: 4 },
        ExecutionShapeClass::DecodeBatch { max_batch: 16 },
        ExecutionShapeClass::PrefillBucket { tokens: 4096 },
        ExecutionShapeClass::PrefillBucket { tokens: 8192 },
        ExecutionShapeClass::ChunkedPrefill { chunk_tokens: 1024 },
        ExecutionShapeClass::MixedBatch,
        ExecutionShapeClass::DiffusionForward {
            max_canvas_tokens: 16384,
        },
    ]
}

/// Returns `true` when the given variants cover every required shape class
/// (i.e. `missing_shape_classes` is empty).
pub fn is_coverage_complete(variants: &[ShapeVariantDefinition]) -> bool {
    analyze_coverage(variants).missing_shape_classes.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::execution_shape::ExecutionShapeClass;

    fn make_variant(variant_id: &str, shape_class: ExecutionShapeClass) -> ShapeVariantDefinition {
        ShapeVariantDefinition {
            variant_id: variant_id.to_string(),
            shape_class,
            description: String::new(),
            max_batch: None,
            max_tokens: None,
            required_hardware_features: vec![],
            target_profile_label: "test-profile".to_string(),
            program_hash: 0,
            program_data: vec![],
        }
    }

    #[test]
    fn empty_variants_reports_no_coverage() {
        let report = analyze_coverage(&[]);
        assert_eq!(report.total_variants, 0);
        assert!(report.covered_shape_classes.is_empty());
        assert!(!report.missing_shape_classes.is_empty());
        assert_eq!(report.coverage_ratio, 0.0);
        assert_eq!(report.overlapping_variants.len(), 0);
    }

    #[test]
    fn single_variant_partial_coverage() {
        let variants = vec![make_variant("v1", ExecutionShapeClass::Decode1)];
        let report = analyze_coverage(&variants);
        assert_eq!(report.total_variants, 1);
        assert_eq!(report.covered_shape_classes.len(), 1);
        assert!(report.coverage_ratio > 0.0);
        assert!(report.coverage_ratio < 1.0);
    }

    #[test]
    fn full_coverage_succeeds() {
        let required = required_shape_classes();
        let variants: Vec<ShapeVariantDefinition> = required
            .iter()
            .enumerate()
            .map(|(i, sc)| make_variant(&format!("v{i}"), sc.clone()))
            .collect();
        assert!(is_coverage_complete(&variants));
        let report = analyze_coverage(&variants);
        assert_eq!(report.missing_shape_classes.len(), 0);
        assert!((report.coverage_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn overlapping_variants_detected() {
        let variants = vec![
            make_variant("decode_a", ExecutionShapeClass::Decode1),
            make_variant("decode_b", ExecutionShapeClass::Decode1),
            make_variant(
                "prefill_a",
                ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            ),
            make_variant(
                "prefill_b",
                ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            ),
            make_variant("mixed", ExecutionShapeClass::MixedBatch),
        ];
        let report = analyze_coverage(&variants);
        assert_eq!(report.overlapping_variants.len(), 2);
        for overlap in &report.overlapping_variants {
            assert!(overlap.variant_ids.len() >= 2);
        }
    }

    #[test]
    fn coverage_is_deterministic() {
        let variants: Vec<ShapeVariantDefinition> = (0..5)
            .map(|i| {
                make_variant(
                    &format!("v{i}"),
                    if i < 2 {
                        ExecutionShapeClass::Decode1
                    } else if i < 4 {
                        ExecutionShapeClass::PrefillBucket { tokens: 4096 }
                    } else {
                        ExecutionShapeClass::MixedBatch
                    },
                )
            })
            .collect();

        let report_a = analyze_coverage(&variants);
        let report_b = analyze_coverage(&variants);

        assert_eq!(report_a.total_variants, report_b.total_variants);
        assert_eq!(report_a.coverage_ratio, report_b.coverage_ratio);
        assert_eq!(
            report_a.missing_shape_classes.len(),
            report_b.missing_shape_classes.len()
        );
        assert_eq!(
            report_a.overlapping_variants.len(),
            report_b.overlapping_variants.len()
        );
        // Same ids in the same order.
        for (a, b) in report_a
            .overlapping_variants
            .iter()
            .zip(report_b.overlapping_variants.iter())
        {
            assert_eq!(a.variant_ids, b.variant_ids);
        }
    }
}
