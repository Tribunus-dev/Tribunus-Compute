//! Determinism and coverage tests for the variant system and coverage
//! analysis.
//!
//! These tests verify that:
//! - `analyze_coverage` is a pure, deterministic function
//! - Coverage correctly identifies missing and overlapping shape classes
//! - The coverage completeness check matches the required shape-class set
//! - Variant selection produces consistent results given identical inputs

use tribunus_compute_core::compute_image::execution_shape::ExecutionShapeClass;
use tribunus_compute_core::compute_image::variants::{
    analyze_coverage, is_coverage_complete, required_shape_classes, ShapeVariantDefinition,
};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a [`ShapeVariantDefinition`] with minimal fields for testing.
fn make_variant(id: &str, class: ExecutionShapeClass) -> ShapeVariantDefinition {
    ShapeVariantDefinition {
        variant_id: id.to_string(),
        shape_class: class,
        description: String::new(),
        max_batch: None,
        max_tokens: None,
        required_hardware_features: vec![],
        target_profile_label: "test-profile".to_string(),
        program_hash: 0,
        program_data: vec![],
    }
}

/// Build a [`ShapeVariantDefinition`] with an explicit `program_hash` so
/// that the caller can distinguish "different program, same shape" variants.
fn make_variant_with_hash(
    id: &str,
    class: ExecutionShapeClass,
    hash: u64,
) -> ShapeVariantDefinition {
    ShapeVariantDefinition {
        variant_id: id.to_string(),
        shape_class: class,
        description: String::new(),
        max_batch: None,
        max_tokens: None,
        required_hardware_features: vec![],
        target_profile_label: "test-profile".to_string(),
        program_hash: hash,
        program_data: vec![],
    }
}

// ── Test 1: Variant selection / coverage determinism ───────────────────────

#[test]
fn test_variant_selection_determinism() {
    let variants: Vec<ShapeVariantDefinition> = vec![
        make_variant("decode_v1", ExecutionShapeClass::Decode1),
        make_variant(
            "prefill_v1",
            ExecutionShapeClass::PrefillBucket { tokens: 4096 },
        ),
        make_variant("mixed_v1", ExecutionShapeClass::MixedBatch),
    ];

    let report_a = analyze_coverage(&variants);
    let report_b = analyze_coverage(&variants);

    assert_eq!(
        report_a.total_variants, report_b.total_variants,
        "total_variants must be stable",
    );
    assert_eq!(
        report_a.missing_shape_classes.len(),
        report_b.missing_shape_classes.len(),
        "missing count must be stable",
    );
    assert_eq!(
        report_a.overlapping_variants.len(),
        report_b.overlapping_variants.len(),
        "overlap count must be stable",
    );
    assert_eq!(
        report_a.coverage_ratio, report_b.coverage_ratio,
        "ratio must be identical",
    );

    for (a, b) in report_a
        .covered_shape_classes
        .iter()
        .zip(report_b.covered_shape_classes.iter())
    {
        assert_eq!(
            a.variant_name(),
            b.variant_name(),
            "covered classes must match in order",
        );
    }

    for (oa, ob) in report_a
        .overlapping_variants
        .iter()
        .zip(report_b.overlapping_variants.iter())
    {
        assert_eq!(oa.variant_ids, ob.variant_ids, "overlap ids must be stable",);
    }
}

// ── Test 2: Basic coverage analysis ───────────────────────────────────────

#[test]
fn test_coverage_basic() {
    let variants = vec![make_variant("d1", ExecutionShapeClass::Decode1)];
    let report = analyze_coverage(&variants);

    assert_eq!(report.total_variants, 1);
    assert_eq!(report.covered_shape_classes.len(), 1);

    assert!(report
        .covered_shape_classes
        .iter()
        .any(|sc| matches!(sc, ExecutionShapeClass::Decode1)));

    assert!(!report.missing_shape_classes.is_empty());
    assert!(report.missing_shape_classes.len() > 3);

    assert_eq!(report.overlapping_variants.len(), 0);

    let required_count = required_shape_classes().len();
    let expected_ratio = 1.0 / required_count as f64;
    assert!(
        (report.coverage_ratio - expected_ratio).abs() < f64::EPSILON,
        "expected coverage_ratio {expected_ratio}, got {}",
        report.coverage_ratio,
    );
}

// ── Test 3: Coverage completeness ─────────────────────────────────────────

#[test]
fn test_coverage_completeness() {
    let required = required_shape_classes();
    let variants: Vec<ShapeVariantDefinition> = required
        .iter()
        .enumerate()
        .map(|(i, sc)| make_variant(&format!("v{i}"), sc.clone()))
        .collect();

    assert!(
        is_coverage_complete(&variants),
        "an exhaustive variant set must produce complete coverage",
    );

    let report = analyze_coverage(&variants);
    assert_eq!(
        report.missing_shape_classes.len(),
        0,
        "exhaustive set must have zero missing shape classes",
    );
    assert!(
        (report.coverage_ratio - 1.0).abs() < f64::EPSILON,
        "coverage_ratio must be 1.0 for exhaustive set",
    );

    // Removing one variant produces incomplete coverage.
    let incomplete: Vec<ShapeVariantDefinition> = variants[1..].to_vec();
    assert!(
        !is_coverage_complete(&incomplete),
        "a set missing one required shape class must report incomplete coverage",
    );

    let report_incomplete = analyze_coverage(&incomplete);
    assert_eq!(
        report_incomplete.missing_shape_classes.len(),
        1,
        "exactly one shape class should be missing when one variant is removed",
    );
    assert!(
        report_incomplete.coverage_ratio < 1.0,
        "incomplete coverage must have ratio < 1.0",
    );
}

// ── Test 4: Overlap detection ────────────────────────────────────────────

#[test]
fn test_coverage_overlap_detection() {
    let variants = vec![
        make_variant("decode_metal", ExecutionShapeClass::Decode1),
        make_variant("decode_accelerate", ExecutionShapeClass::Decode1),
        make_variant("decode_coreml", ExecutionShapeClass::Decode1),
        make_variant(
            "prefill_v1",
            ExecutionShapeClass::PrefillBucket { tokens: 4096 },
        ),
        make_variant(
            "prefill_v2",
            ExecutionShapeClass::PrefillBucket { tokens: 4096 },
        ),
        make_variant("mixed_v1", ExecutionShapeClass::MixedBatch),
    ];

    let report = analyze_coverage(&variants);

    assert_eq!(
        report.overlapping_variants.len(),
        2,
        "expected two overlapping shape classes, got {}",
        report.overlapping_variants.len(),
    );

    for overlap in &report.overlapping_variants {
        assert!(
            overlap.variant_ids.len() >= 2,
            "overlap for {:?} has {} ids, expected at least 2",
            overlap.shape_class,
            overlap.variant_ids.len(),
        );
    }

    let decode_overlap = report
        .overlapping_variants
        .iter()
        .find(|o| matches!(o.shape_class, ExecutionShapeClass::Decode1))
        .expect("Decode1 must be in the overlap list");

    assert_eq!(decode_overlap.variant_ids.len(), 3);
    assert!(decode_overlap
        .variant_ids
        .contains(&"decode_metal".to_string()));
    assert!(decode_overlap
        .variant_ids
        .contains(&"decode_accelerate".to_string()));
    assert!(decode_overlap
        .variant_ids
        .contains(&"decode_coreml".to_string()));

    let prefill_overlap = report
        .overlapping_variants
        .iter()
        .find(|o| {
            matches!(
                o.shape_class,
                ExecutionShapeClass::PrefillBucket { tokens: 4096 }
            )
        })
        .expect("PrefillBucket must be in the overlap list");

    assert_eq!(prefill_overlap.variant_ids.len(), 2);
    assert!(prefill_overlap
        .variant_ids
        .contains(&"prefill_v1".to_string()));
    assert!(prefill_overlap
        .variant_ids
        .contains(&"prefill_v2".to_string()));

    let mixed_overlap = report
        .overlapping_variants
        .iter()
        .find(|o| matches!(o.shape_class, ExecutionShapeClass::MixedBatch));
    assert!(
        mixed_overlap.is_none(),
        "MixedBatch with a single variant should not appear in overlaps",
    );
}

// ── Test 5: Determinism with different program hashes ─────────────────────

#[test]
fn test_variant_hash_determinism() {
    let variants = vec![
        make_variant_with_hash("decode_fast", ExecutionShapeClass::Decode1, 100),
        make_variant_with_hash("decode_precise", ExecutionShapeClass::Decode1, 200),
        make_variant_with_hash(
            "prefill_default",
            ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            300,
        ),
    ];

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

    // Overlap details must be identical.
    for (oa, ob) in report_a
        .overlapping_variants
        .iter()
        .zip(report_b.overlapping_variants.iter())
    {
        assert_eq!(oa.variant_ids, ob.variant_ids);
    }
}
