//! Weight object descriptors and residency classification for
//! SealedComputeImageExecutable.
//!
//! The [`ResidencyClassifier`] assigns a [`ResidencyClass`] to each
//! weight object based on its usage pattern across phases and variants.
//! The output [`WeightObject`] list feeds into the compiler's prefetch
//! scheduling and eviction analysis passes.

use serde::{Deserialize, Serialize};

use crate::compute_image::content_store::index::{ArtifactConsumerRef, ContentObjectEntry, ResidencyClass};
use crate::integration::ContentHash;

/// Descriptor for a weight tensor whose residency has been classified.
///
/// Each entry captures the identity, size, assigned residency class,
/// and the set of phases that consume this weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightObject {
    /// Stable identifier for this weight tensor within the compute image.
    pub object_id: String,

    /// Content hash for deduplication and integrity verification.
    pub content_hash: ContentHash,

    /// Number of bytes this weight occupies in resident memory.
    pub byte_size: u64,

    /// Compiler-assigned residency class governing load/lifecycle.
    pub residency_class: ResidencyClass,

    /// Phases (within the current variant) that reference this weight.
    pub consumer_phases: Vec<String>,

    /// Phases during which this weight is expected to remain resident.
    pub estimated_lifetime_phases: Vec<String>,
}

/// Classifies weight objects into residency classes based on their
/// consumption pattern across phases and variants.
///
/// The classifier inspects each [`ContentObjectEntry`]'s consumer list
/// and compares it against the phase IDs of the current variant:
///
/// | Condition | Residency |
/// |---|---|
/// | Referenced by 0 phases in current variant | `DiskOnly` |
/// | Referenced by >80% of phases | `MandatoryAtSessionStart` |
/// | Referenced by exactly 1 phase | `MandatoryBeforePhase` |
/// | Referenced by phases outside current variant too | `ReusablePinned` |
/// | Otherwise | `EvictableAfterPhase` |
pub struct ResidencyClassifier;

impl ResidencyClassifier {
    /// Create a new classifier.
    pub fn new() -> Self {
        Self
    }

    /// Classify a single weight object based on its usage across phases.
    ///
    /// `all_phase_ids` must contain every phase identifier belonging to
    /// the current program variant.
    pub fn classify(
        &self,
        object: &ContentObjectEntry,
        all_phase_ids: &[String],
    ) -> ResidencyClass {
        let total = all_phase_ids.len();

        // Count how many phases in the current variant reference this object.
        let reference_count = all_phase_ids
            .iter()
            .filter(|phase_id| {
                object
                    .consumers
                    .iter()
                    .any(|c| &c.consumer_stage == *phase_id)
            })
            .count();

        // Not used in the current variant at all.
        if reference_count == 0 {
            return ResidencyClass::DiskOnly;
        }

        // Used by nearly every phase -- always resident.
        if (reference_count as f64) > (total as f64 * 0.8) {
            return ResidencyClass::MandatoryAtSessionStart;
        }

        // Used by exactly one phase -- load right before that phase.
        if reference_count == 1 {
            return ResidencyClass::MandatoryBeforePhase;
        }

        // Used by phases outside the current variant too -- keep pinned
        // for reuse when the variant switches.
        let referenced_in_other_variants = object
            .consumers
            .iter()
            .any(|c| !all_phase_ids.contains(&c.consumer_stage));

        if referenced_in_other_variants {
            return ResidencyClass::ReusablePinned;
        }

        // Used in this variant but not critically -- evictable.
        ResidencyClass::EvictableAfterPhase
    }

    /// Classify all weight objects for a given set of phases.
    ///
    /// Returns a [`WeightObject`] per entry, ordered by the input
    /// slice position.
    pub fn classify_all(
        &self,
        objects: &[ContentObjectEntry],
        all_phase_ids: &[String],
    ) -> Vec<WeightObject> {
        objects
            .iter()
            .map(|obj| {
                let residency_class = self.classify(obj, all_phase_ids);

                let consumer_phases: Vec<String> = all_phase_ids
                    .iter()
                    .filter(|phase_id| {
                        obj.consumers
                            .iter()
                            .any(|c| &c.consumer_stage == *phase_id)
                    })
                    .cloned()
                    .collect();

                let estimated_lifetime_phases = estimate_lifetime(&consumer_phases, all_phase_ids);

                WeightObject {
                    object_id: obj.object_id.clone(),
                    content_hash: obj.content_hash,
                    byte_size: obj.payload_bytes,
                    residency_class,
                    consumer_phases,
                    estimated_lifetime_phases,
                }
            })
            .collect()
    }
}

impl Default for ResidencyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate the phases during which a weight is resident.
///
/// For `MandatoryAtSessionStart` weights the lifetime covers all phases.
/// For other classes the lifetime spans from the first consuming phase
/// to the last consuming phase (inclusive), matching the order in
/// `all_phase_ids`.
fn estimate_lifetime(
    consumer_phases: &[String],
    all_phase_ids: &[String],
) -> Vec<String> {
    if consumer_phases.is_empty() || all_phase_ids.is_empty() {
        return vec![];
    }

    // Find the first and last consuming phase by position in all_phase_ids.
    let positions: Vec<usize> = consumer_phases
        .iter()
        .filter_map(|p| all_phase_ids.iter().position(|id| id == p))
        .collect();

    if positions.is_empty() {
        return vec![];
    }

    let first = *positions.iter().min().unwrap();
    let last = *positions.iter().max().unwrap();
    all_phase_ids[first..=last].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_consumer(phase: &str) -> ArtifactConsumerRef {
        ArtifactConsumerRef {
            artifact_id: format!("artifact_{phase}"),
            artifact_kind: "weight".into(),
            consumer_stage: phase.into(),
        }
    }

    fn make_object(object_id: &str, consumers: Vec<ArtifactConsumerRef>) -> ContentObjectEntry {
        ContentObjectEntry {
            object_id: object_id.into(),
            content_hash: ContentHash(0),
            object_kind: crate::compute_image::content_store::index::ContentObjectKind::CanonicalWeight,
            target_layout_id: "layout_1".into(),
            segment_id: "seg_0".into(),
            segment_offset: 0,
            payload_bytes: 1024,
            aligned_bytes: 1024,
            alignment: 64,
            logical_shape: vec![64, 64],
            storage_shape: vec![64, 64],
            physical_strides: vec![64, 1],
            dtype: "f16".into(),
            quantization: None,
            checksum: ContentHash(0),
            consumers,
            residency_class: crate::compute_image::content_store::index::ResidencyClass::DiskOnly,
        }
    }

    #[test]
    fn test_disk_only_when_no_phases_reference() {
        let phases = vec!["phase_a".into(), "phase_b".into(), "phase_c".into()];
        let obj = make_object("w1", vec![]);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::DiskOnly
        );
    }

    #[test]
    fn test_mandatory_at_session_start_when_every_phase_references() {
        let phases = vec!["p1".into(), "p2".into(), "p3".into(), "p4".into(), "p5".into()];
        // All 5 phases reference the object (>80% of 5 = >4 → all 5 qualifies)
        let consumers = vec![
            make_consumer("p1"),
            make_consumer("p2"),
            make_consumer("p3"),
            make_consumer("p4"),
            make_consumer("p5"),
        ];
        let obj = make_object("w2", consumers);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::MandatoryAtSessionStart
        );
    }

    #[test]
    fn test_mandatory_at_session_start_when_above_threshold() {
        // 6 phases, 5 reference → 5/6 ≈ 83.3% > 80%
        let phases = vec![
            "p1".into(), "p2".into(), "p3".into(),
            "p4".into(), "p5".into(), "p6".into(),
        ];
        let consumers = vec![
            make_consumer("p1"),
            make_consumer("p2"),
            make_consumer("p3"),
            make_consumer("p4"),
            make_consumer("p5"),
        ];
        let obj = make_object("w3", consumers);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::MandatoryAtSessionStart
        );
    }

    #[test]
    fn test_mandatory_before_phase_when_exactly_one_phase() {
        let phases = vec![
            "p1".into(), "p2".into(), "p3".into(), "p4".into(),
        ];
        let consumers = vec![make_consumer("p3")];
        let obj = make_object("w4", consumers);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::MandatoryBeforePhase
        );
    }

    #[test]
    fn test_reusable_pinned_when_used_in_other_variants() {
        // Object referenced by p3 in current variant AND by phases outside.
        let phases = vec!["p1".into(), "p2".into(), "p3".into()];
        let consumers = vec![
            make_consumer("p3"),
            make_consumer("other_variant_phase"),
        ];
        let obj = make_object("w5", consumers);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::ReusablePinned
        );
    }

    #[test]
    fn test_evictable_after_phase_when_some_but_not_exceptional() {
        let phases = vec!["p1".into(), "p2".into(), "p3".into(), "p4".into()];
        // Referenced by 2 phases (50% — not >80%, not ==1, not 0, and not cross-variant)
        let consumers = vec![make_consumer("p1"), make_consumer("p3")];
        let obj = make_object("w6", consumers);
        let classifier = ResidencyClassifier::new();
        assert_eq!(
            classifier.classify(&obj, &phases),
            ResidencyClass::EvictableAfterPhase
        );
    }

    #[test]
    fn test_classify_all_produces_correct_count_and_order() {
        let phases = vec!["p1".into(), "p2".into()];
        let objects = vec![
            make_object("o1", vec![]),
            make_object("o2", vec![make_consumer("p1")]),
            make_object("o3", vec![
                make_consumer("p1"),
                make_consumer("p2"),
                make_consumer("outside"),
            ]),
        ];
        let classifier = ResidencyClassifier::new();
        let results = classifier.classify_all(&objects, &phases);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].object_id, "o1");
        assert_eq!(results[0].residency_class, ResidencyClass::DiskOnly);
        assert_eq!(results[1].residency_class, ResidencyClass::MandatoryBeforePhase);
        assert_eq!(results[2].residency_class, ResidencyClass::ReusablePinned);
    }

    #[test]
    fn test_lifetime_spans_first_to_last_consumer() {
        let phases = vec![
            "p1".into(), "p2".into(), "p3".into(),
            "p4".into(), "p5".into(),
        ];
        let consumers = vec!["p2".into(), "p4".into()];
        let lifetime = estimate_lifetime(&consumers, &phases);
        assert_eq!(lifetime, vec!["p2", "p3", "p4"]);
    }

    #[test]
    fn test_lifetime_empty_when_no_consumers() {
        assert_eq!(estimate_lifetime(&[], &["a".into()]), Vec::<String>::new());
    }
}
