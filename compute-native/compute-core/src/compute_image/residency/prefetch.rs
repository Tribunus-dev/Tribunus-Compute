//! Prefetch schedule builder for SealedComputeImageExecutable.
//!
//! The [`PrefetchScheduleBuilder`] produces an ordered list of
//! [`PrefetchAction`] items from the compiler's required weight
//! objects and the execution order of phases.
//!
//! The schedule builder applies the following rules:
//!
//! 1. [`ResidencyClass::MandatoryAtSessionStart`] — weights that are
//!    loaded at session start.  No prefetch action is emitted.
//!
//! 2. [`ResidencyClass::MandatoryBeforePhase`] — weights that must be
//!    resident before a specific phase.  The builder emits a high-
//!    priority prefetch action before the consuming phase.
//!
//! 3. [`ResidencyClass::PrefetchCandidate`] — weights that improve
//!    performance but are not required for correctness.  The builder
//!    emits a low-priority prefetch action.
//!
//! Other residency classes ([`ResidencyClass::ReusablePinned`],
//! [`ResidencyClass::EvictableAfterPhase`], [`ResidencyClass::DiskOnly`])
//! do not produce prefetch actions — they are handled by other parts
//! of the runtime memory manager.

use crate::compute_image::residency::plan::{
    PrefetchAction, PrefetchPriority, RequiredWeightObject, RequiredWeightObjectId,
};
use crate::compute_image::residency::ResidencyClass;

/// Builds a compiled prefetch schedule from the compiler's required
/// weight objects and the execution-phase topology.
///
/// The builder is a stateless utility — all inputs come via method
/// parameters — so it supports a single [`new`](Self::new) constructor
/// and no internal configuration.
#[derive(Debug, Clone, Default)]
pub struct PrefetchScheduleBuilder;

impl PrefetchScheduleBuilder {
    /// Create a new prefetch schedule builder.
    pub fn new() -> Self {
        Self
    }

    /// Build a prefetch schedule from the list of required weight objects
    /// and the execution order of phases.
    ///
    /// The resulting [`Vec<PrefetchAction>`] is ordered by phase —
    /// actions targeting earlier phases appear first.  Within the same
    /// phase, high-priority actions are emitted before low-priority
    /// actions.
    ///
    /// # Schedule rules
    ///
    /// | Residency class              | Prefetch action                      |
    /// |------------------------------|--------------------------------------|
    /// | `MandatoryAtSessionStart`    | None (loaded at session start)       |
    /// | `MandatoryBeforePhase`       | High priority, before consumer phase |
    /// | `PrefetchCandidate`          | Low priority, before first phase     |
    /// | `ReusablePinned`             | None                                 |
    /// | `EvictableAfterPhase`        | None                                 |
    /// | `DiskOnly`                   | None                                 |
    ///
    /// When the consuming phase for a `MandatoryBeforePhase` weight
    /// is not known, the builder conservatively schedules it before
    /// the first phase in `phase_order`.
    pub fn build_schedule(
        &self,
        required_objects: &[RequiredWeightObject],
        phase_order: &[String],
    ) -> Vec<PrefetchAction> {
        if required_objects.is_empty() || phase_order.is_empty() {
            return Vec::new();
        }

        // Determine which phases each weight is consumed by.
        // For MandatoryBeforePhase weights, we look at the weight name /
        // object_id patterns to infer the consuming phase, falling back
        // to the first phase when inference is not possible.
        let first_phase = &phase_order[0];

        let mut actions: Vec<PrefetchAction> = Vec::new();

        for obj in required_objects {
            match obj.residency_class {
                // Already loaded at session start — no prefetch needed.
                ResidencyClass::MandatoryAtSessionStart => {
                    continue;
                }

                // Must be resident before its consuming phase.
                ResidencyClass::MandatoryBeforePhase => {
                    let phase = self.infer_consumer_phase(&obj.object_id, phase_order);

                    // Deduplicate: skip if an action for this object
                    // already targets the same phase.
                    if actions
                        .iter()
                        .any(|a| a.object_id == obj.object_id && a.prefetch_before_phase == phase)
                    {
                        continue;
                    }

                    actions.push(PrefetchAction {
                        object_id: obj.object_id.clone(),
                        prefetch_before_phase: phase.clone(),
                        priority: PrefetchPriority::High,
                    });
                }

                // Performance improvement — load opportunistically.
                ResidencyClass::PrefetchCandidate => {
                    // Deduplicate across prefetch candidates.
                    if actions.iter().any(|a| a.object_id == obj.object_id) {
                        continue;
                    }

                    // Schedule low-priority before the first phase so
                    // the runtime can asynchronously start loading.
                    actions.push(PrefetchAction {
                        object_id: obj.object_id.clone(),
                        prefetch_before_phase: first_phase.clone(),
                        priority: PrefetchPriority::Low,
                    });
                }

                // These classes do not produce prefetch actions.
                ResidencyClass::ReusablePinned
                | ResidencyClass::EvictableAfterPhase
                | ResidencyClass::DiskOnly => {
                    continue;
                }
            }
        }

        // Order by phase index, then high-priority before low.
        self.sort_schedule(&mut actions, phase_order);
        actions
    }

    /// Insert a prefetch action for a weight object before a given phase.
    ///
    /// The action is inserted at the correct position in the schedule
    /// to maintain phase ordering: it will appear after all actions
    /// that target earlier phases and before all actions that target
    /// later phases.
    ///
    /// If an action with the same `object_id` and `phase_id` already
    /// exists in the schedule, this is a no-op (no duplicate entry
    /// is created).
    pub fn prefetch_before(
        &self,
        schedule: &mut Vec<PrefetchAction>,
        object_id: &str,
        phase_id: &str,
        priority: PrefetchPriority,
    ) {
        // Deduplicate.
        if schedule
            .iter()
            .any(|a| a.object_id == object_id && a.prefetch_before_phase == phase_id)
        {
            return;
        }

        let new_action = PrefetchAction {
            object_id: object_id.to_string(),
            prefetch_before_phase: phase_id.to_string(),
            priority,
        };

        // Find the insertion point: insert before the first action
        // whose phase comes after `phase_id`, or at the end.
        let insert_at = schedule
            .iter()
            .position(|a| a.prefetch_before_phase.as_str() > phase_id)
            .unwrap_or(schedule.len());

        schedule.insert(insert_at, new_action);
    }

    // ── Private helpers ──────────────────────────────────────────────

    /// Infer the consuming phase for a weight object.
    ///
    /// Looks for a phase whose ID is a substring in the object_id or
    /// vice versa.  Falls back to the first phase when no match is
    /// found.
    fn infer_consumer_phase(&self, object_id: &str, phase_order: &[String]) -> String {
        // Try to match the object_id against phase names by checking
        // if the phase ID appears as a substring in the object_id.
        for phase in phase_order {
            if object_id.contains(phase.as_str()) || phase.contains(object_id) {
                return phase.clone();
            }
        }

        // Fall back to the first phase.
        phase_order[0].clone()
    }

    /// Sort a prefetch schedule by phase order and priority.
    ///
    /// Actions targeting earlier phases appear first.  Within the
    /// same phase, high-priority actions precede low-priority ones.
    fn sort_schedule(&self, schedule: &mut Vec<PrefetchAction>, phase_order: &[String]) {
        // Build a phase-index lookup.
        use std::collections::HashMap;
        let phase_index: HashMap<&str, usize> = phase_order
            .iter()
            .enumerate()
            .map(|(i, p)| (p.as_str(), i))
            .collect();

        // Any phase not in phase_order gets a sentinel index.
        let default_index = phase_order.len();

        schedule.sort_by(|a, b| {
            let a_idx = phase_index
                .get(a.prefetch_before_phase.as_str())
                .copied()
                .unwrap_or(default_index);
            let b_idx = phase_index
                .get(b.prefetch_before_phase.as_str())
                .copied()
                .unwrap_or(default_index);

            a_idx
                .cmp(&b_idx)
                .then_with(|| a.priority.priority_rank().cmp(&b.priority.priority_rank()))
        });
    }


}

// ── Priority ordering helpers ────────────────────────────────────────

impl PrefetchPriority {
    /// Numeric rank for sorting: 0 = High (must come first), 1 = Low.
    fn priority_rank(&self) -> u8 {
        match self {
            PrefetchPriority::High => 0,
            PrefetchPriority::Low => 1,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn make_weight(id: &str, class: ResidencyClass, bytes: u64) -> RequiredWeightObject {
        RequiredWeightObject {
            object_id: RequiredWeightObjectId::from(id),
            residency_class: class,
            estimated_bytes: bytes,
        }
    }

    fn phase_order(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_empty_schedule() {
        let builder = PrefetchScheduleBuilder::new();
        let schedule = builder.build_schedule(&[], &phase_order(&["phase_1"]));
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_empty_phase_order() {
        let builder = PrefetchScheduleBuilder::new();
        let weight = make_weight("w_attn", ResidencyClass::MandatoryBeforePhase, 1024);
        let schedule = builder.build_schedule(&[weight], &[]);
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_mandatory_at_session_start_skipped() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight(
            "w_embed",
            ResidencyClass::MandatoryAtSessionStart,
            4096,
        )];
        let schedule = builder.build_schedule(&weights, &phase_order(&["phase_prefill"]));
        assert!(
            schedule.is_empty(),
            "MandatoryAtSessionStart weights must not produce prefetch actions"
        );
    }

    #[test]
    fn test_single_mandatory_before_phase() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight(
            "w_attn",
            ResidencyClass::MandatoryBeforePhase,
            1024,
        )];
        let phases = phase_order(&["phase_prefill", "phase_decode"]);
        let schedule = builder.build_schedule(&weights, &phases);

        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0].object_id, "w_attn");
        assert_eq!(schedule[0].prefetch_before_phase, "phase_prefill");
        assert_eq!(schedule[0].priority, PrefetchPriority::High);
    }

    #[test]
    fn test_prefetch_candidate_low_priority() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight("w_aux", ResidencyClass::PrefetchCandidate, 512)];
        let phases = phase_order(&["phase_decode"]);
        let schedule = builder.build_schedule(&weights, &phases);

        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0].object_id, "w_aux");
        assert_eq!(schedule[0].priority, PrefetchPriority::Low);
    }

    #[test]
    fn test_multiple_weights_spread_across_phases() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![
            make_weight("w_q", ResidencyClass::MandatoryBeforePhase, 512),
            make_weight("w_k", ResidencyClass::MandatoryBeforePhase, 512),
            make_weight("w_v", ResidencyClass::MandatoryBeforePhase, 512),
            make_weight("w_o", ResidencyClass::MandatoryBeforePhase, 512),
        ];
        let phases = phase_order(&["phase_attn", "phase_mlp"]);
        let schedule = builder.build_schedule(&weights, &phases);

        // All four are MandatoryBeforePhase — they all map to
        // `phase_attn` (the first phase) since no object_id matches
        // a phase name.  Each gets High priority.
        assert_eq!(schedule.len(), 4);
        for action in &schedule {
            assert_eq!(action.priority, PrefetchPriority::High);
            assert_eq!(action.prefetch_before_phase, "phase_attn");
        }
    }

    #[test]
    fn test_weights_across_phases_with_name_match() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![
            make_weight("phase_attn.w_q", ResidencyClass::MandatoryBeforePhase, 512),
            make_weight("phase_mlp.w_ff", ResidencyClass::MandatoryBeforePhase, 1024),
            make_weight("embed", ResidencyClass::MandatoryAtSessionStart, 4096),
            make_weight("aux_head", ResidencyClass::PrefetchCandidate, 256),
        ];
        let phases = phase_order(&["phase_attn", "phase_mlp"]);
        let schedule = builder.build_schedule(&weights, &phases);

        // `embed` (MandatoryAtSessionStart) → no action.
        // `phase_attn.w_q` → High, before `phase_attn`.
        // `phase_mlp.w_ff` → High, before `phase_mlp`.
        // `aux_head` → Low, before first phase (`phase_attn`).
        assert_eq!(schedule.len(), 3);

        // The schedule is sorted by phase, then priority.
        // phase_attn gets w_q (High) first, then aux_head (Low),
        // then phase_mlp gets w_ff (High).
        assert_eq!(schedule[0].object_id, "phase_attn.w_q");
        assert_eq!(schedule[0].prefetch_before_phase, "phase_attn");
        assert_eq!(schedule[0].priority, PrefetchPriority::High);

        assert_eq!(schedule[1].object_id, "aux_head");
        assert_eq!(schedule[1].prefetch_before_phase, "phase_attn");
        assert_eq!(schedule[1].priority, PrefetchPriority::Low);

        assert_eq!(schedule[2].object_id, "phase_mlp.w_ff");
        assert_eq!(schedule[2].prefetch_before_phase, "phase_mlp");
        assert_eq!(schedule[2].priority, PrefetchPriority::High);
    }

    #[test]
    fn test_prefetch_priority_assignment() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![
            make_weight("w_critical", ResidencyClass::MandatoryBeforePhase, 2048),
            make_weight("w_async", ResidencyClass::PrefetchCandidate, 1024),
        ];
        let phases = phase_order(&["phase_compute"]);
        let schedule = builder.build_schedule(&weights, &phases);

        assert_eq!(schedule.len(), 2);

        // High-priority action should be sorted before low-priority.
        assert_eq!(schedule[0].object_id, "w_critical");
        assert_eq!(schedule[0].priority, PrefetchPriority::High);

        assert_eq!(schedule[1].object_id, "w_async");
        assert_eq!(schedule[1].priority, PrefetchPriority::Low);
    }

    #[test]
    fn test_prefetch_before_inserts_in_order() {
        let builder = PrefetchScheduleBuilder::new();
        let mut schedule = Vec::new();

        builder.prefetch_before(&mut schedule, "w_mlp", "phase_mlp", PrefetchPriority::High);
        builder.prefetch_before(
            &mut schedule,
            "w_attn",
            "phase_attn",
            PrefetchPriority::High,
        );

        // `w_attn` targets an earlier phase → should be first.
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0].object_id, "w_attn");
        assert_eq!(schedule[1].object_id, "w_mlp");
    }

    #[test]
    fn test_prefetch_before_no_duplicate() {
        let builder = PrefetchScheduleBuilder::new();
        let mut schedule = Vec::new();

        builder.prefetch_before(
            &mut schedule,
            "w_attn",
            "phase_attn",
            PrefetchPriority::High,
        );
        builder.prefetch_before(&mut schedule, "w_attn", "phase_attn", PrefetchPriority::Low);

        // Second call with same object_id + phase_id is a no-op.
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0].priority, PrefetchPriority::High);
    }

    #[test]
    fn test_deduplication_built_schedule() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![
            make_weight("w_shared", ResidencyClass::MandatoryBeforePhase, 1024),
            make_weight("w_shared", ResidencyClass::MandatoryBeforePhase, 1024),
        ];
        let phases = phase_order(&["phase_1"]);
        let schedule = builder.build_schedule(&weights, &phases);

        // Duplicate weights with the same object_id and phase → one action.
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_reusable_pinned_skipped() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight("w_buf", ResidencyClass::ReusablePinned, 256)];
        let schedule = builder.build_schedule(&weights, &phase_order(&["phase_1"]));
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_evictable_after_phase_skipped() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight(
            "w_temp",
            ResidencyClass::EvictableAfterPhase,
            128,
        )];
        let schedule = builder.build_schedule(&weights, &phase_order(&["phase_1"]));
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_disk_only_skipped() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight("w_rare", ResidencyClass::DiskOnly, 64)];
        let schedule = builder.build_schedule(&weights, &phase_order(&["phase_1"]));
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_infer_consumer_phase_fallback() {
        let builder = PrefetchScheduleBuilder::new();
        let weights = vec![make_weight(
            "w_unknown",
            ResidencyClass::MandatoryBeforePhase,
            512,
        )];
        let phases = phase_order(&["phase_1", "phase_2"]);
        let schedule = builder.build_schedule(&weights, &phases);

        assert_eq!(schedule.len(), 1);
        // Fallback to first phase when no name match.
        assert_eq!(schedule[0].prefetch_before_phase, "phase_1");
    }
}
