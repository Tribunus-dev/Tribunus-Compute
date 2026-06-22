//! Variant selection — pick the best precompiled program variant for a given
//! request shape, without modifying or creating new variants.
//!
//! The selection algorithm:
//!
//! 1. **Filter** programs by shape-class compatibility via [`shapes_are_compatible`].
//! 2. **Select** the tightest-fitting compatible variant — the one whose
//!    capacity (batch or token count) is closest to the request without
//!    going under.
//! 3. **Reject** with a diagnostic error when no compatible variant exists.

use serde::{Deserialize, Serialize};

use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::compute_image::program::phase_program::SerializedPhaseProgram;

/// Error returned when [`select_program_variant`] cannot find a precompiled
/// program to satisfy the request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantSelectionRefusal {
    /// No variant exists with a compatible shape category for the request.
    NoMatchingVariant,

    /// A same-category variant exists but its shape parameters are
    /// insufficient for the request.  Carries the requested shape and
    /// the maximum shape the closest variant can support.
    ShapeOutOfBounds {
        /// The execution shape class that was requested.
        requested: ExecutionShapeClass,
        /// The execution shape class of the best-matching variant —
        /// its parameter(s) are the maximum the runtime can serve for
        /// this category.
        max_supported: ExecutionShapeClass,
    },

    /// A batched-decode variant exists but its batch capacity is exceeded.
    BatchSizeExceeded {
        /// Batch size requested.
        requested: u32,
        /// Maximum batch size among available variants in this category.
        max_supported: u32,
    },

    /// A prefill variant exists but its sequence-length capacity is exceeded.
    SequenceLengthExceeded {
        /// Token count requested.
        requested: u32,
        /// Maximum token count among available variants in this category.
        max_supported: u32,
    },

    /// The request requires a feature (e.g. a specific backend or precision)
    /// that none of the compiled variants provide.
    MissingRequiredFeature(String),
}

// ── Public API ───────────────────────────────────────────────────────────

/// Select the best precompiled program variant for the given request shape.
///
/// The function **only chooses among the provided `programs`** — it never
/// creates, compiles, or modifies variants.  If no variant satisfies the
/// request, it returns a [`VariantSelectionRefusal`] with a diagnostic
/// explaining why.
///
/// # Selection strategy
///
/// When multiple programs are compatible, the **tightest-fit** variant is
/// selected — the one whose capacity (batch or token count) is the smallest
/// that still satisfies the request.  This minimises waste while ensuring
/// correctness.
pub fn select_program_variant<'a>(
    programs: &'a [SerializedPhaseProgram],
    request_shape: &ExecutionShapeClass,
) -> Result<&'a SerializedPhaseProgram, VariantSelectionRefusal> {
    // ── Phase 1: find compatible programs ──
    let compatible: Vec<&'a SerializedPhaseProgram> = programs
        .iter()
        .filter(|p| shapes_are_compatible(&p.shape_class, request_shape))
        .collect();

    if !compatible.is_empty() {
        // ── Phase 2: pick the tightest fit ──
        return compatible
            .into_iter()
            .min_by_key(|p| capacity_score(&p.shape_class))
            .ok_or(VariantSelectionRefusal::NoMatchingVariant);
    }

    // ── Phase 3: no compatible variant — attempt a diagnostic ──
    //
    // Look for variants that share the same shape *category* as the
    // request.  If one exists, the request's parameters exceed the
    // maximum available capacity and we can return a specific bounds
    // error instead of a generic "no match".
    let same_category: Vec<&'a SerializedPhaseProgram> = programs
        .iter()
        .filter(|p| same_shape_category(&p.shape_class, request_shape))
        .collect();

    if let Some(best) = same_category
        .into_iter()
        .max_by_key(|p| capacity_score(&p.shape_class))
    {
        emit_bounds_error(request_shape, &best.shape_class)
    } else {
        Err(VariantSelectionRefusal::NoMatchingVariant)
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Returns `true` if a variant's compiled `ExecutionShapeClass` can satisfy a
/// request shape.  This is the shape-to-shape compatibility used by the
/// selection algorithm (as opposed to the full-variant-definition check in
/// [`super::compatibility::is_shape_compatible`]).
fn shapes_are_compatible(
    variant_shape: &ExecutionShapeClass,
    request_shape: &ExecutionShapeClass,
) -> bool {
    use ExecutionShapeClass::*;
    match (variant_shape, request_shape) {
        (&Decode1, &Decode1) => true,
        (&DecodeBatch { max_batch: ref v }, &DecodeBatch { max_batch: ref r }) => *v >= *r,
        (&PrefillBucket { tokens: ref v }, &PrefillBucket { tokens: ref r }) => *v >= *r,
        (&ChunkedPrefill { chunk_tokens: ref v }, &ChunkedPrefill { chunk_tokens: ref r }) => *v == *r,
        (&MixedBatch, &MixedBatch) => true,
        (&DiffusionForward { max_canvas_tokens: ref v }, &DiffusionForward { max_canvas_tokens: ref r }) => {
            *v >= *r
        }
        _ => false,
    }
}

/// Returns `true` if both shapes belong to the same top-level category
/// (decode, prefill, chunked prefill, mixed batch, or diffusion forward).
fn same_shape_category(a: &ExecutionShapeClass, b: &ExecutionShapeClass) -> bool {
    use ExecutionShapeClass::*;
    matches!(
        (a, b),
        (&Decode1, &Decode1)
            | (&DecodeBatch { .. }, &DecodeBatch { .. })
            | (&PrefillBucket { .. }, &PrefillBucket { .. })
            | (&ChunkedPrefill { .. }, &ChunkedPrefill { .. })
            | (&MixedBatch, &MixedBatch)
            | (&DiffusionForward { .. }, &DiffusionForward { .. })
    )
}

/// Score a shape class by its parameterised capacity, used for ordering.
/// Returns a single `u64` that increases with capacity.
fn capacity_score(shape: &ExecutionShapeClass) -> u64 {
    use ExecutionShapeClass::*;
    match shape {
        Decode1 => 1,
        DecodeBatch { max_batch } => *max_batch as u64,
        PrefillBucket { tokens } => *tokens as u64,
        ChunkedPrefill { chunk_tokens } => *chunk_tokens as u64,
        MixedBatch => u64::MAX,
        DiffusionForward { max_canvas_tokens } => *max_canvas_tokens as u64,
    }
}

/// Produce a diagnostic error for a same-category mismatch, extracting
/// the specific bound that was exceeded.
fn emit_bounds_error(
    request: &ExecutionShapeClass,
    max_supported: &ExecutionShapeClass,
) -> Result<&'static SerializedPhaseProgram, VariantSelectionRefusal> {
    use ExecutionShapeClass::*;
    match (request, max_supported) {
        (&DecodeBatch { max_batch: ref r }, &DecodeBatch { max_batch: ref v }) => {
            Err(VariantSelectionRefusal::BatchSizeExceeded {
                requested: *r,
                max_supported: *v,
            })
        }
        (&PrefillBucket { tokens: ref r }, &PrefillBucket { tokens: ref v }) => {
            Err(VariantSelectionRefusal::SequenceLengthExceeded {
                requested: *r,
                max_supported: *v,
            })
        }
        _ => Err(VariantSelectionRefusal::ShapeOutOfBounds {
            requested: request.clone(),
            max_supported: max_supported.clone(),
        }),
    }
}
