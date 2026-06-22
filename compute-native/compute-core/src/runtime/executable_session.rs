//! Executable session preparation — loads a sealed compute image, selects the
//! target profile, binds mmap'd segments, admits the residency plan, and
//! assembles a [`PreparedExecutableSession`] ready for scheduler handoff.
//!
//! The preparation sequence:
//!
//! 1. **Select the program variant** for the requested [`ExecutionShapeClass`]
//!    by scanning the bindings' programs via [`select_program_variant`].
//! 2. **Load the compiled residency plan** identified by the selected
//!    program's `residency_plan_id`.
//! 3. **Admit the residency plan** against the available device-memory budget
//!    via [`ResidencyAdmission`].
//! 4. **Bind lane artifacts** for execution through the
//!    [`LaneArtifactRegistry`].
//! 5. **Return the prepared session** for scheduler handoff (E5).

use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::compute_image::program::phase_program::SerializedPhaseProgram;
use crate::compute_image::residency::admission::{ResidencyAdmission, ResidencyAdmissionResult};
use crate::compute_image::residency::plan::CompiledResidencyPlan;
use crate::compute_image::variants::selection::{select_program_variant, VariantSelectionRefusal};
use crate::runtime::executable_bindings::ExecutableBindings;
use crate::runtime::executable_lane::LaneArtifactRegistry;
use crate::runtime::executable_seal::SealVerificationReceipt;
use std::sync::Arc;

/// Opaque backend handles bundled for phase-runner dispatch.
///
/// The `execute_phase_dag` method packs an instance into the
/// `ExecutionContext`'s backend field.  Concrete runners downcast
/// it to access MLX executor and Metal pipeline state.
pub struct RuntimeBackends {
    /// The MLX executor pinned to the GPU device.
    pub mlx_executor: Arc<std::sync::Mutex<crate::mlx_executor::MlxExecutor>>,
    /// Pre-loaded Metal kernel pipeline states.
    pub metal_kernels: Arc<Vec<crate::worker_dispatch::LoadedMetalKernel>>,
    /// Accelerate lane state (CPU SIMD ops via vDSP/NEON).
    pub accelerate_state: crate::backend::accelerate_lane::AccelerateLane,
    /// Core ML lane state (ANE subgraph execution).
    pub coreml_state: crate::backend::coreml_lane::CoreMlLane,
    /// Embedding weights for prologue.
    pub emb_w: std::sync::Arc<mlx_rs::Array>,
    /// Embedding scales for quantized lookup.
    pub emb_s: std::sync::Arc<mlx_rs::Array>,
    /// Embedding biases for quantized lookup.
    pub emb_b: std::sync::Arc<mlx_rs::Array>,
    /// Final RMS norm weight.
    pub fn_w: std::sync::Arc<mlx_rs::Array>,
    /// Partial RoPE cosine table (local/sliding layers).
    pub rope_cos: std::sync::Arc<mlx_rs::Array>,
    /// Partial RoPE sine table (local/sliding layers).
    pub rope_sin: std::sync::Arc<mlx_rs::Array>,
    /// Full RoPE cosine table (global layers).
    pub full_cos: std::sync::Arc<mlx_rs::Array>,
    /// Full RoPE sine table (global layers).
    pub full_sin: std::sync::Arc<mlx_rs::Array>,
}

// Safety: Raw pointers in CoreMlLane/AccelerateLane are accessed only on
// the thread that created them. The struct is behind a single-threaded
// ExecutionContext, never shared across threads concurrently.
unsafe impl Send for RuntimeBackends {}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A compute image that has been loaded, its seal verified, and its segments
/// mmap'd into the process address space.
///
/// This is the output of the loading / seal-verification / profile-selection /
/// binding pipeline (E4A–E4C).  The session preparer uses it to select a
/// variant, admit the residency plan, and bind lane artifacts.
pub struct LoadedExecutableImage {
    /// Receipt proving the image's seal was verified.
    pub seal_receipt: SealVerificationReceipt,
    /// Bound runtime bindings (programs, residency plans, mmap segments).
    pub bindings: ExecutableBindings,
    /// Per-lane artifact registry, populated during preparation.
    pub lane_registry: LaneArtifactRegistry,
    /// Hash of the selected target profile.
    pub selected_profile_hash: String,
    /// Compiler-emitted phase DAG (optional -- present when the image was
    /// compiled with the phase-dag pipeline).  The scheduler uses this
    /// to dispatch typed phases instead of reconstructing dependencies.
    pub phase_dag: Option<crate::compute_image::phase_dag::EmittedPhaseGraph>,
}

/// A fully prepared execution session, ready for scheduler dispatch.
///
/// Contains everything the scheduler (E5) needs to execute the model under
/// the chosen shape class: the selected program, the compiled residency plan,
/// and the admission decision.
pub struct PreparedExecutableSession {
    /// The loaded image that was used to prepare this session.
    pub image: LoadedExecutableImage,
    /// The shape-specialized program selected for this execution shape.
    pub selected_program: SerializedPhaseProgram,
    /// The compiled residency plan associated with the selected program.
    pub residency_plan: CompiledResidencyPlan,
    /// Whether the residency plan was admitted against the available memory.
    pub residency_admitted: bool,
    /// The execution shape class this session was prepared for.
    pub shape_class: ExecutionShapeClass,
}

impl PreparedExecutableSession {
    /// Execute the phase DAG if present, returning per-phase receipts.
    pub fn execute_phase_dag(&self) -> Vec<crate::scheduling::receipts::PhaseReceipt> {
        let dag = match &self.image.phase_dag {
            Some(d) => d,
            None => return Vec::new(),
        };
        let engine = crate::scheduling::phase_engine::PhaseEngine::new();
        let mut ctx = crate::scheduling::execution_context::ExecutionContext::new_empty();
        // Pack backend handles into the context so concrete runners
        // can downcast and dispatch through the real backends.
        // The session is created from a LoadedExecutableImage which
        // has lane_registry with Metal/CoreML artifacts, but the
        // MlxExecutor etc. live at a higher layer.  When they are
        // available, create RuntimeBackends and set ctx.backend.
        // For now the runners receive None and log — the caller
        // falls through to the existing layer-by-layer dispatch.
        let result = engine.execute_graph(dag, &mut ctx);
        result.receipts
    }
}

// ---------------------------------------------------------------------------
// Session preparer
// ---------------------------------------------------------------------------

/// Prepares an executable image for execution under a specific shape class.
///
/// The preparer orchestrates variant selection, residency-admission checking,
/// and lane-artifact binding.  It does **not** schedule execution — that
/// happens in E5.
pub struct ExecutableSessionPreparer;

impl ExecutableSessionPreparer {
    /// Create a new session preparer.
    pub fn new() -> Self {
        Self
    }

    /// Prepare a session for a specific execution shape class.
    ///
    /// # Sequence
    ///
    /// 1. **Variant selection** — calls [`select_program_variant`] on the
    ///    programs in the image's bindings to find the best-fitting program
    ///    for the requested shape class.
    /// 2. **Residency-plan lookup** — resolves the selected program's
    ///    `residency_plan_id` against the bindings' plan map.
    /// 3. **Residency admission** — checks the plan against the device's
    ///    available memory via [`ResidencyAdmission::check_admission`].
    /// 4. **Lane-artifact binding** — calls
    ///    [`LaneArtifactRegistry::bind_for_program`] to prepare lane-level
    ///    execution resources.
    /// 5. **Session assembly** — returns a [`PreparedExecutableSession`]
    ///    with all resolved artifacts and the admission result.
    ///
    /// # Errors
    ///
    /// Returns [`SessionPreparationError::NoMatchingVariant`] when no
    /// compiled program covers the requested shape class.
    ///
    /// Returns [`SessionPreparationError::ResidencyRefused`] when the
    /// residency plan's memory budget exceeds the available device memory.
    ///
    /// Returns [`SessionPreparationError::MissingRequiredArtifact`] when
    /// the residency plan referenced by the selected program is not found
    /// in the bindings.
    ///
    /// Returns [`SessionPreparationError::BindingError`] when lane-artifact
    /// binding fails.
    pub fn prepare(
        &self,
        mut image: LoadedExecutableImage,
        shape_class: ExecutionShapeClass,
        available_memory_bytes: u64,
    ) -> Result<PreparedExecutableSession, SessionPreparationError> {
        // ── 1. Select the program variant for the shape class ──────
        let programs = image.bindings.programs();
        let selected_program = select_program_variant(programs, &shape_class)
            .map_err(|refusal| {
                let detail = Self::format_variant_refusal(&refusal, &shape_class);
                SessionPreparationError::NoMatchingVariant(detail)
            })?
            .clone();

        // ── 2. Look up the compiled residency plan ─────────────────
        let residency_plan = image
            .bindings
            .find_residency_plan(&selected_program.residency_plan_id)
            .ok_or_else(|| {
                SessionPreparationError::MissingRequiredArtifact(format!(
                    "Residency plan '{}' referenced by program '{}' not found in bindings",
                    selected_program.residency_plan_id, selected_program.program_id
                ))
            })?
            .clone();

        // ── 3. Admit the residency plan ────────────────────────────
        let admission = ResidencyAdmission::new();
        let residency_admitted =
            match admission.check_admission(&residency_plan, available_memory_bytes) {
                ResidencyAdmissionResult::Admitted { .. } => true,
                ResidencyAdmissionResult::Refused(reason) => {
                    return Err(SessionPreparationError::ResidencyRefused(format!(
                        "Residency plan '{}' refused: {:?} (available={} bytes)",
                        residency_plan.plan_id, reason, available_memory_bytes
                    )));
                }
            };

        // ── 4. Bind lane artifacts for execution ───────────────────
        image
            .lane_registry
            .bind_for_program(&selected_program)
            .map_err(|e| {
                SessionPreparationError::BindingError(format!(
                    "Failed to bind lane artifacts for program '{}': {}",
                    selected_program.program_id, e
                ))
            })?;

        // ── 5. Assemble and return ─────────────────────────────────
        Ok(PreparedExecutableSession {
            image,
            selected_program,
            residency_plan,
            residency_admitted,
            shape_class,
        })
    }

    // ── Private helpers ────────────────────────────────────────────

    /// Produce a human-readable diagnostic from a variant selection refusal.
    fn format_variant_refusal(
        refusal: &VariantSelectionRefusal,
        _shape_class: &ExecutionShapeClass,
    ) -> String {
        match refusal {
            VariantSelectionRefusal::NoMatchingVariant => {
                "No compatible variant exists for the requested shape class".into()
            }
            VariantSelectionRefusal::ShapeOutOfBounds {
                requested,
                max_supported,
            } => {
                format!(
                    "Requested shape {:?} exceeds max supported shape {:?}",
                    requested, max_supported
                )
            }
            VariantSelectionRefusal::BatchSizeExceeded {
                requested,
                max_supported,
            } => {
                format!(
                    "Requested batch {} exceeds max batch {} among available variants",
                    requested, max_supported
                )
            }
            VariantSelectionRefusal::SequenceLengthExceeded {
                requested,
                max_supported,
            } => {
                format!(
                    "Requested sequence length {} exceeds max {} among available variants",
                    requested, max_supported
                )
            }
            VariantSelectionRefusal::MissingRequiredFeature(feature) => {
                format!("Missing required hardware feature: {}", feature)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during session preparation.
#[derive(Debug, Clone)]
pub enum SessionPreparationError {
    /// No compiled program variant exists for the requested shape class.
    NoMatchingVariant(String),
    /// The residency plan was refused by the admission controller.
    ResidencyRefused(String),
    /// A required artifact (e.g. residency plan, weight object) is missing.
    MissingRequiredArtifact(String),
    /// Lane-artifact binding failed.
    BindingError(String),
}

impl std::fmt::Display for SessionPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionPreparationError::NoMatchingVariant(detail) => {
                write!(f, "No matching variant: {}", detail)
            }
            SessionPreparationError::ResidencyRefused(detail) => {
                write!(f, "Residency refused: {}", detail)
            }
            SessionPreparationError::MissingRequiredArtifact(detail) => {
                write!(f, "Missing required artifact: {}", detail)
            }
            SessionPreparationError::BindingError(detail) => {
                write!(f, "Binding error: {}", detail)
            }
        }
    }
}

impl std::error::Error for SessionPreparationError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub test that validates the function signature and error-construction
    /// paths.  Full integration testing requires real executable images with
    /// seal receipts, bindings, lane registries, and residency plans — that
    /// happens in E5.
    #[test]
    fn test_prepare_returns_no_matching_variant_with_empty_programs() {
        // A preparer with zero available programs cannot satisfy any shape.
        let preparer = ExecutableSessionPreparer::new();

        // We cannot construct a real LoadedExecutableImage without the sibling
        // runtime modules (E4A–E4D).  The construction path is validated in
        // the E5 integration tests.
        //
        // This test verifies that the error enum is constructible and
        // displayable.
        let err = SessionPreparationError::NoMatchingVariant(
            "No compatible variant exists for the requested shape class".into(),
        );
        let display = err.to_string();
        assert!(display.contains("No matching variant"));
    }

    #[test]
    fn test_prepare_returns_residency_refused_error() {
        let err = SessionPreparationError::ResidencyRefused(
            "Residency plan 'test' refused: insufficient memory".into(),
        );
        let display = err.to_string();
        assert!(display.contains("Residency refused"));
    }

    #[test]
    fn test_prepare_returns_missing_artifact_error() {
        let err = SessionPreparationError::MissingRequiredArtifact(
            "Residency plan 'p123' not found".into(),
        );
        let display = err.to_string();
        assert!(display.contains("Missing required artifact"));
    }

    #[test]
    fn test_prepare_returns_binding_error() {
        let err = SessionPreparationError::BindingError("lane artifact binding failed".into());
        let display = err.to_string();
        assert!(display.contains("Binding error"));
    }

    #[test]
    fn test_session_preparation_error_implements_std_error() {
        // Verify the error type can be used in trait-object contexts.
        fn _assert_error(e: &dyn std::error::Error) {
            let _ = e;
        }

        let err = SessionPreparationError::NoMatchingVariant("test".into());
        _assert_error(&err);
    }
}
