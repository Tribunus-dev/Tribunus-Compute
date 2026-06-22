use crate::inference::execution_image_state::ComputeImageState;
use crate::inference::inference_session_state::InferenceSessionState;
use crate::inference::inference_step_state::{InferenceStepState, InferenceStepOutput, InferenceMode};
use crate::scheduling::phase_engine::PhaseEngine;

/// Thin adapter that turns ProfiledInferenceSession methods into
/// PhaseEngine invocations.
///
/// This adapter is the bridge between the current imperative inference
/// loop and the PhaseEngine-driven execution path.
pub struct PhaseEngineAdapter;

impl PhaseEngineAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Execute prefill through the PhaseEngine.
    pub async fn execute_prefill(
        &self,
        engine: &PhaseEngine,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        step.mode = InferenceMode::Prefill;
        engine.execute_until_terminal(image, session, step).await
    }

    /// Execute decode through the PhaseEngine.
    pub async fn execute_decode(
        &self,
        engine: &PhaseEngine,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        step.mode = InferenceMode::Decode;
        engine.execute_until_terminal(image, session, step).await
    }
}
