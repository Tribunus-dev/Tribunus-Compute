use crate::compute_image::phase_dag::{EmittedPhase, EmittedPhaseGraph};
use crate::compute_image::phase_graph::ResolvedPhaseBinding;
use crate::inference::execution_image_state::ComputeImageState;
use crate::inference::inference_session_state::InferenceSessionState;
use crate::inference::inference_step_state::InferenceStepState;
use crate::scheduling::phase_engine_state::RuntimeWorkItemHandle;

/// Narrow typed invocation object passed to every PhaseRunner.
///
/// Runners access only through this object — no reaching back into
/// ProfiledInferenceSession or other global state.
pub struct PhaseInvocation<'a> {
    pub image: &'a ComputeImageState,
    pub session: &'a mut InferenceSessionState,
    pub step: &'a mut InferenceStepState,
    pub phase: &'a EmittedPhase,
    pub resolved_binding: &'a ResolvedPhaseBinding,
    pub work_item: &'a RuntimeWorkItemHandle,
}

impl<'a> PhaseInvocation<'a> {
    pub fn new(
        image: &'a ComputeImageState,
        session: &'a mut InferenceSessionState,
        step: &'a mut InferenceStepState,
        phase: &'a EmittedPhase,
        resolved_binding: &'a ResolvedPhaseBinding,
        work_item: &'a RuntimeWorkItemHandle,
    ) -> Self {
        Self {
            image,
            session,
            step,
            phase,
            resolved_binding,
            work_item,
        }
    }
}
