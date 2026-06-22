use crate::compute_image::phase_graph::{
    ArtifactBindingId, PhaseId, ResolvedPhaseBinding, TensorLayoutContract,
};
use serde::{Deserialize, Serialize};

/// Registry that resolves artifact bindings for phases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseBindingRegistry {
    pub bindings: Vec<ResolvedPhaseBinding>,
}

impl PhaseBindingRegistry {
    pub fn resolve(&self, phase_id: &PhaseId) -> Option<&ResolvedPhaseBinding> {
        self.bindings.iter().find(|b| &b.phase_id == phase_id)
    }

    pub fn register(&mut self, binding: ResolvedPhaseBinding) {
        self.bindings.push(binding);
    }
}
