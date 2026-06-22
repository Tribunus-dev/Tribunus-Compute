//! Dependency contract types for serialized phase programs.

#[derive(Debug, Clone)]
pub struct PhaseDependencyContract {
    pub dependencies_satisfied: bool,
}

#[derive(Debug, Clone)]
pub struct PhaseCompletionContract {
    pub must_emit_receipt: bool,
    pub must_release_regions: bool,
    pub must_advance_epoch: bool,
}
