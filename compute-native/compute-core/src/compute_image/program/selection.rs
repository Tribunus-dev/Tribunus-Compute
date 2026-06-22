//! Program artifact selection and fallback chain types.

#[derive(Debug, Clone)]
pub struct ProgramArtifactSelection {
    pub artifact_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DeclaredFallbackChain {
    pub primary: Vec<String>,
    pub fallbacks: Vec<FallbackStep>,
}

#[derive(Debug, Clone)]
pub struct FallbackStep {
    pub artifact_id: String,
    pub condition: String,
}
