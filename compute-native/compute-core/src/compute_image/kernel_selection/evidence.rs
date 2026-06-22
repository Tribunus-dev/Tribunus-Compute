use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBenchmarkEvidence {
    pub candidate_id: String,
    pub operation_name: String,
    pub shape_class: String,
    pub median_latency_ns: u64,
    pub p95_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub sample_count: u32,
    pub resource_fit_pass: bool,
    pub numerical_verification_pass: bool,
    pub measurement_environment: MeasurementEnvironment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementEnvironment {
    pub hardware_model: String,
    pub os_version: String,
    pub gpu_name: String,
    pub driver_version: String,
    pub memory_bandwidth_gbps: f64,
    pub thermal_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCandidateEvidence {
    pub operation: String,
    pub shape_class: String,
    pub candidates: Vec<CandidateBenchmarkEvidence>,
    pub selected_candidate_id: String,
    pub selection_confidence: SelectionConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionConfidence {
    High,
    Medium,
    Low,
    Insufficient,
}

impl Default for CompiledCandidateEvidence {
    fn default() -> Self {
        Self {
            operation: String::new(),
            shape_class: String::new(),
            candidates: Vec::new(),
            selected_candidate_id: String::new(),
            selection_confidence: SelectionConfidence::Insufficient,
        }
    }
}
