//! HardwareAssessmentPass — compile-time hardware profiling and kernel selection.
//! Runs during ComputeImage build to determine optimal kernel variants
//! for the target device.

use serde::{Serialize, Deserialize};

/// Result of profiling a single kernel variant on the target hardware.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelBenchResult {
    pub variant_name: String,
    pub backend: String,           // "mlx", "accelerate", "coreml", "neon"
    pub op_type: String,           // "matmul", "rms_norm", "softmax", "rope"
    pub shape: Vec<u32>,
    pub dtype: String,
    pub median_latency_ns: u64,
    pub min_latency_ns: u64,
    pub p90_latency_ns: u64,
    pub bandwidth_gbps: f64,
    pub throughput_ops_per_sec: f64,
    pub numerical_error: f64,      // relative error vs reference
    pub compile_time_ms: f64,
}

/// Per-lane benchmark result for a single op type, used in placement reports.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LaneBenchResult {
    pub lane: String,       // "mlx", "accelerate", "coreml"
    pub median_ns: u64,
    pub min_ns: u64,
    pub bandwidth_gbps: f64,
    pub numerical_error: f64,
}

/// Placement decision for a single op type, comparing all candidate lanes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlacementReport {
    pub op_type: String,
    pub shape: Vec<u32>,
    pub winner: String,                    // which lane won
    pub winner_latency_ns: u64,
    pub runner_up: String,
    pub runner_up_latency_ns: u64,
    pub ratio: f64,                        // runner_up / winner
    pub hazard_count: u32,                  // synchronization events needed
    pub total_transfer_bytes: u64,          // bytes moved between lanes
    pub lane_results: Vec<LaneBenchResult>,
}

/// A candidate kernel variant to benchmark.
#[derive(Clone, Debug)]
pub struct KernelCandidate {
    pub name: String,
    pub backend: String,
    pub op_type: String,
    pub function_constants: Vec<(String, u32)>,  // tile sizes, etc.
    pub threadgroup_size: Option<[u32; 3]>,
    pub metal_function: Option<String>,
    pub vdsp_function: Option<String>,
    pub coreml_subgraph: Option<String>,
}

/// Selected kernel configuration promoted to the live process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelSelection {
    pub op_type: String,
    pub shape_range: Vec<[u32; 2]>,  // min/max per dim
    pub selected_backend: String,
    pub selected_variant: String,
    pub expected_latency_ns: u64,
    pub fallback_backend: String,
    pub assessment_id: String,        // hash of the benchmark receipt
}

/// Complete hardware assessment receipt stored in ComputeImage manifest.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssessmentReceipt {
    pub target_device: String,
    pub device_family: String,
    pub has_unified_memory: bool,
    pub max_threadgroup_size: u32,
    pub thread_execution_width: u32,
    pub max_buffer_length: u64,
    pub recommended_max_working_set_size: u64,
    pub has_ane: bool,
    pub num_ane_cores: u32,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub selections: Vec<KernelSelection>,
    pub benchmark_results: Vec<KernelBenchResult>,
    pub placement_reports: Vec<PlacementReport>,
    pub assessment_duration_ms: u64,
    pub assessment_timestamp: String,
}

impl AssessmentReceipt {
    pub fn summary(&self) -> String {
        let selected = self.selections.len();
        let fastest_mlx = self.benchmark_results.iter()
            .filter(|r| r.backend == "mlx")
            .map(|r| r.median_latency_ns)
            .min().unwrap_or(0);
        format!(
            "HW-Assessment: {} selections, fastest MLX: {} ns, device: {}",
            selected, fastest_mlx, self.target_device
        )
    }
}

/// Hardware probe result (no FFI needed — struct only).
pub struct HardwareProbe {
    pub device_name: String,
    pub device_family: String,
    pub has_unified_memory: bool,
    pub max_threads_per_threadgroup: u32,
    pub thread_execution_width: u32,
    pub max_buffer_length: u64,
    pub recommended_max_working_set_size: u64,
    pub has_ane: bool,
    pub num_ane_cores: u32,
    pub supports_f16: bool,
    pub supports_bf16: bool,
}

impl HardwareProbe {
    /// Stub probe — returns generic Apple Silicon M1 capabilities.
    /// Real implementation calls Metal API at runtime.
    pub fn probe() -> Self {
        HardwareProbe {
            device_name: "Apple M1 (stub)".into(),
            device_family: "apple7".into(),
            has_unified_memory: true,
            max_threads_per_threadgroup: 1024,
            thread_execution_width: 32,
            max_buffer_length: 1024 * 1024 * 1024,
            recommended_max_working_set_size: 12 * 1024 * 1024 * 1024,
            has_ane: cfg!(target_os = "macos"),
            num_ane_cores: 16,
            supports_f16: true,
            supports_bf16: true,
        }
    }
}
