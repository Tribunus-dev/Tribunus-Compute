//! Core ML execution lane — compiled subgraph accelerator.
//!
//! Core ML compiles subgraphs (MLP bundles, projection sets, fixed-shape
//! prefill segments) into .mlmodelc packages with explicit input/output
//! tensor contracts. The lane invokes them on the ANE when shapes match
//! and dispatch overhead is acceptable.
//!
//! This is NOT an op-by-op backend. Core ML subgraphs must be shape-stable
//! and large enough to amortize the compilation and dispatch cost.
//!
//! Lane state includes subgraph compilation status, timing telemetry, and
//! availability probes. The caller (scheduler / compute-image phase) is
//! responsible for submitting subgraphs for compilation via the full
//! MIL → coremlc pipeline and checking `can_execute` before dispatch.

use std::path::Path;
use std::time::Instant;

use tempfile::TempDir;

use crate::compute_image::hw_assessment::KernelBenchResult;
use crate::coreml_pipeline;

/// Compute profile for the Core ML execution lane.
///
/// Maps to Apple's [`MLComputeUnits`] values used when loading Core ML models.
/// The recommended default for flexible CPU/ANE execution is [`CpuAndNeuralEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeProfile {
    CpuOnly,
    CpuAndNeuralEngine,
    NeuralEngineOnly,
    GpuOnly,
    All,
}

impl ComputeProfile {
    pub fn name(&self) -> &'static str {
        match self {
            ComputeProfile::CpuOnly => "cpuOnly",
            ComputeProfile::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            ComputeProfile::NeuralEngineOnly => "neuralEngine",
            ComputeProfile::GpuOnly => "gpuOnly",
            ComputeProfile::All => "all",
        }
    }
}

/// Status of a Core ML compiled subgraph.
#[derive(Clone, Debug)]
pub enum CoreMlSubgraphStatus {
    /// Compiled and ready for inference
    Compiled { model_path: String },
    /// Compilation failed — will fallback to MLX
    CompileFailed { reason: String },
    /// Not attempted yet
    Pending,
    /// Shape mismatch — subgraph cannot run on this input
    ShapeMismatch {
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
}

/// A compiled Core ML subgraph.
pub struct CoreMlSubgraph {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub status: CoreMlSubgraphStatus,
    pub compile_time_ms: f64,
    pub inference_time_ms: f64,
}

impl CoreMlSubgraph {
    pub fn new(name: &str) -> Self {
        CoreMlSubgraph {
            name: name.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            status: CoreMlSubgraphStatus::Pending,
            compile_time_ms: 0.0,
            inference_time_ms: 0.0,
        }
    }

    /// Compile this subgraph via coremlc.
    pub fn compile(&mut self, _mil_text: &str, _output_dir: &Path) -> Result<(), String> {
        // Stub: real compilation would call xcrun coremlc compile.
        // Since Core ML compilation requires the full ML pipeline,
        // this is deferred to the Core ML compute-image compile pass.
        Err("Core ML subgraph compilation requires the full compute-image pipeline".to_string())
    }

    /// Run inference on this compiled subgraph.
    ///
    /// If the subgraph has `Compiled` status, loads the .mlmodelc via the
    /// coreml bridge and runs prediction. Measures inference wall time.
    /// Returns inference time in milliseconds.
    pub fn infer(&self, input_data: &[f32], output_data: &mut [f32]) -> Result<f64, String> {
        let model_path = match &self.status {
            CoreMlSubgraphStatus::Compiled { model_path } => model_path.clone(),
            _ => return Err("Core ML subgraph not compiled".to_string()),
        };
        let dim = input_data.len();
        if output_data.len() != dim {
            return Err(format!(
                "Core ML infer: input/output size mismatch: {} vs {}",
                dim,
                output_data.len()
            ));
        }

        let start = Instant::now();
        let model = crate::coreml_bridge::CoreMlModel::load(&model_path)?;

        let input_arena = crate::arena_info::ArenaInfo {
            width: 1,
            height: dim as i32,
            logical_dim0: 1,
            logical_dim1: dim as i32,
            pixel_format: 0,
            byte_size: (dim as i32) * 4,
            bytes_per_row: (dim as i32) * 4,
            base_address: input_data.as_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };
        let output_arena = crate::arena_info::ArenaInfo {
            width: 1,
            height: dim as i32,
            logical_dim0: 1,
            logical_dim1: dim as i32,
            pixel_format: 0,
            byte_size: (dim as i32) * 4,
            bytes_per_row: (dim as i32) * 4,
            base_address: output_data.as_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };

        model.predict("input", &input_arena, "matmul_1", &output_arena)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        Ok(elapsed)
    }
}

/// Core ML execution lane.
pub struct CoreMlLane {
    pub name: String,
    pub subgraphs: Vec<CoreMlSubgraph>,
    pub is_available: bool,
    pub compute_profile: ComputeProfile,
}

impl CoreMlLane {
    pub fn new() -> Self {
        // Probe for Core ML availability
        let is_available = cfg!(target_os = "macos");
        CoreMlLane {
            name: "coreml-ane".into(),
            subgraphs: Vec::new(),
            is_available,
            compute_profile: ComputeProfile::CpuAndNeuralEngine,
        }
    }

    /// Check if a subgraph is compiled and ready for the given input shape.
    pub fn can_execute(&self, subgraph_name: &str) -> bool {
        self.subgraphs.iter().any(|sg| {
            sg.name == subgraph_name && matches!(sg.status, CoreMlSubgraphStatus::Compiled { .. })
        })
    }

    pub fn add_subgraph(&mut self, subgraph: CoreMlSubgraph) {
        self.subgraphs.push(subgraph);
    }

    /// Compile a minimal test subgraph and benchmark it.
    ///
    /// Compiles a 256x256 F32 matmul via [`coreml_pipeline::build_matmul_region`]
    /// (which uses `cpuAndNeuralEngine` compute profile), loads the compiled
    /// `.mlmodelc`, runs 1 warmup + 10 timed iterations, and returns measured
    /// latency statistics.
    ///
    /// Returns None if Core ML is unavailable, compilation fails, or inference fails.
    pub fn bench_minimal_subgraph(&self) -> Option<KernelBenchResult> {
        eprintln!(
            "[coreml-bench] using {} compute profile",
            self.compute_profile.name()
        );

        if !self.is_available {
            return None;
        }

        // Compile a minimal matmul benchmark model with cpuAndNeuralEngine profile.
        eprintln!("[coreml-bench] compiling benchmark subgraph...");
        let compile_dir = TempDir::new().ok()?;
        let compile_dir_path = compile_dir.path().to_path_buf();

        let receipt = coreml_pipeline::build_matmul_region(
            "input",
            &[256, 256],
            "weight",
            &[1.0f32; 256 * 256],
            &[256, 256],
            &compile_dir_path,
            "coreml-bench-identity",
        )
        .ok()?;

        eprintln!(
            "[coreml-bench] compiled: {} (hash={})",
            receipt.compiled_modelc_path, receipt.compiled_hash
        );

        let model = crate::coreml_bridge::CoreMlModel::load(&receipt.compiled_modelc_path).ok()?;

        // 256x256 float32 — large enough to measure real ANE dispatch.
        let dim = 256u32;
        let n = (dim * dim) as usize;

        let input_data = vec![1.0f32; n];
        let mut output_data = vec![0.0f32; n];

        let input_arena = crate::arena_info::ArenaInfo {
            width: dim as i32,
            height: dim as i32,
            logical_dim0: dim as i32,
            logical_dim1: dim as i32,
            pixel_format: 0,
            byte_size: (n as i32) * 4,
            bytes_per_row: (dim as i32) * 4,
            base_address: input_data.as_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };
        let output_arena = crate::arena_info::ArenaInfo {
            width: dim as i32,
            height: dim as i32,
            logical_dim0: dim as i32,
            logical_dim1: dim as i32,
            pixel_format: 0,
            byte_size: (n as i32) * 4,
            bytes_per_row: (dim as i32) * 4,
            base_address: output_data.as_mut_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };

        // Warmup: one inference to prime ANE caches and avoid cold-start bias.
        model
            .predict("input", &input_arena, "matmul_1", &output_arena)
            .ok()?;

        // Timed iterations.
        const ITERATIONS: u32 = 10;
        let mut total_ns: u64 = 0;
        let mut min_ns: u64 = u64::MAX;
        let mut latencies = Vec::with_capacity(ITERATIONS as usize);

        for _ in 0..ITERATIONS {
            let t0 = Instant::now();
            model
                .predict("input", &input_arena, "matmul_1", &output_arena)
                .ok()?;
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            total_ns = total_ns.wrapping_add(elapsed_ns);
            min_ns = min_ns.min(elapsed_ns);
            latencies.push(elapsed_ns);
        }

        latencies.sort();
        let median_ns = latencies[latencies.len() / 2];
        let p90_idx = ((latencies.len() as f64) * 0.9) as usize;
        let p90_ns = latencies[p90_idx.min(latencies.len() - 1)];
        let avg_ns = total_ns / ITERATIONS as u64;

        // Bandwidth: 2x buffer (read input + write output) * 4 bytes per f32
        let bandwidth_gbps = (n as f64 * 4.0 * 2.0) / avg_ns as f64 * 1e3;
        let throughput_ops_per_sec = n as f64 / avg_ns as f64 * 1e9;

        Some(KernelBenchResult {
            variant_name: "coreml-bench-identity".into(),
            backend: "coreml".into(),
            op_type: "matmul".into(),
            shape: vec![dim, dim],
            dtype: "f32".into(),
            median_latency_ns: median_ns,
            min_latency_ns: min_ns,
            p90_latency_ns: p90_ns,
            bandwidth_gbps,
            throughput_ops_per_sec,
            numerical_error: 0.0,
            compile_time_ms: 0.0,
        })
    }
}

impl Default for CoreMlLane {
    fn default() -> Self {
        Self::new()
    }
}
