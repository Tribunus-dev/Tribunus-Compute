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
    ShapeMismatch { expected: Vec<u32>, actual: Vec<u32> },
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
    pub fn infer(&self, _inputs: &[&[f32]], _outputs: &mut [&mut [f32]]) -> Result<f64, String> {
        Err("Core ML inference requires compiled .mlmodelc and Core ML runtime".to_string())
    }
}

/// Core ML execution lane.
pub struct CoreMlLane {
    pub name: String,
    pub subgraphs: Vec<CoreMlSubgraph>,
    pub is_available: bool,
}

impl CoreMlLane {
    pub fn new() -> Self {
        // Probe for Core ML availability
        let is_available = cfg!(target_os = "macos");
        CoreMlLane {
            name: "coreml-ane".into(),
            subgraphs: Vec::new(),
            is_available,
        }
    }

    /// Check if a subgraph is compiled and ready for the given input shape.
    pub fn can_execute(&self, subgraph_name: &str) -> bool {
        self.subgraphs.iter().any(|sg| {
            sg.name == subgraph_name
                && matches!(sg.status, CoreMlSubgraphStatus::Compiled { .. })
        })
    }

    pub fn add_subgraph(&mut self, subgraph: CoreMlSubgraph) {
        self.subgraphs.push(subgraph);
    }
}

impl Default for CoreMlLane {
    fn default() -> Self {
        Self::new()
    }
}
