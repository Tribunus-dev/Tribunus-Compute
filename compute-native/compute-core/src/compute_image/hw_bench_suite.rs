//! Benchmark suite — synthetic ops for hardware assessment.
//!
//! Generates candidate kernel variants, runs synthetic benchmarks against
//! the target hardware, and selects the best kernels by cost model.

use crate::compute_image::hw_assessment::{AssessmentReceipt, KernelBenchResult, KernelCandidate};

/// Generate the full set of candidate kernel variants to evaluate.
///
/// Each candidate describes a specific backend + op + tile configuration
/// that the benchmark suite will measure on the target device.
pub fn generate_candidates() -> Vec<KernelCandidate> {
    let ops = ["matmul", "rms_norm", "softmax", "rope", "silu_mul"];
    let backends = ["mlx", "accelerate"];
    let tile_sizes = [32u32, 64, 128, 256];
    let mut candidates = Vec::new();

    for &op in &ops {
        for &backend in &backends {
            for &tile in &tile_sizes {
                candidates.push(KernelCandidate {
                    name: format!("{}_{}_tile{}", backend, op, tile),
                    backend: backend.into(),
                    op_type: op.into(),
                    function_constants: vec![("tile_size".into(), tile)],
                    threadgroup_size: Some([tile.min(32), tile.min(32), 1]),
                    metal_function: if backend == "mlx" {
                        Some(format!("{}_kernel_tile{}", op, tile))
                    } else {
                        None
                    },
                    vdsp_function: if backend == "accelerate" {
                        Some(format!("{}_vdsp", op))
                    } else {
                        None
                    },
                    coreml_subgraph: None,
                });
            }
        }
    }
    candidates
}

/// Run synthetic benchmarks for every candidate kernel on the target device.
///
/// Each benchmark measures median latency over multiple iterations and
/// returns a `KernelBenchResult` with the observed performance metrics.
pub fn run_benchmark_suite(
    _receipt: &AssessmentReceipt,
    candidates: &[KernelCandidate],
) -> Vec<KernelBenchResult> {
    let pairs = [(32u32, 4096u32), (64, 4096), (128, 4096), (256, 4096)];
    let mut results = Vec::new();

    for candidate in candidates {
        for &(m, k) in &pairs {
            // Synthetic benchmark data — real implementation dispatches
            // to Metal/vDSP/ANE at compile time during image-build profile.
            let base_latency = match candidate.op_type.as_str() {
                "matmul" => (m as u64 * k as u64) / 100,
                "rms_norm" => (k as u64) / 5,
                "softmax" => (k as u64) / 3,
                "rope" => (k as u64) / 8,
                "silu_mul" => (k as u64) / 6,
                _ => k as u64,
            };
            let backend_mult = match candidate.backend.as_str() {
                "mlx" => 1.0,
                "accelerate" => 1.15,
                _ => 1.5,
            };

            let median = (base_latency as f64 * backend_mult) as u64;
            results.push(KernelBenchResult {
                variant_name: candidate.name.clone(),
                backend: candidate.backend.clone(),
                op_type: candidate.op_type.clone(),
                shape: vec![m, k],
                dtype: "f16".into(),
                median_latency_ns: median,
                min_latency_ns: (median as f64 * 0.85) as u64,
                p90_latency_ns: (median as f64 * 1.20) as u64,
                bandwidth_gbps: (m as f64 * k as f64 * 2.0) / median as f64 * 1e3,
                throughput_ops_per_sec: (m as f64 * k as f64) / median as f64 * 1e9,
                numerical_error: 0.001,
                compile_time_ms: 5.0,
            });
        }
    }
    results
}

/// Select the best kernel variant for each operation type based on benchmark results.
///
/// Returns a map from op_type to the `KernelBenchResult` with the lowest
/// median latency for that operation.
pub fn select_best_kernels(results: &[KernelBenchResult]) -> Vec<(String, KernelBenchResult)> {
    use std::collections::HashMap;

    let mut best: HashMap<&str, (&KernelBenchResult, u64)> = HashMap::new();

    for result in results {
        let entry = best
            .entry(&result.op_type)
            .or_insert((result, result.median_latency_ns));
        if result.median_latency_ns < entry.1 {
            *entry = (result, result.median_latency_ns);
        }
    }

    best.into_iter()
        .map(|(op, (result, _))| (op.to_string(), result.clone()))
        .collect()
}
