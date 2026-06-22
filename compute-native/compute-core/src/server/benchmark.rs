//! System benchmark for backend routing decisions.
//! Measures latency and throughput for each available backend.

use std::time::Instant;

/// Results from benchmarking a single operation
#[derive(Debug, Clone)]
pub struct OpBenchmark {
    pub op_name: &'static str,
    pub mlx_us: f64,        // microseconds for MLX backend
    pub accelerate_us: f64, // microseconds for Accelerate backend (0 if unavailable)
    pub mlx_available: bool,
    pub accelerate_available: bool,
}

/// Full system benchmark results
#[derive(Debug, Clone)]
pub struct SystemBenchmark {
    pub chip: String,
    pub ram_gb: u64,
    pub ops: Vec<OpBenchmark>,
    pub recommend_accelerate_for: Vec<&'static str>,
    pub recommend_mlx_for: Vec<&'static str>,
}

/// Run the full system benchmark
pub fn run_benchmark() -> SystemBenchmark {
    let chip = get_chip_name();
    let ram_gb = get_ram_gb();

    let mut ops = Vec::new();

    // Benchmark matmul at various sizes
    ops.push(bench_op("matmul_1024x1024", 1024, 1024, 1024));
    ops.push(bench_op("matmul_4096x4096", 4096, 4096, 4096));

    // Benchmark rms_norm
    ops.push(bench_op("rms_norm_4096", 4096, 1, 4096));

    // Benchmark attention
    ops.push(bench_op("attention_32x128", 32, 128, 128));

    // Determine routing recommendations
    let mut recommend_accelerate = Vec::new();
    let mut recommend_mlx = Vec::new();
    for op in &ops {
        if op.accelerate_available && op.accelerate_us < op.mlx_us * 0.8 {
            recommend_accelerate.push(op.op_name);
        } else if op.accelerate_available && op.mlx_us < op.accelerate_us * 0.8 {
            recommend_mlx.push(op.op_name);
        }
    }

    SystemBenchmark {
        chip,
        ram_gb,
        ops,
        recommend_accelerate_for: recommend_accelerate,
        recommend_mlx_for: recommend_mlx,
    }
}

fn get_chip_name() -> String {
    // Try sysctl for Apple Silicon info
    #[cfg(target_os = "macos")]
    {
        if let Ok(name) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            if let Ok(s) = String::from_utf8(name.stdout) {
                return s.trim().to_string();
            }
        }
        if let Ok(name) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.model")
            .output()
        {
            if let Ok(s) = String::from_utf8(name.stdout) {
                return s.trim().to_string();
            }
        }
    }
    "unknown".to_string()
}

fn get_ram_gb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(mem) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(s) = String::from_utf8(mem.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes / 1024 / 1024 / 1024;
                }
            }
        }
    }
    0
}

fn bench_op(name: &'static str, _m: usize, _n: usize, _k: usize) -> OpBenchmark {
    // For now, return placeholder timing.
    // Real benchmark would create backends and run actual matmul ops.
    OpBenchmark {
        op_name: name,
        mlx_us: 100.0,
        accelerate_us: 150.0,
        mlx_available: true,
        accelerate_available: cfg!(target_os = "macos"),
    }
}
