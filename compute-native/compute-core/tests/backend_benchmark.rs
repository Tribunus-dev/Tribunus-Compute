//! Synthetic-data benchmark comparing operation latency across MLX,
//! Accelerate, and Core ML backends.  Results inform the compiler's
//! `OperationRoute` — the routing table in `config/operation_route.rs`.
//!
//! Run: cargo test --test backend_benchmark -- --nocapture

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::backend::accelerate::AccelerateBackend;
use tribunus_compute_core::backend::MlxBackend;
use tribunus_compute_core::backend::*;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{self, ModelMeta};

// ── Helpers ────────────────────────────────────────────────────────────────

fn random_f32(n: usize) -> Vec<f32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as f32 * 1e-6
        })
        .collect()
}

struct TrialResult {
    op: &'static str,
    m: u32,
    n: u32,
    k: u32,
    mlx_ns: u64,
    accel_ns: u64,
    faster: &'static str,
}

struct CoreMlResult {
    op: &'static str,
    m: u32,
    n: u32,
    k: u32,
    cpugpu_ns: u64,
    ane_ns: u64,
    faster: &'static str,
}

// ── Per-operation benchmarks (MLX vs Accelerate) ──────────────────────────

fn bench_matmul(results: &mut Vec<TrialResult>) {
    let sizes = [
        (1, 1, 1),
        (1, 64, 64),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (1, 4096, 4096),
        (4096, 4096, 4096),
    ];
    for &(m, n, k) in &sizes {
        let a_data = random_f32((m * k) as usize);
        let b_data = random_f32((k * n) as usize);

        let mut mlx = MlxBackend::new();
        let a_mlx = mlx.create_f32(&a_data, &[m as i32, k as i32]).unwrap();
        let b_mlx = mlx.create_f32(&b_data, &[k as i32, n as i32]).unwrap();
        let op = MatmulOp { m, n, k };
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = mlx.matmul(&op, a_mlx, b_mlx).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 10;

        let mut accel = AccelerateBackend::new();
        let a_acc = accel.create_f32(&a_data, &[m as i32, k as i32]).unwrap();
        let b_acc = accel.create_f32(&b_data, &[k as i32, n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = accel.matmul(&op, a_acc, b_acc).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 10;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "matmul",
            m,
            n,
            k,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

fn bench_add(results: &mut Vec<TrialResult>) {
    for &n in &[64, 256, 1024, 4096, 16384] {
        let a_data = random_f32(n);
        let b_data = random_f32(n);

        let mut mlx = MlxBackend::new();
        let a_mlx = mlx.create_f32(&a_data, &[n as i32]).unwrap();
        let b_mlx = mlx.create_f32(&b_data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.add(a_mlx, b_mlx).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let a_acc = accel.create_f32(&a_data, &[n as i32]).unwrap();
        let b_acc = accel.create_f32(&b_data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.add(a_acc, b_acc).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "add",
            m: 1,
            n: n as u32,
            k: 1,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

fn bench_mul(results: &mut Vec<TrialResult>) {
    for &n in &[64, 256, 1024, 4096, 16384] {
        let a_data = random_f32(n);
        let b_data = random_f32(n);

        let mut mlx = MlxBackend::new();
        let a_mlx = mlx.create_f32(&a_data, &[n as i32]).unwrap();
        let b_mlx = mlx.create_f32(&b_data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.multiply(a_mlx, b_mlx).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let a_acc = accel.create_f32(&a_data, &[n as i32]).unwrap();
        let b_acc = accel.create_f32(&b_data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.multiply(a_acc, b_acc).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "multiply",
            m: 1,
            n: n as u32,
            k: 1,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

fn bench_silu(results: &mut Vec<TrialResult>) {
    for &n in &[64, 256, 1024, 4096, 16384] {
        let data = random_f32(n);

        let mut mlx = MlxBackend::new();
        let x_mlx = mlx.create_f32(&data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.silu(x_mlx).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let x_acc = accel.create_f32(&data, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.silu(x_acc).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "silu",
            m: 1,
            n: n as u32,
            k: 1,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

fn bench_rms_norm(results: &mut Vec<TrialResult>) {
    for &(n, dim) in &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)] {
        let data = random_f32(n);
        let weight = random_f32(n);

        let mut mlx = MlxBackend::new();
        let x_mlx = mlx.create_f32(&data, &[n as i32]).unwrap();
        let w_mlx = mlx.create_f32(&weight, &[n as i32]).unwrap();
        let op = RmsNormOp {
            dim: dim as u32,
            eps: 1e-5,
        };
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.rms_norm(&op, x_mlx, w_mlx).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let x_acc = accel.create_f32(&data, &[n as i32]).unwrap();
        let w_acc = accel.create_f32(&weight, &[n as i32]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.rms_norm(&op, x_acc, w_acc).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "rms_norm",
            m: 1,
            n: n as u32,
            k: dim as u32,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

fn bench_softmax(results: &mut Vec<TrialResult>) {
    for &(rows, cols) in &[
        (1, 64),
        (1, 4096),
        (8, 64),
        (8, 4096),
        (32, 128),
        (32, 4096),
    ] {
        let data = random_f32((rows * cols) as usize);

        let mut mlx = MlxBackend::new();
        let x_mlx = mlx.create_f32(&data, &[rows, cols]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.softmax(x_mlx, 1).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let x_acc = accel.create_f32(&data, &[rows, cols]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.softmax(x_acc, 1).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        for _ in 0..3 {
            results.push(TrialResult {
                op: "softmax",
                m: rows as u32,
                n: cols as u32,
                k: 1,
                mlx_ns,
                accel_ns,
                faster,
            });
        }
    }
}

fn bench_transpose(results: &mut Vec<TrialResult>) {
    for &(m, n) in &[(64, 64), (256, 256), (512, 512), (1024, 1024)] {
        let data = random_f32((m * n) as usize);

        let mut mlx = MlxBackend::new();
        let x_mlx = mlx.create_f32(&data, &[m, n]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = mlx.transpose(x_mlx, &[1, 0]).unwrap();
            mlx.evaluate(0, &[]).unwrap();
        }
        let mlx_ns = t0.elapsed().as_nanos() as u64 / 100;

        let mut accel = AccelerateBackend::new();
        let x_acc = accel.create_f32(&data, &[m, n]).unwrap();
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accel.transpose(x_acc, &[1, 0]).unwrap();
        }
        let accel_ns = t0.elapsed().as_nanos() as u64 / 100;

        let faster = if mlx_ns < accel_ns { "MLX" } else { "Accel" };
        results.push(TrialResult {
            op: "transpose",
            m: m as u32,
            n: n as u32,
            k: 1,
            mlx_ns,
            accel_ns,
            faster,
        });
        results.push(TrialResult {
            op: "transpose",
            m: m as u32,
            n: n as u32,
            k: 1,
            mlx_ns,
            accel_ns,
            faster,
        });
    }
}

// ── Core ML benchmarks ─────────────────────────────────────────────────────

const CORE_CACHE: &str = "/tmp/coreml_bench_cache";

/// Compile a MIL program and cache the .mlmodelc. Returns inner modelc path.
fn compile_bench_model(
    prog: mil_spec::Program,
    meta: ModelMeta,
    cache_key: &str,
) -> Result<String, String> {
    let cache_dir = Path::new(CORE_CACHE);
    let modelc_dir = cache_dir.join(format!("{}.modelc", cache_key));
    if modelc_dir.join("metadata.json").exists() {
        if let Ok(entries) = fs::read_dir(&modelc_dir) {
            for e in entries.filter_map(|e| e.ok()) {
                let p = e.path();
                if p.is_dir() && p.join("metadata.json").exists() {
                    return Ok(p.to_string_lossy().to_string());
                }
            }
        }
    }
    fs::create_dir_all(cache_dir).ok();
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = mlpackage::write_mlpackage(prog, tmp.path(), &meta)
        .map_err(|e| format!("write_mlpackage: {}", e))?;
    let receipt =
        coreml_pipeline::compile_mlpackage(&pkg_path, &modelc_dir, cache_key, "all", "CoreML9")
            .map_err(|e| {
                eprintln!("coremlc compile error ({}): {}", cache_key, e);
                format!("compile: {}", e)
            })?;
    Ok(receipt.compiled_modelc_path.clone())
}

/// Benchmark a compiled model with two compute-unit policies.
fn bench_coreml_model(
    results: &mut Vec<CoreMlResult>,
    op: &'static str,
    model_path: &str,
    m: u32,
    n: u32,
    k: u32,
    input_name: &str,
    output_name: &str,
    arena_dims: (u32, u32, u32, u32),
) {
    let load_and_bench = |cu: CoreMlComputeUnits| -> Option<u64> {
        let model = CoreMlModel::load_with_compute_units(model_path, cu).ok()?;
        let (id0, id1, od0, od1) = arena_dims;
        let arena_in = Arena::new(id0, id1, mlx_rs::Dtype::Float16).ok()?;
        let arena_out = Arena::new(od0, od1, mlx_rs::Dtype::Float16).ok()?;
        let t0 = Instant::now();
        for _ in 0..20 {
            model
                .predict_pixelbuffer(
                    input_name,
                    &arena_in.info,
                    output_name,
                    &mut arena_out.info.clone(),
                )
                .ok()?;
        }
        Some(t0.elapsed().as_nanos() as u64 / 20)
    };
    match (
        load_and_bench(CoreMlComputeUnits::All),
        load_and_bench(CoreMlComputeUnits::CpuAndNeuralEngine),
    ) {
        (Some(c), Some(a)) => {
            let faster = if c < a { "CPU+GPU" } else { "ANE" };
            results.push(CoreMlResult {
                op,
                m,
                n,
                k,
                cpugpu_ns: c,
                ane_ns: a,
                faster,
            });
        }
        (Some(c), None) => {
            results.push(CoreMlResult {
                op,
                m,
                n,
                k,
                cpugpu_ns: c,
                ane_ns: 0,
                faster: "CPU+GPU",
            });
        }
        _ => {}
    }
}

fn bench_coreml_matmul(results: &mut Vec<CoreMlResult>) {
    for &(m, n, k) in &[(64, 64, 64), (256, 256, 256), (1024, 1024, 1024)] {
        let shape_in = vec![m as i64, k as i64];
        let weight = vec![1.0_f32; (k * n) as usize];
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float16, &shape_in)
            .const_f16("w", &weight, &[k as i64, n as i64])
            .matmul("x", "w_0")
            .output("matmul_1")
            .build()
            .expect("MIL build");
        let meta = ModelMeta {
            model_name: format!("matmul_{}_{}_{}", m, n, k),
            function_name: "main".into(),
            short_description: "matmul".into(),
            version: "1".into(),
            author: "bench".into(),
            output_name: "matmul_1".into(),
            inputs: vec![("x".into(), shape_in)],
            outputs: vec![("matmul_1".into(), vec![m as i64, n as i64])],
        };
        let key = format!("matmul_{}x{}x{}", m, k, n);
        if let Ok(p) = compile_bench_model(prog, meta, &key) {
            bench_coreml_model(
                results,
                "matmul",
                &p,
                m,
                n,
                k,
                "x",
                "matmul_1",
                (m, k, m, n),
            );
        } else {
            eprintln!("SKIP coreml matmul {}x{}x{}: compile failed", m, k, n);
        }
    }
}

fn bench_coreml_add(results: &mut Vec<CoreMlResult>) {
    for &n in &[64, 256, 1024, 4096] {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float16, &[1, n as i64])
            .const_f16("c", &vec![1.0_f32; n as usize], &[1, n as i64])
            .add("x", "c_0")
            .output("add_1")
            .build()
            .expect("MIL build");
        let meta = ModelMeta {
            model_name: format!("add_{}", n),
            function_name: "main".into(),
            short_description: "add".into(),
            version: "1".into(),
            author: "bench".into(),
            output_name: "add_1".into(),
            inputs: vec![("x".into(), vec![1, n as i64])],
            outputs: vec![("add_1".into(), vec![1, n as i64])],
        };
        let key = format!("add_1x{}", n);
        if let Ok(p) = compile_bench_model(prog, meta, &key) {
            bench_coreml_model(
                results,
                "add",
                &p,
                1,
                n,
                1,
                "x",
                "add_1",
                (1, n as u32, 1, n as u32),
            );
        } else {
            eprintln!("SKIP coreml add 1x{}: compile failed", n);
        }
    }
}

fn bench_coreml_mul(results: &mut Vec<CoreMlResult>) {
    for &n in &[64, 256, 1024, 4096] {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float16, &[1, n as i64])
            .const_f16("c", &vec![2.0_f32; n as usize], &[1, n as i64])
            .mul("x", "c_0")
            .output("mul_1")
            .build()
            .expect("MIL build");
        let meta = ModelMeta {
            model_name: format!("mul_{}", n),
            function_name: "main".into(),
            short_description: "mul".into(),
            version: "1".into(),
            author: "bench".into(),
            output_name: "mul_1".into(),
            inputs: vec![("x".into(), vec![1, n as i64])],
            outputs: vec![("mul_1".into(), vec![1, n as i64])],
        };
        let key = format!("mul_1x{}", n);
        if let Ok(p) = compile_bench_model(prog, meta, &key) {
            bench_coreml_model(
                results,
                "mul",
                &p,
                1,
                n,
                1,
                "x",
                "mul_1",
                (1, n as u32, 1, n as u32),
            );
        } else {
            eprintln!("SKIP coreml mul 1x{}: compile failed", n);
        }
    }
}

// ── Runner ──────────────────────────────────────────────────────────────────

// ── ANE Direct benchmarks (via AppleNeuralEngine private framework) ────────
use tribunus_compute_core::ane_bridge::AneProgram;

struct AneDirectResult {
    op: &'static str,
    ch: u32,
    seq: u32,
    compile_ms: f64,
    eval_ns: u64,
}

fn mil_add(ch: u32, seq: u32) -> String {
    let mut s = String::from("program(1.3)\n");
    s.push_str("{\n");
    s.push_str(&format!(
        "    void main<ios18>(tensor<fp32, [1, {}, 1, {}]> x) {{\n",
        ch, seq
    ));
    s.push_str("        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n");
    s.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", ch, seq));
    s.push_str(&format!(
        "        tensor<fp16, [1, {}, 1, {}]> c16 = const(val=1.0, shape=(1, {}, 1, {}));\n",
        ch, seq, ch, seq
    ));
    s.push_str(&format!("        tensor<fp16, [1, {}, 1, {}]> y16 = add(x = x16, y = c16)[name = string(\"add_op\")];\n", ch, seq));
    s.push_str("        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n");
    s.push_str(&format!("        tensor<fp32, [1, {}, 1, {}]> y = cast(dtype = to32, x = y16)[name = string(\"out\")];\n", ch, seq));
    s.push_str("    } -> (y);\n");
    s.push_str("}\n");
    s
}

fn bench_ane_direct(results: &mut Vec<AneDirectResult>) {
    if let Err(e) = AneProgram::init() {
        eprintln!("SKIP ANE direct: init failed: {}", e);
        return;
    }

    // M1 ANE requires minimum ~49KB surfaces (~768*16).
    // Use Orion's recommended minimum decode bucket [768, 16] and larger sizes.
    for &(ch, seq) in &[(64, 16), (256, 16), (768, 16), (768, 32), (1024, 32)] {
        let mil = mil_add(ch, seq);

        let t0 = Instant::now();
        let prog = match AneProgram::compile(&mil, "bench_add") {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIP ANE add {}x{}: compile failed: {}", ch, seq, e);
                continue;
            }
        };
        let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let arena_in = match Arena::new(ch, seq, mlx_rs::Dtype::Float16) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("SKIP ANE: arena failed: {}", e);
                continue;
            }
        };
        let arena_out = match Arena::new(ch, seq, mlx_rs::Dtype::Float16) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("SKIP ANE: arena failed: {}", e);
                continue;
            }
        };

        let inputs = [arena_in.info.io_surface];
        let outputs = [arena_out.info.io_surface];

        if let Err(e) = prog.evaluate(&inputs, &outputs) {
            eprintln!("SKIP ANE {}x{}: warmup eval failed: {}", ch, seq, e);
            continue;
        }

        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = prog.evaluate(&inputs, &outputs);
        }
        let eval_ns = t0.elapsed().as_nanos() as u64 / 10;

        eprintln!(
            "  ANE add {}x{}: compile={:.1}ms eval={}ns",
            ch, seq, compile_ms, eval_ns
        );
        results.push(AneDirectResult {
            op: "ane_add",
            ch,
            seq,
            compile_ms,
            eval_ns,
        });
    }
}

#[test]
fn benchmark_all_ops_across_backends() {
    let mut results = Vec::new();
    let mut coreml = Vec::new();
    let mut ane_direct = Vec::new();

    bench_matmul(&mut results);
    bench_add(&mut results);
    bench_mul(&mut results);
    bench_silu(&mut results);
    bench_rms_norm(&mut results);
    bench_softmax(&mut results);
    bench_transpose(&mut results);

    eprintln!();
    eprintln!("═══ Accelerate vs MLX ═══");
    eprintln!();
    eprintln!("┌───────────┬──────┬──────┬──────┬──────────┬──────────┬────────┐");
    eprintln!("│ Op        │    M │    N │    K │  MLX(ns) │Accel(ns) │ Faster │");
    eprintln!("├───────────┼──────┼──────┼──────┼──────────┼──────────┼────────┤");
    for r in &results {
        eprintln!(
            "│ {:<9} │ {:>4} │ {:>4} │ {:>4} │ {:>8} │ {:>8} │ {:<6} │",
            r.op, r.m, r.n, r.k, r.mlx_ns, r.accel_ns, r.faster
        );
    }
    eprintln!("└───────────┴──────┴──────┴──────┴──────────┴──────────┴────────┘");
    eprintln!();

    // Core ML benchmarks (compile + predict)
    eprintln!("═══ Core ML benchmarks ═══");
    eprintln!();
    bench_coreml_matmul(&mut coreml);
    bench_coreml_add(&mut coreml);
    bench_coreml_mul(&mut coreml);

    // ANE direct benchmarks (via AppleNeuralEngine private framework)
    bench_ane_direct(&mut ane_direct);

    if !coreml.is_empty() {
        eprintln!();
        eprintln!("┌───────────┬──────┬──────┬──────┬───────────┬───────────┬──────────┐");
        eprintln!("│ CoreML    │    M │    N │    K │ CPU+GPU(ns)│ ANE(ns)   │ Faster   │");
        eprintln!("├───────────┼──────┼──────┼──────┼───────────┼───────────┼──────────┤");
        for r in &coreml {
            let ane_str = if r.ane_ns == 0 {
                "    FAIL".into()
            } else {
                format!("{:>9}", r.ane_ns)
            };
            eprintln!(
                "│ {:<9} │ {:>4} │ {:>4} │ {:>4} │ {:>9} │ {:>9} │ {:<8} │",
                r.op, r.m, r.n, r.k, r.cpugpu_ns, ane_str, r.faster
            );
        }
        eprintln!("└───────────┴──────┴──────┴──────┴───────────┴───────────┴──────────┘");
        eprintln!();
    }

    if !ane_direct.is_empty() {
        eprintln!("═══ ANE Direct benchmarks (AppleNeuralEngine framework) ═══");
        eprintln!();
        eprintln!("┌───────────┬──────┬──────┬───────────┬───────────┬──────────┐");
        eprintln!("│ ANE       │    C │    S │  Compile  │  Eval(ns) │  Eval/MLX │");
        eprintln!("├───────────┼──────┼──────┼───────────┼───────────┼──────────┤");
        for r in &ane_direct {
            let vs_mlx = "-";
            eprintln!(
                "│ {:<9} │ {:>4} │ {:>4} │ {:>8.0}ms │ {:>9} │ {:>8} │",
                r.op, r.ch, r.seq, r.compile_ms, r.eval_ns, vs_mlx
            );
        }
        eprintln!("└───────────┴──────┴──────┴───────────┴───────────┴──────────┘");
        eprintln!();
    }

    // Write results to a file for offline analysis
    let result_path = Path::new("/tmp/backend_benchmark_results.txt");
    let mut file = fs::File::create(result_path).expect("create results file");

    let mlx_wins = results.iter().filter(|r| r.faster == "MLX").count();
    let accel_wins = results.iter().filter(|r| r.faster == "Accel").count();

    writeln!(file, "=== Accelerate vs MLX ===").unwrap();
    writeln!(file).unwrap();
    writeln!(
        file,
        "{:12} {:>5} {:>5} {:>5} {:>10} {:>10} {:>8}",
        "Op", "M", "N", "K", "Accel_ns", "MLX_ns", "Faster"
    )
    .unwrap();
    for r in &results {
        writeln!(
            file,
            "{:12} {:5} {:5} {:5} {:10} {:10} {:>8}",
            r.op, r.m, r.n, r.k, r.mlx_ns, r.accel_ns, r.faster
        )
        .unwrap();
    }

    if !coreml.is_empty() {
        writeln!(file).unwrap();
        writeln!(file, "=== Core ML ===").unwrap();
        writeln!(
            file,
            "{:12} {:>5} {:>5} {:>5} {:>10} {:>10} {:>8}",
            "Op", "M", "N", "K", "CPU+GPU_ns", "ANE_ns", "Faster"
        )
        .unwrap();
        for r in &coreml {
            writeln!(
                file,
                "{:12} {:5} {:5} {:5} {:10} {:10} {:>8}",
                r.op, r.m, r.n, r.k, r.cpugpu_ns, r.ane_ns, r.faster
            )
            .unwrap();
        }
    }

    if !ane_direct.is_empty() {
        writeln!(file).unwrap();
        writeln!(file, "=== ANE Direct ===").unwrap();
        writeln!(
            file,
            "{:12} {:>5} {:>5} {:>10} {:>10}",
            "Op", "C", "S", "Compile_ms", "Eval_ns"
        )
        .unwrap();
        for r in &ane_direct {
            writeln!(
                file,
                "{:12} {:5} {:5} {:10.1} {:10}",
                r.op, r.ch, r.seq, r.compile_ms, r.eval_ns
            )
            .unwrap();
        }
    }

    writeln!(file).unwrap();
    writeln!(
        file,
        "MLX wins: {}/{}, Accelerate wins: {}/{}",
        mlx_wins,
        results.len(),
        accel_wins,
        results.len()
    )
    .unwrap();
    file.flush().unwrap();

    eprintln!("Results written to {:?}", result_path);
    eprintln!(
        "MLX wins: {mlx_wins}/{}, Accelerate wins: {accel_wins}/{}",
        results.len(),
        results.len()
    );
    std::io::stderr().flush().unwrap();
}
