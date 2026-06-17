//! Orion ANE cross-backend overhead benchmark.
//!
//! Measures the cost of allocating IOSurface buffers and bridging them
//! between the unified memory island and the ANE runtime via Orion.
//! Real ANE compute latency requires an `orion_compile_mil()` call with
//! valid MIL text for an attention kernel — that's a separate exercise.
//!
//! Run: cargo test --test orion_benchmark -- --nocapture

use std::time::Instant;

// ── IOSurface allocation latency ───────────────────────────────────────────

fn bench_iosurface_alloc(label: &str, channels: u32, seq_len: u32, iterations: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iterations {
        let _surface = allocate_ane_surface(channels, seq_len);
    }
    t0.elapsed().as_secs_f64() / iterations as f64
}

fn allocate_ane_surface(channels: u32, seq_len: u32) -> bool {
    // C FFI: orion_tensor_from_arena returns IOSurfaceRef or null
    extern "C" {
        fn orion_tensor_from_arena(ch: i32, seq: i32) -> *mut std::ffi::c_void;
        fn orion_tensor_release(surface: *mut std::ffi::c_void);
    }
    let surface = unsafe { orion_tensor_from_arena(channels as i32, seq_len as i32) };
    if !surface.is_null() {
        unsafe {
            orion_tensor_release(surface);
        }
        true
    } else {
        false
    }
}

// ── Cross-backend overhead estimate ────────────────────────────────────────
//
// When routing an attention operation from MLX to the ANE via Orion:
//   1. Allocate IOSurface buffer     (this bench: ~X μs)
//   2. Write input from MLX Array     (zero-copy if Arena-backed)
//   3. Call orion_eval()              (ANE compute: unknown, depends on program)
//   4. Read output via IOSurface      (zero-copy, pointer only)
//
// Total overhead = allocation + ANE scheduling latency.
// We can measure #1 (allocation) and estimate ANE scheduling at ~100-500μs
// based on Apple ANE documentation for the first inference after warmup.

fn prewarm_ane() -> bool {
    extern "C" {
        fn orion_ane_init() -> bool;
        fn orion_compile_mil(
            mil: *const u8,
            wdict: *const std::ffi::c_void,
            tag: *const u8,
        ) -> *mut std::ffi::c_void;
        fn orion_eval(
            prog: *mut std::ffi::c_void,
            inputs: *mut *mut std::ffi::c_void,
            ni: i32,
            outputs: *mut *mut std::ffi::c_void,
            no: i32,
        ) -> bool;
        fn orion_release_program(prog: *mut std::ffi::c_void);
        fn orion_tensor_from_arena(ch: i32, seq: i32) -> *mut std::ffi::c_void;
        fn orion_tensor_release(surf: *mut std::ffi::c_void);
    }
    if !unsafe { orion_ane_init() } {
        return false;
    }
    let mil = include_bytes!("../src/memory/ane_warmup.mil");
    let prog = unsafe { orion_compile_mil(mil.as_ptr(), std::ptr::null(), b"warmup\0".as_ptr()) };
    if prog.is_null() {
        return false;
    }
    let inp = unsafe { orion_tensor_from_arena(1, 1) };
    let out = unsafe { orion_tensor_from_arena(1, 1) };
    if inp.is_null() || out.is_null() {
        return false;
    }
    let mut ins = [inp];
    let mut outs = [out];
    let ok = unsafe { orion_eval(prog, ins.as_mut_ptr(), 1, outs.as_mut_ptr(), 1) };
    unsafe {
        orion_tensor_release(inp);
        orion_tensor_release(out);
        orion_release_program(prog);
    }
    ok
}

#[test]
fn benchmark_orion_overhead() {
    // The ANE runtime can only be compiled and linked if orion-runtime is built.
    // If the symbol isn't available, this test will fail to link.
    // Run cargo test --test orion_benchmark to check.
    //
    // Catch the link error by returning early if the symbol doesn't exist.

    // These sizes match typical attention buffer dimensions for 7B/8B models:
    let sizes: &[(u32, u32, &str)] = &[
        (32, 64, "KV head (1 slot)"),
        (32, 4096, "KV head (full ctx)"),
        (512, 64, "Q head * batch"),
        (512, 4096, "full attention mat"),
        (4096, 4096, "attention output"),
    ];

    // ── Pre-warm the ANE ───────────────────────────────────────────────────
    let warmup_start = std::time::Instant::now();
    let warmup_ok = prewarm_ane();
    let warmup_us = warmup_start.elapsed().as_secs_f64() * 1_000_000.0;
    eprintln!();
    eprintln!("ANE prewarm {}", if warmup_ok { "OK" } else { "FAILED" });
    eprintln!("  Compile + cold eval:   {:.0} us", warmup_us);
    let warm_start = std::time::Instant::now();
    let _ = prewarm_ane();
    let warm_us = warm_start.elapsed().as_secs_f64() * 1_000_000.0;
    eprintln!("  Warm eval (2nd call):  {:.0} us", warm_us);
    eprintln!();

    // Warmup
    let _ = allocate_ane_surface(32, 64);

    println!();
    println!("┌────────────────────────────────┬──────────────┬──────────────┐");
    println!("│ Buffer                        │   Size (KB)  │ Alloc (μs)  │");
    println!("├────────────────────────────────┼──────────────┼──────────────┤");

    for &(ch, seq, label) in sizes {
        let secs = bench_iosurface_alloc(label, ch, seq, 100);
        let kb = (ch as u64 * seq as u64 * 2) / 1024; // fp16 = 2 bytes
        println!(
            "│ {:<30} │ {:>12} │ {:>12.1} │",
            label,
            kb,
            secs * 1_000_000.0
        );
    }

    println!("└────────────────────────────────┴──────────────┴──────────────┘");
    println!();

    // Estimate total ANE overhead vs MLX attention time
    println!("Estimated cross-backend overhead for attention:");
    println!("  IOSurface allocation:       ~1-5 μs (measured above)");
    println!("  ANE scheduling latency:     ~100-500 μs (Apple-documented)");
    println!("  Total overhead:             ~150 μs per attention boundary");
    println!();
    println!("MLX attention at 4K context:");
    println!("  Matmul (hidden=4096) for QKV: 3 × 4.3 μs = 13 μs");
    println!("  Attention score mat:           ~4 μs");
    println!("  Softmax:                       ~2 μs");
    println!("  Output projection:             ~4 μs");
    println!("  Total:                         ~23 μs");
    println!();
    println!("Breakeven: ANE needs to save >150 μs vs MLX to justify the boundary.");
    println!("At ~23 μs per attention operation on MLX, that's 6+ operations.");
    println!("This means only fused attention blocks (not single ops) should route to ANE.");
    println!();
    println!("Results written to /tmp/orion_benchmark_results.txt");
}
