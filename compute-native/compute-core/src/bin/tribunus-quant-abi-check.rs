//! Standalone quantized ABI debug tool.
//! Compares native MLX quantization against Tribunus compiler quantization byte-for-byte.
//! Run: cargo run --no-default-features --features mlx-backend --bin tribunus-quant-abi-check

use mlx_rs::{ops, Array, Device, DeviceType};

fn main() {
    let rows = 4u32;
    let cols = 64u32;
    let n = (rows * cols) as usize;
    let matrix: Vec<f32> = (0..n)
        .map(|i| {
            let r = (i / cols as usize) as f32;
            let c = (i % cols as usize) as f32;
            (r * 0.5 + c * 0.3 + 0.1).sin()
        })
        .collect();

    Device::set_default(&Device::new(DeviceType::Cpu, 0));

    // 1. Native MLX quantization
    let src = Array::from_slice(&matrix, &[rows as i32, cols as i32]);
    let (w_mlx, s_mlx, b_mlx) = ops::quantize(&src, 64, 4).unwrap();
    let _ = w_mlx.eval().unwrap();
    let _ = s_mlx.eval().unwrap();
    let _ = b_mlx.eval().unwrap();

    let mlx_w_shape: Vec<i32> = w_mlx.shape().to_vec();
    let mlx_w: Vec<u32> = w_mlx.as_slice::<u32>().to_vec();
    let mlx_s: Vec<f32> = s_mlx.as_slice::<f32>().to_vec();
    let mlx_b: Vec<f32> = b_mlx.as_slice::<f32>().to_vec();

    // 2. Tribunus quantization (each row is one group of group_size=64)
    let mut tri_w = Vec::new();
    let mut tri_s = Vec::new();
    let mut tri_b = Vec::new();
    for r in 0..rows as usize {
        let start = r * cols as usize;
        let (pw, s, b) = tribunus_compute_core::compute_image::compile::quantize_int4_group(
            &matrix[start..start + cols as usize],
        );
        tri_w.extend(pw);
        tri_s.push(s);
        tri_b.push(b);
    }

    // 3. Diagnostic output
    println!("\n=== QUANTIZED ABI DEBUG ===");
    println!("Matrix: {}x{}, {} values", rows, cols, n);

    println!("\n--- Native MLX (quantize bits=4 group_size=64) ---");
    println!("  w shape: {:?}  words: {}", mlx_w_shape, mlx_w.len());
    println!("  w U32 hex: {:08x?}", mlx_w);
    println!("  scales: {:?}", mlx_s);
    println!("  biases: {:?}", mlx_b);

    println!("\n--- Tribunus (quantize_int4_group, per-row) ---");
    println!("  words: {}", tri_w.len());
    println!("  U32 hex: {:08x?}", tri_w);
    println!("  scales: {:?}", tri_s);
    println!("  biases: {:?}", tri_b);

    // 4. Byte comparison
    println!("\n--- Byte comparison ---");
    let n_groups = mlx_s.len().min(tri_s.len());
    if mlx_w.len() == tri_w.len() {
        let matched = mlx_w == tri_w;
        println!("  WEIGHT: {}", if matched { "MATCH" } else { "DIFFER" });
        if !matched {
            for (i, (&m, &t)) in mlx_w.iter().zip(tri_w.iter()).enumerate().take(8) {
                if m != t {
                    println!(
                        "    word[{:2}]: MLX={:08x} Tri={:08x} xor={:08x}",
                        i,
                        m,
                        t,
                        m ^ t
                    );
                }
            }
        }
    } else {
        println!(
            "  WEIGHT: SHAPE DIFFER (MLX: {} words, Tri: {} words)",
            mlx_w.len(),
            tri_w.len()
        );
    }

    // Per-row scale/bias comparison
    for g in 0..n_groups {
        let sd = (mlx_s[g] - tri_s[g]).abs();
        let bd = (mlx_b[g] - tri_b[g]).abs();
        println!(
            "  ROW[{}]: SCALE {} ({:.8} vs {:.8} diff={:.8e}) BIAS {} ({:.8} vs {:.8} diff={:.8e})",
            g,
            if sd < 1e-4 { "MATCH" } else { "DIFFER" },
            mlx_s[g],
            tri_s[g],
            sd,
            if bd < 1e-4 { "MATCH" } else { "DIFFER" },
            mlx_b[g],
            tri_b[g],
            bd
        );
    }

    // 5. Dequantize round-trip (compare against original matrix)
    println!("\n--- MLX Dequantization error ---");
    match ops::dequantize(&w_mlx, &s_mlx, &b_mlx, 64, 4) {
        Ok(deq) => {
            let _ = deq.eval().unwrap();
            let vals = deq.as_slice::<f32>().to_vec();
            let max_diff = matrix
                .iter()
                .zip(vals.iter())
                .map(|(o, d)| (o - d).abs())
                .fold(0.0f32, f32::max);
            println!("  max abs diff: {:.8}", max_diff);
            println!("  First 4 values:");
            for (i, (o, d)) in matrix.iter().zip(vals.iter()).enumerate().take(4) {
                println!(
                    "    [{:2}] orig={:+.6} deq={:+.6} diff={:.6}",
                    i,
                    o,
                    d,
                    (o - d).abs()
                );
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    println!();
}
