//! MLX quantized ABI isolation test.
//!
//! Compares the output of `mlx_rs::ops::quantize` against the tribunus
//! compiler's `quantize_int4_group` on a deterministic f32 matrix.
//!
//! Run with:
//!   cargo test -p tribunus-compute-core -- quant_abi --nocapture

use mlx_rs::ops;
use mlx_rs::Array;

/// Generate a deterministic f32 matrix for quantization testing.
fn test_matrix(rows: u32, cols: u32) -> Vec<f32> {
    (0..(rows * cols) as usize)
        .map(|i| {
            let r = (i / cols as usize) as f32;
            let c = (i % cols as usize) as f32;
            // Deterministic non-zero values with variety
            (r * 0.5 + c * 0.3 + 0.1).sin()
        })
        .collect()
}

/// Native MLX quantization reference.
fn mlx_quantize_reference(matrix: &[f32], rows: u32, cols: u32) -> (Array, Array, Array) {
    let src = Array::from_slice(matrix, &[rows as i32, cols as i32]);
    ops::quantize(&src, 64, 4).unwrap()
}

/// Tribunus compiler quantization reference.
fn tribunus_quantize_int4(matrix: &[f32]) -> (Vec<u32>, f32, f32) {
    crate::compute_image::compile::quantize_int4_group(matrix)
}

#[test]
fn quant_abi_compare_bytes() {
    let rows = 4u32;
    let cols = 8u32;
    let matrix = test_matrix(rows, cols);

    // 1. Native MLX quantization
    let (w_mlx, s_mlx, b_mlx) = mlx_quantize_reference(&matrix, rows, cols);

    // Evaluate MLX arrays so try_as_slice works.
    w_mlx.eval().unwrap();
    s_mlx.eval().unwrap();
    b_mlx.eval().unwrap();

    // 2. Tribunus quantization (same matrix as one group)
    //    For group_size=64 and 32 elements total, there is 1 group.
    let (tri_w, tri_scale, tri_bias) = tribunus_quantize_int4(&matrix);

    // 3. Extract native MLX array properties via safe typed slices
    let mlx_w_shape = w_mlx.shape().to_vec();
    let mlx_w_dtype = w_mlx.dtype();
    let mlx_w: Vec<u32> = w_mlx.try_as_slice::<u32>().unwrap().to_vec();
    let mlx_s: Vec<f32> = s_mlx.try_as_slice::<f32>().unwrap().to_vec();
    let mlx_b: Vec<f32> = b_mlx.try_as_slice::<f32>().unwrap().to_vec();

    // Also get raw byte representations for ABI debugging
    let mlx_w_bytes: Vec<u8> = mlx_w.iter().flat_map(|&w| w.to_le_bytes()).collect();
    let mlx_s_bytes: Vec<u8> = mlx_s.iter().flat_map(|&s| s.to_le_bytes()).collect();
    let mlx_b_bytes: Vec<u8> = mlx_b.iter().flat_map(|&b| b.to_le_bytes()).collect();

    // 4. Tribunus bytes (packed U32 words as LE bytes)
    let tri_w_bytes: Vec<u8> = tri_w.iter().flat_map(|&w| w.to_le_bytes()).collect();
    let tri_s_bytes: Vec<u8> = tri_scale.to_le_bytes().to_vec();
    let tri_b_bytes: Vec<u8> = tri_bias.to_le_bytes().to_vec();

    // 5. Diagnostic output
    println!();
    println!("=== QUANTIZED ABI DEBUG ===");
    println!("Matrix: {}x{}", rows, cols);
    println!();
    println!("--- Native MLX ---");
    println!("  w shape: {:?}", mlx_w_shape);
    println!("  w dtype: {:?}", mlx_w_dtype);
    println!("  w values (len={}): {:08x?}", mlx_w.len(), &mlx_w);
    println!(
        "  w bytes (len={}): {:02x?}",
        mlx_w_bytes.len(),
        &mlx_w_bytes
    );
    println!("  s values (len={}): {:?}", mlx_s.len(), &mlx_s);
    println!(
        "  s bytes (len={}): {:02x?}",
        mlx_s_bytes.len(),
        &mlx_s_bytes
    );
    println!("  b values (len={}): {:?}", mlx_b.len(), &mlx_b);
    println!(
        "  b bytes (len={}): {:02x?}",
        mlx_b_bytes.len(),
        &mlx_b_bytes
    );
    println!();
    println!("--- Tribunus ---");
    println!("  packed words: {:08x?}", tri_w);
    println!(
        "  w bytes (len={}): {:02x?}",
        tri_w_bytes.len(),
        &tri_w_bytes
    );
    println!("  scale: {} (bytes: {:02x?})", tri_scale, tri_s_bytes);
    println!("  bias: {} (bytes: {:02x?})", tri_bias, tri_b_bytes);
    println!();

    // 6. Comparison
    println!("--- Comparison ---");
    println!("  MLX packed u32: {:08x?}", mlx_w);
    println!("  Tri packed u32: {:08x?}", tri_w);

    if mlx_w == tri_w {
        println!("  WEIGHT BYTES: MATCH");
    } else {
        println!("  WEIGHT BYTES: DIFFER");
        // Diagnose bit-level difference
        for (i, (&m, &t)) in mlx_w.iter().zip(tri_w.iter()).enumerate() {
            if m != t {
                println!(
                    "    word[{}]: MLX={:08x} Tri={:08x} xor={:08x}",
                    i,
                    m,
                    t,
                    m ^ t
                );
                // Check if it's a byte-swap issue
                let m_swapped = m.swap_bytes();
                if m_swapped == t {
                    println!("    -> Byte order swapped (endianness)");
                }
                // Check if it's a bit-shift issue
                for shift in 1..32 {
                    if m.rotate_left(shift) == t || m.rotate_right(shift) == t {
                        println!("    -> Bit rotation by {}", shift);
                    }
                }
            }
        }
    }

    // Compare scales (at least one scale for one group)
    if !mlx_s.is_empty() && tri_scale.to_le_bytes() == mlx_s[0].to_le_bytes() {
        println!("  SCALE BYTES: MATCH");
    } else {
        println!("  SCALE BYTES: DIFFER");
        if !mlx_s.is_empty() {
            println!(
                "    MLX scale[0]: {} ({:02x?})",
                mlx_s[0],
                mlx_s[0].to_le_bytes()
            );
        }
        println!("    Tri scale:   {} ({:02x?})", tri_scale, tri_s_bytes);
    }

    // Compare biases
    if !mlx_b.is_empty() && tri_bias.to_le_bytes() == mlx_b[0].to_le_bytes() {
        println!("  BIAS BYTES: MATCH");
    } else {
        println!("  BIAS BYTES: DIFFER");
        if !mlx_b.is_empty() {
            println!(
                "    MLX bias[0]: {} ({:02x?})",
                mlx_b[0],
                mlx_b[0].to_le_bytes()
            );
        }
        println!("    Tri bias:   {} ({:02x?})", tri_bias, tri_b_bytes);
    }

    // 7. Structural comparison (shapes)
    println!();
    println!("--- Structural ---");
    println!(
        "  MLX w shape: {:?} -> {} elements",
        mlx_w_shape,
        mlx_w.len()
    );
    println!("  Tri w:        {} elements", tri_w.len());
    if mlx_w.len() == tri_w.len() {
        println!("  WEIGHT COUNT: MATCH ({})", mlx_w.len());
    } else {
        println!(
            "  WEIGHT COUNT: DIFFER MLX={} Tri={}",
            mlx_w.len(),
            tri_w.len()
        );
    }

    println!(
        "  MLX s count: {} (expected {} groups)",
        mlx_s.len(),
        if rows * cols > 64 {
            (rows * cols + 63) / 64
        } else {
            1
        }
    );
    println!("  Tri scale count: 1 (one group)");
    println!();
}
