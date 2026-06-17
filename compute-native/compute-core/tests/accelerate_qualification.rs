//! Accelerate native qualification tests.
//!
//! Proves cblas_sgemm correctness, shape validation, handle lifecycle,
//! and memory accounting.  Must run under inference-research profile
//! for native FFI behavior.

use tribunus_compute_core::backend::accelerate::AccelerateBackend;
use tribunus_compute_core::backend::MatmulOp;
use tribunus_compute_core::backend::TensorBackend;
use tribunus_compute_core::backend::{
    DType, QuantizedMatmulOp, QuantizedWeightHandle, RmsNormOp, RoPEOp,
};

fn accel() -> AccelerateBackend {
    AccelerateBackend::new()
}

// ── Known-answer 2×3 @ 3×4 ────────────────────────────────────────────

#[test]
fn known_answer_2x3_times_3x4() {
    let mut be = accel();
    let a = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let b = be
        .create_f32(
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            &[3, 4],
        )
        .unwrap();

    let op = MatmulOp { m: 2, n: 4, k: 3 };
    let c = be.matmul(&op, a, b).unwrap();
    let shape = be.shape(c).unwrap();
    assert_eq!(shape, vec![2, 4]);

    let result = be.read_f32(c).unwrap();
    // [1 2 3; 4 5 6] @ [7 8 9 10; 11 12 13 14; 15 16 17 18]
    // row0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
    //     = [74, 80, 86, 92]
    // row1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
    //     = [173, 188, 203, 218]
    let expected = [74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0];
    for (i, (&got, &exp)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.001,
            "c[{i}] expected {exp}, got {got}"
        );
    }

    be.release(c).unwrap();
    be.release(b).unwrap();
    be.release(a).unwrap();
}

// ── Shape validation ──────────────────────────────────────────────────

#[test]
fn matmul_rejects_3d_tensor() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 24], &[2, 3, 4]).unwrap();
    let b = be.create_f32(&[1.0; 24], &[2, 3, 4]).unwrap();
    let op = MatmulOp { m: 2, n: 4, k: 3 };
    let err = be.matmul(&op, a, b);
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("exactly 2D"));
    be.release(b).unwrap();
    be.release(a).unwrap();
}

#[test]
fn create_f32_rejects_negative_dim() {
    let mut be = accel();
    assert!(be.create_f32(&[1.0; 6], &[-2, 3]).is_err());
}

#[test]
fn create_f32_rejects_zero_dim() {
    let mut be = accel();
    assert!(be.create_f32(&[1.0; 0], &[0, 3]).is_err());
}

#[test]
fn create_f32_rejects_shape_product_mismatch() {
    let mut be = accel();
    assert!(be.create_f32(&[1.0; 5], &[2, 3]).is_err());
}

#[test]
fn matmul_rejects_dimension_mismatch() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    let b = be.create_f32(&[1.0; 8], &[4, 2]).unwrap();
    let op = MatmulOp { m: 2, n: 2, k: 3 };
    assert!(be.matmul(&op, a, b).is_err());
    be.release(b).unwrap();
    be.release(a).unwrap();
}

// ── Stale handle rejection ────────────────────────────────────────────

#[test]
fn stale_handle_rejected_on_matmul() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    let b = be.create_f32(&[1.0; 12], &[3, 4]).unwrap();
    be.release(a).unwrap(); // now stale
    let op = MatmulOp { m: 2, n: 4, k: 3 };
    assert!(be.matmul(&op, a, b).is_err());
    be.release(b).unwrap();
}

#[test]
fn double_release_rejected() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    be.release(a).unwrap();
    assert!(be.release(a).is_err());
}

#[test]
fn stale_evaluate_output_rejected() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    be.release(a).unwrap();
    let receipt = be.evaluate(0, &[a]);
    assert!(receipt.is_err());
}

// ── Memory accounting ─────────────────────────────────────────────────

#[test]
fn memory_returns_to_zero_after_release() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 600], &[20, 30]).unwrap();
    let b = be.create_f32(&[1.0; 1200], &[30, 40]).unwrap();
    let (active, _) = be.active_memory();
    assert!(active > 0);

    be.release(b).unwrap();
    be.release(a).unwrap();
    let (active, _) = be.active_memory();
    assert_eq!(active, 0, "memory must return to zero after all releases");
}

// ── Repeated execution ────────────────────────────────────────────────

#[test]
fn repeated_matmul_consistent_output() {
    let mut be = accel();
    let a = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let b = be
        .create_f32(
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            &[3, 4],
        )
        .unwrap();
    let op = MatmulOp { m: 2, n: 4, k: 3 };

    let c1 = be.matmul(&op, a, b).unwrap();
    let r1 = be.read_f32(c1).unwrap();
    let c2 = be.matmul(&op, a, b).unwrap();
    let r2 = be.read_f32(c2).unwrap();

    assert_eq!(r1.data.len(), r2.data.len());
    for (i, (&x, &y)) in r1.data.iter().zip(r2.data.iter()).enumerate() {
        assert!((x - y).abs() < 0.001, "mismatch at {i}: {x} vs {y}");
    }
}

// ── Generation reuse ──────────────────────────────────────────────────

#[test]
fn generation_increment_on_reuse() {
    let mut be = accel();
    let a = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    let gen1 = a.generation;
    be.release(a).unwrap();

    let b = be.create_f32(&[1.0; 6], &[2, 3]).unwrap();
    // Slot may be reused; generation must differ
    if b.slot == a.slot {
        assert_ne!(b.generation, gen1, "reused slot must have new generation");
    }
    be.release(b).unwrap();
}
// ── Element-wise arithmetic ─────────────────────────────────────────────

#[test]
fn test_add_basic() {
    let mut be = accel();
    let a = be.create_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = be.create_f32(&[5.0, 6.0, 7.0, 8.0], &[4]).unwrap();
    let c = be.add(a, b).unwrap();
    let result = be.read_f32(c).unwrap();
    assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    be.release(c).unwrap();
    be.release(b).unwrap();
    be.release(a).unwrap();
}

#[test]
fn test_add_shape_mismatch() {
    let mut be = accel();
    let a = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let b = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
        .unwrap();
    let err = be.add(a, b);
    assert!(err.is_err());
    let msg = err.unwrap_err();
    assert!(msg.contains("differ") || msg.contains("length"));
    be.release(b).unwrap();
    be.release(a).unwrap();
}

#[test]
fn test_multiply_basic() {
    let mut be = accel();
    let a = be.create_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = be.create_f32(&[5.0, 6.0, 7.0, 8.0], &[4]).unwrap();
    let c = be.multiply(a, b).unwrap();
    let result = be.read_f32(c).unwrap();
    assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
    be.release(c).unwrap();
    be.release(b).unwrap();
    be.release(a).unwrap();
}

#[test]
fn test_multiply_shape_mismatch() {
    let mut be = accel();
    let a = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let b = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
        .unwrap();
    let err = be.multiply(a, b);
    assert!(err.is_err());
    let msg = err.unwrap_err();
    assert!(msg.contains("differ") || msg.contains("length"));
    be.release(b).unwrap();
    be.release(a).unwrap();
}

// ── Activations ────────────────────────────────────────────────────────

#[test]
fn test_silu_basic() {
    let mut be = accel();
    let x = be.create_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    let result = be.silu(x).unwrap();
    let vals = be.read_f32(result).unwrap();
    let expected = [-0.2384, -0.2689, 0.0, 0.7311, 1.7616];
    for (i, (&got, &exp)) in vals.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "silu[{i}] expected {exp}, got {got}"
        );
    }
    be.release(result).unwrap();
    be.release(x).unwrap();
}

// ── Transpose ──────────────────────────────────────────────────────────

#[test]
fn test_transpose_2x3() {
    let mut be = accel();
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let y = be.transpose(x, &[1, 0]).unwrap();
    let shape = be.shape(y).unwrap();
    assert_eq!(shape, vec![3, 2]);
    let data = be.read_f32(y).unwrap();
    assert_eq!(data.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    be.release(y).unwrap();
    be.release(x).unwrap();
}

#[test]
fn test_transpose_not_2d() {
    let mut be = accel();
    let x = be.create_f32(&[1.0; 6], &[6]).unwrap();
    let err = be.transpose(x, &[0]);
    assert!(err.is_err());
    be.release(x).unwrap();
}

// ── Reshape ────────────────────────────────────────────────────────────

#[test]
fn test_reshape_valid() {
    let mut be = accel();
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let y = be.reshape(x, &[3, 2]).unwrap();
    let shape = be.shape(y).unwrap();
    assert_eq!(shape, vec![3, 2]);
    let data = be.read_f32(y).unwrap();
    assert_eq!(data.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    be.release(y).unwrap();
    be.release(x).unwrap();
}

#[test]
fn test_reshape_invalid() {
    let mut be = accel();
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let err = be.reshape(x, &[2, 4]);
    assert!(err.is_err());
    let msg = err.unwrap_err();
    assert!(msg.contains("mismatch") || msg.contains("count"));
    be.release(x).unwrap();
}

// ── Softmax ────────────────────────────────────────────────────────────

#[test]
fn test_softmax_basic() {
    let mut be = accel();
    let x = be.create_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let y = be.softmax(x, 0).unwrap();
    let vals = be.read_f32(y).unwrap();
    let expected = [0.0900, 0.2447, 0.6652];
    for (i, (&got, &exp)) in vals.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "softmax[{i}] expected {exp}, got {got}"
        );
    }
    // Probabilities should sum to ~1.0
    let sum: f32 = vals.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "softmax probabilities sum to {sum}, expected ~1.0"
    );
    be.release(y).unwrap();
    be.release(x).unwrap();
}

#[test]
fn test_softmax_2d() {
    let mut be = accel();
    // 2x3 matrix: [[1,2,3],[1,2,3]]
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3])
        .unwrap();
    // softmax along axis=1 (last dim)
    let y = be.softmax(x, 1).unwrap();
    let vals = be.read_f32(y).unwrap();
    let shape = be.shape(y).unwrap();
    assert_eq!(shape, vec![2, 3]);
    // Each row should sum to ~1.0
    for row in 0..2 {
        let row_sum: f32 = vals.data[row * 3..(row + 1) * 3].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 0.01,
            "softmax row {row} sums to {row_sum}, expected ~1.0"
        );
    }
    be.release(y).unwrap();
    be.release(x).unwrap();
}

// ── Index select ───────────────────────────────────────────────────────

#[test]
fn test_index_select_1d() {
    let mut be = accel();
    let x = be
        .create_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5])
        .unwrap();
    let y = be.index_select(x, &[0, 2, 4], 0).unwrap();
    let vals = be.read_f32(y).unwrap();
    assert_eq!(vals.data, vec![10.0, 30.0, 50.0]);
    be.release(y).unwrap();
    be.release(x).unwrap();
}

#[test]
fn test_index_select_2d_axis0() {
    let mut be = accel();
    // 2x3 matrix: [[1,2,3],[4,5,6]]
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    let y = be.index_select(x, &[1], 0).unwrap();
    let vals = be.read_f32(y).unwrap();
    assert_eq!(vals.data, vec![4.0, 5.0, 6.0]);
    let shape = be.shape(y).unwrap();
    assert_eq!(shape, vec![1, 3]);
    be.release(y).unwrap();
    be.release(x).unwrap();
}

#[test]
fn test_index_select_empty_indices() {
    let mut be = accel();
    let x = be.create_f32(&[10.0, 20.0, 30.0], &[3]).unwrap();
    let y = be.index_select(x, &[], 0).unwrap();
    let vals = be.read_f32(y).unwrap();
    assert!(vals.data.is_empty());
    be.release(y).unwrap();
    be.release(x).unwrap();
}

// ── Quantized matmul ───────────────────────────────────────────────────

#[test]
fn test_quantized_matmul_8bit() {
    let mut be = accel();
    // Create input x = [[1,1,1],[2,2,2]]
    let x = be
        .create_f32(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3])
        .unwrap();
    // Register a quantized weight: raw u8 values [1,2,3,4,5,6], shape [3,2], group_size=3, bits=8
    // This represents dequantized [[1,2],[3,4],[5,6]] with scale=1.0, bias=0.0 per group
    let w_raw: Vec<u8> = (1..=6).collect();
    let w = be.register_quantized_weight(&w_raw, 3, 8, &[3, 2]);
    // Scales and biases: 2 groups (k*e / group_size = 6/3 = 2)
    let scales = be.create_f32(&[1.0; 2], &[2]).unwrap();
    let biases = be.create_f32(&[0.0; 2], &[2]).unwrap();
    let op = QuantizedMatmulOp {
        m: 2, // x rows
        n: 2, // weight cols
        k: 3,
        input_dtype: DType::F32,
        weight_dtype: DType::F32,
        scale_dtype: DType::F32,
        bias_dtype: DType::F32,
        output_dtype: DType::F32,
        group_size: 3,
        bits: 8,
        transpose: false,
    };
    let c = be
        .quantized_matmul(&op, x, w, scales, biases)
        .expect("quantized_matmul should succeed");
    let shape = be.shape(c).unwrap();
    assert_eq!(shape, vec![2, 2], "output shape should be [2,2]");
    let result = be.read_f32(c).unwrap();
    // x @ W = [[1,1,1];[2,2,2]] @ [[1,2],[3,4],[5,6]]
    // Row 0: [1*1+1*3+1*5, 1*2+1*4+1*6] = [9, 12]
    // Row 1: [2*1+2*3+2*5, 2*2+2*4+2*6] = [18, 24]
    let expected = [9.0, 12.0, 18.0, 24.0];
    for (i, (&got, &exp)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.001,
            "c[{i}] expected {exp}, got {got}"
        );
    }
    be.release_weight(w).unwrap();
    be.release(c).unwrap();
    be.release(biases).unwrap();
    be.release(scales).unwrap();
    be.release(x).unwrap();
}

// ── 4-bit quantized matmul ───────────────────────────────────────────────

#[test]
fn test_quantized_matmul_4bit() {
    let mut be = accel();
    let x = be.create_f32(&[1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
    // 4-bit quantized: all ones, shape [2,3], group_size=3
    // bytes_per_group = ceil(3*4/8) = 2, total = 2 groups * 2 bytes = 4 bytes
    // Group 0: [1,1,1] → byte0=0x11 (low=1,high=1), byte1=0x01 (low=1)
    // Group 1: [1,1,1] → byte2=0x11 (low=1,high=1), byte3=0x01 (low=1)
    let w_raw: Vec<u8> = vec![0x11, 0x01, 0x11, 0x01];
    let w = be.register_quantized_weight(&w_raw, 3, 4, &[2, 3]);
    let scales = be.create_f32(&[1.0; 2], &[2]).unwrap();
    let biases = be.create_f32(&[0.0; 2], &[2]).unwrap();
    let op = QuantizedMatmulOp {
        m: 2,
        n: 3,
        k: 2,
        input_dtype: DType::F32,
        weight_dtype: DType::F32,
        scale_dtype: DType::F32,
        bias_dtype: DType::F32,
        output_dtype: DType::F32,
        group_size: 3,
        bits: 4,
        transpose: false,
    };
    let c = be
        .quantized_matmul(&op, x, w, scales, biases)
        .expect("4-bit quantized_matmul should succeed");
    let shape = be.shape(c).unwrap();
    assert_eq!(shape, vec![2, 3], "output shape should be [2,3]");
    // W = [[1,1,1],[1,1,1]], x = [[1,1],[2,2]]
    // Row 0: [1+1, 1+1, 1+1] = [2,2,2]
    // Row 1: [2+2, 2+2, 2+2] = [4,4,4]
    let expected = [2.0, 2.0, 2.0, 4.0, 4.0, 4.0];
    let result = be.read_f32(c).unwrap();
    for (i, (&got, &exp)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.001,
            "4-bit c[{i}] expected {exp}, got {got}"
        );
    }
    be.release_weight(w).unwrap();
    be.release(c).unwrap();
    be.release(biases).unwrap();
    be.release(scales).unwrap();
    be.release(x).unwrap();
}

// ── 2-bit quantized matmul ───────────────────────────────────────────────

#[test]
fn test_quantized_matmul_2bit() {
    let mut be = accel();
    let x = be.create_f32(&[1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
    // 2-bit quantized: all ones, shape [2,4], group_size=4
    // bytes_per_group = ceil(4*2/8) = 1, total = 2 groups * 1 byte = 2 bytes
    // Group 0: [1,1,1,1] → byte0 = 0b01010101 = 0x55
    // Group 1: [1,1,1,1] → byte1 = 0b01010101 = 0x55
    let w_raw: Vec<u8> = vec![0x55, 0x55];
    let w = be.register_quantized_weight(&w_raw, 4, 2, &[2, 4]);
    let scales = be.create_f32(&[1.0; 2], &[2]).unwrap();
    let biases = be.create_f32(&[0.0; 2], &[2]).unwrap();
    let op = QuantizedMatmulOp {
        m: 2,
        n: 4,
        k: 2,
        input_dtype: DType::F32,
        weight_dtype: DType::F32,
        scale_dtype: DType::F32,
        bias_dtype: DType::F32,
        output_dtype: DType::F32,
        group_size: 4,
        bits: 2,
        transpose: false,
    };
    let c = be
        .quantized_matmul(&op, x, w, scales, biases)
        .expect("2-bit quantized_matmul should succeed");
    let shape = be.shape(c).unwrap();
    assert_eq!(shape, vec![2, 4], "output shape should be [2,4]");
    // W = [[1,1,1,1],[1,1,1,1]], x = [[1,1],[2,2]]
    // Row 0: [1+1, 1+1, 1+1, 1+1] = [2,2,2,2]
    // Row 1: [2+2, 2+2, 2+2, 2+2] = [4,4,4,4]
    let expected = [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
    let result = be.read_f32(c).unwrap();
    for (i, (&got, &exp)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.001,
            "2-bit c[{i}] expected {exp}, got {got}"
        );
    }
    be.release_weight(w).unwrap();
    be.release(c).unwrap();
    be.release(biases).unwrap();
    be.release(scales).unwrap();
    be.release(x).unwrap();
}

// ── External binding ───────────────────────────────────────────────────

#[test]
fn test_bind_external() {
    let mut be = accel();
    let data: Vec<u8> = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let h = be.bind_external(0, &data, &[4], DType::F32).unwrap();
    let shape = be.shape(h).unwrap();
    assert_eq!(shape, vec![4]);
    let vals = be.read_f32(h).unwrap();
    for (i, (&got, &exp)) in vals
        .data
        .iter()
        .zip([1.0, 2.0, 3.0, 4.0].iter())
        .enumerate()
    {
        assert!(
            (got - exp).abs() < 0.001,
            "bind_external[{i}] expected {exp}, got {got}"
        );
    }
    be.release(h).unwrap();
}

// ── RMS norm ───────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_basic() {
    let mut be = accel();
    let x = be.create_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let w = be.create_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let op = RmsNormOp { dim: 4, eps: 1e-5 };
    let y = be.rms_norm(&op, x, w).unwrap();
    let vals = be.read_f32(y).unwrap();
    let expected = [0.3651, 0.7303, 1.0954, 1.4606];
    for (i, (&got, &exp)) in vals.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "rms_norm[{i}] expected {exp}, got {got}"
        );
    }
    be.release(y).unwrap();
    be.release(w).unwrap();
    be.release(x).unwrap();
}

// ── RoPE ───────────────────────────────────────────────────────────────

#[test]
fn test_rope_basic() {
    let mut be = accel();
    // 2 positions, head_dim=4
    let x = be
        .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
        .unwrap();
    let op = RoPEOp {
        head_dim: 4,
        positions: vec![0, 1],
    };
    let y = be.rope(&op, x).unwrap();
    let vals = be.read_f32(y).unwrap();
    // Position 0: theta=0 → cos=1, sin=0 → output matches input for first 4 elements
    for i in 0..4 {
        assert!(
            (vals.data[i] - [1.0, 2.0, 3.0, 4.0][i]).abs() < 0.001,
            "rope position 0 element {i}: expected unchanged, got {}",
            vals.data[i]
        );
    }
    // Position 1: should be rotated (different from input)
    // First half-dim (i=0): theta=1.0, cos≈0.5403, sin≈0.8415
    //   out[4] = 5*0.5403 - 6*0.8415 = 2.7015 - 5.049 = -2.3475
    //   out[5] = 5*0.8415 + 6*0.5403 = 4.2075 + 3.2418 = 7.4493
    let expected_rotated = [-2.3475, 7.4493];
    for i in 0..2 {
        assert!(
            (vals.data[4 + i] - expected_rotated[i]).abs() < 0.01,
            "rope position 1 element {i}: expected {}, got {}",
            expected_rotated[i],
            vals.data[4 + i]
        );
    }
    be.release(y).unwrap();
    be.release(x).unwrap();
}
