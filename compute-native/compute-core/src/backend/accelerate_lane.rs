//! Accelerate CPU execution lane — selected ops for CPU on Apple Silicon.
//!
//! Receives ArenaView references into the UnifiedExecutionArena.
//! No memory copies — operates on CPU-accessible arena pages directly.
//!
//! Uses NEON intrinsics on aarch64, vDSP through accelerate_ffi on macOS,
//! and falls back to scalar Rust loops everywhere else.

// ── NEON-accelerated RMSNorm ───────────────────────────────────────────────

/// NEON-optimized 4-wide RMSNorm loop for aarch64.
///
/// Computes x * inv_rms * w where inv_rms = 1 / sqrt(mean(x^2) + eps).
/// The sum-of-squares phase uses `vfmaq_f32` to fuse multiply-accumulate, and
/// `vaddvq_f32` for the horizontal reduction. The write-back phase multiplies
/// the scaled value by the weight vector, all four elements at a time.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn rms_norm_neon(x: *const f32, w: *const f32, out: *mut f32, n: usize, eps: f32) {
    unsafe {
        use std::arch::aarch64::*;

        // ── Phase 1: sum of squares ──
        let mut sum_sq = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let vx = vld1q_f32(x.add(i));
            // Fused multiply-accumulate: sum_sq += vx * vx
            sum_sq = vfmaq_f32(sum_sq, vx, vx);
            i += 4;
        }
        // Horizontal add across the single vector
        let r = vaddvq_f32(sum_sq);
        let mut rsum = r;
        while i < n {
            let v = *x.add(i);
            rsum += v * v;
            i += 1;
        }

        let inv_rms = (rsum / n as f32 + eps).sqrt().recip();

        // ── Phase 2: scale by weight ──
        let mut i = 0usize;
        while i + 4 <= n {
            let vx = vld1q_f32(x.add(i));
            let vw = vld1q_f32(w.add(i));
            let vs = vmulq_n_f32(vx, inv_rms);
            vst1q_f32(out.add(i), vmulq_f32(vs, vw));
            i += 4;
        }
        while i < n {
            *out.add(i) = *x.add(i) * inv_rms * *w.add(i);
            i += 1;
        }
    }
}

/// Scalar RMSNorm fallback (no NEON or when n < 4).
///
/// out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
pub fn rms_norm_scalar(
    x_ptr: *const f32,
    w_ptr: *const f32,
    out_ptr: *mut f32,
    dim: usize,
    eps: f32,
) {
    unsafe {
        let mut sum_sq = 0.0f32;
        for i in 0..dim {
            let v = *x_ptr.add(i);
            sum_sq += v * v;
        }
        let inv_rms = ((sum_sq / dim as f32) + eps).sqrt().recip();
        for i in 0..dim {
            *out_ptr.add(i) = *x_ptr.add(i) * inv_rms * *w_ptr.add(i);
        }
    }
}

// ── Softmax (combined max+sum pattern from ggml) ──────────────────────────

/// Combined single-pass softmax using the ggml pattern.
///
/// This is strictly better than the traditional two-pass (max, then exp+sum)
/// because it walks the array only twice (once for max, once for exp+sum+norm,
/// vs three walks in the original two-pass form).  For very short arrays the
/// difference is negligible; for long arrays it saves a full traversal.
///
/// See ggml_vec_soft_max_f32 in ggml/src/ggml-cpu/vec/vec.cpp.
pub fn softmax_pass(logits: &mut [f32]) -> Result<(), String> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }

    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Single pass: exp the shifted values and accumulate sum
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    if sum <= 0.0 {
        return Err("softmax sum is zero — all-NaN or all -inf logits".into());
    }

    // Normalize
    for v in logits.iter_mut() {
        *v /= sum;
    }
    Ok(())
}

// ── vDSP helpers (macOS only, delegating to super::accelerate_ffi) ────────

/// Element-wise addition via vDSP_vadd.
#[cfg(target_os = "macos")]
pub fn add_vdsp(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    unsafe {
        super::accelerate_ffi::vDSP_vadd(a.as_ptr(), 1, b.as_ptr(), 1, c.as_mut_ptr(), 1, n as i32);
    }
}

/// Element-wise multiplication via vDSP_vmul.
#[cfg(target_os = "macos")]
pub fn mul_vdsp(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    unsafe {
        super::accelerate_ffi::vDSP_vmul(a.as_ptr(), 1, b.as_ptr(), 1, c.as_mut_ptr(), 1, n as i32);
    }
}

/// Scalar-vector multiply via vDSP_vsmul:  c[i] = a[i] * b
#[cfg(target_os = "macos")]
pub fn scale_vdsp(a: &[f32], b: f32, c: &mut [f32]) {
    let n = a.len();
    assert_eq!(c.len(), n);
    unsafe {
        super::accelerate_ffi::vDSP_vsmul(a.as_ptr(), 1, &b, c.as_mut_ptr(), 1, n as i32);
    }
}

/// Vector sum via vDSP_sve.
#[cfg(target_os = "macos")]
pub fn sum_vdsp(a: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        super::accelerate_ffi::vDSP_sve(a.as_ptr(), 1, &mut result, a.len() as i32);
    }
    result
}

// ── Scalar fallbacks for non-macOS ─────────────────────────────────────────

/// Scalar element-wise addition (non-macOS fallback).
#[cfg(not(target_os = "macos"))]
pub fn add_vdsp(a: &[f32], b: &[f32], c: &mut [f32]) {
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        c[i] = av + bv;
    }
}

/// Scalar element-wise multiplication (non-macOS fallback).
#[cfg(not(target_os = "macos"))]
pub fn mul_vdsp(a: &[f32], b: &[f32], c: &mut [f32]) {
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        c[i] = av * bv;
    }
}

/// Scalar scale (non-macOS fallback).
#[cfg(not(target_os = "macos"))]
pub fn scale_vdsp(a: &[f32], b: f32, c: &mut [f32]) {
    for (i, &av) in a.iter().enumerate() {
        c[i] = av * b;
    }
}

/// Scalar sum (non-macOS fallback).
#[cfg(not(target_os = "macos"))]
pub fn sum_vdsp(a: &[f32]) -> f32 {
    a.iter().sum()
}

// ── 4x4 f32 microkernel for small matmuls (NEON) ──────────────────────────

/// 4x4 f32 matrix multiply microkernel using NEON.
///
/// Computes C[4][4] = A[4][K] * B[K][4] where both A and B are row-major.
/// Uses four NEON accumulators (one per output row) and streams four B columns
/// at a time for inner-dimension parallelism.
///
/// Falls back to scalar when NEON is unavailable.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn matmul_4x4_neon(c: &mut [f32], a: &[f32], b: &[f32], k: usize) {
    assert!(c.len() >= 16, "c must be at least 4x4 (16 elements)");
    assert!(a.len() >= 4 * k, "a must be at least 4*k elements");
    assert!(b.len() >= k * 4, "b must be at least k*4 elements");

    unsafe {
        use std::arch::aarch64::*;

        // Four accumulators, one per output row
        let mut acc = [vdupq_n_f32(0.0); 4];

        let mut i = 0usize;
        while i + 4 <= k {
            // Load four B columns (each is 4 floats = one column-vector in B)
            let b0 = vld1q_f32(b.as_ptr().add(i * 4));
            let b1 = vld1q_f32(b.as_ptr().add((i + 1) * 4));
            let b2 = vld1q_f32(b.as_ptr().add((i + 2) * 4));
            let b3 = vld1q_f32(b.as_ptr().add((i + 3) * 4));

            for r in 0..4 {
                let av0 = vdupq_n_f32(a[r * k + i]);
                let av1 = vdupq_n_f32(a[r * k + i + 1]);
                let av2 = vdupq_n_f32(a[r * k + i + 2]);
                let av3 = vdupq_n_f32(a[r * k + i + 3]);
                acc[r] = vfmaq_f32(acc[r], av0, b0);
                acc[r] = vfmaq_f32(acc[r], av1, b1);
                acc[r] = vfmaq_f32(acc[r], av2, b2);
                acc[r] = vfmaq_f32(acc[r], av3, b3);
            }
            i += 4;
        }

        // Residual K iterations (when k % 4 != 0)
        while i < k {
            let b_col = vld1q_f32(b.as_ptr().add(i * 4));
            for r in 0..4 {
                let av = vdupq_n_f32(a[r * k + i]);
                acc[r] = vfmaq_f32(acc[r], av, b_col);
            }
            i += 1;
        }

        // Store results
        for r in 0..4 {
            vst1q_f32(c.as_mut_ptr().add(r * 4), acc[r]);
        }
    }
}

/// Scalar 4x4 f32 matmul fallback (used when NEON is unavailable or k is tiny).
pub fn matmul_4x4_scalar(c: &mut [f32], a: &[f32], b: &[f32], k: usize) {
    for r in 0..4 {
        for cc in 0..4 {
            let mut s = 0.0f32;
            for i in 0..k {
                s += a[r * k + i] * b[i * 4 + cc];
            }
            c[r * 4 + cc] = s;
        }
    }
}

/// Dispatch: try NEON 4x4 matmul, fall back to scalar.
pub fn matmul_4x4(c: &mut [f32], a: &[f32], b: &[f32], k: usize) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return matmul_4x4_neon(c, a, b, k);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        matmul_4x4_scalar(c, a, b, k);
    }
}

// ── Free functions (retained for backward compat) ─────────────────────────

/// Accelerate-based reduction sum.
///
/// Uses vDSP_sve on macOS, plain iterator sum everywhere else.
pub fn sum_accelerate(data: &[f32]) -> f32 {
    sum_vdsp(data)
}

/// Legacy RMSNorm entry point — dispatches to NEON or scalar.
///
/// Prefer [`AccelerateLane::rms_norm`] for the safe slice-based API.
#[cfg(target_arch = "aarch64")]
pub fn rms_norm_accelerate(
    x_ptr: *mut f32,
    w_ptr: *const f32,
    out_ptr: *mut f32,
    dim: usize,
    eps: f32,
) {
    #[cfg(target_feature = "neon")]
    {
        rms_norm_neon(x_ptr as *const f32, w_ptr, out_ptr, dim, eps);
        return;
    }
    #[cfg(not(target_feature = "neon"))]
    {
        rms_norm_scalar(x_ptr as *const f32, w_ptr, out_ptr, dim, eps);
    }
}

// ── AccelerateLane scheduler ──────────────────────────────────────────────

/// Accelerate lane scheduler.
///
/// Owns no memory — receives `&[f32]` views that are backed by the
/// shared unified arena.
pub struct AccelerateLane {
    pub name: String,
}

impl AccelerateLane {
    pub fn new() -> Self {
        AccelerateLane {
            name: "accelerate-cpu".into(),
        }
    }

    /// Run RMSNorm via NEON (preferred) or scalar fallback.
    ///
    /// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
    pub fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        eps: f32,
    ) -> Result<(), String> {
        if x.len() != weight.len() || x.len() != out.len() {
            return Err(format!(
                "RMSNorm dim mismatch: x={}, weight={}, out={}",
                x.len(),
                weight.len(),
                out.len()
            ));
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            rms_norm_neon(x.as_ptr(), weight.as_ptr(), out.as_mut_ptr(), x.len(), eps);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            rms_norm_scalar(x.as_ptr(), weight.as_ptr(), out.as_mut_ptr(), x.len(), eps);
        }
        Ok(())
    }

    /// Run softmax over logits.
    ///
    /// Uses the single-pass ggml pattern (max, then exp+sum+normalize).
    pub fn softmax(&self, logits: &mut [f32]) -> Result<(), String> {
        softmax_pass(logits)
    }

    /// Element-wise addition via vDSP or scalar fallback.
    ///
    /// c[i] = a[i] + b[i]
    pub fn add(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), String> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err("add: dimension mismatch".into());
        }
        add_vdsp(a, b, c);
        Ok(())
    }

    /// Element-wise multiplication via vDSP or scalar fallback.
    ///
    /// c[i] = a[i] * b[i]
    pub fn mul(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), String> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err("mul: dimension mismatch".into());
        }
        mul_vdsp(a, b, c);
        Ok(())
    }

    /// Scale vector by scalar via vDSP or scalar fallback.
    ///
    /// c[i] = a[i] * b
    pub fn scale(&self, a: &[f32], b: f32, c: &mut [f32]) -> Result<(), String> {
        if a.len() != c.len() {
            return Err("scale: dimension mismatch".into());
        }
        scale_vdsp(a, b, c);
        Ok(())
    }

    /// Small matmul (4x4) using NEON microkernel or scalar fallback.
    ///
    /// `c` must be at least 16 elements (4x4), `a` at least 4*k, `b` at least k*4.
    /// For larger matmuls, prefer cblas_sgemm through the full accelerate backend.
    pub fn matmul(&self, c: &mut [f32], a: &[f32], b: &[f32], k: usize) -> Result<(), String> {
        if c.len() < 16 {
            return Err("matmul 4x4: c must have at least 16 elements".into());
        }
        if a.len() < 4 * k {
            return Err("matmul 4x4: a must have at least 4*k elements".into());
        }
        if b.len() < k * 4 {
            return Err("matmul 4x4: b must have at least k*4 elements".into());
        }
        matmul_4x4(c, a, b, k);
        Ok(())
    }

    /// Sum all elements.
    ///
    /// Uses vDSP_sve on macOS, plain iterator sum everywhere else.
    pub fn sum(&self, data: &[f32]) -> f32 {
        sum_vdsp(data)
    }

    /// Sample a token from logits at the given temperature.
    ///
    /// * `temperature` near 1.0 applies softmax as-is.
    /// * `temperature` near 0.0 or negative clamps to greedy argmax.
    pub fn sample(&self, logits: &[f32], temperature: f32) -> Result<u32, String> {
        if logits.is_empty() {
            return Err("empty logits".into());
        }
        // Greedy argmax for temperature <= 0 or near 0
        if temperature <= 1e-6 {
            let idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            return Ok(idx as u32);
        }
        let mut scaled: Vec<f32>;
        let probs: &[f32] = if (temperature - 1.0).abs() > 1e-6 {
            scaled = logits.iter().map(|l| l / temperature).collect();
            let _ = self.softmax(&mut scaled);
            &scaled
        } else {
            logits
        };
        // Simple argmax for now (greedy)
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(idx as u32)
    }
}

impl Default for AccelerateLane {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RMSNorm ──────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_basic() {
        let lane = AccelerateLane::new();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut out = vec![0.0f32; 4];
        lane.rms_norm(&x, &w, &mut out, 1e-6).unwrap();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0 + 1e-6).sqrt();
        let inv = 1.0 / rms;
        for i in 0..4 {
            let expected = x[i] * inv * w[i];
            assert!((out[i] - expected).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_rms_norm_dim_mismatch() {
        let lane = AccelerateLane::new();
        let x = vec![1.0f32; 10];
        let w = vec![0.5f32; 5];
        let mut out = vec![0.0f32; 10];
        assert!(lane.rms_norm(&x, &w, &mut out, 1e-6).is_err());
    }

    #[test]
    fn test_rms_norm_large_dim() {
        let lane = AccelerateLane::new();
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) / n as f32).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + 0.5 * (i as f32) / n as f32).collect();
        let mut out = vec![0.0f32; n];
        lane.rms_norm(&x, &w, &mut out, 1e-6).unwrap();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + 1e-6).sqrt();
        for i in 0..n {
            let expected = x[i] * inv_rms * w[i];
            assert!((out[i] - expected).abs() < 1e-4, "mismatch at {i}");
        }
    }

    #[test]
    fn test_rms_norm_scalar_matches_neon() {
        // Verify scalar and (when available) NEON produce identical results.
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32)).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 / (i + 1) as f32).collect();
        let eps = 1e-6;

        // Scalar
        let mut out_scalar = vec![0.0f32; n];
        rms_norm_scalar(x.as_ptr(), w.as_ptr(), out_scalar.as_mut_ptr(), n, eps);

        // Via AccelerateLane (may use NEON)
        let mut out_lane = vec![0.0f32; n];
        AccelerateLane::new()
            .rms_norm(&x, &w, &mut out_lane, eps)
            .unwrap();

        for i in 0..n {
            assert!(
                (out_scalar[i] - out_lane[i]).abs() < 1e-5,
                "mismatch at {i}: scalar={} lane={}",
                out_scalar[i],
                out_lane[i]
            );
        }
    }

    // ── Softmax ──────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_basic() {
        let lane = AccelerateLane::new();
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        lane.softmax(&mut logits).unwrap();
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax does not sum to 1");
    }

    #[test]
    fn test_softmax_all_neginf() {
        let lane = AccelerateLane::new();
        let mut logits = vec![f32::NEG_INFINITY; 4];
        assert!(lane.softmax(&mut logits).is_err());
    }

    #[test]
    fn test_softmax_negative_input() {
        let lane = AccelerateLane::new();
        let mut logits = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0];
        lane.softmax(&mut logits).unwrap();
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax of negatives does not sum to 1"
        );
        // Highest input should produce highest probability
        assert!(logits[0] > logits[1], "exp(-1) > exp(-2)");
        assert!(logits[0] > logits[4], "exp(-1) > exp(-5)");
    }

    #[test]
    fn test_softmax_pass_basic() {
        let mut logits = vec![0.0f32, 1.0, 2.0, 3.0];
        softmax_pass(&mut logits).unwrap();
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_pass_empty() {
        let mut logits: Vec<f32> = vec![];
        assert!(softmax_pass(&mut logits).is_err());
    }

    #[test]
    fn test_softmax_pass_all_neginf() {
        let mut logits = vec![f32::NEG_INFINITY; 4];
        assert!(softmax_pass(&mut logits).is_err());
    }

    // ── Sum ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_accelerate() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((sum_accelerate(&data) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_method() {
        let lane = AccelerateLane::new();
        let data = vec![10.0f32, 20.0, 30.0];
        assert!((lane.sum(&data) - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_empty() {
        let data: Vec<f32> = vec![];
        assert!((sum_accelerate(&data) - 0.0).abs() < 1e-6);
    }

    // ── vDSP operations ──────────────────────────────────────────────────

    #[test]
    fn test_vdsp_add() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut c = vec![0.0f32; 3];
        add_vdsp(&a, &b, &mut c);
        assert!((c[0] - 5.0).abs() < 1e-5);
        assert!((c[1] - 7.0).abs() < 1e-5);
        assert!((c[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_vdsp_mul() {
        let a = vec![2.0f32, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0];
        let mut c = vec![0.0f32; 3];
        mul_vdsp(&a, &b, &mut c);
        assert!((c[0] - 10.0).abs() < 1e-5);
        assert!((c[1] - 18.0).abs() < 1e-5);
        assert!((c[2] - 28.0).abs() < 1e-5);
    }

    #[test]
    fn test_vdsp_scale() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut c = vec![0.0f32; 3];
        scale_vdsp(&a, 2.5, &mut c);
        assert!((c[0] - 2.5).abs() < 1e-5);
        assert!((c[1] - 5.0).abs() < 1e-5);
        assert!((c[2] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_vdsp_sum() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = sum_vdsp(&a);
        assert!((s - 10.0).abs() < 1e-5);
    }

    // ── AccelerateLane methods ───────────────────────────────────────────

    #[test]
    fn test_lane_add() {
        let lane = AccelerateLane::new();
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let mut c = vec![0.0f32; 2];
        lane.add(&a, &b, &mut c).unwrap();
        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_lane_add_dim_mismatch() {
        let lane = AccelerateLane::new();
        assert!(lane.add(&[1.0], &[1.0, 2.0], &mut [0.0; 2]).is_err());
    }

    #[test]
    fn test_lane_mul() {
        let lane = AccelerateLane::new();
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut c = vec![0.0f32; 2];
        lane.mul(&a, &b, &mut c).unwrap();
        assert!((c[0] - 8.0).abs() < 1e-5);
        assert!((c[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_lane_mul_dim_mismatch() {
        let lane = AccelerateLane::new();
        assert!(lane.mul(&[1.0], &[1.0, 2.0], &mut [0.0; 2]).is_err());
    }

    #[test]
    fn test_lane_scale() {
        let lane = AccelerateLane::new();
        let a = vec![1.0f32, 2.0, 3.0];
        let mut c = vec![0.0f32; 3];
        lane.scale(&a, 3.0, &mut c).unwrap();
        assert!((c[0] - 3.0).abs() < 1e-5);
        assert!((c[1] - 6.0).abs() < 1e-5);
        assert!((c[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_lane_scale_dim_mismatch() {
        let lane = AccelerateLane::new();
        assert!(lane.scale(&[1.0, 2.0], 2.0, &mut [0.0]).is_err());
    }

    // ── 4x4 Matmul ──────────────────────────────────────────────────────

    #[test]
    fn test_matmul_4x4_scalar_identity() {
        // A = 4x4 identity, B = [0..15], expected C = B
        let k = 4;
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_4x4_scalar(&mut c, &a, &b, k);
        for i in 0..16 {
            assert!((c[i] - b[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_matmul_4x4_dispatch_identity() {
        let k = 4;
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_4x4(&mut c, &a, &b, k);
        for i in 0..16 {
            assert!((c[i] - b[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_matmul_4x4_dispatch_all_ones() {
        let k = 4;
        let a: Vec<f32> = vec![1.0f32; 16];
        let b: Vec<f32> = vec![2.0f32; 16];
        let mut c = vec![0.0f32; 16];
        matmul_4x4(&mut c, &a, &b, k);
        // Each row of A (all 1s) dot each column of B (all 2s) = 8.0
        for i in 0..16 {
            assert!((c[i] - 8.0).abs() < 1e-5, "mismatch at {i}: got {}", c[i]);
        }
    }

    #[test]
    fn test_lane_matmul() {
        let lane = AccelerateLane::new();
        let k = 4;
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; 16];
        lane.matmul(&mut c, &a, &b, k).unwrap();
        for i in 0..16 {
            assert!((c[i] - b[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_lane_matmul_bad_dims() {
        let lane = AccelerateLane::new();
        let mut c = vec![0.0f32; 4]; // too small
        assert!(lane.matmul(&mut c, &[1.0], &[2.0], 1).is_err());
    }

    // ── Sample ───────────────────────────────────────────────────────────

    #[test]
    fn test_sample_greedy() {
        let lane = AccelerateLane::new();
        let logits = vec![0.1f32, 0.2, 10.0, 0.3];
        let token = lane.sample(&logits, 0.0).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_empty_error() {
        let lane = AccelerateLane::new();
        assert!(lane.sample(&[], 1.0).is_err());
    }

    // ── Scalar fallback correctness ──────────────────────────────────────

    #[test]
    fn test_scalar_rms_norm_matches_direct() {
        let x: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let w: Vec<f32> = vec![0.5f32; 16];
        let mut out = vec![0.0f32; 16];
        let eps = 1e-6;

        rms_norm_scalar(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), 16, eps);

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / 16.0 + eps).sqrt();
        for i in 0..16 {
            assert!((out[i] - x[i] * inv_rms * w[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_matmul_4x4_scalar_random() {
        let k = 3;
        let a: Vec<f32> = (0..12).map(|i| (i + 1) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..12).map(|i| (i + 1) as f32 * 0.2).collect();
        let mut c = vec![0.0f32; 16];
        matmul_4x4_scalar(&mut c, &a, &b, k);

        // Manual reference
        let mut expected = [0.0f32; 16];
        for r in 0..4 {
            for cc in 0..4 {
                let mut s = 0.0;
                for i in 0..k {
                    s += a[r * k + i] * b[i * 4 + cc];
                }
                expected[r * 4 + cc] = s;
            }
        }
        for i in 0..16 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {i}: {} vs {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_4x4_dispatch_scalar_fallback() {
        // Stress the dispatch wrapper: should match scalar regardless of platform
        let k = 2;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![0.5, 0.25, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let mut c_dispatch = vec![0.0f32; 16];
        let mut c_scalar = vec![0.0f32; 16];
        matmul_4x4(&mut c_dispatch, &a, &b, k);
        matmul_4x4_scalar(&mut c_scalar, &a, &b, k);
        for i in 0..16 {
            assert!(
                (c_dispatch[i] - c_scalar[i]).abs() < 1e-5,
                "dispatch vs scalar mismatch at {i}"
            );
        }
    }
}
