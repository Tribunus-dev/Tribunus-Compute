//! Accelerate CPU execution lane — selected ops for CPU on Apple Silicon.
//!
//! Receives ArenaView references into the UnifiedExecutionArena.
//! No memory copies — operates on CPU-accessible arena pages directly.

/// Accelerate-based reduction sum.
pub fn sum_accelerate(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Perform RMSNorm using Accelerate-style scalar loop.
/// For aarch64 with NEON, the caller should prefer vDSP via the full
/// `accelerate` backend.  This is a safe fallback for CPU-accessible memory.
#[cfg(target_arch = "aarch64")]
pub fn rms_norm_accelerate(
    x_ptr: *mut f32,
    w_ptr: *const f32,
    out_ptr: *mut f32,
    dim: usize,
    eps: f32,
) {
    let sum_sq = unsafe {
        let mut sum = 0.0f32;
        for i in 0..dim {
            let v = *x_ptr.add(i);
            sum += v * v;
        }
        sum
    };
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    unsafe {
        for i in 0..dim {
            *out_ptr.add(i) = *x_ptr.add(i) * inv_rms * *w_ptr.add(i);
        }
    }
}

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

    /// Run RMSNorm via scalar loop.
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
        #[cfg(target_arch = "aarch64")]
        {
            rms_norm_accelerate(
                x.as_ptr() as *mut f32,
                weight.as_ptr(),
                out.as_mut_ptr(),
                x.len(),
                eps,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / x.len() as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            for (i, v) in x.iter().enumerate() {
                out[i] = v * inv_rms * weight[i];
            }
        }
        Ok(())
    }

    /// Run softmax over logits.
    pub fn softmax(&self, logits: &mut [f32]) -> Result<(), String> {
        let max_val = logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum <= 0.0 {
            return Err("softmax sum is zero — all-NaN or all -inf logits".into());
        }
        for v in logits.iter_mut() {
            *v /= sum;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_sum_accelerate() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((sum_accelerate(&data) - 15.0).abs() < 1e-6);
    }

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
}
