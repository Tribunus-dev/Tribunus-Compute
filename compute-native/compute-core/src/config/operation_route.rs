//! Per-operation backend routing for decoder layers.
//!
//! Each operation group in a decoder layer is assigned to the optimal
//! execution backend during compilation.  This is the compiler's single
//! source of truth for heterogeneous dispatch — the runtime reads these
//! assignments to route each operation to the correct backend via the
//! IOSurface unified memory island.
//!
//! Backend IDs:
//!   0 = MLX (GPU matmul, attention, softmax, default)
//!   1 = Accelerate (element-wise: RMS norm, SiLU, transpose, add, reshape)
//!   2 = Core ML (stateful ANE islands)
//!   3 = Orion/ANE private runtime (dense attention, decoder layers)

use serde::{Deserialize, Serialize};

/// Per-operation backend routing for a single decoder layer.
///
/// Default for all fields is 0 (MLX) — always safe, always correct.
/// During compilation the BackendAssessmentPass sets each field to the
/// optimal backend based on operation family and hardware capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRoute {
    /// Layer-input RMS normalization.
    /// Accelerate(1) ideal — `vDSP_vsma` + `vvfrsqrtf`.
    #[serde(default)]
    pub rms_norm: u32,

    /// SiLU activation: `x * sigmoid(x)`.
    /// Accelerate(1) ideal — `vDSP_vsigmoid` + `vDSP_vmul`.
    #[serde(default)]
    pub silu: u32,

    /// Matrix multiplication (QKV projections, MLP gate/up/down).
    /// MLX(0) ideal for GPU — `cblas_sgemm` via Accelerate(1) is fallback.
    #[serde(default)]
    pub matmul: u32,

    /// Scaled dot-product attention.
    /// Orion(3) ideal for full/global attention — ANE's matrix engine.
    /// MLX(0) for sliding/local attention (irregular access patterns).
    #[serde(default)]
    pub attention: u32,

    /// Softmax on attention scores.
    /// MLX(0) ideal — Accelerate(1) possible via `vDSP_vmax` + `vvfexpf`.
    #[serde(default)]
    pub softmax: u32,

    /// Rotary Position Embedding.
    /// MLX(0) ideal — Accelerate(1) possible via `vDSP_zvmul`.
    #[serde(default)]
    pub rope: u32,

    /// Element-wise addition (residual connections, bias add).
    /// Accelerate(1) ideal — `vDSP_vadd`.
    #[serde(default)]
    pub add: u32,

    /// Element-wise multiplication (gating, scaling).
    /// Accelerate(1) ideal — `vDSP_vmul`.
    #[serde(default)]
    pub multiply: u32,

    /// Matrix transpose.
    /// Accelerate(1) ideal — `vDSP_mtrans`.
    #[serde(default)]
    pub transpose: u32,

    /// Tensor reshape (metadata-only, no data movement).
    /// Accelerate(1) ideal — Accelerate backend has the fastest reshape
    /// (it's a no-op in the storage layer).
    #[serde(default)]
    pub reshape: u32,
}

impl OperationRoute {
    /// True if any operation group is routed to the ANE backend.
    pub fn has_ane_backend(&self) -> bool {
        const ANE: u32 = 3;
        self.rms_norm == ANE
            || self.silu == ANE
            || self.matmul == ANE
            || self.attention == ANE
            || self.softmax == ANE
            || self.rope == ANE
            || self.add == ANE
            || self.multiply == ANE
            || self.transpose == ANE
            || self.reshape == ANE
    }

    /// Return the dominant (most-frequently-assigned) backend across all
    /// operation groups.  Used as the fast-dispatch hint at runtime.
    pub fn dominant_backend(&self) -> u32 {
        let counts = [
            self.rms_norm,
            self.silu,
            self.matmul,
            self.attention,
            self.softmax,
            self.rope,
            self.add,
            self.multiply,
            self.transpose,
            self.reshape,
        ];
        let mut freq = [0u32; 4]; // backends 0..3
        for &b in &counts {
            if b < 4 {
                freq[b as usize] += 1;
            }
        }
        freq.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Override the route so the given backend is dominant.
    ///
    /// Sets matrix-heavy operations (matmul, attention, softmax, rope, silu,
    /// transpose) to the target backend so `dominant_backend()` returns it.
    /// Element-wise ops (rms_norm, add, multiply, reshape) are left on
    /// their optimal backend (Accelerate) unless the target is Orion/ANE
    /// private runtime (backend 3) which sets all operations.
    pub fn set_dominant_backend(&mut self, backend_id: u32) {
        match backend_id {
            0 => {
                // MLX (GPU): matrix ops work best on GPU
                self.matmul = 0;
                self.attention = 0;
                self.softmax = 0;
                self.rope = 0;
                self.silu = 0;
                self.transpose = 0;
                // rms_norm=1, add=1, multiply=1, reshape=1 preserved (Accelerate)
            }
            1 => {
                // Accelerate: element-wise
                self.rms_norm = 1;
                self.add = 1;
                self.multiply = 1;
                self.reshape = 1;
            }
            2 => {
                // Core ML / ANE: matrix ops for ANE's high-throughput matmul
                self.matmul = 2;
                self.attention = 2;
                self.softmax = 2;
                self.rope = 2;
                self.silu = 2;
                self.transpose = 2;
                // rms_norm=1, add=1, multiply=1, reshape=1 preserved (Accelerate)
            }
            3 => {
                // Orion/ANE private runtime: all operations
                self.rms_norm = 3;
                self.silu = 3;
                self.matmul = 3;
                self.attention = 3;
                self.softmax = 3;
                self.rope = 3;
                self.add = 3;
                self.multiply = 3;
                self.transpose = 3;
                self.reshape = 3;
            }
            _ => {}
        }
    }
}

impl Default for OperationRoute {
    /// Evidence-based defaults from backend_benchmark results on M1:
    /// MLX wins: matmul (>64), SiLU (>64), softmax, transpose (>64)
    /// Accelerate wins: element-wise (add, mul), RMS norm, transpose (≤64)
    fn default() -> Self {
        Self {
            rms_norm: 1,  // Accelerate: vDSP_vsma + vvfrsqrtf (wins at 64-1024)
            silu: 0,      // MLX: GPU sigmoid (MLX 3.7μs vs Accel 24μs at 1K)
            matmul: 0,    // MLX: GPU matmul (MLX 4μs vs Accel 2.6ms at 4K)
            attention: 0, // MLX: GPU attention (overridden for full_attention→Orion)
            softmax: 0,   // MLX: GPU softmax (1.6μs vs Accel 138μs at 4K)
            rope: 0,      // MLX: GPU RoPE (sin/cos table faster on GPU)
            add: 1,       // Accelerate: vDSP_vadd (wins at 64-1024)
            multiply: 1,  // Accelerate: vDSP_vmul (wins at 64-1024)
            transpose: 0, // MLX: GPU transpose (MLX 2.1μs vs Accel 971μs at 1024)
            reshape: 1,   // Accelerate: no-op in storage layer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominant_backend_defaults() {
        let route = OperationRoute::default();
        // Counts: MLX=6 (silu,matmul,attention,softmax,rope,transpose) vs Accel=4 (rms_norm,add,mul,reshape)
        assert_eq!(
            route.dominant_backend(),
            0,
            "MLX should be dominant in default route (6 vs 4)"
        );
    }

    #[test]
    fn test_dominant_backend_orion_preferred() {
        let route = OperationRoute {
            attention: 3, // Orion
            matmul: 3,    // Orion
            ..Default::default()
        };
        // Orion 3: attention=3, matmul=3 (2)
        // Accelerate 1: rms_norm, silu, add, mul, transpose, reshape (6)
        assert_eq!(
            route.dominant_backend(),
            1,
            "Accelerate still dominant by count"
        );
    }

    #[test]
    fn test_serde_round_trip() {
        let route = OperationRoute {
            attention: 3,
            matmul: 0,
            ..Default::default()
        };
        let json = serde_json::to_string(&route).unwrap();
        let back: OperationRoute = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attention, 3);
        assert_eq!(back.matmul, 0);
        assert_eq!(back.rms_norm, 1); // from default
    }
}
