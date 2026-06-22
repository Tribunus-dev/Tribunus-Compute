//! ANE-based attention sink detector.
//!
//! Monitors attention weight entropy and predicts whether the adaptive
//! window needs to grow.  When running on the ANE, a tiny MLP predicts
//! window sufficiency from the last layer's attention weights.  A CPU-side
//! entropy heuristic provides the fallback path.
//!
//! The MLP (compiled offline as a Core ML model) accepts the last layer's
//! attention weight distribution [1, seq_len] and outputs a scalar in [0, 1]
//! where >0.5 means "window should grow".

use crate::arena::Arena;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};

/// Input feature name in the compiled Core ML model.
const INPUT_NAME: &str = "attention_weights";

/// Output feature name in the compiled Core ML model.
const OUTPUT_NAME: &str = "window_grow_prob";

/// ANE-based attention sink detector.
///
/// Runs a tiny MLP on ANE that predicts from attention weights whether
/// we are in a high-uncertainty region requiring a larger adaptive window.
pub struct AneSinkDetector {
    /// Core ML model (optional — falls back to CPU heuristic when None).
    model: Option<CoreMlModel>,
    /// Input arena for attention weight distribution.
    input_arena: Option<Arena>,
    /// Output arena for prediction result.
    output_arena: Option<Arena>,
    /// Maximum sequence length the detector supports.
    max_seq_len: u32,
    /// Number of predictions made.
    total_predictions: u64,
    /// Number of times the detector recommended growing the window.
    grow_recommendations: u64,
}

impl AneSinkDetector {
    /// Create a new ANE sink detector.
    ///
    /// When `model_path` is provided and the model loads successfully, ANE
    /// acceleration is used.  Otherwise the detector falls back to a CPU-side
    /// entropy heuristic.
    pub fn new(model_path: Option<&str>, max_seq_len: u32) -> Result<Self, String> {
        let (model, input_arena, output_arena) = if let Some(path) = model_path {
            match CoreMlModel::load_with_compute_units(path, CoreMlComputeUnits::CpuAndNeuralEngine)
            {
                Ok(m) => {
                    let inp = Arena::new(1, max_seq_len, mlx_rs::Dtype::Float16)?;
                    let out = Arena::new(1, 1, mlx_rs::Dtype::Float16)?;
                    (Some(m), Some(inp), Some(out))
                }
                Err(e) => {
                    eprintln!(
                        "[sink-detector] WARNING: failed to load ANE model at {}: {} \
                         (falling back to CPU entropy heuristic)",
                        path, e,
                    );
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };

        Ok(Self {
            model,
            input_arena,
            output_arena,
            max_seq_len,
            total_predictions: 0,
            grow_recommendations: 0,
        })
    }

    /// Check if the adaptive window should grow based on attention weights.
    ///
    /// `attention_weights` is a flat slice of f32 softmax probabilities
    /// (one head's distribution over the KV sequence).
    ///
    /// Returns `true` if the window should grow (high uncertainty / high entropy).
    pub fn check(&mut self, attention_weights: &[f32]) -> Result<bool, String> {
        self.total_predictions += 1;

        // Prefer ANE path when a model is loaded.
        if let Some(model) = self.model.take() {
            let result = self.check_ane(&model, attention_weights);
            self.model = Some(model);
            return result;
        }

        // CPU fallback: compute entropy heuristic.
        Ok(self.check_cpu(attention_weights))
    }

    /// Run the ANE Core ML model to predict window sufficiency.
    fn check_ane(
        &mut self,
        model: &CoreMlModel,
        attention_weights: &[f32],
    ) -> Result<bool, String> {
        let input_arena = self
            .input_arena
            .as_ref()
            .ok_or_else(|| "input arena not initialized".to_string())?;
        let output_arena = self
            .output_arena
            .as_mut()
            .ok_or_else(|| "output arena not initialized".to_string())?;

        let n = attention_weights.len().min(self.max_seq_len as usize);

        // Write FP16 attention weights into the input IOSurface.
        input_arena.lock()?;
        unsafe {
            let ptr = input_arena.base_ptr() as *mut u16;
            for i in 0..n {
                ptr.add(i).write(f32_to_f16(attention_weights[i]));
            }
            // Zero-fill remaining slots.
            for i in n..(self.max_seq_len as usize) {
                ptr.add(i).write(0u16);
            }
        }
        input_arena.unlock()?;

        // Run ANE prediction.
        let input_info = input_arena.info;
        let mut output_info = output_arena.info;
        model.predict_pixelbuffer(INPUT_NAME, &input_info, OUTPUT_NAME, &mut output_info)?;
        output_arena.info = output_info;

        // Read the scalar output: probability that window should grow.
        output_arena.lock()?;
        let grow_prob: f32;
        unsafe {
            let ptr = output_arena.base_ptr() as *const u16;
            grow_prob = f16_to_f32(ptr.read());
        }
        output_arena.unlock()?;

        let should_grow = grow_prob > 0.5;
        if should_grow {
            self.grow_recommendations += 1;
        }

        Ok(should_grow)
    }

    /// CPU-side entropy heuristic: compare distribution entropy to a threshold.
    fn check_cpu(&self, attention_weights: &[f32]) -> bool {
        let n = attention_weights.len();
        if n < 2 {
            return false;
        }

        let mut entropy = 0.0f32;
        for &p in attention_weights {
            if p > 0.0 {
                entropy -= p * p.log(std::f32::consts::E);
            }
        }

        // Normalize by max possible entropy (uniform distribution).
        let max_entropy = (n as f32).ln();
        let normalized = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Threshold: >0.8 means highly uncertain (near-uniform) → grow window.
        normalized > 0.8
    }

    /// Report detection statistics.
    pub fn report_stats(&self) -> String {
        format!(
            "AneSinkDetector: {}/{} grow recommendations ({:.1}%)",
            self.grow_recommendations,
            self.total_predictions,
            if self.total_predictions > 0 {
                (self.grow_recommendations as f64 / self.total_predictions as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

// ---------------------------------------------------------------------------
// FP16 conversion helpers (mirroring hot_row_predictor.rs)
// ---------------------------------------------------------------------------

/// Convert `f32` to packed `u16` in IEEE 754 fp16 format.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }

    let new_mant = mant >> 13;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
}

/// Convert packed IEEE 754 fp16 `u16` to `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let leading = mant.leading_zeros() - 21;
        let norm_exp = ((127 - 15 - leading as i32) as u32) << 23;
        let norm_mant = (mant << (leading + 1)) & 0x7F_FFFF;
        f32::from_bits((sign << 31) | norm_exp | norm_mant)
    } else if exp == 31 {
        let mant32 = if mant == 0 {
            0
        } else {
            (mant << 13) | 0x7F_FFFF
        };
        f32::from_bits((sign << 31) | 0x7F80_0000 | mant32)
    } else {
        let exp32 = (exp + (127 - 15)) << 23;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | exp32 | mant32)
    }
}
