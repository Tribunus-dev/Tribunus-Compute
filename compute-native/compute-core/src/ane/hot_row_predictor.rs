//! ANE-based next-token predictor for LM head weight prefetch.
//!
//! Uses a tiny 1-layer MLP on the Neural Engine that maps the current hidden
//! state to a short list of candidate token IDs.  The candidates' weight rows
//! in the LM head are then pre-loaded into ANE SRAM so the GPU reads them
//! from IOSurface-backed memory rather than main memory.
//!
//! # Architecture
//!
//! ```text
//!  GPU (decode layers)               ANE (predictor)
//!        │                               │
//!        ├── hidden state ──► IOSurface ──► 1-layer MLP ──► IOSurface ──► candidate IDs
//!        │                               │        (CpuAndNeuralEngine)
//!        └── prefetch rows ◄── predicted IDs ───┘
//! ```
//!
//! The MLP is compiled offline as a Core ML model (`.mlmodelc`).  It accepts
//! `[1, hidden_size]` FP16 input and produces `[num_candidates, 2]` FP16
//! output pairs `(token_id, confidence)`.  The input and output move through
//! IOSurface-backed [`Arena`] instances — zero CPU copy.

use crate::arena::Arena;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};

// ---------------------------------------------------------------------------
// Feature-name constants for Core ML binding I/O
// ---------------------------------------------------------------------------

/// Input feature name in the compiled predictor Core ML model.
const INPUT_NAME: &str = "hidden";

/// Output feature name in the compiled predictor Core ML model.
const OUTPUT_NAME: &str = "candidates";

// ---------------------------------------------------------------------------
// FP16 ↔ f32 conversion (IEEE 754, matching the independent helpers in
// `ane::draft_model` — kept local to avoid cross-module coupling.)
// ---------------------------------------------------------------------------

/// Convert `f32` to packed `u16` in IEEE 754 fp16 format.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // NaN or Inf — preserve sign, set fp16 exponent all-ones
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        // f32 subnormal / zero → flush to fp16 zero
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow → zero
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

// ---------------------------------------------------------------------------
// HotRowPredictor
// ---------------------------------------------------------------------------

/// Predicts the most likely next token(s) given the current hidden state.
///
/// Uses a tiny 1-layer MLP on ANE that maps the hidden state to a short list
/// of candidate token IDs.  The candidates' LM head rows are then pre-loaded
/// into ANE SRAM for zero-latency GPU access.
pub struct HotRowPredictor {
    /// ANE-based predictor Core ML model.
    model: CoreMlModel,
    /// Input arena: takes the hidden state as FP16 values, shaped `[1, hidden_size]`.
    input_arena: Arena,
    /// Output arena: returns `[num_candidates, 2]` FP16 pairs (token_id, confidence).
    output_arena: Arena,
    /// Number of candidate prediction slots.
    pub num_candidates: u32,
    /// Previous predictions for debug/statistics.
    pub last_prediction: Vec<u32>,
    /// Statistics: how often the next token was in the predicted set.
    pub prediction_hit_rate: f64,
    pub total_predictions: u64,
    pub hits: u64,
}

impl HotRowPredictor {
    /// Create a new predictor.
    ///
    /// The MLP model must be compiled as a Core ML package at `model_path`.
    /// It is loaded with `CpuAndNeuralEngine` so inference runs entirely on ANE.
    ///
    /// # Parameters
    ///
    /// * `model_path` — filesystem path to the compiled Core ML predictor
    ///   (`.mlmodelc` directory).
    /// * `hidden_size` — model's hidden state dimension (e.g. 3840).
    /// * `num_candidates` — number of candidate token IDs to predict (e.g. 64).
    pub fn new(model_path: &str, hidden_size: u32, num_candidates: u32) -> Result<Self, String> {
        let model = CoreMlModel::load_with_compute_units(
            model_path,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )?;

        // Input arena: one FP16 value per hidden dimension, shaped [1, hidden_size].
        let input_arena = Arena::new(1, hidden_size, mlx_rs::Dtype::Float16)?;

        // Output arena: [num_candidates, 2] — each slot is (token_id, confidence).
        let output_arena = Arena::new(num_candidates, 2, mlx_rs::Dtype::Float16)?;

        Ok(Self {
            model,
            input_arena,
            output_arena,
            num_candidates,
            last_prediction: Vec::new(),
            prediction_hit_rate: 0.0,
            total_predictions: 0,
            hits: 0,
        })
    }

    /// Write hidden state values into the input IOSurface arena as FP16.
    fn write_hidden_state(&self, hidden_state: &[f32]) -> Result<(), String> {
        let n = hidden_state.len();
        let arena_elements = self.input_arena.element_count();
        if n > arena_elements {
            return Err(format!(
                "HotRowPredictor: hidden state length {} exceeds arena capacity {}",
                n, arena_elements
            ));
        }

        self.input_arena.lock()?;
        unsafe {
            let ptr = self.input_arena.base_ptr() as *mut u16;
            for (i, &v) in hidden_state.iter().enumerate() {
                ptr.add(i).write(f32_to_f16(v));
            }
            // Zero-fill any remaining slots so stale data is not presented.
            for i in n..arena_elements {
                ptr.add(i).write(0u16);
            }
        }
        self.input_arena.unlock()?;
        Ok(())
    }

    /// Read candidate token IDs from the output IOSurface arena.
    ///
    /// Each candidate is a pair `(token_id, confidence)` stored as FP16 values.
    /// Tokens are returned sorted by confidence descending.
    fn read_candidates(&self) -> Result<Vec<u32>, String> {
        let n = self.num_candidates as usize;
        let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(n);

        self.output_arena.lock()?;
        unsafe {
            let ptr = self.output_arena.base_ptr() as *const u16;
            for i in 0..n {
                let token_id = f16_to_f32(ptr.add(i * 2).read()) as u32;
                let confidence = f16_to_f32(ptr.add(i * 2 + 1).read());
                pairs.push((token_id, confidence));
            }
        }
        self.output_arena.unlock()?;

        // Sort by confidence descending so the most likely token is first.
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter out zero/empty slots and deduplicate.
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::with_capacity(n);
        for (tid, _conf) in pairs {
            if tid != 0 && seen.insert(tid) {
                result.push(tid);
            }
        }

        Ok(result)
    }

    /// Given the current hidden state slice, predict the top candidate tokens.
    ///
    /// 1. Write hidden state to input IOSurface.
    /// 2. Run ANE prediction via Core ML.
    /// 3. Read candidate token IDs from output IOSurface.
    /// 4. Update statistics.
    pub fn predict(&mut self, hidden_state: &[f32]) -> Result<Vec<u32>, String> {
        // 1. Write hidden state to input IOSurface
        self.write_hidden_state(hidden_state)?;

        // 2. Run ANE prediction
        let input_info = self.input_arena.info;
        let mut output_info = self.output_arena.info;
        self.model
            .predict_pixelbuffer(INPUT_NAME, &input_info, OUTPUT_NAME, &mut output_info)?;
        self.output_arena.info = output_info;

        // 3. Read candidate token IDs from output
        let candidates = self.read_candidates()?;

        // 4. Update statistics (prediction_hit_rate is updated by the caller
        //    after it samples the actual next token)
        self.last_prediction = candidates.clone();
        self.total_predictions += 1;

        Ok(candidates)
    }

    /// Record whether the sampled next token was in the predicted set.
    pub fn record_outcome(&mut self, actual_token: u32) {
        if self.last_prediction.contains(&actual_token) {
            self.hits += 1;
        }
        let total = self.total_predictions.max(1);
        self.prediction_hit_rate = self.hits as f64 / total as f64;
    }

    /// Report hit rate statistics.
    pub fn report_hit_rate(&self) -> String {
        format!(
            "HotRowPredictor: {}/{} hits ({:.1}%)",
            self.hits,
            self.total_predictions,
            if self.total_predictions > 0 {
                self.prediction_hit_rate * 100.0
            } else {
                0.0
            }
        )
    }
}
