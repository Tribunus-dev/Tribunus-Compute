# oQ Dynamic Quantization — Port Plan

Source: `omlx/omlx/oq.py` (4.2K lines)
Status: Reference copied to `ref/omlx/oq.py`

## What it is

oQ (oMLX Universal Dynamic Quantization) is a load-time mixed-precision quantization
system that combines three approaches:

1. **GGUF K-quant layer position strategy** — which layers to quantize vs keep full precision
2. **Unsloth Dynamic 2.0 selective non-quantization** — skip quantizing sensitive weights
3. **Bits-and-Bytes MSE-optimal clipping** — find the optimal clip threshold per tensor

## Features

- Levels: oQ2, oQ2.5, oQ2.7, oQ3, oQ3.5, oQ4, oQ5, oQ6, oQ8
- Fractional levels add expert down_proj boost (Super Weights protection)
- Load-time application (modifies model weights before inference starts)
- Sensitivity model for adaptive quantization
- No calibration data needed (unlike GPTQ/AWQ)

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/quantization/oq.rs`

```rust
/// oQ quantization level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OqLevel {
    Oq2,
    Oq2_5,
    Oq2_7,
    Oq3,
    Oq3_5,
    Oq4,
    Oq5,
    Oq6,
    Oq8,
}

impl OqLevel {
    pub fn base_bits(&self) -> u32 { /* 2, 2, 2, 3, 3, 4, 5, 6, 8 */ }
    pub fn protection_level(&self) -> &str { /* KQUANT_2..KQUANT_6 */ }
}

/// Configuration for oQ quantization
pub struct OqConfig {
    pub level: OqLevel,
    pub group_size: usize,        // default 64
    pub dtype: OqDtype,           // bfloat16 or float16
    pub sensitivity_model: Option<String>,  // path to proxy model
}

/// Apply oQ quantization to model weights
pub fn apply_oq(
    weights: &mut HashMap<String, Tensor>,
    config: &OqConfig,
) -> Result<(), OqError> {
    // 1. Build layer quant plan from level + K-quant strategy
    // 2. Apply MSE-optimal clipping per group
    // 3. Quantize weights, store scale + zero-point
    // 4. Apply expert down_proj boost for fractional levels
}
```

### Key types

```rust
pub struct QuantizedTensor {
    pub data: Tensor,        // quantized data (packed bits)
    pub scale: Tensor,       // per-group scale factors
    pub zero_point: Tensor,  // per-group zero points (or None for symmetric)
    pub group_size: usize,
    pub orig_shape: Vec<usize>,
}

pub enum OqDtype { Bf16, F16 }
```

### Integration

- Called during model load in `compute-core/src/loading.rs`
- Applied before KV cache initialization
- Compatible with existing mlx-rs `QuantizedArray` type for on-device execution
