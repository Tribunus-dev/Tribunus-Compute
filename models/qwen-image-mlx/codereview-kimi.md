# Qwen-Image-MLX Critical Code Review & Performance Enhancement Plan

**Date**: 2026-01-29
**Reviewer**: Claude Code (kimi-k2.5)
**Scope**: Full codebase review with architecture analysis and performance roadmap

---

## Executive Summary

The `qwen-image-mlx` crate is a Rust implementation of the Qwen-Image-2512 text-to-image diffusion model using Apple's MLX framework. The codebase successfully implements the model with correct algorithms but requires significant cleanup and optimization before production deployment.

**Overall Assessment**: Functional but needs refinement
- **Architecture**: Sound
- **Code Quality**: Needs cleanup (debug artifacts, duplication)
- **Performance**: Good foundation, significant headroom for optimization
- **Safety**: No unsafe code, proper error handling

---

## 1. Architecture Overview

### 1.1 High-Level Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Text Prompt    │────▶│  QwenTextEncoder │────▶│  Quantized      │────▶│  QwenVAE    │────▶ RGB Image
│  (Tokenizer)    │     │  (28 layers GQA) │     │  Transformer    │     │  Decoder    │
└─────────────────┘     └──────────────────┘     │  (60 blocks)    │     └─────────────┘
                                                  │  Flow Matching  │
                                                  └─────────────────┘
```

### 1.2 Component Breakdown

| Component | Location | Purpose | Lines |
|-----------|----------|---------|-------|
| **Text Encoder** | `src/text_encoder.rs` | Qwen2.5-VL style 28-layer transformer with GQA | 578 |
| **Quantized Transformer** | `src/qwen_quantized.rs` | Main 60-block DiT with 4/8-bit quantization | 1096 |
| **Full-Precision Transformer** | `src/qwen_full_precision.rs` | FP32/BF16 variant for training | - |
| **VAE** | `src/vae/vae.rs` | 3D causal convolutional encoder/decoder | 316 |
| **Pipeline** | `src/pipeline.rs` | Flow-match Euler scheduler + generation loop | 268 |
| **Weight Loading** | `src/weights.rs` | SafeTensors loading with name mapping | 207 |

### 1.3 Model Specifications

**Text Encoder** (`src/text_encoder.rs:17-44`):
- Hidden Size: 3584
- Layers: 28
- Query Heads: 28, KV Heads: 4 (GQA)
- Head Dim: 128
- Vocab Size: 152064

**Diffusion Transformer** (`src/qwen_quantized.rs:18-45`):
- Layers: 60 transformer blocks
- Inner Dim: 3072 (24 heads × 128 head_dim)
- Patch Size: 2×2
- Quantization: 4-bit or 8-bit
- Joint image-text attention

**VAE** (`src/vae/vae.rs:30-33`):
- Base Channels: 96
- Stage Multipliers: [1, 1, 2, 4, 4]
- Latent Channels: 16
- Downsampling: 8×

---

## 2. Critical Issues

### 2.1 Code Duplication (HIGH PRIORITY)

**Problem**: Two parallel transformer implementations exist with significant duplication.

| Component | Quantized Location | Full-Precision Location |
|-----------|-------------------|------------------------|
| FeedForward | `src/qwen_quantized.rs:71-124` | `src/transformer/feedforward.rs` |
| Attention | `src/qwen_quantized.rs:126-336` | `src/transformer/attention.rs` |
| TransformerBlock | `src/qwen_quantized.rs:338-362` | `src/transformer/block.rs` |

**Impact**: Maintenance burden, risk of divergence between implementations.

**Recommendation**: Extract common traits or use generic types to share logic:
```rust
pub trait TransformerBlock {
    type Linear: LinearLayer;
    fn new(dim: i32, num_heads: i32) -> Self;
    fn forward(&mut self, x: &Array, text: &Array) -> Result<(Array, Array), Exception>;
}
```

### 2.2 Debug Code in Production (HIGH PRIORITY)

**Problem**: Multiple static atomic flags control debug printing throughout the codebase.

**Locations Found**:
- `src/qwen_quantized.rs:93-102` - DEBUG_FFN
- `src/qwen_quantized.rs:220-243` - DEBUG_BEFORE_NORM
- `src/qwen_quantized.rs:252-260` - DEBUG_NORM
- `src/qwen_quantized.rs:273-287` - DEBUG_QK
- `src/qwen_quantized.rs:386-394` - DEBUG_BLOCK_INPUT
- `src/qwen_quantized.rs:397-405` - DEBUG_TEMB
- `src/qwen_quantized.rs:412-419` - DEBUG_MOD
- `src/qwen_quantized.rs:432-439` - DEBUG_IMG_NORMED
- `src/qwen_quantized.rs:444-451` - DEBUG_IMG_MODULATED
- `src/qwen_quantized.rs:458-469` - DEBUG_GATE
- `src/qwen_quantized.rs:481-490` - DEBUG_ATTN
- `src/qwen_quantized.rs:555-610` - DEBUG_TS
- `src/qwen_quantized.rs:775-803` - DEBUG_IMG_IN, DEBUG_TXT_RAW, DEBUG_TXT_NORMED
- `src/qwen_quantized.rs:850-861` - DEBUG_BLOCK0
- `src/qwen_quantized.rs:865-886` - DEBUG_PRE_NORM, DEBUG_POST_NORM
- `src/qwen_quantized.rs:891-898` - DEBUG_FINAL
- `src/qwen_quantized.rs:941-951` - DEBUG_MODULATE

**Example**:
```rust
static DEBUG_FFN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
let debug_ffn = !DEBUG_FFN.swap(true, std::sync::atomic::Ordering::SeqCst);
if debug_ffn {
    mlx_rs::transforms::eval([x, &hidden]).ok();
    eprintln!("[DEBUG FFN] input: [{:.2}, {:.2}]...", ...);
}
```

**Impact**:
- Runtime overhead from atomic operations on hot paths
- Console noise in production builds
- Unprofessional output for end users

**Recommendation**:
1. Remove all debug printing infrastructure
2. Use proper logging crate (e.g., `tracing` or `log`) with compile-time filters
3. Gate debug code behind `#[cfg(debug_assertions)]` or feature flags

### 2.3 Dead Code and Unused Imports (MEDIUM PRIORITY)

**File**: `src/pipeline.rs:11-13`
```rust
use crate::transformer::QwenTransformer;  // UNUSED - only uses quantized version
use crate::vae::QwenVAE;
```

The `QwenImagePipeline` struct at `src/pipeline.rs:82-86` uses the full-precision `QwenTransformer`, but the example at `examples/generate_qwen_image.rs:200` only uses `QwenQuantizedTransformer`. The pipeline is essentially orphaned code.

**Recommendation**: Either integrate `QwenTransformer` properly or remove it and consolidate on the quantized version.

### 2.4 Hardcoded Magic Numbers (MEDIUM PRIORITY)

**File**: `examples/generate_qwen_image.rs:380`
```rust
let template = "<|im_start|>system\nDescribe the image...";
let drop_idx = 34;  // Assumes template is exactly 34 tokens
```

**File**: `examples/generate_qwen_image.rs:393`
```rust
let max_input_len = 77 + 34;  // Magic numbers without explanation
```

**Recommendation**: Define constants with documentation:
```rust
const TEMPLATE_TOKEN_COUNT: usize = 34;
const MAX_OUTPUT_TOKENS: usize = 77;
const MAX_INPUT_TOKENS: usize = MAX_OUTPUT_TOKENS + TEMPLATE_TOKEN_COUNT;
```

### 2.5 Disabled Code Blocks (MEDIUM PRIORITY)

**File**: `src/qwen_quantized.rs:221`
```rust
if false {  // Entire debug block disabled but present
    mlx_rs::transforms::eval([&img_q, &txt_q]).ok();
    // ... 20 lines of dead code
}
```

Multiple `if false` blocks exist throughout the codebase (lines 221, 253, 274, 398, etc.).

**Recommendation**: Remove or convert to proper feature flags:
```rust
#[cfg(feature = "debug-attention")]
{
    // Debug code here
}
```

### 2.6 Inefficient Weight Mapping (LOW PRIORITY)

**File**: `src/weights.rs:53-91`
```rust
impl TransformerWeightMapper {
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();
        name = name.replace("transformer_blocks.", "transformer_blocks.");  // No-op!
        name = name.replace("to_q.weight", "to_q.weight");  // No-op!
        // ... many no-op replacements
    }
}
```

The weight mapper performs many no-op string replacements.

**Recommendation**: Replace with actual mappings or remove no-ops:
```rust
static WEIGHT_MAPPINGS: &[(&str, &str)] = &[
    (".attn1.", ".attn."),
    ("to_out.0.", "attn_to_out."),
    ("ff.net.0.proj.", "mlp_in."),
    // ... actual mappings only
];
```

---

## 3. Architecture Strengths

### 3.1 Proper Quantization Support

The quantized transformer at `src/qwen_quantized.rs:708-733` correctly uses `QuantizedLinear` with configurable bits (4/8) and group size:

```rust
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenQuantizedTransformer {
    #[param]
    pub img_in: QuantizedLinear,
    #[param]
    pub transformer_blocks: Vec<QwenTransformerBlock>,
    // ... 60 transformer blocks
}
```

### 3.2 Correct VAE Latent Normalization

The VAE properly handles pre-computed normalization constants at `src/vae/vae.rs:18-28`:

```rust
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, ...
];
pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, ...
];
```

### 3.3 Flow Matching Implementation

The scheduler at `src/pipeline.rs:14-79` correctly implements flow matching with time shifting:

```rust
pub fn step(&self, model_output: &Array, timestep_idx: usize, sample: &Array)
    -> Result<Array, Exception> {
    let dt = self.sigmas[timestep_idx + 1] - self.sigmas[timestep_idx];
    ops::add(sample, &ops::multiply(model_output, &Array::from_f32(dt))?)
}
```

### 3.4 Batched CFG Support

The example at `examples/generate_qwen_image.rs:537-604` implements an optimization for classifier-free guidance that batches conditional and unconditional passes when possible, providing 2x speedup for the forward pass.

### 3.5 Correct 3D Convolution Implementation

The VAE causal convolution at `src/vae/conv3d.rs:62-87` properly handles the NCTHW ↔ NTHWC transpositions required by MLX's conv3d:

```rust
// Transpose from NCTHW to NTHWC
let input = padded.transpose_axes(&[0, 2, 3, 4, 1])?;
// Transpose weight from [out, in, kT, kH, kW] to [out, kT, kH, kW, in]
let weight = self.weight.transpose_axes(&[0, 2, 3, 4, 1])?;
```

---

## 4. Detailed Component Analysis

### 4.1 Text Encoder (`src/text_encoder.rs`)

**Architecture**: 28-layer transformer with Grouped Query Attention (GQA)

**Strengths**:
- Correct RoPE implementation at `src/text_encoder.rs:123-184`
- Proper causal masking with padding support at `src/text_encoder.rs:379-414`
- SwiGLU activation in MLP at `src/text_encoder.rs:289-313`

**Issues**:
- Custom `Linear` implementation at `src/text_encoder.rs:72-103` duplicates `mlx_rs::nn::Linear`
- `repeat_kv` at `src/text_encoder.rs:272-287` could use MLX's built-in broadcast

### 4.2 Quantized Transformer (`src/qwen_quantized.rs`)

**Architecture**: 60-block DiT (Diffusion Transformer) with joint image-text attention

**Block Structure** (per `src/qwen_quantized.rs:338-362`):
1. Image modulation linear (6x dim for shift/scale/gate)
2. Text modulation linear (6x dim for shift/scale/gate)
3. Joint attention (QKV for both streams, concatenated)
4. Image FFN (4x expansion)
5. Text FFN (4x expansion)

**Key Algorithm - Modulation** (`src/qwen_quantized.rs:933-959`):

...

...

...
*Review generated by Claude Code (kimi-k2.5) on 2026-01-29*
