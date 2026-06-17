# TurboQuant KV Cache — Port Plan

Source: `omlx/omlx/turboquant_kv.py` (495 lines)
Status: Reference copied to `ref/omlx/turboquant_kv.py`

## What it is

A wrapper around mlx-vlm's TurboQuant for KV cache quantization that adds
continuous-batching batch support. Supported quantization types:

1. **Polar** — Sign-preserving polar quant (complex number decomposition)
2. **Prod** — Product quant (decompose into two lower-bit values)
3. **Split** — Split quant (separate by head dimension)
4. **PolarProd** — Combined polar + product for extreme compression
5. **MSEState** — MSE-optimal state selection per batch

## Key capabilities

- Quantizes KV cache to 2-4 bits instead of full precision (FP16/BF16)
- Batch-aware: different cache states per request in the batch
- Configurable codec selection per-layer
- MSE-optimal clipping for minimal perplexity impact
- Composability: multiple quant modes layered for extreme compression

## Rust Implementation Plan

### Location: `compute-native/compute-core/src/quantization/turboquant_kv.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KvQuantMode {
    Polar(u32),       // n_bits
    Prod(u32),        // n_bits
    Split(u32),       // n_bits
    PolarProd(u32),   // Total combined bits
    Mse { bits: u32, state_bits: u32 },
}

pub struct TurboQuantKvCache {
    quant_mode: KvQuantMode,
    codec: QuantCodec,
    state: QuantizedState,
}

pub struct QuantCodec {
    // Encoder/decoder for the chosen quant mode
    encode: fn(&[f32], &mut [u8]),
    decode: fn(&[u8], &mut [f32]),
}

pub struct QuantizedState {
    data: Vec<u8>,
    scale: Vec<f32>,
    zero_point: Option<Vec<f32>>,
    bits: u32,
    group_size: usize,
}
```
