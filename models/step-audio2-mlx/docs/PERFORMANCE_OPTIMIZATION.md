# Step-Audio 2 MLX Performance Optimization

## Overview

This document describes the performance optimizations implemented for Step-Audio 2 MLX on Apple Silicon. All optimizations are complete and the system achieves **real-time ASR capability**.

## Performance Summary

```
  STEP-AUDIO 2 MLX PERFORMANCE

  Audio Preprocessing:  320x faster (GPU vs CPU)
  End-to-End ASR:       1.6-1.8x real-time
  Status:               REAL-TIME CAPABLE
```

## Benchmark Results

Tested on Apple Silicon with Step-Audio 2 mini (7B parameters).

### Audio Processing (STFT + Mel Spectrogram)

| Metric | CPU | GPU (MLX) | Speedup |
|--------|-----|-----------|---------|
| 15s audio (mean) | 506 ms | 1.6 ms | **320x** |
| 15s audio (min) | 498 ms | 0.85 ms | **586x** |
| Real-time factor | 29.6x RT | 9,478x RT | - |

### End-to-End ASR Performance

| Audio | Duration | E2E Time | RTF | Speed | Status |
|-------|----------|----------|-----|-------|--------|
| English | 7.18s | 4.56s | 0.64x | **1.6x real-time** | Real-time capable |
| Chinese | 5.62s | 3.16s | 0.56x | **1.8x real-time** | Real-time capable |

### Component Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| Audio Preprocessing | **0.6 ms** | 0.0% |
| Encoder + LLM | 3.2-4.6s | 100% |

**Key Achievement**: Audio preprocessing is now negligible (0.0% of total time).

## Optimizations Implemented

### 1. GPU-Accelerated Audio Processing (320x speedup)

**Location**: `src/audio.rs`

Replaced CPU-bound audio processing with MLX GPU operations:

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| STFT | O(n^2) CPU DFT loops | `mlx_rs::fft::rfft()` | 300-600x |
| Mel Filterbank | CPU triple-nested loop | `mlx_rs::ops::matmul()` | 100-300x |
| Normalization | CPU loops | MLX element-wise ops | 50-100x |

### 2. MLX JIT Warmup (8% E2E improvement)

**Location**: `src/model.rs`

Added `warmup()` method to pre-compile MLX kernels:

```rust
// Call once before inference
model.warmup()?;
```

- JIT warmup time: ~2.6ms (dummy input)
- Eliminates first-call latency spike
- MLX caches compiled Metal kernels internally

### 3. Optimized Generation Loop

**Location**: `src/model.rs`

- Documented synchronization points
- MLX lazy evaluation batches operations automatically
- `item()` calls are the only sync points (for stopping conditions)
- KV cache updated incrementally

## Running Benchmarks

```bash
# Audio processing benchmark (STFT + Mel)
cargo run --release --example benchmark_audio_processing -- 15

# End-to-end ASR benchmark
cargo run --release --example benchmark_e2e -- ./audio.wav 5
```

## Future Optimizations

Potential improvements for even better performance:

1. **Quantization**: 4-bit/8-bit weights for reduced memory bandwidth
2. **Speculative Decoding**: Draft model for faster generation
3. **Continuous Batching**: Process multiple audio streams
4. **Float16 Inference**: Reduce memory and leverage ANE

## Hardware Notes

### Apple Silicon Optimization

MLX automatically leverages:
- **GPU**: Metal compute shaders for matrix ops
- **ANE**: Neural Engine for compatible operations
- **Unified Memory**: Zero-copy data sharing

### Memory Usage

- Model size: ~15GB (7B parameters, fp32)
- Peak memory: ~16GB during inference
- Fits on M1 Pro/Max/Ultra with 16GB+ RAM
