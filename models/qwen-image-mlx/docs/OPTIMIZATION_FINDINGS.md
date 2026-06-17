# Qwen-Image MLX Optimization Findings

*Last updated: 2026-01-29*

## Executive Summary

This document captures all optimization work done on the Qwen-Image MLX full-precision implementation, including successful optimizations, failed attempts, and critical lessons learned about MLX performance.

## Performance Summary

### Diffusion Time Only (excluding model loading)

| Steps | Diffusion Time | Per Step | Quality | vs Baseline |
|-------|----------------|----------|---------|-------------|
| 20 | ~80s | ~4.0s | Excellent | 1.8x faster |
| **10** | **~40s** | **~4.0s** | **Excellent** | **3.6x faster** |
| 8 | ~32s | ~4.0s | Very Good | 4.5x faster |

*Baseline: 145s for 20 steps before optimization*

### Total Time (including ~25s overhead for loading/encoding/VAE)

| Steps | Total Time | Quality | Recommendation |
|-------|------------|---------|----------------|
| 20 | ~105s | Excellent | Best quality |
| **10** | **~65s** | **Excellent** | **Recommended** |
| 8 | ~57s | Very Good | Fast mode |

*Note: Per-step time (~4s) is consistent. Thermal throttling can add variance.*

See [STEP_REDUCTION_RESULTS.md](./STEP_REDUCTION_RESULTS.md) for detailed analysis.

---

## Successful Optimizations

### 1. RoPE Pre-computation
**Impact: HIGH**

- Pre-compute rotary position embeddings once at initialization
- Eliminates redundant computation across all 60 transformer blocks x 20 steps
- Frequencies cached in `TimestepEmbedder.cached_freqs`

```rust
// In TimestepEmbedder::new()
let freqs: Vec<f32> = (0..half_dim)
    .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
    .collect();
let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);
```

### 2. Fast SDPA (Scaled Dot-Product Attention)
**Impact: HIGH**

- Use MLX's optimized `mlx_rs::fast::scaled_dot_product_attention`
- Hardware-accelerated with Flash Attention-style memory efficiency

```rust
let out = fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?;
```

### 3. Fast RMS Norm
**Impact: MEDIUM**

- Use MLX's `mlx_rs::fast::rms_norm` instead of manual implementation
- Fused kernel eliminates intermediate allocations

### 4. Lazy Evaluation Preservation
**Impact: CRITICAL**

The single most important optimization is **not breaking MLX's lazy evaluation**:
- **Do NOT call `eval()` on every step**
- **Do NOT call `mlx_clear_cache()` frequently**
- Only eval when necessary

---

## Failed/Rejected Optimizations

### 1. Fused Modulate Metal Kernel
**Status**: Working but slower than MLX ops

| Metric | Manual Implementation | Fused Kernel |
|--------|----------------------|--------------|
| Time per step | **4.31s** | 4.52s |

**Why it's slower**: MLX's built-in ops with lazy evaluation are already highly optimized.
The custom kernel adds overhead that outweighs the fusion benefit.

### 2. Cache Clearing Optimizations
**Status**: Rejected - adds overhead

| Attempted | Result |
|-----------|--------|
| `mlx_clear_cache()` every 5 steps | Slowed from 76s to 86s |
| `mlx_clear_cache()` every 15 blocks | Additional slowdown |
| `eval()` on every step | Slowed to ~110s |

### 3. Constant Caching with OnceLock
**Status**: Build failed - Array doesn't implement Sync.

### 4. Moving Constants Outside Loop
**Status**: Slower - creating Arrays inside the loop allows MLX to incorporate them into the computation graph for better fusion.

### 5. ANE (Apple Neural Engine) Offloading
**Status**: Not possible with MLX - MLX only supports CPU and GPU (Metal).

---

## Architecture Reference

### DiT (Diffusion Transformer) Value Ranges

| Stage | Value Range | Notes |
|-------|-------------|-------|
| After img_in | [-15, +16] | Normal |
| After 60 blocks | [-51M, +57M] | Expected explosion |
| After norm_out | [-16, +14] | LayerNorm normalizes |
| After proj_out | [-4.4, +4.4] | Final output |

### Modulation Formula

```
output = (1 + scale) * LayerNorm(x) + shift
```

Where:
- `LayerNorm` has no learnable parameters (`elementwise_affine=False`)
- `scale` and `shift` come from timestep embedding projection
- Called 4x per block x 60 blocks = 240 times per forward pass

### Timestep Convention

- Pass sigma (in [0, 1] range) to transformer
- `get_timestep_embedding` internally scales by 1000
- Identical to mflux and diffusers conventions

---

## Memory Usage

| Component | Memory |
|-----------|--------|
| Full precision model | ~13GB |
| Peak during generation | ~15-16GB |
| After text encoder release | ~12-13GB |

**Tip**: Release text encoder after encoding to save ~2-3GB:
```rust
drop(text_encoder);
```

---

## Benchmarking Commands

```bash
# Full precision, 20 steps, 512x512
cargo run --release --example generate_fp32 -- \
  --prompt "a fluffy cat" \
  --height 512 --width 512 \
  --steps 20 \
  --output output.png

# Quick test (3 steps)
cargo run --release --example generate_fp32 -- \
  --prompt "a fluffy cat" \
  --height 512 --width 512 \
  --steps 3 \
  --output output_test.png
```
