# Step-Audio 2 MLX Development Plan

## Executive Summary

This plan outlines the implementation of Step-Audio 2 mini in MLX (Rust), leveraging existing components from the OminiX-MLX codebase. The implementation is structured in 4 phases, progressing from ASR-only to full speech-to-speech capabilities.

**Target**: Step-Audio 2 mini (8B parameters) with Think mode support

**Estimated Total Effort**: ~4,700 lines of new/adapted code
**Reuse Rate**: ~55% from existing codebase

---

## Existing Components to Leverage

### From `mlx-rs-core`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| KVCache | `src/cache.rs` | 100% | None |
| Sampler | `src/sampler.rs` | 100% | None |
| RoPE init | `src/utils.rs` | 100% | None |
| Attention masks | `src/utils.rs` | 100% | None |
| SDPA | `src/utils.rs` | 100% | None |
| Token generation | `src/generate/` | 100% | None |
| Audio I/O | `src/audio.rs` | 90% | Add 128-mel config |
| Mel spectrogram | `src/audio.rs` | 80% | Update for 128 mels |

### From `qwen3-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| Qwen2 Model | `src/qwen2.rs` | 85% | Scale dimensions, extend vocab |
| Qwen2 Attention | `src/qwen2.rs` | 90% | Verify bias settings |
| Qwen2 MLP | `src/qwen2.rs` | 100% | None |
| Weight loading | `src/qwen2.rs` | 80% | Update key mapping |
| Quantization | `src/qwen2.rs` | 100% | None |

### From `funasr-nano-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| WhisperEncoder | `src/whisper_encoder.rs` | 80% | Add AvgPool, update config |
| WhisperAttention | `src/whisper_encoder.rs` | 100% | None |
| WhisperMLP | `src/whisper_encoder.rs` | 100% | None |
| Model integration | `src/model.rs` | 60% | New adaptor, TTS path |
| Audio preprocessing | `src/audio.rs` | 70% | 128 mels |

### From `gpt-sovits-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| HiFi-GAN Generator | `src/models/vits.rs` | 70% | Extract, adapt upsampling |
| ResBlock | `src/models/vits.rs` | 100% | None |
| VQ Codebook | `src/models/vits.rs` | 60% | Different codebook size |

### From `flux-klein-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| FluxSampler | `src/sampler.rs` | 80% | Adapt for CosyVoice2 |
| Rectified flow | `src/sampler.rs` | 90% | Same algorithm |
| Timestep schedule | `src/sampler.rs` | 80% | Linear schedule |
