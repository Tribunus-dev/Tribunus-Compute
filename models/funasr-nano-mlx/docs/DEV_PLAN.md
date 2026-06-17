# Development Plan: funasr-nano-mlx

**Created**: 2026-01-25
**Target**: Production-ready ASR library for Apple Silicon

---

## Goals

1. **Performance**: Achieve 30x+ real-time transcription (currently 16x)
2. **Features**: Streaming API, batch processing, advanced sampling
3. **Quality**: Comprehensive error handling, tests, documentation
4. **Reliability**: Handle edge cases, validate inputs

---

## Lessons from funasr-mlx (Paraformer Implementation)

The funasr-mlx crate provides a well-structured reference implementation. Key patterns to adopt:

### 1. FFT with Cached Planner (45x speedup)

**funasr-mlx/src/paraformer.rs:155-222**
```rust
use rustfft::{num_complex::Complex, FftPlanner};

pub struct MelFrontend {
    // Cached FFT instance for efficient repeated STFT computation
    fft: Arc<dyn rustfft::Fft<f32>>,
    mel_filters: Vec<f32>,  // Pre-computed filterbank
    window: Vec<f32>,       // Pre-computed Hamming window
}

impl MelFrontend {
    pub fn new(config: &Config) -> Self {
        // Pre-create FFT planner for efficient repeated use
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        // ...
    }
}
```

### 2. Error Handling with thiserror

**funasr-mlx/src/error.rs**
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),  // Auto-convert from MLX errors

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),
}
```

### 3. Clean Configuration Struct

**funasr-mlx/src/paraformer.rs:55-108**
```rust
#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    // Audio frontend
    pub sample_rate: i32,
    pub n_mels: i32,
    pub n_fft: i32,
    pub hop_length: i32,
    pub lfr_m: i32,
    pub lfr_n: i32,

    // Encoder
    pub encoder_dim: i32,
    pub encoder_layers: i32,
    // ...
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,      // 25ms window - documented!
            hop_length: 160, // 10ms hop
            // ...
        }
    }
}
```

### 4. Weight Loading Helpers

**funasr-mlx/src/paraformer.rs:1286-1298**
```rust
fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Missing weight: {}", key)))
}

fn get_conv_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    let weight = get_weight(weights, key)?;
    weight.transpose_axes(&[0, 2, 1])
        .map_err(|e| Error::Model(format!("Failed to transpose: {}", e)))
}
```

### 5. High-Level API

**funasr-mlx/src/lib.rs:121-138**
```rust
/// High-level transcription function
pub fn transcribe(
    model: &mut Paraformer,
    audio: &[f32],
    vocab: &Vocabulary,
) -> Result<String> {
    let audio_array = Array::from_slice(audio, &[audio.len() as i32]);
    let token_ids = model.transcribe(&audio_array)?;
    eval([&token_ids])?;

    let token_ids_vec: Vec<i32> = token_ids.try_as_slice::<i32>()?.to_vec();
    Ok(vocab.decode(&token_ids_vec))
}
```

### 6. Vocabulary with Special Token Filtering

**funasr-mlx/src/lib.rs:62-110**
```rust
pub struct Vocabulary {
    tokens: Vec<String>,
}

impl Vocabulary {
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids.iter()
            .filter_map(|&id| {
                let token = &self.tokens[id as usize];
                // Filter special tokens
                if token == "<blank>" || token == "<s>" || token == "</s>" {
                    None
                } else {
                    Some(token.clone())
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
```

### 7. Architecture Documentation in Code

**funasr-mlx/src/paraformer.rs:1-28**
```rust
//! # Architecture
//!
//! ```text
//! Audio (16kHz)
//!     |
//! [Mel Frontend] - 80 bins, 25ms window, 10ms hop, LFR 7/6
//!     |
//! [SAN-M Encoder] - 50 layers, 512 hidden, 4 heads
//!     |
//! [CIF Predictor] - Continuous Integrate-and-Fire
//!     |
//! [Bidirectional Decoder] - 16 layers, 512 hidden, 4 heads
//!     |
//! Tokens [batch, num_tokens]
//! ```
```

---

## Milestones

### Milestone 1: Performance Optimization (P0)
**Target**: 2x speedup (16x -> 32x real-time)

### Milestone 2: Core Features (P1)
**Target**: Streaming & batch APIs

### Milestone 3: Robustness (P2)
**Target**: Production-ready error handling & tests

### Milestone 4: Polish (P3)
**Target**: Documentation & minor features

---

## Phase 1: Performance (P0)

### Task 1.1: Replace DFT with FFT
**Priority**: P0 | **Effort**: Medium | **Impact**: HIGH (10-100x audio processing speedup)

```rust
// Add to Cargo.toml:
rustfft = "6.2"

// Replace audio.rs:154-163 with:
use rustfft::{FftPlanner, num_complex::Complex};

fn compute_fft(frame: &[f32], n_fft: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    // ... FFT implementation
}
```

**Files**: `src/audio.rs`
**Tests**: Verify mel spectrogram output matches current implementation

---

### Task 1.2: Use Optimized SDPA in Encoder
**Priority**: P0 | **Effort**: Low | **Impact**: HIGH (2-3x encoder speedup)

```rust
// Replace sensevoice_encoder.rs:232-240 with:
let attn_out = mlx_rs::fast::scaled_dot_product_attention(
    q, k, v, self.scale, None::<ScaledDotProductAttentionMask>
)?;
```

**Files**: `src/sensevoice_encoder.rs`
**Tests**: Compare encoder output before/after

---

### Task 1.3: Reduce Array Cloning
**Priority**: P0 | **Effort**: Medium | **Impact**: MEDIUM (memory reduction)

```rust
// Before (qwen.rs:263):
let residual = x.clone();
let h = self.input_layernorm.forward(&x)?;

// After:
let h = self.input_layernorm.forward(x)?;
let h = self.self_attn.forward_with_cache(&h, cache, mask)?;
x.add(&h)?  // x is still valid
```

**Files**: `src/qwen.rs`, `src/sensevoice_encoder.rs`, `src/adaptor.rs`
**Tests**: Memory profiling before/after

---

### Task 1.4: GPU-Accelerated LFR
**Priority**: P0 | **Effort**: Medium | **Impact**: MEDIUM

```rust
// Replace CPU-based LFR (audio.rs:241-307) with MLX operations:
pub fn apply_lfr_gpu(mel: &Array, lfr_m: i32, lfr_n: i32) -> Result<Array> {
    // Use MLX reshape, pad, concatenate instead of CPU Vec operations
}
```

**Files**: `src/audio.rs`

---

## Phase 2: Core Features (P1)

### Task 2.1: Streaming Transcription API
**Priority**: P1 | **Effort**: High | **Impact**: HIGH

```rust
// New API in model.rs:

/// Streaming context holding encoder state and partial results
pub struct StreamingContext {
    encoder_state: Option<Array>,
    cache: Vec<Option<KVCache>>,
    pending_audio: Vec<f32>,
    partial_text: String,
}

impl FunASRNano {
    /// Create a new streaming context
    pub fn create_streaming_context(&self) -> StreamingContext;

    /// Process an audio chunk (16kHz f32 samples)
    /// Returns partial transcription if available
    pub fn transcribe_chunk(
        &mut self,
        ctx: &mut StreamingContext,
        chunk: &[f32]
    ) -> Result<Option<String>>;

    /// Finalize streaming and return complete transcription
    pub fn finalize_stream(&mut self, ctx: StreamingContext) -> Result<String>;
}

...
