# Qwen3-TTS-MLX

High-performance Qwen3-TTS inference on Apple Silicon in pure Rust, powered by [MLX](https://github.com/ml-explore/mlx).

Part of the [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) ecosystem.

## Highlights

- **~2.3x realtime** on Apple Silicon (M1/M2/M3/M4)
- **9 preset speakers** across 12 languages (CustomVoice model)
- **Voice cloning** from a short reference audio clip (Base model)
- **Streaming** audio output — start playback before generation finishes
- **Deterministic** generation with seed control
- **8-bit quantized** — 1.7B parameter model fits comfortably in unified memory
- **Zero Python dependencies** at inference time

## Quick Start

```rust
use qwen3_tts_mlx::{Synthesizer, SynthesizeOptions, save_wav, normalize_audio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let mut synth = Synthesizer::load("./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")?;

    // Synthesize with a preset speaker
    let opts = SynthesizeOptions {
        speaker: "vivian",
        language: "english",
        ..Default::default()
    };
    let samples = synth.synthesize("Hello! Welcome to Qwen3 TTS.", &opts)?;

    // Save to WAV (24kHz, 16-bit PCM, mono)
    let samples = normalize_audio(&samples, 0.95);
    save_wav(&samples, synth.sample_rate, "output.wav")?;
    Ok(())
}
```

Or from the command line:

```bash
cargo run --release --example synthesize -- \
    --model-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "Hello! Welcome to Qwen3 TTS." \
    --speaker vivian --language english \
    --output output.wav
```

## Building from Source

```bash
cargo build --release
cargo run --release --example synthesize -- --help
```
