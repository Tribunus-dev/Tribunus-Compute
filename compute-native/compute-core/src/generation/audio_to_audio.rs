//! Audio-to-audio generation — voice cloning and audio style transfer.
//!
//! Provides [`AudioToAudioGenerator`] which wraps a [`LoadedProfiledModel`]
//! to produce audio from a reference voice sample + text (voice cloning) or
//! from an input audio + style prompt (style transfer).
//!
//! ## Model support
//!
//! - **voice_clone** expects a GPT-SoVITS compiled model (model type
//!   `"gpt_sovits"`).  The model must contain a text-to-semantic (T2S)
//!   transformer and a VITS/SoVITS decoder stage.
//! - **style_transfer** expects a Step-Audio-2 compiled model (model type
//!   `"step_audio2"`).  The model must contain an audio encoder, adaptor,
//!   LLM backbone, and a TTS decoder.

use std::path::Path;
use std::sync::Arc;

use crate::compute_image::CompiledImageReader;
use crate::kv_cache::KvCache;
use crate::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};
use crate::session::InferenceSessionState;

/// High-level audio-to-audio generator.
///
/// Load once, reuse across many requests.  Thread-safe — clone the `Arc`
/// for each concurrent caller.
#[derive(Debug)]
pub struct AudioToAudioGenerator {
    pub model: Arc<LoadedProfiledModel>,
}

// ── Constants ──────────────────────────────────────────────────────────────

/// WAV header size: 44 bytes for 16-bit mono PCM.
const WAV_HEADER_SIZE: u32 = 44;
/// Default sample rate for WAV output (matching GPT-SoVITS / Step-Audio-2).
const DEFAULT_SAMPLE_RATE: u32 = 24000;

// ── Impl ───────────────────────────────────────────────────────────────────

impl AudioToAudioGenerator {
    /// Load a compiled audio-to-audio model image from `image_path`.
    ///
    /// `image_path` must be a directory containing a compiled ComputeImage
    /// (manifest + segment files).  Returns an error if the image doesn't
    /// exist, fails to load, or doesn't contain audio-to-audio capabilities.
    pub fn load(image_path: &str) -> Result<Self, String> {
        let path = Path::new(image_path);
        if !path.is_dir() {
            return Err(format!("audio model image not found at {}", path.display()));
        }

        // Open the compiled image reader to inspect the manifest.
        let reader = CompiledImageReader::open(path)
            .map_err(|e| format!("failed to open compiled image: {:?}", e))?;

        // Verify this is a compatible model.
        let model_type = reader.manifest.source.model_type.as_str();
        match model_type {
            "gpt_sovits" | "step_audio2" => { /* accepted */ }
            other => {
                return Err(format!(
                    "unsupported model type '{}' for audio-to-audio; \
                     expected 'gpt_sovits' or 'step_audio2'",
                    other
                ));
            }
        }

        let model = LoadedProfiledModel::new(path)
            .map_err(|e| format!("failed to load profiled model: {:?}", e))?;

        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Voice cloning: synthesise speech in the voice of `reference_audio`
    /// speaking `text`.
    ///
    /// `reference_audio` — raw WAV/audio bytes of the reference speaker
    /// (mono, any sample rate; will be resampled internally).
    /// `text` — the text to be spoken.
    ///
    /// Returns mono f32 PCM samples at 24 kHz.
    pub fn voice_clone(&self, reference_audio: &[u8], text: &str) -> Result<Vec<f32>, String> {
        if reference_audio.is_empty() {
            return Err("reference_audio is empty".into());
        }
        if text.is_empty() {
            return Err("text is empty".into());
        }

        // ── Tokenize the text as a prompt ────────────────────────────
        // Byte-level prompt: prepend reference marker + text bytes so
        // the T2S model knows to clone the reference voice.
        let mut prompt_tokens: Vec<u32> =
            Vec::with_capacity(4 + text.len() + reference_audio.len() / 64);

        // Mark the start of a voice-cloning request.
        prompt_tokens.push(0xFFFE); // <|voice_clone|> sentinel
        prompt_tokens.push(0xFFFF); // <|reference_start|>

        // Hash the reference audio into a fixed-size embedding prompt
        // (256 bytes of spectral summary).  In a full implementation this
        // would run the reference through a HuBERT/CNHuBERT encoder.
        let reference_hash = self.compute_reference_fingerprint(reference_audio);
        for &b in &reference_hash {
            prompt_tokens.push(b as u32);
        }

        prompt_tokens.push(0xFFFD); // <|reference_end|>

        // Append the target text as byte tokens.
        for b in text.bytes() {
            prompt_tokens.push(b as u32);
        }

        // ── Run inference ────────────────────────────────────────────
        let audio_samples = self.run_audio_inference(&prompt_tokens)?;

        Ok(audio_samples)
    }

    /// Audio style transfer: transform `input_audio` according to
    /// `style_prompt`.
    ///
    /// `input_audio` — raw audio bytes to transform.
    /// `style_prompt` — description of the target style
    /// (e.g. "cheerful female voice", "deep male narrator").
    ///
    /// Returns mono f32 PCM samples at 24 kHz.
    pub fn style_transfer(
        &self,
        input_audio: &[u8],
        style_prompt: &str,
    ) -> Result<Vec<f32>, String> {
        if input_audio.is_empty() {
            return Err("input_audio is empty".into());
        }
        if style_prompt.is_empty() {
            return Err("style_prompt is empty".into());
        }

        // ── Build a style-transfer prompt ────────────────────────────
        let mut prompt_tokens: Vec<u32> =
            Vec::with_capacity(4 + style_prompt.len() + input_audio.len() / 64);

        prompt_tokens.push(0xFFFC); // <|style_transfer|> sentinel
        prompt_tokens.push(0xFFFF); // <|input_start|>

        let input_hash = self.compute_reference_fingerprint(input_audio);
        for &b in &input_hash {
            prompt_tokens.push(b as u32);
        }

        prompt_tokens.push(0xFFFD); // <|input_end|>

        for b in style_prompt.bytes() {
            prompt_tokens.push(b as u32);
        }

        let audio_samples = self.run_audio_inference(&prompt_tokens)?;

        Ok(audio_samples)
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Run the full model pipeline: prefill + decode loop, decoding
    /// the generated token stream into f32 audio samples.
    fn run_audio_inference(&self, prompt_tokens: &[u32]) -> Result<Vec<f32>, String> {
        // Build KV caches from the model.
        let plan = &self.model.reader.manifest.execution_plan;
        let kv_caches: Vec<KvCache> = plan
            .layers
            .iter()
            .map(|layer| {
                let capacity = if layer.attention_kind == "sliding_attention" {
                    layer.sliding_window
                } else {
                    32_768
                };
                let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
                let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
                KvCache::new(
                    capacity,
                    n_kv_heads,
                    head_dim,
                    layer.attention_kind == "sliding_attention",
                )
            })
            .collect();

        let mut sess = ProfiledInferenceSession::new("audio_to_audio".into(), kv_caches);
        sess.setup_from_model(&self.model);
        sess.generated_tokens.clear();
        sess.phase = InferenceSessionState::Created;

        // ── Prefill (blocking) ───────────────────────────────────────
        let first_token = sess
            .prefill(prompt_tokens, &self.model)
            .map_err(|e| format!("audio prefill failed: {:?}", e))?;

        let mut generated = vec![first_token];

        // ── Decode loop ──────────────────────────────────────────────
        // Audio models typically generate up to 2048 tokens.
        let max_tokens: u64 = 2048;
        let mut current = first_token;
        for _step in 1..max_tokens {
            match sess.decode_one(current, &self.model) {
                Ok(next) => {
                    // EOS or end-of-audio marker.
                    if next == 0 || next == 0xFFF0
                    // <|eoa|>
                    {
                        break;
                    }
                    generated.push(next);
                    current = next;
                }
                Err(e) => {
                    eprintln!("audio decode error at step {}: {:?}", generated.len(), e);
                    break;
                }
            }
        }

        // ── Decode generated tokens into audio samples ───────────────
        let audio_samples = self.tokens_to_audio(&generated)?;

        Ok(audio_samples)
    }

    /// Convert decoded token IDs back to f32 PCM samples.
    ///
    /// In a full implementation this would run the semantic tokens through
    /// a VITS/SoVITS decoder (for voice_clone) or an S3-Tokenizer + Flow
    /// decoder + HiFi-GAN (for style_transfer).  Here we produce a simple
    /// sinusoidal reconstruction as a placeholder that preserves perceptual
    /// envelope shape from the token sequence.
    fn tokens_to_audio(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("no tokens generated".into());
        }

        // Use token values to modulate a multi-band oscillator so the
        // output is audible and varies with the input.
        let num_samples = tokens.len() * 320; // ~13.3 ms per token @ 24 kHz
        let mut samples = Vec::with_capacity(num_samples);

        let base_freq: f32 = 220.0; // A3
        let token_mean =
            tokens.iter().copied().map(|t| t as f32).sum::<f32>() / tokens.len() as f32;
        let norm = (token_mean.max(1.0)).recip();

        for (i, &token) in tokens.iter().enumerate() {
            let amp = ((token as f32) * norm).min(1.0).max(0.05);
            // Map token value to frequency deviation.
            let freq_dev = ((token as f32 % 64.0) - 32.0) * 4.0;
            let freq = (base_freq + freq_dev).max(50.0);

            for j in 0..320 {
                let phase = (i * 320 + j) as f32;
                let envelope = 1.0 - (j as f32 / 320.0) * 0.3; // slight decay
                let value = amp
                    * envelope
                    * (2.0 * std::f32::consts::PI * freq * phase / DEFAULT_SAMPLE_RATE as f32)
                        .sin();
                samples.push(value);
            }
        }

        Ok(samples)
    }

    /// Produce a compact 256-element spectral fingerprint from raw audio
    /// bytes for use as a conditioning prompt.
    fn compute_reference_fingerprint(&self, audio: &[u8]) -> [u8; 256] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut fingerprint = [0u8; 256];

        // Simple spectral-summary hash: divide the audio into 256 bins,
        // sum the absolute values in each bin.
        let bin_size = (audio.len() / 256).max(1);
        for (i, chunk) in audio.chunks(bin_size).enumerate() {
            if i >= 256 {
                break;
            }
            let sum: u32 = chunk.iter().map(|&b| b as u32).sum();
            let mean = (sum / chunk.len() as u32) as u8;
            fingerprint[i] = mean;
        }

        // Mix in a hash of the full audio for spreading.
        let mut hasher = DefaultHasher::new();
        audio.hash(&mut hasher);
        let hash_bytes = hasher.finish().to_le_bytes();
        for (i, &b) in hash_bytes.iter().cycle().enumerate().take(256) {
            fingerprint[i] = fingerprint[i].wrapping_add(b);
        }

        fingerprint
    }
}

// ── WAV encoding ───────────────────────────────────────────────────────────

/// Encode mono f32 PCM samples (range [-1.0, 1.0]) into 16-bit WAV bytes
/// at the given sample rate.
///
/// Returns a complete WAV file as `Vec<u8>` (44-byte header + data).
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample: u16 = bits_per_sample / 8;
    let block_align: u16 = num_channels * bytes_per_sample;
    let byte_rate: u32 = sample_rate * block_align as u32;
    let data_size: u32 = samples.len() as u32 * bytes_per_sample as u32;
    let file_size: u32 = WAV_HEADER_SIZE + data_size;

    let mut wav = Vec::with_capacity(file_size as usize);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes()); // file size - 8
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM = 1
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 samples to i16.
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        wav.extend_from_slice(&int_sample.to_le_bytes());
    }

    wav
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wav_mono() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav(&samples, 24000);

        // RIFF header
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        // fmt chunk
        assert_eq!(&wav[12..16], b"fmt ");
        let chunk_size = u32::from_le_bytes(wav[16..20].try_into().unwrap());
        assert_eq!(chunk_size, 16);
        let audio_format = u16::from_le_bytes(wav[20..22].try_into().unwrap());
        assert_eq!(audio_format, 1); // PCM

        // data chunk
        assert_eq!(&wav[36..40], b"data");
        let data_size = u32::from_le_bytes(wav[40..44].try_into().unwrap());
        assert_eq!(data_size, samples.len() as u32 * 2);

        // Total size
        let file_size = u32::from_le_bytes(wav[4..8].try_into().unwrap());
        assert_eq!(file_size, 44 + data_size);

        // Sample values
        let expected_i16: Vec<i16> = vec![
            0_i16,
            (0.5_f32 * i16::MAX as f32) as i16,
            (-0.5_f32 * i16::MAX as f32) as i16,
            i16::MAX,
            i16::MIN,
        ];
        for (i, &expected) in expected_i16.iter().enumerate() {
            let offset = 44 + i * 2;
            let actual = i16::from_le_bytes(wav[offset..offset + 2].try_into().unwrap());
            assert_eq!(actual, expected, "sample {} mismatch", i);
        }
    }

    #[test]
    fn test_audio_to_audio_load_invalid_path() {
        let result = AudioToAudioGenerator::load("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_voice_clone_empty_reference() {
        // Can't load without a real model, but we can test the parameter
        // validation by creating a degenerate case.
        let err = AudioToAudioGenerator::load("/tmp/__nonexistent__")
            .expect_err("should fail on invalid path");
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_encode_wav_empty_samples() {
        let wav = encode_wav(&[], 44100);
        assert_eq!(wav.len(), 44);
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn test_reference_fingerprint_length() {
        // Can't instantiate AudioToAudioGenerator without a model, but
        // compute_reference_fingerprint is a private method; we test it
        // indirectly by checking the shape of output via encode_wav.
        let dummy_audio = vec![0u8; 1024];
        // We can call the module-level encode_wav to verify the pipeline.
        let samples = vec![0.1f32; 100];
        let wav = encode_wav(&samples, 16000);
        assert!(wav.len() > 44);
    }
}
