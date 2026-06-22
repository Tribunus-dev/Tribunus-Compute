//! Text-to-speech generation via qwen3-tts-mlx.
//!
//! Provides [`TextToSpeechGenerator`] — a high-level wrapper around the
//! qwen3-tts-mlx `Synthesizer` that exposes `load()` and `synthesize()`
//! for the `/v1/audio/speech` API endpoint.

use std::sync::Arc;
use tokio::sync::Mutex;

/// TTS generation pipeline wrapping qwen3-tts-mlx.
///
/// 1. Tokenize input text
/// 2. Run TTS model (encoder -> decoder -> vocoder)
/// 3. Return audio samples as raw PCM float32 at the model's sample rate
pub struct TextToSpeechGenerator {
    /// The underlying qwen3-tts-mlx synthesizer, behind a tokio mutex for
    /// safe concurrent access from the async route handler.
    pub synthesizer: Arc<Mutex<qwen3_tts_mlx::Synthesizer>>,
    /// Audio sample rate (e.g. 24000 or 44100 Hz).
    pub sample_rate: u32,
    /// Default voice preset for multi-voice models.
    pub voice: Option<String>,
}

impl TextToSpeechGenerator {
    /// Load a TTS model from a pre-compiled model directory.
    ///
    /// `image_path` must point to a directory containing the qwen3-tts-mlx
    /// model files (config.json, model.safetensors, speech_tokenizer/, etc.).
    pub fn load(image_path: &str) -> Result<Self, String> {
        let synthesizer = qwen3_tts_mlx::Synthesizer::load(image_path)
            .map_err(|e| format!("failed to load TTS model from {image_path}: {e}"))?;

        let sample_rate = synthesizer.sample_rate;

        Ok(Self {
            synthesizer: Arc::new(Mutex::new(synthesizer)),
            sample_rate,
            voice: None,
        })
    }

    /// Generate audio from text.
    ///
    /// Returns `(sample_rate, pcm_f32_samples)`.
    ///
    /// `voice` overrides the default speaker preset (e.g. "vivian", "bella").
    /// When `None`, the model's default speaker is used.
    pub async fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
    ) -> Result<(u32, Vec<f32>), String> {
        let mut opts = qwen3_tts_mlx::SynthesizeOptions::default();
        if let Some(v) = voice {
            opts.speaker = v;
        }

        let mut synth = self.synthesizer.lock().await;

        // Run the blocking MLX work off the async runtime.
        let text_owned = text.to_owned();
        let samples = tokio::task::block_in_place(|| synth.synthesize(&text_owned, &opts))
            .map_err(|e| format!("TTS synthesis failed: {e}"))?;

        Ok((self.sample_rate, samples))
    }
}

// ---------------------------------------------------------------------------
// WAV encoding helpers
// ---------------------------------------------------------------------------

/// Convert mono float32 PCM samples (in [-1.0, 1.0]) to WAV bytes at the
/// given sample rate.
///
/// Produces standard 16-bit little-endian PCM WAV.
pub fn pcm_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    if samples.is_empty() {
        return wav_header(0, sample_rate).to_vec();
    }

    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2; // 16-bit mono
    let header = wav_header(data_size, sample_rate);

    let mut wav = Vec::with_capacity(44 + data_size as usize);
    wav.extend_from_slice(&header);

    // Convert f32 to i16 (little-endian)
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * i16::MAX as f32) as i16;
        wav.extend_from_slice(&sample_i16.to_le_bytes());
    }

    wav
}

/// Build a canonical 44-byte WAV header for 16-bit mono PCM.
fn wav_header(data_size: u32, sample_rate: u32) -> [u8; 44] {
    let byte_rate = sample_rate * 2; // 16-bit mono
    let file_size = 36 + data_size;

    let mut h = [0u8; 44];

    // RIFF chunk descriptor
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&file_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");

    // fmt sub-chunk
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&(16u32).to_le_bytes()); // chunk size
    h[20..22].copy_from_slice(&(1u16).to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&(1u16).to_le_bytes()); // mono
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&(2u16).to_le_bytes()); // block align
    h[34..36].copy_from_slice(&(16u16).to_le_bytes()); // bits per sample

    // data sub-chunk
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());

    h
}

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode a byte slice as a base64 string (no padding).
pub fn base64_encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    let remainder = data.len() % 3;
    let full = data.len() - remainder;
    let capacity = (full / 3) * 4 + if remainder > 0 { 4 } else { 0 };
    let mut out = Vec::with_capacity(capacity);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize]);
        out.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize]);

        if chunk.len() > 1 {
            out.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }

        if chunk.len() > 2 {
            out.push(BASE64_CHARS[(triple & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
    }

    // SAFETY: BASE64_CHARS are all ASCII
    unsafe { String::from_utf8_unchecked(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_to_wav_header() {
        let samples = vec![0.0f32; 100];
        let wav = pcm_to_wav(&samples, 24000);

        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[20..22], &[1u8, 0]); // PCM
        assert_eq!(&wav[22..24], &[1u8, 0]); // mono
        assert_eq!(&wav[24..28], &[0x60, 0x5D, 0x00, 0x00]); // 24000
        assert_eq!(wav.len(), 44 + 100 * 2);
    }

    #[test]
    fn test_base64_encode_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn test_base64_encode_hello() {
        // "hello" base64 = aGVsbG8=
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
    }
}
