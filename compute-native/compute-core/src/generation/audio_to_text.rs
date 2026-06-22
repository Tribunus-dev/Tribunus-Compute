//! ASR (speech-to-text) pipeline wrapping funasr-mlx / qwen3-asr-mlx.
//!
//! 1. Load audio file from path/URL
//! 2. Preprocess (resample to 16kHz, mel spectrogram)
//! 3. Run ASR model (encoder → decoder/CTC)
//! 4. Return transcribed text

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use funasr_mlx::audio::load_audio_for_paraformer;
use funasr_mlx::paraformer::{parse_cmvn_file, Paraformer};
use funasr_mlx::Vocabulary;

/// Compiled model directory handle — keeps a reference to the model
/// assets on disk so downstream consumers (model registry, telemetry)
/// can inspect the path without holding the inference model.
///
/// For ASR models the internal representation is a loaded Paraformer
/// wrapped behind a Mutex for `&self`-based transcription.
struct ModelHandle {
    /// Path to the model directory.
    image_dir: String,
}

/// ASR (speech-to-text) pipeline wrapping funasr-mlx / qwen3-asr-mlx.
///
/// 1. Load audio file from path/URL
/// 2. Preprocess (resample to 16kHz, mel spectrogram)
/// 3. Run ASR model (encoder → decoder/CTC)
/// 4. Return transcribed text
pub struct AudioToTextGenerator {
    /// Handle to the compiled model assets on disk.
    /// Reserved for future ComputeImage integration.
    pub model: Arc<ModelHandle>,
    pub sample_rate: u32,
    pub language: Option<String>,

    // Internal inference state
    paraformer: Mutex<Paraformer>,
    vocab: Vocabulary,
}

impl AudioToTextGenerator {
    /// Load an ASR model from a directory containing model assets.
    ///
    /// The directory must contain:
    /// - `paraformer.safetensors` — model weights
    /// - `tokens.txt` — vocabulary (one token per line)
    /// - `am.mvn` — optional CMVN normalisation file
    ///
    /// # Errors
    /// Returns an error if any required file is missing or loading fails.
    pub fn load(image_path: &str) -> Result<Self, String> {
        let image_dir = Path::new(image_path);

        // 1. Locate the safetensors weights file
        let safetensors_candidates = [
            image_dir.join("paraformer.safetensors"),
            image_dir.join("model.safetensors"),
            image_dir.join("model-00001-of-00002.safetensors"),
        ];
        let weights_path = safetensors_candidates
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| {
                format!(
                    "no safetensors model file found in {} \
                     (tried paraformer.safetensors, model.safetensors, sharded)",
                    image_path
                )
            })?;

        // 2. Load model weights via the funasr-mlx public loader.
        //    The loader creates a Paraformer with default config and populates
        //    all weights from the safetensors file.
        let mut paraformer = funasr_mlx::paraformer::load_model(&weights_path)
            .map_err(|e| format!("load paraformer model: {e}"))?;

        // 3. Load CMVN if available (am.mvn is optional but recommended for
        //    proper normalisation).
        let cmvn_path = image_dir.join("am.mvn");
        if cmvn_path.exists() {
            let (addshift, rescale) =
                parse_cmvn_file(&cmvn_path).map_err(|e| format!("parse CMVN: {e}"))?;
            paraformer.set_cmvn(addshift, rescale);
        }

        // 4. Load vocabulary (one token per line)
        let vocab_path = image_dir.join("tokens.txt");
        let vocab = Vocabulary::load(&vocab_path).map_err(|e| format!("load vocabulary: {e}"))?;

        let handle = ModelHandle {
            image_dir: image_path.to_string(),
        };

        Ok(Self {
            model: Arc::new(handle),
            sample_rate: 16000,
            language: None,
            paraformer: Mutex::new(paraformer),
            vocab,
        })
    }

    /// Transcribe audio to text.
    ///
    /// `audio_path` — local file path or URL to an audio file.
    /// `language` — optional language hint (e.g. "Chinese", "English").
    ///              When `None`, defaults to "Chinese" for Paraformer.
    ///
    /// # Errors
    /// Returns an error if the audio file cannot be loaded or transcription fails.
    pub fn transcribe(&self, audio_path: &str, language: Option<&str>) -> Result<String, String> {
        // 1. Load audio (load_audio_for_paraformer resamples to 16kHz
        //    automatically if needed).
        let (samples, _sample_rate): (Vec<f32>, u32) =
            load_audio_for_paraformer(audio_path).map_err(|e| format!("load audio: {e}"))?;

        // 2. Acquire the model and transcribe
        let mut guard = self
            .paraformer
            .lock()
            .map_err(|e| format!("lock paraformer: {e}"))?;

        let text = funasr_mlx::transcribe(&mut *guard, &samples, &self.vocab)
            .map_err(|e| format!("transcribe: {e}"))?;

        // Language hint handled at the model level:
        // Paraformer currently defaults to Chinese recognition;
        // future backends (qwen3-asr-mlx) will use the hint.
        let _ = language;

        Ok(text)
    }

    /// Transcribe audio from raw PCM samples at 16kHz.
    ///
    /// Useful when the audio has already been decoded externally.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, String> {
        let mut guard = self
            .paraformer
            .lock()
            .map_err(|e| format!("lock paraformer: {e}"))?;

        let text = funasr_mlx::transcribe(&mut *guard, samples, &self.vocab)
            .map_err(|e| format!("transcribe samples: {e}"))?;

        Ok(text)
    }
}
