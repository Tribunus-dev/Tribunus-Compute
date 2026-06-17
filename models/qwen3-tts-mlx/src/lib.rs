//! Qwen3-TTS: Text-to-speech on Apple Silicon using MLX.
//!
//! Supports the `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` model
//! with 9 preset speakers and multilingual support.

/// Crate version (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod error;
pub mod generate;
pub mod metal_kernels;
pub mod mrope;
pub mod sampling;
pub mod speaker_encoder;
pub mod speech_encoder;
pub mod speech_tokenizer;
pub mod talker;

use std::path::Path;
use std::time::Instant;

use tracing::info;

use config::{GenerationConfig, Qwen3TtsConfig, SpeechTokenizerConfig};
use error::{Error, Result};
use generate::{build_codec_prefix, build_codec_prefix_voice_design, generate, generate_voice_clone, generate_voice_clone_instruct, generate_voice_design, GenerationState};
use speech_tokenizer::SpeechTokenizerDecoder;
use talker::Talker;

// Re-exports
pub use config::GenerationConfig as GenConfig;
pub use config::ModelType;
pub use error::Error as TtsError;
pub use generate::GenerationTiming;

/// Default chunk size for streaming (10 frames = ~833ms at 12Hz)
pub const DEFAULT_CHUNK_FRAMES: usize = 10;

/// High-level text-to-speech synthesizer.
pub struct Synthesizer {
    pub talker: Talker,
    pub decoder: SpeechTokenizerDecoder,
    pub tts_config: Qwen3TtsConfig,
    pub gen_config: GenerationConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub sample_rate: u32,
    /// Optional speaker encoder for voice cloning (Base model only)
    pub speaker_encoder: Option<speaker_encoder::SpeakerEncoder>,
    /// Optional speech encoder for ICL voice cloning (Base model only)
    pub speech_encoder: Option<speech_encoder::SpeechEncoder>,
}

/// Configuration for synthesis.
pub struct SynthesizeOptions<'a> {
    pub speaker: &'a str,
    pub language: &'a str,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub max_new_tokens: Option<i32>,
    pub seed: Option<u64>,
    /// Speed factor: > 1.0 = faster, < 1.0 = slower. Default 1.0.
    pub speed_factor: Option<f32>,
    /// Repetition penalty (e.g. 1.05). Overrides generation_config default if set.
    pub repetition_penalty: Option<f32>,
}

impl Default for SynthesizeOptions<'_> {
    fn default() -> Self {
        Self {
            speaker: "vivian",
            language: "english",
            temperature: None,
            top_k: None,
            top_p: None,
            max_new_tokens: None,
            seed: None,
            speed_factor: None,
            repetition_penalty: None,
        }
    }
}

/// Timing breakdown for synthesis.
#[derive(Debug, Clone)]
pub struct SynthesisTiming {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
    pub decode_ms: f64,
    pub total_ms: f64,
}

impl Synthesizer {
    /// Load models from a directory.
    /// The directory should contain:
    /// - config.json, generation_config.json
    /// - model.safetensors (or with index)
    /// - vocab.json, merges.txt (BPE tokenizer)
    /// - speech_tokenizer/ subdirectory with its own model.safetensors and config.json
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        info!("Loading TTS config...");
        let tts_config = Qwen3TtsConfig::load(model_dir)?;
        let gen_config = GenerationConfig::load(model_dir)?;
        let st_config = SpeechTokenizerConfig::load(model_dir)?;

        let quant = tts_config.quant_config().cloned();

        info!("Loading text tokenizer...");
        let tokenizer = load_bpe_tokenizer(model_dir)?;

        if let Some(ref q) = quant {
            info!("Loading talker model ({}-bit)...", q.bits);
        } else {
            info!("Loading talker model (float)...");
        }
        let talker = talker::load_talker(model_dir, &tts_config.talker_config, quant.as_ref(), tts_config.tts_pad_token_id)?;

        info!("Loading speech tokenizer decoder...");
        let decoder =
            speech_tokenizer::load_speech_tokenizer(model_dir, &st_config.decoder_config)?;

        // Load speaker encoder if present (Base model only)
        let model_type = tts_config.model_type();
        let (spk_encoder, spch_encoder) = if model_type == config::ModelType::Base {
            info!("Loading speaker encoder (ECAPA-TDNN)...");
            let weights = talker::load_all_weights(model_dir)?;

            let spk = if speaker_encoder::has_speaker_encoder_weights(&weights) {
                let enc_dim = tts_config
                    .speaker_encoder_config
                    .as_ref()
                    .map(|c| c.enc_dim)
                    .unwrap_or(tts_config.talker_config.hidden_size);
                let se_config = speaker_encoder::SpeakerEncoderConfig::from_enc_dim(enc_dim);
                match speaker_encoder::load_speaker_encoder(&weights, &se_config) {
                    Ok(enc) => {
                        info!("Speaker encoder loaded (enc_dim={})", enc_dim);
                        Some(enc)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load speaker encoder: {}", e);
                        None
                    }
                }
            } else {
                tracing::warn!("Base model but no speaker_encoder.* weights found");
                None
            };

            // Load speech encoder (Mimi) for ICL voice cloning
            let st_model_path = model_dir.join("speech_tokenizer").join("model.safetensors");
            let spch = if st_model_path.exists() {
                info!("Loading speech encoder (Mimi) for ICL voice cloning...");
                let st_weights = mlx_rs::Array::load_safetensors(&st_model_path)?;
                if speech_encoder::has_encoder_weights(&st_weights) {
                    match speech_encoder::load_speech_encoder(&st_weights) {
                        Ok(enc) => {
                            info!("Speech encoder (Mimi) loaded");
                            Some(enc)
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load speech encoder: {}", e);
                            None
                        }
                    }
                } else {
                    tracing::info!("No encoder.* weights in speech_tokenizer — ICL mode unavailable");
                    None
                }
            } else {
                None
            };

            (spk, spch)
        } else {
            (None, None)
        };

        info!("Models loaded successfully (type: {})", model_type);

        Ok(Self {
            talker,
            decoder,
            tts_config,
            gen_config,
            tokenizer,
            sample_rate: st_config.output_sample_rate,
            speaker_encoder: spk_encoder,
            speech_encoder: spch_encoder,
        })
    }

    /// Swap the talker (and voice-cloning encoders) to a different model variant,
    /// keeping the shared decoder, tokenizer, and sample rate.
    ///
    /// This is much faster than a full `load()` because the speech tokenizer
    /// decoder (~651 MB) and BPE tokenizer are reused. Only the talker weights
    /// (~2.2-2.3 GB) plus optional speaker/speech encoders are reloaded.
    pub fn swap_talker(&mut self, model_dir: impl AsRef<Path>) -> Result<()> {
        let model_dir = model_dir.as_ref();

        info!("Swapping talker from model dir: {}", model_dir.display());

        // Load new configs
        let tts_config = Qwen3TtsConfig::load(model_dir)?;
        let gen_config = GenerationConfig::load(model_dir)?;
        let quant = tts_config.quant_config().cloned();

        // Load new talker first, then swap — if loading fails we keep the old one.
        if let Some(ref q) = quant {
            info!("Loading talker model ({}-bit)...", q.bits);
        } else {
            info!("Loading talker model (float)...");
        }
        let new_talker = talker::load_talker(
            model_dir,
            &tts_config.talker_config,
            quant.as_ref(),
            tts_config.tts_pad_token_id,
        )?;

        // Success — drop old talker + encoders and install new one
        self.speaker_encoder = None;
        self.speech_encoder = None;
        self.talker = new_talker;

        // Load voice-cloning encoders if Base model
        let model_type = tts_config.model_type();
        if model_type == config::ModelType::Base {
            info!("Loading speaker encoder (ECAPA-TDNN)...");
            let weights = talker::load_all_weights(model_dir)?;

            if speaker_encoder::has_speaker_encoder_weights(&weights) {
                let enc_dim = tts_config
                    .speaker_encoder_config
                    .as_ref()
                    .map(|c| c.enc_dim)
                    .unwrap_or(tts_config.talker_config.hidden_size);
                let se_config = speaker_encoder::SpeakerEncoderConfig::from_enc_dim(enc_dim);
                match speaker_encoder::load_speaker_encoder(&weights, &se_config) {
                    Ok(enc) => {
                        info!("Speaker encoder loaded (enc_dim={})", enc_dim);
                        self.speaker_encoder = Some(enc);
                    }
                    Err(e) => tracing::warn!("Failed to load speaker encoder: {}", e),
                }
            }

            // Load speech encoder (Mimi) for ICL voice cloning
            let st_model_path = model_dir.join("speech_tokenizer").join("model.safetensors");
            if st_model_path.exists() {
                info!("Loading speech encoder (Mimi) for ICL voice cloning...");
                let st_weights = mlx_rs::Array::load_safetensors(&st_model_path)?;
                if speech_encoder::has_encoder_weights(&st_weights) {
                    match speech_encoder::load_speech_encoder(&st_weights) {
                        Ok(enc) => {
                            info!("Speech encoder (Mimi) loaded");
                            self.speech_encoder = Some(enc);
                        }
                        Err(e) => tracing::warn!("Failed to load speech encoder: {}", e),
                    }
                }
            }
        }

        self.tts_config = tts_config;
        self.gen_config = gen_config;

        info!("Talker swapped successfully (type: {})", model_type);
        Ok(())
    }

    /// Detected model type (Base, CustomVoice, VoiceDesign).
    pub fn model_type(&self) -> config::ModelType {
        self.tts_config.model_type()
    }

    /// Whether this model supports preset speakers.
    pub fn supports_preset_speakers(&self) -> bool {
        self.model_type().supports_preset_speakers()
    }

    /// Whether this model supports voice cloning.
    pub fn supports_voice_cloning(&self) -> bool {
        self.model_type().supports_voice_cloning()
    }

    /// Whether this model supports voice design via text instructions.
    pub fn supports_voice_design(&self) -> bool {
        self.model_type().supports_voice_design()
    }

    /// Synthesize speech from text.
    /// Returns audio samples as f32 in [-1, 1] at 24kHz.
    pub fn synthesize(
        &mut self,
        text: &str,
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let (samples, _) = self.synthesize_with_timing(text, opts)?;
        Ok(samples)
    }

    /// Synthesize with timing breakdown.
    pub fn synthesize_with_timing(
        &mut self,
        text: &str,
        opts: &SynthesizeOptions,
    ) -> Result<(Vec<f32>, SynthesisTiming)> {
        let t0 = Instant::now();

        // Override generation config with per-call options
        let gen_config = self.resolve_gen_config(opts);

        // Build codec prefix for CustomVoice mode
        let codec_prefix = build_codec_prefix(
            &self.tts_config.talker_config,
            opts.language,
            opts.speaker,
        )?;

        // Tokenize text
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let t1 = Instant::now();

        // Generate codec frames
        let (codes, gen_timing) = generate(
            &mut self.talker,
            &text_ids,
            &codec_prefix,
            &gen_config,
            &self.tts_config,
            opts.seed,
        )?;

        let t2 = Instant::now();

        // Decode to audio
        let samples = self.decoder.decode(&codes)?;

        let t3 = Instant::now();

        let prefill_ms = gen_timing.prefill_ms;
        let generation_ms = gen_timing.generation_ms;
        let decode_ms = (t3 - t2).as_secs_f64() * 1000.0;
        let total_ms = (t3 - t0).as_secs_f64() * 1000.0;

        Ok((
            samples,
            SynthesisTiming {
                prefill_ms,
                generation_ms,
                generation_frames: gen_timing.generation_frames,
                decode_ms,
                total_ms,
            },
        ))
    }

    /// Synthesize with voice cloning (x-vector mode).
    /// Requires Base model.
    pub fn synthesize_voice_clone(
        &mut self,
        text: &str,
        ref_audio: &[f32],
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let (samples, _) = self.synthesize_voice_clone_with_timing(text, ref_audio, language, opts)?;
        Ok(samples)
    }

    /// Synthesize with voice cloning + timing breakdown.
    pub fn synthesize_voice_clone_with_timing(
        &mut self,
        text: &str,
        ref_audio: &[f32],
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<(Vec<f32>, SynthesisTiming)> {
        let t0 = Instant::now();
        let gen_config = self.resolve_gen_config(opts);
        let eos_token = self.tts_config.talker_config.codec_eos_token_id;
        let bos_id = self.tts_config.talker_config.codec_bos_id;
        let pad_id = self.tts_config.talker_config.codec_pad_id;

        // Extract speaker embedding from reference audio
        let spk_encoder = self.speaker_encoder
            .as_mut()
            .ok_or_else(|| Error::Config("Voice cloning requires Base model with speaker encoder".to_string()))?;
        let spk_embedding = spk_encoder.extract_embedding(ref_audio)?;

        // Tokenize text
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let t1 = Instant::now();

        // Generate with speaker embedding injected
        let (codes, gen_timing) = generate_voice_clone(
            &mut self.talker,
            &text_ids,
            &spk_embedding,
            &gen_config,
            &self.tts_config,
            eos_token,
            bos_id,
            pad_id,
            opts.seed,
        )?;

        let t2 = Instant::now();

        let samples = self.decoder.decode(&codes)?;

        let t3 = Instant::now();

        Ok((
            samples,
            SynthesisTiming {
                prefill_ms: gen_timing.prefill_ms,
                generation_ms: gen_timing.generation_ms,
                generation_frames: gen_timing.generation_frames,
                decode_ms: (t3 - t2).as_secs_f64() * 1000.0,
                total_ms: (t3 - t0).as_secs_f64() * 1000.0,
            },
        ))
    }

    /// Synthesize with voice cloning + emotion/style instruct (experimental).
    /// Combines x-vector cloning with voice instruct.
    pub fn synthesize_voice_clone_instruct(
        &mut self,
        text: &str,
        ref_audio: &[f32],
        instruct: &str,
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let (samples, _) = self.synthesize_voice_clone_instruct_with_timing(text, ref_audio, instruct, language, opts)?;
        Ok(samples)
    }

    /// Synthesize with voice cloning + instruct + timing.
    pub fn synthesize_voice_clone_instruct_with_timing(
        &mut self,
        text: &str,
        ref_audio: &[f32],
        instruct: &str,
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<(Vec<f32>, SynthesisTiming)> {
        let t0 = Instant::now();
        let gen_config = self.resolve_gen_config(opts);

        // Extract speaker embedding
        let spk_encoder = self.speaker_encoder
            .as_mut()
            .ok_or_else(|| Error::Config("Voice cloning requires Base model with speaker encoder".to_string()))?;
        let spk_embedding = spk_encoder.extract_embedding(ref_audio)?;

        // Tokenize text and instruct
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let inst_encoding = self.tokenizer.encode(instruct, true).map_err(|e| {
            Error::Model(format!("Instruct tokenization error: {e}"))
        })?;
        let instruct_ids: Vec<u32> = inst_encoding.get_ids().iter().copied().collect();

        let t1 = Instant::now();

        let eos_token = self.tts_config.talker_config.codec_eos_token_id;
        let bos_id = self.tts_config.talker_config.codec_bos_id;
        let pad_id = self.tts_config.talker_config.codec_pad_id;

        let (codes, gen_timing) = generate_voice_clone_instruct(
            &mut self.talker,
            &text_ids,
            &instruct_ids,
            &spk_embedding,
            &gen_config,
            &self.tts_config,
            eos_token,
            bos_id,
            pad_id,
            opts.seed,
        )?;

        let t2 = Instant::now();

        let samples = self.decoder.decode(&codes)?;

        let t3 = Instant::now();

        Ok((
            samples,
            SynthesisTiming {
                prefill_ms: gen_timing.prefill_ms,
                generation_ms: gen_timing.generation_ms,
                generation_frames: gen_timing.generation_frames,
                decode_ms: (t3 - t2).as_secs_f64() * 1000.0,
                total_ms: (t3 - t0).as_secs_f64() * 1000.0,
            },
        ))
    }

    /// Synthesize with voice design (text-described voice characteristics).
    /// Requires VoiceDesign model.
    pub fn synthesize_voice_design(
        &mut self,
        text: &str,
        voice_description: &str,
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let (samples, _) = self.synthesize_voice_design_with_timing(text, voice_description, language, opts)?;
        Ok(samples)
    }

    /// Synthesize with voice design + timing.
    pub fn synthesize_voice_design_with_timing(
        &mut self,
        text: &str,
        voice_description: &str,
        language: &str,
        opts: &SynthesizeOptions,
    ) -> Result<(Vec<f32>, SynthesisTiming)> {
        let t0 = Instant::now();
        let gen_config = self.resolve_gen_config(opts);

        // Build codec prefix (no speaker token)
        let codec_prefix = build_codec_prefix_voice_design(
            &self.tts_config.talker_config,
            language,
        )?;

        // Tokenize text and voice description
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let desc_encoding = self.tokenizer.encode(voice_description, true).map_err(|e| {
            Error::Model(format!("Voice description tokenization error: {e}"))
        })?;
        let desc_ids: Vec<u32> = desc_encoding.get_ids().iter().copied().collect();

        let t1 = Instant::now();

        let eos_token = self.tts_config.talker_config.codec_eos_token_id;
        let bos_id = self.tts_config.talker_config.codec_bos_id;
        let pad_id = self.tts_config.talker_config.codec_pad_id;

        let (codes, gen_timing) = generate_voice_design(
            &mut self.talker,
            &text_ids,
            &desc_ids,
            &codec_prefix,
            &gen_config,
            &self.tts_config,
            eos_token,
            bos_id,
            pad_id,
            opts.seed,
        )?;

        let t2 = Instant::now();

        let samples = self.decoder.decode(&codes)?;

        let t3 = Instant::now();

        Ok((
            samples,
            SynthesisTiming {
                prefill_ms: gen_timing.prefill_ms,
                generation_ms: gen_timing.generation_ms,
                generation_frames: gen_timing.generation_frames,
                decode_ms: (t3 - t2).as_secs_f64() * 1000.0,
                total_ms: (t3 - t0).as_secs_f64() * 1000.0,
            },
        ))
    }

    /// Synthesize with a preset speaker + style instruction (Speaker+Instruct mode).
    /// Requires CustomVoice model.
    pub fn synthesize_with_speaker_instruct(
        &mut self,
        text: &str,
        instruct: &str,
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let (samples, _) = self.synthesize_with_speaker_instruct_with_timing(text, instruct, opts)?;
        Ok(samples)
    }

    /// Synthesize with speaker + instruct + timing.
    pub fn synthesize_with_speaker_instruct_with_timing(
        &mut self,
        text: &str,
        instruct: &str,
        opts: &SynthesizeOptions,
    ) -> Result<(Vec<f32>, SynthesisTiming)> {
        let t0 = Instant::now();
        let gen_config = self.resolve_gen_config(opts);

        // Build codec prefix for CustomVoice mode (with speaker token)
        let codec_prefix = build_codec_prefix(
            &self.tts_config.talker_config,
            opts.language,
            opts.speaker,
        )?;

        // Tokenize text and instruct
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let inst_encoding = self.tokenizer.encode(instruct, true).map_err(|e| {
            Error::Model(format!("Instruct tokenization error: {e}"))
        })?;
        let instruct_ids: Vec<u32> = inst_encoding.get_ids().iter().copied().collect();

        let t1 = Instant::now();

        let (codes, gen_timing) = generate(
            &mut self.talker,
            &text_ids,
            &codec_prefix,
            &gen_config,
            &self.tts_config,
            opts.seed,
        )?;

        let t2 = Instant::now();

        let samples = self.decoder.decode(&codes)?;

        let t3 = Instant::now();

        Ok((
            samples,
            SynthesisTiming {
                prefill_ms: gen_timing.prefill_ms,
                generation_ms: gen_timing.generation_ms,
                generation_frames: gen_timing.generation_frames,
                decode_ms: (t3 - t2).as_secs_f64() * 1000.0,
                total_ms: (t3 - t0).as_secs_f64() * 1000.0,
            },
        ))
    }

    /// Start a streaming synthesis session that yields audio chunks as they're generated.
    ///
    /// `chunk_frames`: codec frames per chunk (default 10 = ~833ms audio)
    pub fn start_streaming(
        &mut self,
        text: &str,
        opts: &SynthesizeOptions,
        chunk_frames: usize,
    ) -> Result<StreamingSession<'_>> {
        let gen_config = self.resolve_gen_config(opts);

        let codec_prefix = build_codec_prefix(
            &self.tts_config.talker_config,
            opts.language,
            opts.speaker,
        )?;

        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            Error::Model(format!("Tokenization error: {e}"))
        })?;
        let text_ids: Vec<u32> = encoding.get_ids().iter().copied().collect();

        let state = GenerationState::new(
            &mut self.talker,
            &text_ids,
            &codec_prefix,
            &gen_config,
            &self.tts_config,
            opts.seed,
        )?;

        Ok(StreamingSession {
            state: Some(state),
            decoder: &mut self.decoder,
            chunk_frames,
            generated_frames: 0,
            all_codes: Vec::new(),
        })
    }

    /// Available preset speaker names.
    pub fn speakers(&self) -> Vec<String> {
        self.tts_config.talker_config.spk_id.keys().cloned().collect()
    }

    /// Available language names.
    pub fn languages(&self) -> Vec<String> {
        self.tts_config.talker_config.codec_language_id.keys().cloned().collect()
    }

    fn resolve_gen_config(&self, opts: &SynthesizeOptions) -> GenerationConfig {
        let mut gc = self.gen_config.clone();
        if let Some(t) = opts.temperature { gc.temperature = t; }
        if let Some(k) = opts.top_k { gc.top_k = k; }
        if let Some(p) = opts.top_p { gc.top_p = p; }
        if let Some(n) = opts.max_new_tokens { gc.max_new_tokens = n; }
        if let Some(s) = opts.speed_factor { gc.speed_factor = s; }
        if let Some(r) = opts.repetition_penalty { gc.repetition_penalty = r; }
        gc
    }
}

/// Streaming synthesis session.
pub struct StreamingSession<'a> {
    state: Option<GenerationState<'a>>,
    decoder: &'a mut SpeechTokenizerDecoder,
    chunk_frames: usize,
    generated_frames: usize,
    all_codes: Vec<[u32; 16]>,
}

impl StreamingSession<'_> {
    /// Get the next chunk of audio samples.
    /// Returns `None` when generation is complete.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>> {
        let state = self.state.as_mut().ok_or_else(|| {
            Error::Model("Streaming session already consumed".to_string())
        })?;

        let chunk = state.next_chunk(self.chunk_frames)?;
        match chunk {
            Some(codes) => {
                let n = codes.len();
                self.all_codes.extend(codes);
                let samples = self.decoder.decode(&self.all_codes)?;
                self.generated_frames += n;
                Ok(Some(samples))
            }
            None => {
                self.state = None;
                Ok(None)
            }
        }
    }

    /// Number of codec frames generated so far.
    pub fn total_frames(&self) -> usize {
        self.generated_frames
    }

    /// Approximate duration of audio generated so far in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.generated_frames as f32 / 12.0 // 12Hz codec
    }
}

// ── BPE Tokenizer loading ──────────────────────────────────────────────────

fn load_bpe_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    let vocab_path = model_dir.join("vocab.json");
    let merges_path = model_dir.join("merges.txt");

    if vocab_path.exists() && merges_path.exists() {
        let mut tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::from_files(
                vocab_path.to_str().unwrap(),
                merges_path.to_str().unwrap(),
            ).map_err(|e| Error::Model(format!("BPE load error: {e}")))?,
        );
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Config(format!("truncation config: {e}")))?;
        Ok(tokenizer)
    } else {
        // Fallback: try loading tokenizer.json
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            Ok(tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
                Error::Model(format!("tokenizer.json load error: {e}"))
            })?)
        } else {
            Err(Error::Model(format!(
                "No tokenizer found in {}. Need vocab.json + merges.txt or tokenizer.json",
                model_dir.display()
            )))
        }
    }
}

// ── Audio utilities ────────────────────────────────────────────────────────

/// Normalize audio to peak amplitude.
pub fn normalize_audio(samples: &[f32], peak: f32) -> Vec<f32> {
    let max = samples.iter().copied().fold(0.0f32, f32::max);
    let min = samples.iter().copied().fold(0.0f32, |a, b| a.min(b.abs()));
    let scale = if max > 0.0 { peak / max } else { 1.0 };
    samples.iter().map(|&s| (s * scale).clamp(-1.0, 1.0)).collect()
}

/// Save audio samples as a WAV file.
pub fn save_wav(samples: &[f32], sample_rate: u32, path: impl AsRef<std::path::Path>) -> Result<()> {
    use std::io::Write;

    let path = path.as_ref();
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;

    // Convert f32 samples to i16
    let sample_count = samples.len();
    let data_len = sample_count * 2; // 16-bit = 2 bytes per sample
    let file_len = 44 + data_len; // WAV header (44 bytes) + data

    let mut file = std::fs::File::create(path)?;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&(file_len as u32 - 8).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?;  // PCM format
    file.write_all(&1u16.to_le_bytes())?;  // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    file.write_all(&2u16.to_le_bytes())?;  // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&(data_len as u32).to_le_bytes())?;

    // PCM data
    for &sample in samples {
        let clamped = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        file.write_all(&clamped.to_le_bytes())?;
    }

    Ok(())
}
