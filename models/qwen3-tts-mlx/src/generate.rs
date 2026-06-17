//! Dual-track autoregressive generation for Qwen3-TTS.
//!
//! Streaming batched prefill (CustomVoice mode):
//!   Pos 0-2:   role tokens [im_start, assistant, \n] — text projection only, NO codec
//!   Pos 3-7:   tts_pad + codec_embedding([think, think_bos, lang, think_eos, spk])
//!   Pos 8:     tts_bos + codec_pad
//!   Pos 9:     first_text + codec_bos
//!
//! All 10 prefill positions processed in ONE forward pass with causal mask.
//!
//! Autoregressive (streaming text):
//!   Frame i:   codec_embed(prev_codes) + trailing_text[i]
//!   Where trailing_text = [text_token_1, ..., text_token_N-1, tts_eos, tts_pad, tts_pad, ...]

use std::time::Instant;

use mlx_rs::{module::Module, ops::indexing::IndexOp, transforms::eval, Array};
use tracing::{info, warn};

use crate::config::{GenerationConfig, Qwen3TtsConfig, TalkerConfig};
use crate::error::Result;
use crate::sampling::{build_eos_suppression_mask, build_eos_unit_mask, build_suppression_mask, compute_eos_steering_bias, sample_logits_with_mask, RepetitionPenaltyMask, SamplingKey};
use crate::talker::Talker;

// ── Repetition loop detection constants ─────────────────────────────────────
/// Window size (in codec frames) for repetition detection. ~2s at 12Hz.
const REPEAT_WINDOW: usize = 24;
/// If this many frames in the window share the same token0, it's a loop.
const REPEAT_THRESHOLD: usize = 20;

/// Average ratio of generated codec frames to text tokens.
/// Empirically determined from Chinese text: ~4.0 frames per text token.
/// (12Hz codec, typical Chinese text generates ~3.3 frames per character,
/// but BPE tokens are coarser so ratio per token is higher)
const AVG_FRAMES_PER_TEXT_TOKEN: f32 = 4.0;

/// Timing information for each phase of generation.
#[derive(Debug, Clone)]
pub struct GenerationTiming {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
}

/// Build the codec prefix for CustomVoice mode with specified language.
/// Returns [think, think_bos, lang_id, think_eos, spk_id]
pub fn build_codec_prefix(
    talker_config: &TalkerConfig,
    language: &str,
    speaker: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(talker_config, language)?;

    let spk_id = talker_config
        .spk_id
        .get(speaker)
        .copied()
        .ok_or_else(|| {
            crate::error::Error::Config(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker,
                talker_config.spk_id.keys().collect::<Vec<_>>()
            ))
        })?;

    Ok(vec![
        talker_config.codec_think_id,
        talker_config.codec_think_bos_id,
        lang_id,
        talker_config.codec_think_eos_id,
        spk_id,
    ])
}

/// Build the codec prefix for VoiceDesign mode (no speaker token).
/// Returns [think, think_bos, lang_id, think_eos]
pub fn build_codec_prefix_voice_design(
    talker_config: &TalkerConfig,
    language: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(talker_config, language)?;
    Ok(vec![
        talker_config.codec_think_id,
        talker_config.codec_think_bos_id,
        lang_id,
        talker_config.codec_think_eos_id,
    ])
}

/// Build codec prefix for ICL voice cloning (auto-language / nothink mode).
/// Returns [nothink, think_bos, think_eos] — 3 tokens, NO language token.
/// This matches the Python's `language="auto"` path used for voice cloning.
pub fn build_codec_prefix_nothink(talker_config: &TalkerConfig) -> Vec<u32> {
    vec![
        talker_config.codec_nothink_id,    // 2155
        talker_config.codec_think_bos_id,  // 2156
        talker_config.codec_think_eos_id,  // 2157
    ]
}

fn resolve_language_id(talker_config: &TalkerConfig, language: &str) -> Result<u32> {
    talker_config
        .codec_language_id
        .get(language)
        .copied()
        .ok_or_else(|| {
            crate::error::Error::Config(format!(
                "Unknown language '{}'. Available: {:?}",
                language,
                talker_config.codec_language_id.keys().collect::<Vec<_>>()
            ))
        })
}

// ============================================================================
// Shared generation loop — used by all generate_* functions
// ============================================================================

/// Configuration for the shared autoregressive generation loop.
/// Each `generate_*` function does its own unique prefill, then delegates
/// to `run_generation_loop()` for the autoregressive decoding.
struct GenerationLoopParams {
    trailing_len: usize,
    eos_token: u32,
    pad_id: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    max_new_tokens: usize,
    /// Optional EOS steering: (target_frames, speed_factor)
    eos_steering: Option<(usize, f32)>,
}

/// Run the shared autoregressive generation loop.
///
/// Takes initial logits/hidden from prefill and generates codec frames
/// until EOS or max_new_tokens is reached.
fn run_generation_loop(
    talker: &mut Talker,
    mut logits: Array,
    mut hidden: Array,
    trailing_text_embeds: &Array,
    tts_pad_embed: &Array,
    params: &GenerationLoopParams,
    rng_key: &mut Option<SamplingKey>,
    vocab_size: usize,
) -> Result<Vec<[u32; 16]>> {
    let min_new_tokens: usize = 2;

    // Pre-build suppression masks (GPU arrays, built once)
    let suppress_mask = build_suppression_mask(vocab_size, params.eos_token);
    let eos_suppress_mask = build_eos_suppression_mask(vocab_size, params.eos_token);
    let combined_mask = suppress_mask.add(&eos_suppress_mask)?;
    let mut penalty_mask = RepetitionPenaltyMask::new(vocab_size, params.repetition_penalty)?;

    // EOS logit steering (optional)
    let eos_unit_mask = if params.eos_steering.is_some() {
        Some(build_eos_unit_mask(vocab_size, params.eos_token))
    } else {
        None
    };

    let mut all_codes: Vec<[u32; 16]> = Vec::new();
    let mut mem_guard = mlx_rs_core::memory::MemoryGuard::default_guard();

    // Repetition loop detection: if the same token0 repeats too many times
    // in a window, the model is stuck — break early and return what we have.
    let mut recent_tokens: Vec<u32> = Vec::new();

    for step in 0..params.max_new_tokens {
        let base_mask = if step < min_new_tokens {
            &combined_mask
        } else {
            &suppress_mask
        };

        // Apply EOS logit steering bias (avoid cloning: use reference when no bias needed)
        let steered_mask;
        let effective_mask: &Array = if let Some((target, speed)) = params.eos_steering {
            if step >= min_new_tokens {
                let bias = compute_eos_steering_bias(step, target, speed);
                if bias.abs() > 0.01 {
                    let unit_mask = eos_unit_mask.as_ref().unwrap();
                    let bias_mask = unit_mask.multiply(mlx_rs::array!(bias))?;
                    steered_mask = base_mask.add(&bias_mask)?;
                    &steered_mask
                } else {
                    base_mask
                }
            } else {
                base_mask
            }
        } else {
            base_mask
        };

        let token0 = sample_logits_with_mask(
            &logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
            &[],
            rng_key.as_mut(),
            Some(effective_mask),
            Some(&penalty_mask),
        )?;

        if token0 == params.eos_token {
            info!("EOS at step {} (target={:?})", step, params.eos_steering.map(|(t, _)| t));
            break;
        }

        penalty_mask.record_token(token0)?;

        // Repetition loop detection
        recent_tokens.push(token0);
        if recent_tokens.len() > REPEAT_WINDOW {
            recent_tokens.remove(0);
        }
        if recent_tokens.len() == REPEAT_WINDOW && step >= params.trailing_len {
            let mut counts = std::collections::HashMap::new();
            for &t in &recent_tokens {
                *counts.entry(t).or_insert(0usize) += 1;
            }
            let max_count = counts.values().copied().max().unwrap_or(0);
            if max_count >= REPEAT_THRESHOLD {
                warn!(
                    "Repetition loop detected at step {} (token {} repeated {}/{} times), stopping early",
                    step, recent_tokens.last().unwrap_or(&0), max_count, REPEAT_WINDOW
                );
                break;
            }
        }

        // Generate codebooks 1-15 via code predictor
        let hidden_slice = hidden.index((.., -1.., ..));
        let code0_arr = Array::from_slice(&[token0 as i32], &[1, 1]);
        let code0_embed = talker.codec_embedding.forward(&code0_arr)?;
        let sub_codes = talker
            .code_predictor
            .generate_codes(&hidden_slice, &code0_embed)?;

        let mut frame = [params.pad_id; 16];
        frame[0] = token0;
        for (g, &code) in sub_codes.iter().enumerate() {
            frame[g + 1] = code;
        }
        all_codes.push(frame);

        // Build next input: codec_embed(prev_codes) + trailing_text or tts_pad
        let text_embed_indexed;
        let text_embed: &Array = if step < params.trailing_len {
            let s = step as i32;
            text_embed_indexed = trailing_text_embeds.index((.., s..s + 1, ..));
            &text_embed_indexed
        } else {
            tts_pad_embed
        };

        let input_embed = talker.build_generation_embedding_with_text(&frame, text_embed)?;
        let result = talker.forward_step(input_embed)?;
        logits = result.0;
        hidden = result.1;

        mem_guard.step();
    }

    Ok(all_codes)
}

/// Run the full generation loop (streaming text, batched prefill).
///
/// Prefill (10 positions in one forward pass):
///   Pos 0-2: role [im_start, assistant, \n] — text only
///   Pos 3-7: tts_pad + codec [think, think_bos, lang, think_eos, spk]
///   Pos 8:   tts_bos + codec_pad
///   Pos 9:   first_text + codec_bos
///
/// Generation (streaming):
///   Each frame: codec_embed(prev_codes) + trailing_text[frame_idx]
///   Where trailing_text = [text_1, text_2, ..., text_N-1, tts_eos]
///   After trailing text exhausted: tts_pad
///
/// Returns Vec of [u32; 16] code frames.
pub fn generate(
    talker: &mut Talker,
    text_token_ids: &[u32],
    codec_prefix: &[u32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;
    let bos_id = tts_config.talker_config.codec_bos_id;

    // Initialize seeded PRNG if seed is provided
    let mut rng_key = seed
        .map(SamplingKey::new)
        .transpose()?;

    info!(
        "Generating text_length={}, codec_prefix_len={}, max_new_tokens={}",
        text_token_ids.len(),
        codec_prefix.len(),
        gen_config.max_new_tokens,
    );

    if text_token_ids.is_empty() {
        return Err(crate::error::Error::Model("Empty text input".to_string()));
    }

    // Speed control: set RoPE speed factor for EOS steering
    let speed = gen_config.speed_factor;
    talker.set_rope_speed_factor(speed);

    // Build trailing text sequence: text_tokens[1..] + [tts_eos] + [tts_pad, ...]
    // Truncate trailing text to avoid exceeding max_new_tokens (leave room for prefill)
    let max_trailing = (gen_config.max_new_tokens as usize).saturating_sub(codec_prefix.len() + 5);
    let text_tail: Vec<u32> = text_token_ids[1..]
        .iter()
        .copied()
        .take(max_trailing)
        .collect();
    let trailing_len = text_tail.len();
    let trailing_width = (trailing_len + 5).max(trailing_len);
    let mut trailing_text: Vec<u32> = text_tail;
    trailing_text.push(tts_config.tts_eos_token_id);
    while trailing_text.len() < trailing_width {
        trailing_text.push(tts_config.tts_pad_token_id);
    }

    // ========================================================================
    // Prefill: build text + codec embeddings for first 10 positions
    // ========================================================================
    let t0 = Instant::now();

    // Text position embeddings
    let text_role_tokens: Vec<u32> = vec![
        tts_config.im_start_token_id,
        tts_config.assistant_token_id,
        tts_config.tts_bos_token_id, // \n role token
    ];
    let text_pad_token = tts_config.tts_pad_token_id;

    // Gather codec prefix embeddings
    let codec_prefix_embeds: Vec<Array> = codec_prefix
        .iter()
        .map(|&id| {
            let arr = Array::from_slice(&[id as i32], &[1, 1]);
            talker.codec_embedding.forward(&arr)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // tts_pad_embed for non-codec positions
    let tts_pad_arr = Array::from_slice(&[tts_config.tts_pad_token_id as i32], &[1, 1]);
    let tts_pad_embed = talker.text_embedding.forward(&tts_pad_arr)?;

    let tts_bos_arr = Array::from_slice(&[tts_config.tts_bos_token_id as i32], &[1, 1]);
    let text_bos_embed = talker.text_embedding.forward(&tts_bos_arr)?;

    // Build first text embedding
    let first_text_id = text_token_ids[0];
    let text_token_arr = Array::from_slice(&[first_text_id as i32], &[1, 1]);
    let first_text_embed = talker.text_embedding.forward(&text_token_arr)?;

    // Prefill: 10 positions with different codec/text combinations
    let no_codec_positions = 3; // im_start, assistant, \n
    let prefill_len = no_codec_positions + codec_prefix.len() + 2; // +2 for tts_bos+codec_pad and first_text+codec_bos
    trace!("Prefill length: {} (no_codec={} + codec_prefix={} + 2)", prefill_len, no_codec_positions, codec_prefix.len());

    let mut prefill_embeds: Vec<Array> = Vec::with_capacity(prefill_len);

    // Positions 0-2: text only, no codec
    for &role_id in &text_role_tokens {
        let role_arr = Array::from_slice(&[role_id as i32], &[1, 1]);
        let text_emb = talker.text_embedding.forward(&role_arr)?;
        let projected = talker.text_projection.forward(&text_emb)?;
        prefill_embeds.push(projected);
    }

    // Positions 3-7: tts_pad text + codec prefix embeddings
    for codec_emb in &codec_prefix_embeds {
        let projected = talker.text_projection.forward(&tts_pad_embed)?;
        let combined = projected.add(codec_emb)?;
        prefill_embeds.push(combined);
    }

    // Position 8: tts_bos text + codec_pad embedding
    let tts_bos_projected = talker.text_projection.forward(&text_bos_embed)?;
    let codec_pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
    let codec_pad_embed = talker.codec_embedding.forward(&codec_pad_arr)?;
    prefill_embeds.push(tts_bos_projected.add(&codec_pad_embed)?);

    // Position 9: first_text text + codec_bos embedding
    let first_text_projected = talker.text_projection.forward(&first_text_embed)?;
    let codec_bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
    let codec_bos_embed = talker.codec_embedding.forward(&codec_bos_arr)?;
    prefill_embeds.push(first_text_projected.add(&codec_bos_embed)?);

    // Concatenate all prefill embeddings
    let concat_refs: Vec<&Array> = prefill_embeds.iter().collect();
    let input_embeds = Array::concatenate(&concat_refs, 1)?;
    let hidden_size = talker.hidden_size();

    // Batched forward pass through talker transformer
    let (logits, final_hidden) = talker.forward_batch(input_embeds)?;

    // logits: [1, prefill_len, vocab] — take last position
    let logits = logits.index((.., -1.., ..));

    // Build trailing text embeddings (for autoregressive loop)
    let trailing_embeds: Vec<Array> = trailing_text
        .iter()
        .map(|&id| {
            let arr = Array::from_slice(&[id as i32], &[1, 1]);
            let text_emb = talker.text_embedding.forward(&arr);
            talker.text_projection.forward(&text_emb)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let trailing_refs: Vec<&Array> = trailing_embeds.iter().collect();
    let trailing_text_embeds = Array::concatenate(&trailing_refs, 1)?;

    let t1 = Instant::now();
    let prefill_ms = (t1 - t0).as_secs_f64() * 1000.0;
    trace!("Prefill done in {:.1}ms", prefill_ms);

    // ========================================================================
    // Autoregressive generation loop
    // ========================================================================

    // EOS steering for speed control
    let eos_steering = if (speed - 1.0).abs() > 0.01 {
        let target = (text_token_ids.len() as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
        info!("Speed={:.2}x: EOS steering target={} frames", speed, target);
        Some((target, speed))
    } else {
        None
    };

    let loop_params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering,
    };

    let vocab_size = tts_config.talker_config.vocab_size as usize;
    let all_codes = run_generation_loop(
        talker,
        logits,
        final_hidden,
        &trailing_text_embeds,
        &tts_pad_embed,
        &loop_params,
        &mut rng_key,
        vocab_size,
    )?;

    let t2 = Instant::now();
    let generation_ms = (t2 - t1).as_secs_f64() * 1000.0;
    info!("Generated {} codec frames in {:.1}ms", all_codes.len(), generation_ms);

    Ok((all_codes, GenerationTiming {
        prefill_ms,
        generation_ms,
        generation_frames: all_codes.len(),
    }))
}

/// Voice clone generation: inject speaker embedding into the prefill.
pub fn generate_voice_clone(
    talker: &mut Talker,
    text_token_ids: &[u32],
    speaker_embedding: &[f32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    eos_token: u32,
    bos_id: u32,
    pad_id: u32,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let mut rng_key = seed.map(SamplingKey::new).transpose()?;
    let speed = gen_config.speed_factor;
    talker.set_rope_speed_factor(speed);

    // Build trailing text (same as CustomVoice)
    let max_trailing = gen_config.max_new_tokens as usize;
    let text_tail: Vec<u32> = text_token_ids[1..]
        .iter()
        .copied()
        .take(max_trailing)
        .collect();
    let trailing_len = text_tail.len();
    let trailing_width = (trailing_len + 5).max(trailing_len);
    let mut trailing_text: Vec<u32> = text_tail;
    trailing_text.push(tts_config.tts_eos_token_id);
    while trailing_text.len() < trailing_width {
        trailing_text.push(tts_config.tts_pad_token_id);
    }

    let t0 = Instant::now();

    // Prefill: no role tokens, just tts_pad text + speaker embedding + first_text
    // Build codec prefix for voice clone: nothink + think_bos + think_eos
    let codec_prefix = build_codec_prefix_nothink(&tts_config.talker_config);
    let no_codec_positions = 0; // No role tokens for voice clone

    // tts_pad_embed for padding
    let tts_pad_arr = Array::from_slice(&[tts_config.tts_pad_token_id as i32], &[1, 1]);
    let tts_pad_embed = talker.text_embedding.forward(&tts_pad_arr)?;

    // Codec prefix embeddings
    let codec_prefix_embeds: Vec<Array> = codec_prefix
        .iter()
        .map(|&id| {
            let arr = Array::from_slice(&[id as i32], &[1, 1]);
            talker.codec_embedding.forward(&arr)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // First text embedding
    let first_text_id = text_token_ids[0];
    let text_token_arr = Array::from_slice(&[first_text_id as i32], &[1, 1]);
    let first_text_embed = talker.text_embedding.forward(&text_token_arr)?;

    // Prefill: [3 codec tokens] + [tts_bos + codec_pad] + [first_text + codec_bos]
    let prefill_len = codec_prefix.len() + 2;
    let mut prefill_embeds: Vec<Array> = Vec::with_capacity(prefill_len);

    // Speaker embedding is already in the right shape — use as text projection output directly
    let spk_arr = Array::from_slice(speaker_embedding, &[1, 1, talker.hidden_size() as i32]);
    let spk_projected = talker.text_projection.forward(&tts_pad_embed)?;
    // Replace the text projection weights so the spk embedding dominates
    // Actually: we just concatenate spk embedding with codec_emb for those positions
    for cemb in &codec_prefix_embeds {
        let combined = spk_projected.add(cemb)?;
        prefill_embeds.push(combined);
    }

    // tts_bos text + codec_pad
    let tts_bos_arr = Array::from_slice(&[tts_config.tts_bos_token_id as i32], &[1, 1]);
    let text_bos_embed = talker.text_embedding.forward(&tts_bos_arr)?;
    let tts_bos_projected = talker.text_projection.forward(&text_bos_embed)?;
    let codec_pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
    let codec_pad_embed = talker.codec_embedding.forward(&codec_pad_arr)?;
    prefill_embeds.push(tts_bos_projected.add(&codec_pad_embed)?);

    // first_text + codec_bos
    let first_text_projected = talker.text_projection.forward(&first_text_embed)?;
    let codec_bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
    let codec_bos_embed = talker.codec_embedding.forward(&codec_bos_arr)?;
    prefill_embeds.push(first_text_projected.add(&codec_bos_embed)?);

    // Concatenate and forward
    let concat_refs: Vec<&Array> = prefill_embeds.iter().collect();
    let input_embeds = Array::concatenate(&concat_refs, 1)?;
    let (logits, final_hidden) = talker.forward_batch(input_embeds)?;
    let logits = logits.index((.., -1.., ..));

    // Build trailing text embeddings
    let trailing_embeds: Vec<Array> = trailing_text
        .iter()
        .map(|&id| {
            let arr = Array::from_slice(&[id as i32], &[1, 1]);
            let text_emb = talker.text_embedding.forward(&arr);
            talker.text_projection.forward(&text_emb)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let trailing_refs: Vec<&Array> = trailing_embeds.iter().collect();
    let trailing_text_embeds = Array::concatenate(&trailing_refs, 1)?;

    let t1 = Instant::now();
    let prefill_ms = (t1 - t0).as_secs_f64() * 1000.0;

    let eos_steering = if (speed - 1.0).abs() > 0.01 {
        let target = (text_token_ids.len() as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
        Some((target, speed))
    } else {
        None
    };

    let loop_params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering,
    };

    let vocab_size = tts_config.talker_config.vocab_size as usize;
    let all_codes = run_generation_loop(
        talker,
        logits,
        final_hidden,
        &trailing_text_embeds,
        &tts_pad_embed,
        &loop_params,
        &mut rng_key,
        vocab_size,
    )?;

    let t2 = Instant::now();
    let generation_ms = (t2 - t1).as_secs_f64() * 1000.0;

    Ok((all_codes, GenerationTiming {
        prefill_ms,
        generation_ms,
        generation_frames: all_codes.len(),
    }))
}

/// Voice clone + instruct generation.
pub fn generate_voice_clone_instruct(
    talker: &mut Talker,
    text_token_ids: &[u32],
    instruct_token_ids: &[u32],
    speaker_embedding: &[f32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    eos_token: u32,
    bos_id: u32,
    pad_id: u32,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    // Same as voice clone but appends instruct tokens to trailing text
    let mut full_text: Vec<u32> = text_token_ids.to_vec();
    full_text.extend_from_slice(instruct_token_ids);
    generate_voice_clone(
        talker,
        &full_text,
        speaker_embedding,
        gen_config,
        tts_config,
        eos_token,
        bos_id,
        pad_id,
        seed,
    )
}

/// Voice design generation (text-described voice).
pub fn generate_voice_design(
    talker: &mut Talker,
    text_token_ids: &[u32],
    voice_description_ids: &[u32],
    codec_prefix: &[u32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    eos_token: u32,
    bos_id: u32,
    pad_id: u32,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    // VoiceDesign: prepend voice description as trailing text before the actual text
    let mut full_text: Vec<u32> = voice_description_ids.to_vec();
    full_text.extend_from_slice(text_token_ids);
    generate(
        talker,
        &full_text,
        codec_prefix,
        gen_config,
        tts_config,
        seed,
    )
}

// ============================================================================
// Streaming generation state
// ============================================================================

/// State machine for incremental generation.
pub struct GenerationState<'a> {
    talker: &'a mut Talker,
    logits: Array,
    hidden: Array,
    trailing_text_embeds: Array,
    tts_pad_embed: Array,
    params: GenerationLoopParams,
    rng_key: Option<SamplingKey>,
    vocab_size: usize,
    step: usize,
    done: bool,
    all_codes: Vec<[u32; 16]>,
    mem_guard: Option<mlx_rs_core::memory::MemoryGuard>,
    penalty_mask: Option<RepetitionPenaltyMask>,
    eos_unit_mask: Option<Array>,
    suppress_mask: Array,
    eos_suppress_mask: Array,
    combined_mask: Array,
}

impl GenerationState<'_> {
    pub fn new(
        talker: &mut Talker,
        text_token_ids: &[u32],
        codec_prefix: &[u32],
        gen_config: &GenerationConfig,
        tts_config: &Qwen3TtsConfig,
        seed: Option<u64>,
    ) -> Result<GenerationState> {
        let eos_token = tts_config.talker_config.codec_eos_token_id;
        let pad_id = tts_config.talker_config.codec_pad_id;
        let bos_id = tts_config.talker_config.codec_bos_id;
        let speed = gen_config.speed_factor;

        let rng_key = seed
            .map(SamplingKey::new)
            .transpose()?;

        talker.set_rope_speed_factor(speed);

        // Build trailing text
        let max_trailing = gen_config.max_new_tokens as usize;
        let text_tail: Vec<u32> = text_token_ids[1..]
            .iter()
            .copied()
            .take(max_trailing)
            .collect();
        let trailing_len = text_tail.len();
        let trailing_width = (trailing_len + 5).max(trailing_len);
        let mut trailing_text: Vec<u32> = text_tail;
        trailing_text.push(tts_config.tts_eos_token_id);
        while trailing_text.len() < trailing_width {
            trailing_text.push(tts_config.tts_pad_token_id);
        }

        // Prefill
        let no_codec_positions = 3;
        let prefill_len = no_codec_positions + codec_prefix.len() + 2;

        let text_role_tokens: Vec<u32> = vec![
            tts_config.im_start_token_id,
            tts_config.assistant_token_id,
            tts_config.tts_bos_token_id,
        ];

        let codec_prefix_embeds: Vec<Array> = codec_prefix
            .iter()
            .map(|&id| {
                let arr = Array::from_slice(&[id as i32], &[1, 1]);
                talker.codec_embedding.forward(&arr)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let tts_pad_arr = Array::from_slice(&[tts_config.tts_pad_token_id as i32], &[1, 1]);
        let tts_pad_embed = talker.text_embedding.forward(&tts_pad_arr)?;

        let tts_bos_arr = Array::from_slice(&[tts_config.tts_bos_token_id as i32], &[1, 1]);
        let text_bos_embed = talker.text_embedding.forward(&tts_bos_arr)?;

        let first_text_id = text_token_ids[0];
        let text_token_arr = Array::from_slice(&[first_text_id as i32], &[1, 1]);
        let first_text_embed = talker.text_embedding.forward(&text_token_arr)?;

        let mut prefill_embeds: Vec<Array> = Vec::with_capacity(prefill_len);

        for &role_id in &text_role_tokens {
            let role_arr = Array::from_slice(&[role_id as i32], &[1, 1]);
            let text_emb = talker.text_embedding.forward(&role_arr)?;
            let projected = talker.text_projection.forward(&text_emb)?;
            prefill_embeds.push(projected);
        }

        for codec_emb in codec_prefix_embeds.iter() {
            let projected = talker.text_projection.forward(&tts_pad_embed)?;
            let combined = projected.add(codec_emb)?;
            prefill_embeds.push(combined);
        }

        let tts_bos_projected = talker.text_projection.forward(&text_bos_embed)?;
        let codec_pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
        let codec_pad_embed = talker.codec_embedding.forward(&codec_pad_arr)?;
        prefill_embeds.push(tts_bos_projected.add(&codec_pad_embed)?);

        let first_text_projected = talker.text_projection.forward(&first_text_embed)?;
        let codec_bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
        let codec_bos_embed = talker.codec_embedding.forward(&codec_bos_arr)?;
        prefill_embeds.push(first_text_projected.add(&codec_bos_embed)?);

        let concat_refs: Vec<&Array> = prefill_embeds.iter().collect();
        let input_embeds = Array::concatenate(&concat_refs, 1)?;
        let (logits, final_hidden) = talker.forward_batch(input_embeds)?;
        let logits = logits.index((.., -1.., ..));
        let hidden = final_hidden;

        // Build trailing text embeddings
        let trailing_embeds: Vec<Array> = trailing_text
            .iter()
            .map(|&id| {
                let arr = Array::from_slice(&[id as i32], &[1, 1]);
                let text_emb = talker.text_embedding.forward(&arr);
                talker.text_projection.forward(&text_emb)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let trailing_refs: Vec<&Array> = trailing_embeds.iter().collect();
        let trailing_text_embeds = Array::concatenate(&trailing_refs, 1)?;

        let vocab_size = tts_config.talker_config.vocab_size as usize;

        // Suppression masks
        let suppress_mask = build_suppression_mask(vocab_size, eos_token);
        let eos_suppress_mask = build_eos_suppression_mask(vocab_size, eos_token);
        let combined_mask = suppress_mask.add(&eos_suppress_mask)?;
        let penalty_mask = Some(RepetitionPenaltyMask::new(vocab_size, gen_config.repetition_penalty)?);

        let eos_steering = if (speed - 1.0).abs() > 0.01 {
            let target = (text_token_ids.len() as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
            Some((target, speed))
        } else {
            None
        };

        let eos_unit_mask = if eos_steering.is_some() {
            Some(build_eos_unit_mask(vocab_size, eos_token))
        } else {
            None
        };

        let params = GenerationLoopParams {
            trailing_len,
            eos_token,
            pad_id,
            temperature: gen_config.temperature,
            top_k: gen_config.top_k,
            top_p: gen_config.top_p,
            repetition_penalty: gen_config.repetition_penalty,
            max_new_tokens: gen_config.max_new_tokens as usize,
            eos_steering,
        };

        Ok(GenerationState {
            talker,
            logits,
            hidden,
            trailing_text_embeds,
            tts_pad_embed,
            params,
            rng_key,
            vocab_size,
            step: 0,
            done: false,
            all_codes: Vec::new(),
            mem_guard: Some(mlx_rs_core::memory::MemoryGuard::default_guard()),
            penalty_mask,
            eos_unit_mask,
            suppress_mask,
            eos_suppress_mask,
            combined_mask,
        })
    }

    /// Generate the next chunk of codec frames.
    pub fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<Vec<[u32; 16]>>> {
        if self.done {
            return Ok(None);
        }

        let min_new_tokens: usize = 2;
        let mut chunk: Vec<[u32; 16]> = Vec::with_capacity(chunk_size);

        for _ in 0..chunk_size {
            if self.step >= self.params.max_new_tokens {
                self.done = true;
                break;
            }

            let base_mask = if self.step < min_new_tokens {
                &self.combined_mask
            } else {
                &self.suppress_mask
            };

            let steered_mask;
            let effective_mask: &Array = if let Some((target, speed)) = self.params.eos_steering {
                if self.step >= min_new_tokens {
                    let bias = compute_eos_steering_bias(self.step, target, speed);
                    if bias.abs() > 0.01 {
                        let unit_mask = self.eos_unit_mask.as_ref().unwrap();
                        let bias_mask = unit_mask.multiply(mlx_rs::array!(bias))?;
                        steered_mask = base_mask.add(&bias_mask)?;
                        &steered_mask
                    } else {
                        base_mask
                    }
                } else {
                    base_mask
                }
            } else {
                base_mask
            };

            let token0 = sample_logits_with_mask(
                &self.logits,
                self.params.temperature,
                self.params.top_k,
                self.params.top_p,
                self.params.repetition_penalty,
                &[],
                self.rng_key.as_mut(),
                Some(effective_mask),
                self.penalty_mask.as_ref(),
            )?;

            if token0 == self.params.eos_token {
                self.done = true;
                break;
            }

            if let Some(ref mut pm) = self.penalty_mask {
                pm.record_token(token0)?;
            }

            // Generate sub-codes
            let hidden_slice = self.hidden.index((.., -1.., ..));
            let code0_arr = Array::from_slice(&[token0 as i32], &[1, 1]);
            let code0_embed = self.talker.codec_embedding.forward(&code0_arr)?;
            let sub_codes = self.talker.code_predictor.generate_codes(&hidden_slice, &code0_embed)?;

            let mut frame = [self.params.pad_id; 16];
            frame[0] = token0;
            for (g, &code) in sub_codes.iter().enumerate() {
                frame[g + 1] = code;
            }
            chunk.push(frame);
            self.all_codes.push(frame);

            // Build next input
            let text_embed_indexed;
            let text_embed: &Array = if self.step < self.params.trailing_len {
                let s = self.step as i32;
                text_embed_indexed = self.trailing_text_embeds.index((.., s..s + 1, ..));
                &text_embed_indexed
            } else {
                &self.tts_pad_embed
            };

            let input_embed = self.talker.build_generation_embedding_with_text(&frame, text_embed)?;
            let result = self.talker.forward_step(input_embed)?;
            self.logits = result.0;
            self.hidden = result.1;

            self.step += 1;

            if let Some(ref mut mg) = self.mem_guard {
                mg.step();
            }
        }

        if chunk.is_empty() {
            self.done = true;
            Ok(None)
        } else {
            Ok(Some(chunk))
        }
    }
}

fn trace(msg: impl std::fmt::Display) {
    if tracing::enabled!(tracing::Level::TRACE) {
        tracing::trace!("{}", msg);
    }
}
