//! Multi-modal audio support for Gemma 4 Unified.
//!
//! Provides audio preprocessing (mel spectrogram extraction),
//! audio encoder (conformer/transformer-based), and feature injection
//! into the text model's hidden state.

pub mod encoder;
pub mod preprocess;

pub use encoder::AudioEncoder;
pub use preprocess::preprocess_audio;

use mlx_rs::ops;
use mlx_rs::Array;

/// Inject audio features into the text hidden state by concatenating
/// audio feature tokens at the start of the sequence.
///
/// `hidden` — text hidden state, shape `[text_tokens, hidden_size]`.
/// `audio_features` — audio encoder output, shape `[num_frames, projection_dim]`.
///
/// Returns combined hidden state of shape `[text_tokens + num_frames, hidden_size]`.
pub fn inject_audio_features(hidden: &Array, audio_features: &Array) -> Result<Array, String> {
    // Ensure audio features are projected to the text model's hidden dimension.
    // If projection_dim != hidden_size, the audio encoder's output_proj handles
    // the projection, so dimensions should already match.
    let audio_ndim = audio_features.ndim();
    if audio_ndim != 2 {
        return Err(format!(
            "audio_features must be rank 2 [num_frames, hidden_size], got rank {}",
            audio_ndim
        ));
    }

    // Concatenate audio features before text tokens so cross-attention can
    // attend to audio context.
    ops::concatenate(&[audio_features, hidden])
        .map_err(|e| format!("concatenate audio features: {:?}", e))
}
