//! Audio preprocessing — load, resample, mel spectrogram extraction.
//!
//! Converts audio files (WAV) into normalized mel spectrograms
//! suitable for the audio encoder. Supports local WAV files and
//! URL-based audio downloads.

use crate::config::AudioArchitecture;
use mlx_rs::Array;
use std::io::Read;
use std::path::Path;

/// Load and preprocess an audio file for the audio encoder.
///
/// 1. Load audio from path/URL (WAV format)
/// 2. Resample to model's expected sample_rate
/// 3. Convert to mel spectrogram `[1, num_mel_bins, num_frames]`
/// 4. Normalize
pub fn preprocess_audio(path_or_url: &str, config: &AudioArchitecture) -> Result<Array, String> {
    let samples = load_audio(path_or_url, config.sample_rate)?;
    let mel_spec = compute_mel_spectrogram(&samples, config)?;
    Ok(mel_spec)
}

/// Load audio from a local file or URL, resampling to `target_sample_rate`.
fn load_audio(path_or_url: &str, target_sample_rate: u32) -> Result<Vec<f32>, String> {
    if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
        load_audio_from_url(path_or_url, target_sample_rate)
    } else {
        load_audio_from_file(path_or_url, target_sample_rate)
    }
}

/// Load audio from a local file path.
fn load_audio_from_file(path: &str, target_sample_rate: u32) -> Result<Vec<f32>, String> {
    let path = Path::new(path);
    if !path.exists() {
        return Err(format!("Audio file not found: {}", path.display()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav")
        .to_lowercase();

    match ext.as_str() {
        "wav" => load_wav(path, target_sample_rate),
        other => Err(format!(
            "Unsupported audio format: {} (only WAV supported)",
            other
        )),
    }
}

/// Load audio from a URL using std-only HTTP (download via temp file).
fn load_audio_from_url(url: &str, target_sample_rate: u32) -> Result<Vec<f32>, String> {
    // Use a subprocess curl to download since ureq is not available.
    let output = std::process::Command::new("curl")
        .args(["-s", "-L", url])
        .output()
        .map_err(|e| format!("Failed to execute curl: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "curl failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let body = output.stdout;
    let temp_dir = std::env::temp_dir();
    let filename = format!(
        "tribunus_audio_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    let ext = if url.to_lowercase().ends_with(".mp3") {
        "mp3"
    } else if url.to_lowercase().ends_with(".flac") {
        "flac"
    } else {
        "wav"
    };
    let temp_path = temp_dir.join(&filename).with_extension(ext);

    std::fs::write(&temp_path, &body)
        .map_err(|e| format!("Failed to write temp audio file: {}", e))?;

    let result = load_audio_from_file(temp_path.to_str().unwrap_or(""), target_sample_rate);

    let _ = std::fs::remove_file(&temp_path);
    result
}

/// Parse a WAV file header and return PCM F32 mono samples.
///
/// Supports: PCM 8/16/24/32-bit integer, IEEE float 32-bit, mono/stereo.
fn load_wav(path: &Path, target_sample_rate: u32) -> Result<Vec<f32>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read WAV: {}", e))?;

    if buf.len() < 44 {
        return Err("WAV file too short (no header)".into());
    }

    // Parse RIFF header.
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    let mut channels: u16 = 1;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 16;
    let mut data_start: usize = 44;
    let mut data_size: usize = 0;
    let mut fmt_found = false;

    // Parse chunks.
    let mut offset: usize = 12;
    while offset + 8 <= buf.len() {
        let chunk_id: [u8; 4] = buf[offset..offset + 4].try_into().unwrap();
        let chunk_size =
            u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        match &chunk_id {
            b"fmt " => {
                if offset + 16 > buf.len() {
                    return Err("WAV fmt chunk truncated".into());
                }
                let audio_format = u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap());
                channels = u16::from_le_bytes(buf[offset + 2..offset + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(buf[offset + 14..offset + 16].try_into().unwrap());
                // audio_format 1 = PCM, 3 = IEEE float
                if audio_format != 1 && audio_format != 3 {
                    return Err(format!(
                        "Unsupported WAV format code: {} (only PCM/float)",
                        audio_format
                    ));
                }
                fmt_found = true;
            }
            b"data" => {
                data_start = offset;
                data_size = chunk_size.min(buf.len() - offset);
                break; // data chunk is typically last
            }
            _ => {}
        }
        offset += chunk_size;
        // Chunks must be word-aligned.
        if chunk_size % 2 != 0 {
            offset += 1;
        }
    }

    if !fmt_found {
        return Err("WAV fmt chunk not found".into());
    }
    if data_size == 0 {
        return Err("WAV data chunk not found or empty".into());
    }

    let byte_depth = (bits_per_sample / 8) as usize;
    let frame_size = byte_depth * channels as usize;
    let num_frames = data_size / frame_size;
    let data = &buf[data_start..data_start + num_frames * frame_size];

    // Convert to F32 mono.
    let mut samples = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let frame_offset = frame_idx * frame_size;
        let mut frame_sum: f32 = 0.0;

        for ch in 0..channels as usize {
            let ch_offset = frame_offset + ch * byte_depth;
            let sample_f32 = match bits_per_sample {
                8 => {
                    // 8-bit WAV is unsigned.
                    (data[ch_offset] as f32 - 128.0) / 128.0
                }
                16 => {
                    let val =
                        i16::from_le_bytes(data[ch_offset..ch_offset + 2].try_into().unwrap());
                    val as f32 / i16::MAX as f32
                }
                24 => {
                    let mut bytes = [0u8; 4];
                    bytes[..3].copy_from_slice(&data[ch_offset..ch_offset + 3]);
                    let val = if bytes[2] & 0x80 != 0 {
                        // Sign extension
                        bytes[3] = 0xFF;
                        i32::from_le_bytes(bytes)
                    } else {
                        i32::from_le_bytes(bytes)
                    };
                    val as f32 / 8388607.0f32
                }
                32 => {
                    if byte_depth == 4 {
                        // IEEE float or 32-bit int
                        let val =
                            i32::from_le_bytes(data[ch_offset..ch_offset + 4].try_into().unwrap());
                        val as f32 / i32::MAX as f32
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            frame_sum += sample_f32;
        }

        // Average to mono.
        samples.push(frame_sum / channels as f32);
    }

    // Resample if needed.
    if sample_rate != target_sample_rate {
        Ok(resample(&samples, sample_rate, target_sample_rate))
    } else {
        Ok(samples)
    }
}

/// Simple linear interpolation resampler.
fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        if src_idx + 1 < input.len() {
            let sample = input[src_idx] as f64 * (1.0 - frac) + input[src_idx + 1] as f64 * frac;
            output.push(sample as f32);
        } else {
            output.push(*input.last().unwrap_or(&0.0));
        }
    }

    output
}

/// Compute a mel spectrogram from raw PCM samples.
///
/// Returns `[1, num_mel_bins, num_frames]`.
fn compute_mel_spectrogram(samples: &[f32], config: &AudioArchitecture) -> Result<Array, String> {
    let n_fft: usize = 400; // ~25ms window at 16kHz
    let hop_length = config.hop_length as usize;
    let num_mel_bins = config.num_mel_bins as usize;

    // Truncate or pad to max audio length.
    let max_samples = (config.max_audio_length_s as u64 * config.sample_rate as u64) as usize;
    let samples = if samples.len() > max_samples {
        &samples[..max_samples]
    } else {
        samples
    };

    // Hann window.
    let window: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n_fft - 1) as f64).cos()))
        .collect();

    // Compute STFT frames.
    let num_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        1
    };

    let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];

    // Build mel filterbank: [num_mel_bins, n_fft/2 + 1].
    let mel_filterbank = mel_filterbank(n_fft, num_mel_bins, config.sample_rate);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        // Apply window and compute FFT magnitudes.
        let mut fft_bins = vec![0.0f64; n_fft];
        for i in 0..n_fft {
            if start + i < samples.len() {
                fft_bins[i] = (samples[start + i] as f64) * window[i];
            }
        }

        // Compute power spectrum via DFT.
        let half_n = n_fft / 2;
        let mut power_spec = vec![0.0f64; half_n + 1];
        for k in 0..=half_n {
            let mut sum_real = 0.0f64;
            let mut sum_imag = 0.0f64;
            for n in 0..n_fft {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                sum_real += fft_bins[n] * angle.cos();
                sum_imag += fft_bins[n] * angle.sin();
            }
            power_spec[k] = sum_real * sum_real + sum_imag * sum_imag;
        }

        // Apply mel filterbank.
        for mel_idx in 0..num_mel_bins {
            let mut mel_energy = 0.0f64;
            for fft_idx in 0..=half_n {
                mel_energy += power_spec[fft_idx] * mel_filterbank[mel_idx][fft_idx];
            }
            // Log compression and normalize.
            let log_val = (mel_energy.max(1e-10).ln() as f32 - 5.0) / 5.0;
            mel_spec[mel_idx * num_frames + frame_idx] = log_val.max(-10.0).min(10.0) / 10.0;
        }
    }

    // Reshape to [1, num_mel_bins, num_frames].
    Ok(Array::from_slice(
        &mel_spec,
        &[1, num_mel_bins as i32, num_frames as i32],
    ))
}

/// Build a mel filterbank matrix: `[num_mel_bins, n_fft / 2 + 1]`.
fn mel_filterbank(n_fft: usize, num_mel_bins: usize, sample_rate: u32) -> Vec<Vec<f64>> {
    let low_freq_mel = 0.0;
    let high_freq_mel = hz_to_mel(sample_rate as f64 / 2.0);
    let mel_step = (high_freq_mel - low_freq_mel) / (num_mel_bins + 1) as f64;

    let mel_points: Vec<f64> = (0..num_mel_bins + 2)
        .map(|i| mel_to_hz(low_freq_mel + i as f64 * mel_step))
        .collect();

    let fft_bins = n_fft / 2 + 1;
    let mut filterbank = vec![vec![0.0f64; fft_bins]; num_mel_bins];

    for mel_idx in 0..num_mel_bins {
        let left = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let right = mel_points[mel_idx + 2];

        for fft_idx in 0..fft_bins {
            let freq = fft_idx as f64 * sample_rate as f64 / n_fft as f64;
            if freq >= left && freq <= center {
                filterbank[mel_idx][fft_idx] = (freq - left) / (center - left);
            } else if freq >= center && freq <= right {
                filterbank[mel_idx][fft_idx] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}
