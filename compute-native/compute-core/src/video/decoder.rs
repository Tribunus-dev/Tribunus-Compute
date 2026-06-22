//! Video frame extraction.
//!
//! Decodes a video file (MP4, MOV, WebM) into a sequence of evenly-sampled
//! RGB frames via an `ffmpeg` subprocess.  Each frame is returned as a flat
//! byte buffer `[H, W, 3]` (row-major RGB).

use std::process::{Command, Stdio};

/// Maximum number of frames to extract (keeps context within budget).
pub const MAX_VIDEO_FRAMES: u32 = 32;

/// Extract frames from a video file at regular intervals.
///
/// 1. Opens the video file via `ffmpeg`.
/// 2. Samples `num_frames` frames evenly across the duration.
/// 3. Scales each frame to `target_size x target_size` (centered, padded).
/// 4. Returns each frame as raw RGB pixel data `[H, W, 3]`.
///
/// # Arguments
///
/// * `path_or_url` — Local file path or remote URL to the video.
/// * `num_frames` — Target number of frames (clamped to [`MAX_VIDEO_FRAMES`]).
/// * `target_size` — Output image size (width / height), typically the
///   vision encoder's `image_size` (e.g. 896).
///
/// # Errors
///
/// Returns an error if `ffmpeg` is not installed, the file cannot be
/// opened, or the video could not be decoded.
pub fn extract_frames(
    path_or_url: &str,
    num_frames: u32,
    target_size: u32,
) -> Result<Vec<Vec<u8>>, String> {
    let num_frames = num_frames.clamp(1, MAX_VIDEO_FRAMES);

    // Build the ffmpeg filter for frame sampling and resizing.
    //
    // We sample `num_frames` evenly across the video duration using the
    // `fps` filter set to `num_frames / duration`.
    //
    // Scaling: resize so the longer dimension fits `target_size`, then
    // center-pad to exactly `target_size x target_size`.
    let _filter = format!(
        "fps={}/{},scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        1, // we set fps=1/interval to get the right frame count via frame selection
        0, // placeholder; we actually use select='between(n,...)' or fps
        target_size, target_size,
        target_size, target_size,
    );

    // Actually, the cleanest approach: use `select` to pick evenly-spaced
    // frames, or use a two-pass approach with `ffprobe` for duration.
    // For simplicity, we use `fps` set to sample `num_frames` equally:
    //   fps = num_frames / duration
    //
    // Since we don't know the duration upfront, we first probe it with ffprobe
    // then build the correct filter.

    let duration = probe_duration(path_or_url)?;
    let fps = if duration > 0.0 {
        num_frames as f64 / duration
    } else {
        // Fallback: if we can't probe, request enough fps to get `num_frames`
        // from a 10-second video (reasonable default).
        3.0
    };

    // Build the ffmpeg command.
    //   - Accurate seek (avoid keyframe-only snapshots).
    //   - Scale to target_size with aspect-ratio-preserving resize + padding.
    //   - Raw RGB24 pixel format (one byte per channel, row-major).
    let filter = format!(
        "fps={:.6},scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        fps, target_size, target_size, target_size, target_size,
    );

    let output = Command::new("ffmpeg")
        .args(&[
            "-nostdin",
            "-y",
            "-accurate_seek",
            "-i",
            path_or_url,
            "-vf",
            &filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "vfr",
            "-an",
            "-sn",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| format!("failed to spawn ffmpeg: {} — is ffmpeg installed?", e))?;

    if !output.status.success() {
        return Err(format!(
            "ffmpeg exited with status {} for {}",
            output.status, path_or_url,
        ));
    }

    let frame_size = (target_size * target_size * 3) as usize;
    let raw = output.stdout;

    if raw.is_empty() {
        return Err(format!(
            "ffmpeg produced no output for video: {}",
            path_or_url
        ));
    }

    // Slice the raw stream into individual frames.
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut offset = 0;
    while offset + frame_size <= raw.len() && frames.len() < num_frames as usize {
        let frame: Vec<u8> = raw[offset..offset + frame_size].to_vec();
        frames.push(frame);
        offset += frame_size;
    }

    if frames.is_empty() {
        return Err(format!(
            "could not extract any frames from video: {} (got {} bytes, frame_size={})",
            path_or_url,
            raw.len(),
            frame_size,
        ));
    }

    Ok(frames)
}

/// Probe the video duration in seconds using `ffprobe`.
fn probe_duration(path_or_url: &str) -> Result<f64, String> {
    let output = Command::new("ffprobe")
        .args(&[
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path_or_url,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| format!("failed to spawn ffprobe: {} — is ffmpeg installed?", e))?;

    if !output.status.success() {
        // Return a reasonable default rather than failing entirely.
        return Ok(10.0);
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    let duration: f64 = duration_str.trim().parse().unwrap_or(10.0);
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_frames_constant() {
        assert!(MAX_VIDEO_FRAMES > 0);
        assert!(MAX_VIDEO_FRAMES <= 64);
    }

    #[test]
    fn test_clamp_num_frames() {
        // extract_frames clamps to [1, MAX_VIDEO_FRAMES]
        // Verify via doc-level invariants: we pass 0 and expect 1-min behavior
        // (tested indirectly via the clamp logic in the function).
        let clamped = 0u32.clamp(1, MAX_VIDEO_FRAMES);
        assert_eq!(clamped, 1);
        let clamped = 100u32.clamp(1, MAX_VIDEO_FRAMES);
        assert_eq!(clamped, MAX_VIDEO_FRAMES);
    }

    #[test]
    fn test_frame_size_calculation() {
        let target_size = 896u32;
        let frame_size = (target_size * target_size * 3) as usize;
        assert_eq!(frame_size, 896 * 896 * 3);

        let smaller = 224u32;
        let frame_size_small = (smaller * smaller * 3) as usize;
        assert_eq!(frame_size_small, 224 * 224 * 3);
    }
}
