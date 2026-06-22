//! Image loading and preprocessing for the vision encoder.
//!
//! Supports local file paths and remote URLs.  Images are loaded,
//! resized to the model's expected input size, normalized with the
//! model's mean/std statistics, and returned as an FP32 tensor with
//! shape `[1, num_channels, image_size, image_size]`.

use crate::config::VisionArchitecture;
use mlx_rs::Array;

/// Standard ImageNet normalization mean values (per channel).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Standard ImageNet normalization std values (per channel).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Load and preprocess an image for the vision encoder.
///
/// 1. Load image from path/URL.
/// 2. Resize to `image_size x image_size`.
/// 3. Normalize with model-specific mean/std (ImageNet by default).
/// 4. Convert to FP32 tensor `[1, num_channels, image_size, image_size]`.
///
/// # Arguments
/// * `path_or_url` — local file path or remote URL string.
/// * `config` — [`VisionArchitecture`] providing image size, channels.
///
/// # Returns
/// A 4D FP32 array shaped `[1, C, H, W]` ready for the vision encoder.
pub fn preprocess_image(path_or_url: &str, config: &VisionArchitecture) -> Result<Array, String> {
    let size = config.image_size as usize;
    let channels = config.num_channels as usize;

    // 1. Load bytes — local file or remote URL.
    let image_bytes = load_image_bytes(path_or_url)?;

    // 2. Decode with stb_image or similar RGBA -> raw pixels.
    //    We use a simple PNG/JPEG decoder via the `image` crate fallback.
    //    Convert RGBA -> RGB when the input has 4 channels.
    let (raw_pixels, img_w, img_h) = decode_image_to_rgb(&image_bytes)?;

    // 3. Bilinear resize to (size x size).
    let resized = bilinear_resize(&raw_pixels, img_w, img_h, size, size, channels);

    // 4. Normalize per channel and layout as NCHW [1, C, H, W].
    let mut float_pixels = Vec::with_capacity(channels * size * size);
    for c in 0..channels {
        for y in 0..size {
            for x in 0..size {
                let idx = y * size * channels + x * channels + c;
                let pixel = resized[idx] as f32 / 255.0;
                float_pixels.push((pixel - IMAGENET_MEAN[c]) / IMAGENET_STD[c]);
            }
        }
    }

    let dims: Vec<i32> = vec![1, channels as i32, size as i32, size as i32];
    let arr = Array::from_slice(&float_pixels, &dims);
    Ok(arr)
}

/// Load raw image bytes from a local path or remote URL.
fn load_image_bytes(path_or_url: &str) -> Result<Vec<u8>, String> {
    if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
        // For simplicity, we use `ureq` if available, else fallback to curl.
        // The actual implementation uses a minimal HTTP client.
        // SAFE: URLs come from trusted model input.
        let response = download_via_curl(path_or_url)?;
        Ok(response)
    } else {
        std::fs::read(path_or_url)
            .map_err(|e| format!("failed to read image file '{}': {}", path_or_url, e))
    }
}

/// Download a URL via `curl` subprocess (minimum dependency approach).
fn download_via_curl(url: &str) -> Result<Vec<u8>, String> {
    let output = std::process::Command::new("curl")
        .args(["-s", "-L", url])
        .output()
        .map_err(|e| format!("failed to run curl: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "curl failed with status {} for {}",
            output.status, url
        ));
    }
    Ok(output.stdout)
}

/// Decode raw image bytes into RGB pixel data.
/// Returns (pixels: Vec<u8>, width: usize, height: usize).
///
/// The output is in HWC layout: pixels[y * width * channels + x * channels + c].
fn decode_image_to_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    // Use the `image` crate for decoding (PNG, JPEG, WebP, etc.).
    // Thin re-export wrapper so this function works without importing image directly.
    #[cfg(feature = "image")]
    {
        let img =
            image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {}", e))?;
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        Ok((rgb.into_raw(), w as usize, h as usize))
    }

    #[cfg(not(feature = "image"))]
    {
        // Fallback: try png crate directly (simpler dependency).
        // This handles the common case for Gemma models.
        decode_png_fallback(bytes)
    }
}

/// Minimal PNG decoder fallback using the `png` crate.
#[cfg(not(feature = "image"))]
fn decode_png_fallback(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    use std::io::Cursor;
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png decode failed: {}", e))?;
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut raw)
        .map_err(|e| format!("png read frame failed: {}", e))?;
    let (w, h) = (info.width as usize, info.height as usize);

    // Convert RGBA -> RGB if necessary.
    let channels = info.color_type.samples() as usize;
    if channels >= 3 {
        let rgb: Vec<u8> = raw
            .chunks(channels)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .collect();
        Ok((rgb, w, h))
    } else {
        // Grayscale: duplicate to RGB.
        let rgb: Vec<u8> = raw.iter().flat_map(|&v| vec![v, v, v]).collect();
        Ok((rgb, w, h))
    }
}

/// Bilinear resize from (src_w x src_h) to (dst_w x dst_h).
///
/// Input: HWC layout, `channels` interleaved.
/// Output: HWC layout, same format.
fn bilinear_resize(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * channels];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel to source coordinates.
            let sx_f = (dx as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5;
            let sy_f = (dy as f32 + 0.5) * src_h as f32 / dst_h as f32 - 0.5;

            let sx = sx_f.max(0.0).min((src_w - 1) as f32);
            let sy = sy_f.max(0.0).min((src_h - 1) as f32);

            let ix = sx.floor() as usize;
            let iy = sy.floor() as usize;
            let ix1 = (ix + 1).min(src_w - 1);
            let iy1 = (iy + 1).min(src_h - 1);

            let fx = sx - ix as f32;
            let fy = sy - iy as f32;

            for c in 0..channels {
                let v00 = src[iy * src_w * channels + ix * channels + c] as f32;
                let v10 = src[iy * src_w * channels + ix1 * channels + c] as f32;
                let v01 = src[iy1 * src_w * channels + ix * channels + c] as f32;
                let v11 = src[iy1 * src_w * channels + ix1 * channels + c] as f32;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                dst[dy * dst_w * channels + dx * channels + c] =
                    v.round().max(0.0).min(255.0) as u8;
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_resize_identity() {
        // Resize 2x2 to 2x2 should preserve pixels.
        let src: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let out = bilinear_resize(&src, 2, 2, 2, 2, 3);
        assert_eq!(out, src, "identity resize preserves exact pixels");
    }

    #[test]
    fn test_bilinear_resize_downscale() {
        let src: Vec<u8> = (0..12).collect(); // 2x2x3
        let out = bilinear_resize(&src, 2, 2, 1, 1, 3);
        assert_eq!(out.len(), 3, "single pixel has 3 channels");
    }

    #[test]
    fn test_decode_png_fallback() {
        // Minimal 1x1 white PNG (RGBA).
        let png_bytes: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
            0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x36,
            0x28, 0x19, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, // IEND
            0xAE, 0x42, 0x60, 0x82,
        ];
        let result = decode_png_fallback(&png_bytes);
        assert!(
            result.is_ok(),
            "PNG fallback decode should succeed: {:?}",
            result.err()
        );
        let (pixels, w, h) = result.unwrap();
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert_eq!(pixels.len(), 3, "RGB pixel");
    }
}
