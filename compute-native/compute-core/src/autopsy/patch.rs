//! Segment patch writer — applies corrected weights to ComputeImage segments
//! without full recompilation.
//!
//! A [`SegmentPatch`] identifies the segment file, the tensor within it, and
//! the corrected bytes. Applying the patch:
//! 1. Backs up the original segment
//! 2. Replaces the tensor bytes in-place
//! 3. Recomputes the segment SHA-256
//! 4. Updates manifest.json
//! 5. Optionally updates receipt.json

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::autopsy::replay::ReplayResult;
use crate::compute_image::Manifest;

/// A patch to a single segment in a ComputeImage.
/// Applied post-hoc without full recompilation.
#[derive(Debug, Clone)]
pub struct SegmentPatch {
    /// Which segment to patch
    pub segment_filename: String,
    /// Which tensor within the segment to replace
    pub tensor_name: String,
    /// The corrected bytes for this tensor
    pub corrected_bytes: Vec<u8>,
    /// New SHA-256 of the corrected tensor
    pub new_sha256: String,
    /// Why this patch was applied
    pub reason: String,
}

/// Extension appended to the original segment file for backup.
const BACKUP_EXTENSION: &str = ".bak";

impl SegmentPatch {
    /// Create a patch from a replay result that revealed an issue.
    pub fn from_replay(replay: &ReplayResult) -> Self {
        let tensor_name = replay
            .tensor_name
            .clone()
            .unwrap_or_else(|| replay.weight_name.clone());

        // The corrected bytes are a no-op placeholder when the replay didn't
        // actually modify the weights. In production, the reference matmul
        // would produce corrected weight bytes stored here.
        let corrected_bytes = Vec::new();

        Self {
            segment_filename: replay.segment.clone(),
            tensor_name,
            corrected_bytes,
            new_sha256: replay.computed_hash.clone(),
            reason: format!(
                "hash mismatch: expected {}, computed {} (MSE={:.6})",
                replay.original_hash, replay.computed_hash, replay.reference_mse
            ),
        }
    }

    /// Set the corrected bytes for this patch.
    pub fn with_corrected_bytes(mut self, bytes: Vec<u8>) -> Self {
        // Recompute the SHA-256 over the corrected bytes.
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        self.new_sha256 = format!("{:x}", hasher.finalize());
        self.corrected_bytes = bytes;
        self
    }

    /// Apply the patch to the ComputeImage directory.
    ///
    /// 1. Read the segment file
    /// 2. Find and replace the tensor's bytes
    /// 3. Recompute the segment SHA-256
    /// 4. Update manifest.json with new hash
    /// 5. Write the patched segment
    pub fn apply(&self, image_dir: &Path) -> Result<(), String> {
        let segment_path = image_dir.join(&self.segment_filename);
        let manifest_path = image_dir.join("manifest.json");

        // Read the manifest
        let manifest_json =
            std::fs::read_to_string(&manifest_path).map_err(|e| format!("read manifest: {}", e))?;
        let mut manifest: Manifest =
            serde_json::from_str(&manifest_json).map_err(|e| format!("parse manifest: {}", e))?;

        // Find the tensor entry in the manifest
        let tensor = manifest
            .tensor_table
            .iter()
            .find(|t| t.name == self.tensor_name)
            .ok_or_else(|| format!("tensor {} not found in manifest", self.tensor_name))?
            .clone();

        // Read the segment file
        let mut segment_data =
            std::fs::read(&segment_path).map_err(|e| format!("read segment: {}", e))?;

        // Validate tensor bounds
        let offset = tensor.offset as usize;
        let byte_len = tensor.byte_length as usize;
        if offset + byte_len > segment_data.len() {
            return Err(format!(
                "tensor {} at offset {} + byte_len {} exceeds segment size {}",
                self.tensor_name,
                offset,
                byte_len,
                segment_data.len()
            ));
        }

        // Determine the correct bytes to write
        let write_bytes = if self.corrected_bytes.is_empty() {
            // No explicit corrected bytes provided — the tensor is already in
            // the segment and we are just re-verifying.
            segment_data[offset..offset + byte_len].to_vec()
        } else {
            if self.corrected_bytes.len() != byte_len {
                return Err(format!(
                    "corrected_bytes length {} does not match tensor byte_length {}",
                    self.corrected_bytes.len(),
                    byte_len
                ));
            }
            self.corrected_bytes.clone()
        };

        // Create backup of the original segment (only if not already backed up)
        let backup_path = backup_path_for(&segment_path);
        if !backup_path.exists() {
            std::fs::copy(&segment_path, &backup_path)
                .map_err(|e| format!("backup segment: {}", e))?;
        }

        // Replace the tensor bytes in the segment
        segment_data[offset..offset + byte_len].copy_from_slice(&write_bytes);

        // Recompute the segment SHA-256
        let mut hasher = Sha256::new();
        hasher.update(&segment_data);
        let new_segment_hash = format!("{:x}", hasher.finalize());

        // Write the patched segment
        std::fs::write(&segment_path, &segment_data)
            .map_err(|e| format!("write segment: {}", e))?;

        // Update the manifest with the new segment hash
        for seg in &mut manifest.segments {
            if seg.filename == self.segment_filename {
                seg.sha256 = new_segment_hash.clone();
                break;
            }
        }

        // Write the updated manifest
        let updated_manifest = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("serialize manifest: {}", e))?;
        std::fs::write(&manifest_path, &updated_manifest)
            .map_err(|e| format!("write manifest: {}", e))?;

        Ok(())
    }

    /// Apply the patch and also update the receipt.json by bumping its
    /// patch counter.
    pub fn apply_with_receipt_update(&self, image_dir: &Path) -> Result<(), String> {
        self.apply(image_dir)?;

        let receipt_path = image_dir.join("receipt.json");
        if receipt_path.exists() {
            let receipt_text = std::fs::read_to_string(&receipt_path)
                .map_err(|e| format!("read receipt: {}", e))?;
            let mut receipt: serde_json::Value =
                serde_json::from_str(&receipt_text).map_err(|e| format!("parse receipt: {}", e))?;

            // Bump a patch counter, starting at 0 if missing.
            let patch_count = receipt["patch_count"].as_u64().unwrap_or(0);
            receipt["patch_count"] = serde_json::Value::Number((patch_count + 1).into());
            receipt["last_patch"] = serde_json::Value::String(format!(
                "{} @ {}",
                self.tensor_name, self.segment_filename
            ));
            receipt["last_patch_sha256"] = serde_json::Value::String(self.new_sha256.clone());

            std::fs::write(
                &receipt_path,
                serde_json::to_string_pretty(&receipt)
                    .map_err(|e| format!("serialize receipt: {}", e))?,
            )
            .map_err(|e| format!("write receipt: {}", e))?;
        }

        Ok(())
    }

    /// Rollback a patch (restore from the automatic backup).
    pub fn rollback(image_dir: &Path) -> Result<(), String> {
        // Discover segment files that have .bak backups
        let dir_entries =
            std::fs::read_dir(image_dir).map_err(|e| format!("read image dir: {}", e))?;

        let mut restored_any = false;
        for entry in dir_entries {
            let entry = entry.map_err(|e| format!("dir entry: {}", e))?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "bak") {
                let original_path = path.with_extension("bin");
                // Also check for .bin.bak pattern
                let original_path_alt = {
                    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    image_dir.join(name)
                };

                // Try the simple extension swap first
                if original_path.exists() {
                    std::fs::copy(&path, &original_path)
                        .map_err(|e| format!("restore {}: {}", original_path.display(), e))?;
                    std::fs::remove_file(&path)
                        .map_err(|e| format!("remove backup {}: {}", path.display(), e))?;
                    restored_any = true;
                } else if original_path_alt.exists() {
                    // Handle .bin.bak -> .bin
                    let target = path.with_extension("");
                    std::fs::copy(&path, &target)
                        .map_err(|e| format!("restore {}: {}", target.display(), e))?;
                    std::fs::remove_file(&path)
                        .map_err(|e| format!("remove backup {}: {}", path.display(), e))?;
                    restored_any = true;
                }
            }
        }

        if !restored_any {
            return Err("no backup files found to restore".to_string());
        }

        Ok(())
    }
}

/// Compute the backup path for a segment file.
fn backup_path_for(segment_path: &Path) -> PathBuf {
    let mut name = segment_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("segment")
        .to_string();
    name.push_str(BACKUP_EXTENSION);
    segment_path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_path_for_bin() {
        let path = Path::new("/tmp/image/segment_000.bin");
        let backup = backup_path_for(path);
        assert_eq!(backup, Path::new("/tmp/image/segment_000.bin.bak"));
    }
}
