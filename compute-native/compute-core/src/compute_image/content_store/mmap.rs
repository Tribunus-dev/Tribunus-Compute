//! Mmap loader — segment file mapping (stub, file read fallback).

use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedSegment {
    pub segment_id: String,
    pub file_name: String,
    pub mapped_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct MmapRegion {
    pub base_address: *const u8,
    pub length: u64,
    pub is_private: bool,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

#[derive(Debug, Clone)]
pub enum MmapLoadError {
    FileNotFound(String),
    PermissionDenied(String),
    InvalidAlignment(String),
}

impl std::fmt::Display for MmapLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapLoadError::FileNotFound(p) => write!(f, "file not found: {}", p),
            MmapLoadError::PermissionDenied(p) => write!(f, "permission denied: {}", p),
            MmapLoadError::InvalidAlignment(msg) => write!(f, "invalid alignment: {}", msg),
        }
    }
}

impl std::error::Error for MmapLoadError {}

pub struct MmapLoader;

impl MmapLoader {
    pub fn new() -> Self { Self }

    pub fn open_segment(path: &Path) -> Result<MappedSegment, MmapLoadError> {
        if !path.exists() {
            return Err(MmapLoadError::FileNotFound(path.display().to_string()));
        }
        let file_name = path.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let metadata = std::fs::metadata(path)
            .map_err(|_| MmapLoadError::PermissionDenied(path.display().to_string()))?;
        Ok(MappedSegment {
            segment_id: file_name.trim_end_matches(".bin").to_string(),
            file_name,
            mapped_bytes: metadata.len(),
        })
    }

    pub fn close_segment(_segment: MappedSegment) {}

    pub fn advise_random_access(_segment: &MappedSegment) {}
    pub fn advise_sequential(_segment: &MappedSegment) {}
    pub fn advise_will_need(_segment: &MappedSegment, _offset: u64, _length: u64) {}
}
