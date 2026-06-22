//! Integrity verification — content hash computation and consistency checks.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct IntegrityVerifier;

impl IntegrityVerifier {
    pub fn new() -> Self { Self }

    pub fn compute_content_hash(data: &[u8]) -> ContentHash {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        ContentHash(hasher.finish())
    }

    pub fn verify_object(data: &[u8], expected_hash: &ContentHash) -> bool {
        &Self::compute_content_hash(data) == expected_hash
    }

    pub fn combine_hashes(hashes: &[ContentHash]) -> ContentHash {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for h in hashes {
            h.0.hash(&mut hasher);
        }
        ContentHash(hasher.finish())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityResult {
    pub object_id: String,
    pub verified: bool,
    pub expected_hash: ContentHash,
    pub computed_hash: ContentHash,
}

#[derive(Debug, Clone)]
pub struct IntegrityRecord {
    pub object_id: String,
    pub checksum: ContentHash,
    pub verified_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_and_verify() {
        let data = b"hello world";
        let hash = IntegrityVerifier::compute_content_hash(data);
        assert!(IntegrityVerifier::verify_object(data, &hash));
    }

    #[test]
    fn test_mismatch_detected() {
        let data = b"hello world";
        let wrong = ContentHash(0xDEAD);
        assert!(!IntegrityVerifier::verify_object(data, &wrong));
    }

    #[test]
    fn test_combine_hashes() {
        let h1 = ContentHash(1);
        let h2 = ContentHash(2);
        let combined = IntegrityVerifier::combine_hashes(&[h1, h2]);
        let combined2 = IntegrityVerifier::combine_hashes(&[h1, h2]);
        assert_eq!(combined, combined2);
    }
}
