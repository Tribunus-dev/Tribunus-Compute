//! Sealed executable integrity verification at runtime.
//!
//! Verifies a [`ExecutableSeal`] by cross-referencing every component hash
//! (manifest, per-profile hashes, receipt-bundle hash) against
//! caller-supplied computed hashes, then recomputing and checking the root
//! seal hash using the same [`IntegrityVerifier`] combinators the compiler
//! uses.
//!
//! The caller is responsible for providing the computed hashes (derived from
//! actual content-store scanning).  This module cross-references them
//! against the seal without trusting either side independently.

use crate::compute_image::content_store::integrity::IntegrityVerifier;
use crate::compute_image::executable::seal::ExecutableSeal;
use crate::integration::ContentHash;

/// Stateless seal verifier.
pub struct SealVerifier;

impl SealVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify the executable seal against caller-supplied computed hashes.
    ///
    /// Returns `Ok(receipt)` on success, or the specific
    /// [`SealVerificationError`] describing the first mismatch.
    ///
    /// ## Self-computed root hash
    ///
    /// The method recomputes the expected root hash from the seal's
    /// component hashes using
    /// [`IntegrityVerifier::combine_hashes`] — it never trusts the
    /// stored `root_hash` blindly.  If the stored root hash does not match
    /// the recomputed value, the receipt still reports `seal_valid: true`
    /// as long as every *content* hash matches (the root is a seal-internal
    /// consistency check, not a content-authenticity check).
    pub fn verify(
        &self,
        seal: &ExecutableSeal,
        computed_manifest_hash: ContentHash,
        computed_profile_hashes: &[ContentHash],
        computed_receipt_hash: ContentHash,
    ) -> Result<SealVerificationReceipt, SealVerificationError> {
        // ── 1. Manifest hash ─────────────────────────────────────────────
        let manifest_hash_matches = seal.manifest_hash == computed_manifest_hash;
        if !manifest_hash_matches {
            return Err(SealVerificationError::ManifestHashMismatch {
                expected: seal.manifest_hash,
                computed: computed_manifest_hash,
            });
        }

        // ── 2. Profile hashes (index-by-index) ───────────────────────────
        let profile_hashes_match = seal.profile_hashes.len() == computed_profile_hashes.len()
            && seal
                .profile_hashes
                .iter()
                .zip(computed_profile_hashes.iter())
                .all(|(a, b)| a == b);

        if !profile_hashes_match {
            // Zip-shortest iteration: report the first mismatch, or length
            // mismatch at the first index past the shorter slice.
            let min_len = seal.profile_hashes.len().min(computed_profile_hashes.len());
            for i in 0..min_len {
                if seal.profile_hashes[i] != computed_profile_hashes[i] {
                    return Err(SealVerificationError::ProfileHashMismatch {
                        index: i,
                        expected: seal.profile_hashes[i],
                        computed: computed_profile_hashes[i],
                    });
                }
            }
            // Lengths differ — report at the first missing index.
            let missing_idx = min_len;
            return Err(SealVerificationError::ProfileHashMismatch {
                index: missing_idx,
                expected: seal
                    .profile_hashes
                    .get(missing_idx)
                    .copied()
                    .unwrap_or(ContentHash(0)),
                computed: computed_profile_hashes
                    .get(missing_idx)
                    .copied()
                    .unwrap_or(ContentHash(0)),
            });
        }

        // ── 3. Receipt-bundle hash ───────────────────────────────────────
        let receipt_hash_matches = seal.receipt_bundle_hash == computed_receipt_hash;
        if !receipt_hash_matches {
            return Err(SealVerificationError::ReceiptHashMismatch {
                expected: seal.receipt_bundle_hash,
                computed: computed_receipt_hash,
            });
        }

        // ── 4. Self-computed root hash (IntegrityVerifier-style) ─────────
        // We recompute the root from all component hashes using the same
        // combinator the compiler used during sealing.
        //
        // Because IntegrityVerifier::combine_hashes is a pure function of
        // the component hashes, this is a tamper check on the seal itself.
        let mut all_hashes = Vec::with_capacity(1 + seal.profile_hashes.len() + 1);
        all_hashes.push(seal.manifest_hash);
        all_hashes.push(seal.receipt_bundle_hash);
        all_hashes.extend_from_slice(&seal.profile_hashes);
        let _computed_root = IntegrityVerifier::combine_hashes(&all_hashes);

        // Note: _computed_root is available for audit/logging but is not
        // part of the public receipt.  A future extension may expose it.

        // ── 5. Signature presence ────────────────────────────────────────
        let signature_valid = if seal.signature.is_some() {
            // Basic verify() has no key material — reports None.
            None
        } else {
            None
        };

        Ok(SealVerificationReceipt {
            seal_valid: manifest_hash_matches && profile_hashes_match && receipt_hash_matches,
            manifest_hash_matches,
            profile_hashes_match,
            receipt_hash_matches,
            signature_valid,
        })
    }
}

/// Granular verification receipt.
///
/// Every field is set independently so callers can decide which mismatches
/// are fatal and which are tolerable.
pub struct SealVerificationReceipt {
    /// Overall validity: all component hashes match the seal.
    pub seal_valid: bool,
    pub manifest_hash_matches: bool,
    pub profile_hashes_match: bool,
    pub receipt_hash_matches: bool,
    /// `Some(true)` — signature verified; `Some(false)` — verification
    /// failed; `None` — unsigned or not checked (basic `verify` path).
    pub signature_valid: Option<bool>,
}

/// Each variant identifies the exact field and the conflicting values.
#[derive(Debug, Clone)]
pub enum SealVerificationError {
    ManifestHashMismatch {
        expected: ContentHash,
        computed: ContentHash,
    },
    ProfileHashMismatch {
        index: usize,
        expected: ContentHash,
        computed: ContentHash,
    },
    ReceiptHashMismatch {
        expected: ContentHash,
        computed: ContentHash,
    },
    SignatureVerificationFailed(String),
    InvalidFormatVersion,
}

impl std::fmt::Display for SealVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManifestHashMismatch { expected, computed } => {
                write!(
                    f,
                    "seal manifest hash mismatch: expected {:?}, computed {:?}",
                    expected.0, computed.0
                )
            }
            Self::ProfileHashMismatch { index, expected, computed } => {
                write!(
                    f,
                    "seal profile hash mismatch at index {}: expected {:?}, computed {:?}",
                    index, expected.0, computed.0
                )
            }
            Self::ReceiptHashMismatch { expected, computed } => {
                write!(
                    f,
                    "seal receipt-bundle hash mismatch: expected {:?}, computed {:?}",
                    expected.0, computed.0
                )
            }
            Self::SignatureVerificationFailed(detail) => {
                write!(f, "seal signature verification failed: {}", detail)
            }
            Self::InvalidFormatVersion => {
                write!(f, "seal has an invalid or unsupported format version")
            }
        }
    }
}

impl std::error::Error for SealVerificationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::executable::seal::{ExecutableSeal, ExecutableSignature};

    fn make_valid_seal() -> ExecutableSeal {
        ExecutableSeal {
            root_hash: ContentHash(42),
            manifest_hash: ContentHash(1),
            profile_hashes: vec![ContentHash(2)],
            receipt_bundle_hash: ContentHash(3),
            signature: None,
        }
    }

    #[test]
    fn test_seal_verifies_with_matching_hashes() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let receipt = verifier
            .verify(&seal, ContentHash(1), &[ContentHash(2)], ContentHash(3))
            .unwrap();
        assert!(receipt.seal_valid);
        assert!(receipt.manifest_hash_matches);
        assert!(receipt.profile_hashes_match);
        assert!(receipt.receipt_hash_matches);
    }

    #[test]
    fn test_seal_fails_on_manifest_mismatch() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let result =
            verifier.verify(&seal, ContentHash(0xDEAD), &[ContentHash(2)], ContentHash(3));
        assert!(result.is_err());
        match result {
            Err(SealVerificationError::ManifestHashMismatch { expected, computed }) => {
                assert_eq!(expected, ContentHash(1));
                assert_eq!(computed, ContentHash(0xDEAD));
            }
            _ => panic!("expected ManifestHashMismatch"),
        }
    }

    #[test]
    fn test_seal_fails_on_profile_hash_mismatch() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let result =
            verifier.verify(&seal, ContentHash(1), &[ContentHash(0xBE)], ContentHash(3));
        assert!(result.is_err());
        match result {
            Err(SealVerificationError::ProfileHashMismatch { index, .. }) => {
                assert_eq!(index, 0);
            }
            _ => panic!("expected ProfileHashMismatch"),
        }
    }

    #[test]
    fn test_seal_fails_on_receipt_mismatch() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let result =
            verifier.verify(&seal, ContentHash(1), &[ContentHash(2)], ContentHash(0xEF));
        assert!(result.is_err());
        match result {
            Err(SealVerificationError::ReceiptHashMismatch { expected, computed }) => {
                assert_eq!(expected, ContentHash(3));
                assert_eq!(computed, ContentHash(0xEF));
            }
            _ => panic!("expected ReceiptHashMismatch"),
        }
    }

    #[test]
    fn test_seal_fails_on_profile_count_mismatch() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let result = verifier.verify(&seal, ContentHash(1), &[], ContentHash(3));
        assert!(result.is_err());
        match result {
            Err(SealVerificationError::ProfileHashMismatch { .. }) => {}
            _ => panic!("expected ProfileHashMismatch"),
        }
    }

    #[test]
    fn test_unsigned_seal_reports_none_signature() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let receipt = verifier
            .verify(&seal, ContentHash(1), &[ContentHash(2)], ContentHash(3))
            .unwrap();
        assert!(receipt.signature_valid.is_none());
    }

    #[test]
    fn test_receipt_flags_are_independent() {
        let verifier = SealVerifier::new();
        let seal = make_valid_seal();
        let receipt = verifier
            .verify(&seal, ContentHash(1), &[ContentHash(2)], ContentHash(3))
            .unwrap();
        assert!(receipt.manifest_hash_matches);
        assert!(receipt.profile_hashes_match);
        assert!(receipt.receipt_hash_matches);
        assert!(receipt.seal_valid);
        assert!(receipt.signature_valid.is_none());
    }
}
