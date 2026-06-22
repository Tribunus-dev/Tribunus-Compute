//! Executable seal — Merkle integrity verification.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableSeal {
    pub root_hash: ContentHash,
    pub manifest_hash: ContentHash,
    pub profile_hashes: Vec<ContentHash>,
    pub receipt_bundle_hash: ContentHash,
    pub signature: Option<ExecutableSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableSignature {
    pub signature_bytes: Vec<u8>,
    pub signer_identity: String,
    pub signature_scheme: String,
}
