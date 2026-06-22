use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// A content-addressed hash value used to identify objects in the content
/// store.  Wraps a raw `u64` for compact serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// The zero hash — used as a sentinel / unset value.
    pub const ZERO: ContentHash = ContentHash(0);

    /// Create from a raw `u64`.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Create from a hex-encoded SHA-256 digest string.
    ///
    /// This is a simplified conversion for schema compliance; real content
    /// hashes should be full 256-bit digests computed by the content-store
    /// pipeline.
    pub fn from_hex(hex: &str) -> Self {
        // For schema compatibility we simply hash the string down to u64.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hex.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Return the inner `u64`.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", self.0)
    }
}

