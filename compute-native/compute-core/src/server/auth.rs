use std::collections::HashSet;
use std::sync::RwLock;

/// API key validator for Bearer token authentication.
///
/// Maintains a set of valid API keys loaded from environment variables
/// and config files. Supports runtime addition and removal of keys.
pub struct ApiKeyValidator {
    keys: RwLock<HashSet<String>>,
}

impl ApiKeyValidator {
    /// Create an empty validator (no keys loaded yet).
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashSet::new()),
        }
    }

    /// Load keys from the `TRIBUNUS_API_KEYS` environment variable
    /// (comma-separated list of API keys).
    ///
    /// This is a no-op if the variable is unset or empty. Calling this
    /// multiple times adds new keys to the existing set (it does not
    /// replace previously loaded keys).
    pub fn load_from_env(&self) {
        if let Ok(value) = std::env::var("TRIBUNUS_API_KEYS") {
            let mut keys = self.keys.write().expect("RwLock poisoned");
            for part in value.split(',') {
                let trimmed = part.trim().to_string();
                if !trimmed.is_empty() {
                    keys.insert(trimmed);
                }
            }
        }
    }

    /// Verify a Bearer token.
    ///
    /// Returns `true` if `token` is a known API key, `false` otherwise.
    pub fn validate(&self, token: &str) -> bool {
        let keys = self.keys.read().expect("RwLock poisoned");
        // If no keys are configured, allow all requests (dev mode).
        if keys.is_empty() {
            return true;
        }
        keys.contains(token)
    }

    /// Add a key at runtime.
    pub fn add_key(&self, key: String) {
        let mut keys = self.keys.write().expect("RwLock poisoned");
        keys.insert(key);
    }

    /// Remove a key at runtime.
    pub fn remove_key(&self, key: &str) {
        let mut keys = self.keys.write().expect("RwLock poisoned");
        keys.remove(key);
    }

    /// Check if any API keys are configured.
    pub fn is_empty(&self) -> bool {
        let keys = self.keys.read().expect("RwLock poisoned");
        keys.is_empty()
    }
}
