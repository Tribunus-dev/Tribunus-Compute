//! Content aliasing and deduplication.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAliasEntry {
    pub alias: String,
    pub canonical_object_id: String,
    pub canonical_content_hash: ContentHash,
    pub alias_kind: AliasKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AliasKind {
    TiedEmbedding,
    SharedExpert,
    IdenticalProjection,
    ManifestDeclared,
    ContentHashCollision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDedupTable {
    entries: Vec<DedupEntry>,
    aliases: Vec<ContentAliasEntry>,
    object_to_canonical: HashMap<String, String>,
    hash_to_canonical: HashMap<u64, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupEntry {
    pub content_hash: ContentHash,
    pub canonical_object_id: String,
    pub ref_count: u64,
    pub total_storage_bytes: u64,
    pub dedup_savings_bytes: u64,
}

impl ContentDedupTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            aliases: Vec::new(),
            object_to_canonical: HashMap::new(),
            hash_to_canonical: HashMap::new(),
        }
    }

    pub fn register_object(&mut self, object_id: &str, content_hash: ContentHash, storage_bytes: u64) {
        let hash_key = content_hash.0;
        if let Some(canonical_id) = self.hash_to_canonical.get(&hash_key).cloned() {
            self.add_alias(object_id, &canonical_id, content_hash, AliasKind::ContentHashCollision);
            return;
        }
        self.hash_to_canonical.insert(hash_key, object_id.to_string());
        self.object_to_canonical.insert(object_id.to_string(), object_id.to_string());
        self.entries.push(DedupEntry {
            content_hash,
            canonical_object_id: object_id.to_string(),
            ref_count: 1,
            total_storage_bytes: storage_bytes,
            dedup_savings_bytes: 0,
        });
    }

    pub fn add_alias(&mut self, alias: &str, canonical_object_id: &str, content_hash: ContentHash, kind: AliasKind) {
        self.object_to_canonical.insert(alias.to_string(), canonical_object_id.to_string());
        self.aliases.push(ContentAliasEntry {
            alias: alias.to_string(),
            canonical_object_id: canonical_object_id.to_string(),
            canonical_content_hash: content_hash,
            alias_kind: kind,
        });
        if let Some(entry) = self.entries.iter_mut().find(|e| e.canonical_object_id == canonical_object_id) {
            entry.ref_count += 1;
            entry.dedup_savings_bytes += entry.total_storage_bytes;
        }
    }

    pub fn resolve(&self, object_id: &str) -> Option<&str> {
        self.object_to_canonical.get(object_id).map(|s| s.as_str())
    }

    pub fn dedup_ratio(&self) -> f64 {
        let total: u64 = self.entries.iter().map(|e| e.total_storage_bytes * e.ref_count).sum();
        let unique: u64 = self.entries.iter().map(|e| e.total_storage_bytes).sum();
        if total == 0 { return 0.0; }
        (total - unique) as f64 / total as f64
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn alias_count(&self) -> usize { self.aliases.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_table() {
        let table = ContentDedupTable::new();
        assert_eq!(table.entry_count(), 0);
        assert_eq!(table.alias_count(), 0);
    }

    #[test]
    fn test_unique_objects() {
        let mut table = ContentDedupTable::new();
        table.register_object("w1", ContentHash(1), 100);
        table.register_object("w2", ContentHash(2), 100);
        assert_eq!(table.entry_count(), 2);
    }

    #[test]
    fn test_content_hash_dedup() {
        let mut table = ContentDedupTable::new();
        table.register_object("w1", ContentHash(42), 100);
        table.register_object("w2", ContentHash(42), 100);
        assert_eq!(table.entry_count(), 1);
        assert_eq!(table.alias_count(), 1);
    }

    #[test]
    fn test_resolve() {
        let mut table = ContentDedupTable::new();
        table.register_object("w1", ContentHash(1), 100);
        table.add_alias("w1.tied", "w1", ContentHash(1), AliasKind::TiedEmbedding);
        assert_eq!(table.resolve("w1.tied"), Some("w1"));
        assert_eq!(table.resolve("w1"), Some("w1"));
    }
}
