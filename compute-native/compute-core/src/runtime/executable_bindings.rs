//! Runtime executable bindings — opens segment files, loads object data into
//! byte views, and makes content objects addressable by ID.
//!
//! # Flow
//! 1. `BindingManager::load_segments` iterates `store.segments`, calls
//!    `MmapLoader::open_segment` for each, then reads each object's bytes
//!    from the segment file at the correct offset.
//! 2. `view_object` returns a byte slice for a previously loaded `object_id`.

use crate::compute_image::content_store::index::{
    ContentAddressedContentStore, ContentObjectEntry, ContentObjectKind,
};
use crate::compute_image::content_store::mmap::{MappedSegment, MmapLoadError, MmapLoader};
use crate::compute_image::program::phase_program::SerializedPhaseProgram;
use crate::compute_image::residency::plan::CompiledResidencyPlan;
use serde_json;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
/// All mapped segments and fully loaded object views for a sealed executable.
pub struct ExecutableBindings {
    pub segments: Vec<MappedSegment>,
    /// object_id → its raw payload bytes, read from the owning segment file.
    pub object_views: HashMap<String, Vec<u8>>,
    /// Deserialized phase programs loaded from PhaseProgramPayload objects.
    pub programs: Vec<SerializedPhaseProgram>,
    /// Deserialized residency plans indexed by object_id.
    pub residency_plans: HashMap<String, CompiledResidencyPlan>,
}

impl ExecutableBindings {
    /// Return all phase programs registered in this binding set.
    pub fn programs(&self) -> &[SerializedPhaseProgram] {
        &self.programs
    }

    /// Look up a residency plan by its stable identifier.
    pub fn find_residency_plan(&self, id: &str) -> Option<&CompiledResidencyPlan> {
        self.residency_plans.get(id)
    }
}

pub struct BindingManager;

impl BindingManager {
    pub fn new() -> Self {
        Self
    }

    /// Load all segments referenced by the content store, returning
    /// a set of mapped segments ready for view_object calls.
    ///
    /// Each `ImmutableSegment` is mapped via `MmapLoader::open_segment`.
    /// Each `ContentObjectEntry` is then read from the segment file at
    /// `segment.payload_offset + object.segment_offset` and stored in
    /// `object_views` for fast zero-copy access.
    pub fn load_segments(
        &self,
        store: &ContentAddressedContentStore,
        base_path: &Path,
    ) -> Result<ExecutableBindings, MmapLoadError> {
        // --- Phase 1: map every segment file ---------------------------------
        let mut segments = Vec::with_capacity(store.segments.len());

        for seg in &store.segments {
            let path = base_path.join(format!("{}.bin", seg.segment_id));
            let mapped = MmapLoader::open_segment(&path)?;
            segments.push(mapped);
        }

        // --- Phase 2: read each object's bytes from its segment file ---------
        let mut object_views: HashMap<String, Vec<u8>> =
            HashMap::with_capacity(store.objects.len());
        let mut programs: Vec<SerializedPhaseProgram> = Vec::new();
        let mut residency_plans: HashMap<String, CompiledResidencyPlan> = HashMap::new();

        // Group objects by segment_id so we re-open each segment file at most
        // once, regardless of how many objects it contains.
        let mut objects_by_segment: HashMap<&str, Vec<&ContentObjectEntry>> = HashMap::new();
        for obj in &store.objects {
            objects_by_segment.entry(obj.segment_id.as_str()).or_default().push(obj);
        }

        for seg in &store.segments {
            let path = base_path.join(format!("{}.bin", seg.segment_id));

            let file = std::fs::File::open(&path).map_err(|_| {
                MmapLoadError::PermissionDenied(path.display().to_string())
            })?;

            // mutable borrow needed for pread-style seek+read
            let mut file = file;

            let Some(objects) = objects_by_segment.get(seg.segment_id.as_str()) else {
                continue;
            };

            for obj in objects {
                // The object's bytes live at:
                //   segment header offset + object's offset within the segment
                let file_offset = seg.payload_offset + obj.segment_offset;
                let length = obj.payload_bytes as usize;

                let mut buf = vec![0u8; length];
                file.seek(SeekFrom::Start(file_offset)).map_err(|_| {
                    MmapLoadError::InvalidAlignment(format!(
                        "seek to offset {} for object {}",
                        file_offset, obj.object_id,
                    ))
                })?;
                file.read_exact(&mut buf).map_err(|_| {
                    MmapLoadError::InvalidAlignment(format!(
                        "read {} bytes at offset {} for object {}",
                        length, file_offset, obj.object_id,
                    ))
                })?;

                // Deserialize structured objects based on content kind
                match &obj.object_kind {
                    ContentObjectKind::PhaseProgramPayload => {
                        if let Ok(program) = serde_json::from_slice::<SerializedPhaseProgram>(&buf) {
                            programs.push(program);
                        }
                    }
                    ContentObjectKind::ResidencyPlanPayload => {
                        if let Ok(plan) = serde_json::from_slice::<CompiledResidencyPlan>(&buf) {
                            residency_plans.insert(obj.object_id.clone(), plan);
                        }
                    }
                    _ => {}
                }

                object_views.insert(obj.object_id.clone(), buf);
            }
        }

        Ok(ExecutableBindings {
            segments,
            object_views,
            programs,
            residency_plans,
        })
    }

    /// Get a byte view of a content object from its segment mapping.
    ///
    /// Returns `None` when `object_id` was not loaded (object not in the store,
    /// or the segment file was missing and skipped).
    pub fn view_object<'a>(
        &self,
        bindings: &'a ExecutableBindings,
        object_id: &str,
    ) -> Option<&'a [u8]> {
        bindings.object_views.get(object_id).map(|v| &v[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::content_store::index::{
        ContentAddressedContentStore, ContentObjectEntry, ContentObjectKind,
        ContentStoreVersion, ImmutableSegment, ArtifactConsumerRef, ResidencyClass,
    };

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    fn make_empty_store() -> ContentAddressedContentStore {
        ContentAddressedContentStore {
            store_version: ContentStoreVersion { major: 1, minor: 0 },
            segments: vec![],
            objects: vec![],
            aliases: vec![],
            index_hash: Default::default(),
        }
    }

    /// Write a temporary segment file, return its parent directory.
    /// The file contains `data` at `payload_offset` — any bytes before
    /// that offset are zero-filled.
    fn write_segment_file(
        dir: &std::path::Path,
        segment_id: &str,
        payload_offset: u64,
        data: &[u8],
    ) -> std::path::PathBuf {
        let path = dir.join(format!("{}.bin", segment_id));
        let total_len = (payload_offset as usize) + data.len();
        let mut buf = vec![0u8; total_len];
        buf[payload_offset as usize..][..data.len()].copy_from_slice(data);
        std::fs::write(&path, &buf).expect("write test segment file");
        path
    }

    fn make_object(
        object_id: &str,
        segment_id: &str,
        segment_offset: u64,
        payload_bytes: u64,
    ) -> ContentObjectEntry {
        ContentObjectEntry {
            object_id: object_id.to_string(),
            content_hash: Default::default(),
            object_kind: ContentObjectKind::CanonicalWeight,
            target_layout_id: String::new(),
            segment_id: segment_id.to_string(),
            segment_offset,
            payload_bytes,
            aligned_bytes: payload_bytes,
            alignment: 1,
            logical_shape: vec![],
            storage_shape: vec![],
            physical_strides: vec![],
            dtype: "float32".to_string(),
            quantization: None,
            checksum: Default::default(),
            consumers: vec![],
            residency_class: ResidencyClass::MandatoryAtSessionStart,
        }
    }

    // ------------------------------------------------------------------
    // tests
    // ------------------------------------------------------------------

    #[test]
    fn test_load_empty_store() {
        let store = make_empty_store();
        let manager = BindingManager::new();
        let result = manager.load_segments(&store, Path::new("/tmp"));
        assert!(result.is_ok());
        let bindings = result.unwrap();
        assert!(bindings.segments.is_empty());
        assert!(bindings.object_views.is_empty());
    }

    #[test]
    fn test_load_segments_with_objects() {
        let dir = std::env::temp_dir().join(format!(
            "executable_bindings_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");

        // Write one segment with two objects
        let segment_id = "seg_001";
        let obj1_offset = 0u64;
        let obj1_data = b"hello world";
        let obj2_offset = 16u64; // gap
        let obj2_data = b"second object payload";

        let total = (obj2_offset as usize) + obj2_data.len();
        let mut buf = vec![0u8; total];
        buf[obj1_offset as usize..][..obj1_data.len()].copy_from_slice(obj1_data);
        buf[obj2_offset as usize..][..obj2_data.len()].copy_from_slice(obj2_data);
        write_segment_file(&dir, segment_id, 0, &buf);

        let store = ContentAddressedContentStore {
            store_version: ContentStoreVersion { major: 1, minor: 0 },
            segments: vec![ImmutableSegment {
                segment_id: segment_id.to_string(),
                payload_offset: 0,
                payload_length: 0, // unused for reading
                alignment: 1,
                checksum: Default::default(),
            }],
            objects: vec![
                make_object("obj_1", segment_id, obj1_offset, obj1_data.len() as u64),
                make_object("obj_2", segment_id, obj2_offset, obj2_data.len() as u64),
            ],
            aliases: vec![],
            index_hash: Default::default(),
        };

        let manager = BindingManager::new();
        let bindings = manager
            .load_segments(&store, &dir)
            .expect("load_segments should succeed");

        assert_eq!(bindings.segments.len(), 1);
        assert_eq!(bindings.object_views.len(), 2);

        // Verify contents
        assert_eq!(
            manager.view_object(&bindings, "obj_1"),
            Some(&obj1_data[..]),
        );
        assert_eq!(
            manager.view_object(&bindings, "obj_2"),
            Some(&obj2_data[..]),
        );
        assert_eq!(manager.view_object(&bindings, "nonexistent"), None);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_segments_with_payload_offset() {
        // Verify that ImmutableSegment.payload_offset is respected when
        // reading object data from the file.
        let dir = std::env::temp_dir().join(format!(
            "executable_bindings_payload_offset_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let segment_id = "seg_offset_test";
        // data starts at byte 128 in the file (e.g. simulated metadata header)
        let payload_offset = 128u64;
        let obj_data = b"payload content here";
        let obj_offset = 0u64;

        write_segment_file(&dir, segment_id, payload_offset, obj_data);

        let store = ContentAddressedContentStore {
            store_version: ContentStoreVersion { major: 1, minor: 0 },
            segments: vec![ImmutableSegment {
                segment_id: segment_id.to_string(),
                payload_offset,
                payload_length: obj_data.len() as u64,
                alignment: 1,
                checksum: Default::default(),
            }],
            objects: vec![make_object(
                "obj",
                segment_id,
                obj_offset,
                obj_data.len() as u64,
            )],
            aliases: vec![],
            index_hash: Default::default(),
        };

        let manager = BindingManager::new();
        let bindings = manager
            .load_segments(&store, &dir)
            .expect("load_segments should succeed");

        assert_eq!(
            manager.view_object(&bindings, "obj"),
            Some(&obj_data[..]),
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_segments_missing_file_error() {
        let store = ContentAddressedContentStore {
            store_version: ContentStoreVersion { major: 1, minor: 0 },
            segments: vec![ImmutableSegment {
                segment_id: "nonexistent_seg".to_string(),
                payload_offset: 0,
                payload_length: 0,
                alignment: 1,
                checksum: Default::default(),
            }],
            objects: vec![],
            aliases: vec![],
            index_hash: Default::default(),
        };

        let manager = BindingManager::new();
        let result = manager.load_segments(&store, Path::new("/tmp/bogus_path"));
        assert!(result.is_err());
        match result.unwrap_err() {
            MmapLoadError::FileNotFound(_) => {} // expected
            other => panic!("expected FileNotFound, got: {other}"),
        }
    }
}
