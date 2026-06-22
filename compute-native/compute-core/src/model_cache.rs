//! ModelCache — LRU model pool with disk-backed segments.
//!
//! Provides a central cache for loading, using, and evicting models
//! dynamically. Models load lazily from disk (via ComputeImage segments
//! or streaming HuggingFace compilation), stay resident while in use,
//! and are evicted under memory pressure.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};

use crate::compute_image::Manifest;
use crate::config::HardwareTarget;
use crate::lora::SharedWeightTable;
use crate::metrics::{CacheKind, InferenceTelemetry};
use crate::profiled_executor::LoadedProfiledModel;

// ---------------------------------------------------------------------------
// SegmentWatcher — hot-reload for ComputeImage segments
// ---------------------------------------------------------------------------

/// Watches a model's segment directory for file changes.
///
/// When a segment file or `manifest.json` is modified (e.g. by `--diff`
/// recompilation or ModelAutopsy patching), the watcher triggers a callback
/// so the active model cache can reload the affected segment without
/// restarting the server.
pub struct SegmentWatcher {
    pub model_name: String,
    pub manifest_path: PathBuf,
    pub segment_dir: PathBuf,
    /// The underlying filesystem watcher (kept alive for the watcher's lifetime).
    watcher: Option<RecommendedWatcher>,
    /// Previously observed manifest hash (for poll-based update detection).
    prev_manifest_hash: String,
    /// Previously observed segment hashes keyed by filename (for poll-based
    /// update detection).
    prev_segment_hashes: HashMap<String, String>,
}

impl SegmentWatcher {
    /// Create a new watcher for `model_name` whose ComputeImage lives at
    /// `image_dir`. Reads the initial manifest to record baseline hashes.
    pub fn new(model_name: &str, image_dir: &Path) -> Result<Self, String> {
        let segment_dir = image_dir.to_path_buf();
        let manifest_path = image_dir.join("manifest.json");
        let (manifest_hash, segment_hashes) = Self::read_manifest_hashes(&manifest_path)?;

        Ok(Self {
            model_name: model_name.to_string(),
            manifest_path,
            segment_dir,
            watcher: None,
            prev_manifest_hash: manifest_hash,
            prev_segment_hashes: segment_hashes,
        })
    }

    /// Read the current manifest and return (image_hash, map<filename, sha256>).
    fn read_manifest_hashes(
        manifest_path: &Path,
    ) -> Result<(String, HashMap<String, String>), String> {
        let content = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("read manifest '{}': {}", manifest_path.display(), e))?;
        let manifest: Manifest = serde_json::from_str(&content)
            .map_err(|e| format!("parse manifest '{}': {}", manifest_path.display(), e))?;

        let manifest_hash = manifest.image_hash.clone();
        let segment_hashes: HashMap<String, String> = manifest
            .segments
            .iter()
            .map(|s| (s.filename.clone(), s.sha256.clone()))
            .collect();

        Ok((manifest_hash, segment_hashes))
    }

    /// Start watching the segment directory. `on_segment_changed` is called
    /// from a background thread whenever a segment_*.bin file or manifest.json
    /// is modified or created.
    pub fn start<F>(&mut self, on_segment_changed: F) -> Result<(), String>
    where
        F: Fn(&str) + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();

        let mut watcher = RecommendedWatcher::new(tx, notify::Config::default())
            .map_err(|e| format!("create notify watcher: {}", e))?;

        watcher
            .watch(&self.segment_dir, RecursiveMode::NonRecursive)
            .map_err(|e| format!("watch directory '{}': {}", self.segment_dir.display(), e))?;

        // Spawn a thread that dispatches file-change events to the callback.
        let model_name = self.model_name.clone();
        thread::Builder::new()
            .name(format!("seg-watcher-{}", model_name))
            .spawn(move || {
                for event in rx {
                    match event {
                        Ok(ev) => {
                            if matches!(ev.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                                for path in &ev.paths {
                                    if let Some(filename) =
                                        path.file_name().and_then(|n| n.to_str())
                                    {
                                        if filename.starts_with("segment_")
                                            && filename.ends_with(".bin")
                                            || filename == "manifest.json"
                                        {
                                            (on_segment_changed)(filename);
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[segment-watcher-{}] notify error: {}", model_name, e);
                        }
                    }
                }
            })
            .map_err(|e| format!("spawn watcher thread: {}", e))?;

        self.watcher = Some(watcher);
        Ok(())
    }

    /// Poll-based update check: re-read `manifest.json` from disk and compare
    /// segment SHA-256 hashes against the previously recorded values.
    ///
    /// Returns the list of filenames (segment_*.bin or "manifest.json") whose
    /// content has changed since the last call. A manifest hash change implies
    /// that the full recompilation has occurred (all segments may be new).
    pub fn check_for_updates(&mut self) -> Result<Vec<String>, String> {
        let (manifest_hash, segment_hashes) = Self::read_manifest_hashes(&self.manifest_path)?;
        let mut changed = Vec::new();

        // Manifest hash changed -> full re-evaluation (all segments potentially new).
        if manifest_hash != self.prev_manifest_hash {
            changed.push("manifest.json".to_string());
            self.prev_manifest_hash = manifest_hash;
            self.prev_segment_hashes = segment_hashes;
            return Ok(changed);
        }

        // Check individual segment hashes.
        for (filename, hash) in &segment_hashes {
            if self.prev_segment_hashes.get(filename) != Some(hash) {
                changed.push(filename.clone());
            }
        }

        if !changed.is_empty() {
            self.prev_segment_hashes = segment_hashes;
        }

        Ok(changed)
    }
}

// ---------------------------------------------------------------------------
// ModelType — inference pipeline selector
// ---------------------------------------------------------------------------

/// Classification of a model's modality, used to route inference requests
/// to the correct pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Autoregressive text model (Gemma, Qwen, etc.)
    Text,
    /// Vision-language model (captioning, VQA)
    Vision,
    /// Audio model (speech recognition, TTS)
    Audio,
    /// Image generation model (FLUX, etc.)
    ImageGen,
    /// Diffusion language model (DiffusionGemma)
    Diffusion,
}

// ---------------------------------------------------------------------------
// ModelSource — where to find a model's weights
// ---------------------------------------------------------------------------

/// Source specification for model loading.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Path to a compiled ComputeImage directory.
    ImageDir(PathBuf),
    /// HuggingFace hub ID (streaming compile on first load).
    HuggingFace(String),
}

// ---------------------------------------------------------------------------
// CachedModel — a loaded model in the cache
// ---------------------------------------------------------------------------

/// A model that has been loaded into the cache, with LRU tracking metadata.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// The loaded model, shared via Arc for concurrent access.
    pub model: Arc<LoadedProfiledModel>,
    /// Estimated memory footprint in bytes.
    pub size_bytes: u64,
    /// Last access timestamp (for LRU eviction).
    pub last_used: Instant,
    /// Model modality used for dispatch routing.
    pub model_type: ModelType,
}

// ---------------------------------------------------------------------------
// ModelCache — LRU model pool
// ---------------------------------------------------------------------------

/// A cache of loaded models. Models are loaded lazily from disk (via
/// streaming compile or direct segment loading), kept in memory while
/// in use, and evicted when memory pressure triggers.
pub struct ModelCache {
    /// Maximum total memory for loaded models (bytes).
    pub max_memory_bytes: u64,
    /// Current memory usage across all cached models (bytes).
    pub used_memory_bytes: u64,
    /// Shared base model weights for LoRA adapter deduplication.
    /// When set, adapters reference this single copy instead of duplicating
    /// the entire base model per adapter.
    pub shared_base: Option<Arc<SharedWeightTable>>,
    /// Whether weight streaming is enabled (disabled on memory-rich hardware).
    pub enable_weight_streaming: bool,
    /// Whether KV cache disk eviction is enabled (disabled on memory-rich hardware).
    pub enable_kv_eviction: bool,
    /// Cached models indexed by name.
    lru: HashMap<String, CachedModel>,
    /// LRU order: most recently used at the back.
    lru_order: VecDeque<String>,
    /// Model sources tracked for hot-reload (maps model name to its source).
    loaded_sources: HashMap<String, ModelSource>,
    /// Active file watchers for hot-reload, keyed by model name.
    /// Each watcher has a channel receiver for dispatching change events.
    watchers: HashMap<String, (SegmentWatcher, mpsc::Receiver<String>)>,
}

impl ModelCache {
    /// Check if a compiled model matches the current hardware.
    /// If not, indicate whether recompilation is recommended.
    pub fn check_compatibility(manifest: &Manifest) -> HardwareCheck {
        let current = HardwareTarget::detect();
        let Some(compiled_target) = manifest.hardware_target else {
            return HardwareCheck::NoTarget;
        };

        if compiled_target == current {
            HardwareCheck::Compatible
        } else if current.segment_target_size_mb() >= compiled_target.segment_target_size_mb() {
            // Current hardware can handle this just fine (may be over-optimized)
            HardwareCheck::Compatible
        } else {
            // Compiled for richer hardware, running on poorer hardware — recompile recommended
            HardwareCheck::ShouldRecompile {
                from: compiled_target,
                to: current,
            }
        }
    }

    /// Create a new cache with the given memory limit.
    pub fn new(max_memory_mb: u64) -> Self {
        Self {
            max_memory_bytes: max_memory_mb * 1_048_576,
            used_memory_bytes: 0,
            shared_base: None,
            enable_weight_streaming: true,
            enable_kv_eviction: true,
            lru: HashMap::new(),
            lru_order: VecDeque::new(),
            loaded_sources: HashMap::new(),
            watchers: HashMap::new(),
        }
    }

    /// Configure the cache for the detected hardware.
    ///
    /// On memory-rich hardware (M3 Ultra with 512 GB):
    /// - Weight streaming is DISABLED (everything fits in RAM)
    /// - KV cache disk eviction is DISABLED (no need to spill)
    pub fn configure_for_hardware(&mut self) {
        let hw = crate::scheduling::HardwareConfig::detect();
        self.enable_weight_streaming = hw.enable_weight_streaming;
        self.enable_kv_eviction = hw.enable_kv_disk_eviction;
    }

    /// Preload all enabled models into memory at startup.
    ///
    /// On M3 Ultra (memory-rich), loads every registered model source
    /// into the cache so inference starts instantly — no lazy loading.
    pub fn preload_all(&mut self) -> Result<(), String> {
        let hw = crate::scheduling::HardwareConfig::detect();
        if !hw.is_memory_rich {
            return Ok(()); // Only preload on 64 GB+ systems
        }

        let sources = crate::model_cache::default_model_sources();
        for (name, source) in &sources {
            eprintln!("[model-cache] Preloading {}...", name);
            self.get_or_load(name, source, None)?;
        }
        Ok(())
    }

    /// Get a model by name. If not loaded, compile from source.
    /// If memory is full, evict LRU models.
    pub fn get_or_load(
        &mut self,
        name: &str,
        source: &ModelSource,
        telemetry: Option<&InferenceTelemetry>,
    ) -> Result<Arc<LoadedProfiledModel>, String> {
        // Fast path: already cached — touch LRU and return.
        if let Some(cached) = self.lru.get(name) {
            let model = cached.model.clone();
            self.touch_lru(name);
            if let Some(t) = telemetry {
                t.record_cache_hit(CacheKind::Model);
            }
            return Ok(model);
        }

        if let Some(t) = telemetry {
            t.record_cache_miss(CacheKind::Model);
        }

        // Check system memory pressure before loading.
        self.check_pressure()?;

        // Load from the specified source.
        let (model, size, model_type) = match source {
            ModelSource::ImageDir(path) => self.load_from_image(name, path)?,
            ModelSource::HuggingFace(hub_id) => self.load_from_hf(name, hub_id)?,
        };

        // Ensure there is room for the new model.
        self.evict_lru(size)?;

        let cached = CachedModel {
            model: Arc::new(model),
            size_bytes: size,
            last_used: Instant::now(),
            model_type,
        };

        self.used_memory_bytes = self.used_memory_bytes.saturating_add(size);
        self.lru.insert(name.to_string(), cached);
        self.lru_order.push_back(name.to_string());
        self.loaded_sources.insert(name.to_string(), source.clone());

        Ok(self.lru.get(name).unwrap().model.clone())
    }

    /// Try to get a model without loading.
    pub fn get(&mut self, name: &str) -> Option<&mut CachedModel> {
        if self.lru.contains_key(name) {
            self.touch_lru(name);
        }
        self.lru.get_mut(name)
    }

    /// Load a model from a ComputeImage directory using lazy segment
    /// activation (via `CompiledImageReader::open` and mapped-image
    /// zero-copy, rather than loading all tensor arrays into the MLX
    /// allocator upfront).
    fn load_from_image(
        &mut self,
        name: &str,
        image_dir: &Path,
    ) -> Result<(LoadedProfiledModel, u64, ModelType), String> {
        // Use the profiled model loader which opens the image and reads
        // metadata without loading all segment bytes into the MLX allocator.
        // The MappedImage layer within LoadedProfiledModel handles zero-copy
        // segment access, and individual layer tensors are loaded on demand
        // during inference via the existing runtime.
        let model = LoadedProfiledModel::new(image_dir).map_err(|e| {
            format!(
                "load model '{}' from {}: {:?}",
                name,
                image_dir.display(),
                e
            )
        })?;

        // Estimated memory: the mapped and resident weight bytes reported by
        // the loader. The actual resident footprint grows as layers activate.
        let size = model
            .mapped_weight_bytes
            .saturating_add(model.copied_weight_bytes)
            .saturating_add(model.materialized_bytes);

        let model_type = detect_type(name);

        Ok((model, size, model_type))
    }

    /// Load a model from HuggingFace via streaming compile.
    fn load_from_hf(
        &mut self,
        _name: &str,
        _hub_id: &str,
    ) -> Result<(LoadedProfiledModel, u64, ModelType), String> {
        // Placeholder: streaming compile integration will use the existing
        // streaming pipeline to download, compile, and cache the ComputeImage
        // before loading it through the image path.
        Err("HuggingFace streaming compile not yet wired".to_string())
    }

    /// Evict least recently used models until `needed_bytes` are free.
    fn evict_lru(&mut self, needed_bytes: u64) -> Result<(), String> {
        if needed_bytes > self.max_memory_bytes {
            return Err(format!(
                "requested {} bytes exceeds cache capacity {} bytes",
                needed_bytes, self.max_memory_bytes
            ));
        }

        let mut available = self.max_memory_bytes.saturating_sub(self.used_memory_bytes);
        let mut evicted: u64 = 0;

        while available < needed_bytes {
            let victim = match self.lru_order.pop_front() {
                Some(name) => name,
                None => {
                    return Err(format!(
                        "cannot free {} bytes — only {} available after evicting all models",
                        needed_bytes,
                        self.used_memory_bytes + evicted,
                    ));
                }
            };

            if let Some(cached) = self.lru.remove(&victim) {
                let freed = cached.size_bytes;
                self.used_memory_bytes = self.used_memory_bytes.saturating_sub(freed);
                available = available.saturating_add(freed);
                evicted = evicted.saturating_add(freed);
            }
        }

        Ok(())
    }

    /// Unload a specific model from the cache.
    pub fn unload(&mut self, name: &str) -> Result<(), String> {
        if let Some(cached) = self.lru.remove(name) {
            self.used_memory_bytes = self.used_memory_bytes.saturating_sub(cached.size_bytes);
            if let Some(pos) = self.lru_order.iter().position(|n| n == name) {
                self.lru_order.remove(pos);
            }
        }
        Ok(())
    }

    /// Current memory status as a human-readable string.
    pub fn memory_status(&self) -> String {
        let used_mb = self.used_memory_bytes / 1_048_576;
        let max_mb = self.max_memory_bytes / 1_048_576;
        format!(
            "{}/{} MB used ({} models loaded)",
            used_mb,
            max_mb,
            self.lru.len()
        )
    }

    /// Check system memory pressure and evict if >80% of RAM is used.
    pub fn check_pressure(&mut self) -> Result<(), String> {
        let _free_mb = crate::gpu_memory::get_current_wired_limit_mb()
            .map(|m| m as u64)
            .unwrap_or(0);
        let used_mb = self.used_memory_bytes / 1_048_576;
        let total_mb = crate::gpu_memory::total_physical_ram_mb() as u64;

        // If >80% of RAM is consumed by cached models, evict LRU models
        // to free 10% of the current cache usage.
        if total_mb > 0 && used_mb > (total_mb as f64 * 0.8) as u64 {
            let to_free = self.used_memory_bytes / 10;
            self.evict_lru(to_free)?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Number of models currently in the cache.
    pub fn entry_count(&self) -> usize {
        self.lru.len()
    }

    /// Whether any models are currently cached.
    pub fn has_any(&self) -> bool {
        !self.lru.is_empty()
    }

    /// Move `name` to the back of the LRU order (most recently used).
    fn touch_lru(&mut self, name: &str) {
        if let Some(pos) = self.lru_order.iter().position(|n| n == name) {
            let n = self.lru_order.remove(pos).unwrap();
            self.lru_order.push_back(n);
        }
    }

    // ------------------------------------------------------------------
    // Hot-reload support
    // ------------------------------------------------------------------

    /// Return the image directory for a loaded model, if it was loaded from
    /// a local ComputeImage directory.
    fn get_image_dir(&self, name: &str) -> Result<PathBuf, String> {
        self.loaded_sources
            .get(name)
            .and_then(|s| match s {
                ModelSource::ImageDir(path) => Some(path.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("model '{}' is not loaded from an image directory", name))
    }

    /// Enable file-watcher-based hot-reload for a loaded model.
    ///
    /// Starts watching the model's image directory for changes to segment
    /// files and manifest.json. When a change is detected, the affected
    /// segment is reloaded without restarting the server.
    pub fn watch(&mut self, name: &str) -> Result<(), String> {
        // Verify the model is loaded from an image directory.
        let image_dir = self.get_image_dir(name)?;
        if self.watchers.contains_key(name) {
            return Err(format!("watcher already active for model '{}'", name));
        }

        let mut watcher = SegmentWatcher::new(name, &image_dir)?;
        let _model_name = name.to_string();

        // Channel: watcher thread sends changed segment filenames here.
        let (tx, rx) = mpsc::channel::<String>();

        watcher.start(move |segment| {
            let _ = tx.send(segment.to_string());
        })?;

        self.watchers.insert(name.to_string(), (watcher, rx));

        eprintln!(
            "[model-cache] Hot-reload enabled for '{}' at {}",
            name,
            image_dir.display()
        );
        Ok(())
    }

    /// Reload a single segment for a cached model. This unloads and
    /// re-loads the model from its image directory so the updated
    /// segment file takes effect.
    ///
    /// Called either from [`process_pending_reloads`] (event-driven) or
    /// directly by the server after a manual recompile.
    pub fn reload_segment(&mut self, name: &str, segment: &str) -> Result<(), String> {
        let source = self
            .loaded_sources
            .get(name)
            .ok_or_else(|| format!("model '{}' is not tracked in cache", name))?
            .clone();

        // Unload the current model — this drops all tensor handles so the
        // next load picks up the new segment data from disk.
        self.unload(name)?;

        // Re-load from the same source directory (segment files are now
        // updated on disk by the --diff or ModelAutopsy patching).
        self.get_or_load(name, &source, None)?;

        eprintln!(
            "[model-cache] Hot-reloaded segment '{}' for model '{}'",
            segment, name
        );
        Ok(())
    }

    /// Apply an edit patch to a segment and hot-reload it.
    ///
    /// This is the primary integration point for the knowledge editing
    /// system.  Call this AFTER the segment file has been patched on disk
    /// (e.g. by `SegmentPatch::apply`).
    ///
    /// * `name` — model name (e.g. "gemma4").
    /// * `segment` — segment filename (e.g. "segment_005.bin").
    pub fn apply_edit_patch(&mut self, name: &str, segment: &str) -> Result<(), String> {
        // Verify the model is tracked before attempting a reload.
        if !self.loaded_sources.contains_key(name) {
            return Err(format!(
                "cannot apply edit patch: model '{}' is not loaded in cache",
                name
            ));
        }

        self.reload_segment(name, segment)?;

        eprintln!(
            "[model-cache] Edit patch applied and hot-reloaded segment '{}' for model '{}'",
            segment, name
        );
        Ok(())
    }

    /// Process any pending hot-reload events from file watchers.
    /// Drains the channel for each active watcher and calls
    /// [`reload_segment`] for each changed file.
    ///
    /// Should be called periodically from the server's main event loop.
    pub fn process_pending_reloads(&mut self) {
        // Collect pending events first (avoid borrow conflicts with
        // reload_segment which also borrows self mutably).
        let pending: Vec<(String, String)> = self
            .watchers
            .iter()
            .flat_map(|(name, (_watcher, rx))| {
                let mut events: Vec<(String, String)> = Vec::new();
                while let Ok(segment) = rx.try_recv() {
                    events.push((name.clone(), segment));
                }
                events
            })
            .collect();

        for (name, segment) in pending {
            if let Err(e) = self.reload_segment(&name, &segment) {
                eprintln!(
                    "[model-cache] Failed to hot-reload '{}' for '{}': {}",
                    segment, name, e
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Default model sources
// ---------------------------------------------------------------------------

/// Map well-known model names to their disk or HuggingFace sources.
pub fn default_model_sources() -> HashMap<String, ModelSource> {
    let mut m = HashMap::new();
    // Auto-register model from env vars (populated by --model-path / --dev-mode).
    if let Ok(path) = std::env::var("TRIBUNUS_MODEL_PATH") {
        let name = std::env::var("TRIBUNUS_MODEL_NAME").unwrap_or_else(|_| {
            std::path::Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".into())
        });
        m.insert(name, ModelSource::ImageDir(path.into()));
    }
    m.insert(
        "qwen2.5:0.5b".into(),
        ModelSource::ImageDir("compute-native/models/qwen-compiled".into()),
    );
    m.insert(
        "gemma4".into(),
        ModelSource::ImageDir("compute-native/models/gemma4-compiled".into()),
    );
    m.insert(
        "diffusiongemma".into(),
        ModelSource::ImageDir("compute-native/models/diffusiongemma-compiled".into()),
    );
    m.insert(
        "flux".into(),
        ModelSource::ImageDir("compute-native/models/flux-compiled".into()),
    );
    m.insert(
        "funasr".into(),
        ModelSource::HuggingFace("google/funasr-mlx".into()),
    );
    m.insert(
        "qwen-tts".into(),
        ModelSource::HuggingFace("Qwen/qwen3-tts-mlx".into()),
    );
    m.insert(
        "qwen2.5-hw-bench".into(),
        ModelSource::ImageDir("./models/qwen2.5-hw-bench".into()),
    );
    m
}

// ---------------------------------------------------------------------------
// Model type detection
// ---------------------------------------------------------------------------

/// Infer the [`ModelType`] from a model name string.
pub fn detect_type(name: &str) -> ModelType {
    let lower = name.to_lowercase();
    if lower.contains("diffusion") || lower.contains("diffusiongemma") {
        ModelType::Diffusion
    } else if lower.contains("flux") || lower.contains("t2i") || lower.contains("text_to_image") {
        ModelType::ImageGen
    } else if lower.contains("funasr")
        || lower.contains("whisper")
        || lower.contains("asr")
        || lower.contains("qwen-tts")
        || lower.contains("tts")
        || lower.contains("speech")
    {
        ModelType::Audio
    } else if lower.contains("vit") || lower.contains("vision") || lower.contains("siglip") {
        ModelType::Vision
    } else {
        ModelType::Text
    }
}

// ---------------------------------------------------------------------------
// Hardware compatibility check
// ---------------------------------------------------------------------------

/// Result of checking a compiled model's hardware compatibility against the
/// current machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardwareCheck {
    /// Model is compatible with the current hardware.
    Compatible,
    /// Model was compiled for a different target and should be recompiled.
    ShouldRecompile {
        /// Target the model was compiled for.
        from: HardwareTarget,
        /// Current target that would be used for recompilation.
        to: HardwareTarget,
    },
    /// No hardware target information in the manifest.
    NoTarget,
}
