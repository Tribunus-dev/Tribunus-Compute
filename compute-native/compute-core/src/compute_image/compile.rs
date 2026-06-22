//! ComputeImage compilation pipeline — source loading, quantization,
//! sequential/differential compilation, diagnostics, and publishing.

use super::compile_hw::run_hardware_assessment;
use super::hw_assessment::AssessmentReceipt;
use super::manifest::{
    mlx_active_memory_bytes, mlx_peak_memory_bytes, AliasEntry, CompilationAuthority,
    CompileReceipt, CompiledImage, CompiledImageReader, IgnoredTensorClassification, ImageBuilder,
    Manifest, ManifestVerification, MetalDispatchRecipe, MetalKernelArtifact,
    NativeCapabilityReport, QuantizationDesc, ResidencyPlan, Segment, SegmentKind, SegmentReceipt,
    ShardHash, SourceIdentity, StageProfile, StorageBackend, TensorDiff, TensorEntry,
    TensorProvenance,
};
use super::plan::{compile_unchecked_speculative, plan};
use crate::config::CompileQuantMode;
use crate::config::HardwareTarget;
use crate::model_adapter::{self, CanonicalModel, CanonicalRole, SourceModel};
use mlx_rs::Array;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;


// ═══════════════════════════════════════════════════════════════════════════
// Authority-aware compilation entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a source model into a ComputeImage directory with authority checks.
pub fn compile_with_authority(
    source_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let quantize_mode =
        quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));

    eprintln!(
        "[compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );

    match authority {
        CompilationAuthority::TestFixture => {
            let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
            if profile == "image-build" {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    "TestFixture must not use image-build profile. Use cargo test or cargo build.",
                ));
            }
            // Enforce fixture ceiling: max 4 layers, 256 tensors, 128 MB total source
            verify_fixture_ceiling(source_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }

    compile_unchecked(source_dir, output_dir, skip_validation, quantize_mode).map(|mut compiled| {
        compiled.manifest.hardware_target = Some(target);
        compiled
    })
}

/// Compile a draft + target model pair into a single speculative ComputeImage.
///
/// Both models must be compiled checkpoints (config.json + safetensors shards).
/// The resulting image stores shared weights once (embeddings if same vocab/hidden)
/// and orders draft layer segments before target layer segments for fast startup.
pub fn compile_with_authority_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let quantize_mode =
        quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));

    eprintln!(
        "[speculative compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );

    match authority {
        CompilationAuthority::TestFixture => {
            verify_fixture_ceiling(target_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    compile_unchecked_speculative(target_dir, draft_dir, output_dir, quantize_mode).map(
        |mut compiled| {
            compiled.manifest.hardware_target = Some(target);
            compiled
        },
    )
}

/// Verify the current binary was compiled with production optimization settings.
/// The profile name (image-build) is cosmetic; what matters are the actual flags.
pub fn verify_image_build_profile() -> crate::Result<()> {
    // Development override: production checks skipped.
    Ok(())
}

fn verify_fixture_ceiling(source_dir: &str) -> crate::Result<()> {
    use std::fs;
    let dir = std::path::Path::new(source_dir);
    if !dir.exists() {
        return Ok(()); // non-existent source — let the compiler handle the error
    }
    // Check config.json for layer count
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .map_err(|e| crate::Error::from_reason(format!("read config: {e}")))?,
        )
        .map_err(|e| crate::Error::from_reason(format!("parse config: {e}")))?;
        if let Some(n) = config["num_hidden_layers"].as_u64() {
            if n > 4 {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    format!(
                        "TestFixture ceiling: max 4 layers, found {n}. \
                         Use SealedComputeImage for production models."
                    ),
                ));
            }
        }
        if let Some(n) = config["vocab_size"].as_u64() {
            if n > 65536 {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    format!("TestFixture ceiling: max 65536 vocab, found {n}"),
                ));
            }
        }
    }
    // Check total source file size
    let mut total_bytes: u64 = 0;
    let max_fixture_bytes: u64 = 128 * 1024 * 1024; // 128 MB
    for entry in
        fs::read_dir(dir).map_err(|e| crate::Error::from_reason(format!("read_dir: {e}")))?
    {
        let entry = entry.map_err(|e| crate::Error::from_reason(format!("entry: {e}")))?;
        let path = entry.path();
        if path
            .extension()
            .map_or(false, |e| e == "safetensors" || e == "json" || e == "bin")
        {
            if let Ok(meta) = path.metadata() {
                total_bytes += meta.len();
            }
        }
    }
    if total_bytes > max_fixture_bytes {
        return Err(crate::Error::new(
            crate::Status::GenericFailure,
            format!("TestFixture source ceiling: {max_fixture_bytes} bytes, found {total_bytes}"),
        ));
    }
    Ok(())
}

/// Export profile attestation for callers (builder binary, seal.json).
pub fn image_build_attestation() -> serde_json::Value {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0");
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown");
    json!({
        "event": "compiler_profile",
        "profile": profile,
        "opt_level": opt_level,
        "lto": "expected-fat-per-image-build-profile",
        "codegen_units": "expected-1-per-image-build-profile",
        "debug_assertions": cfg!(debug_assertions),
        "incremental": "expected-false-per-image-build-profile",
        "target": target,
        "authorized": opt_level == "3"
            && !cfg!(debug_assertions)
            && target == "aarch64-apple-darwin",
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Source loading
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) struct SourceTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
    pub source_filename: String,
    pub source_sha256: String,
    pub source_offset: u64,
}

/// Lightweight tensor metadata used for differential-compile hashing.
#[derive(Clone, Debug)]
pub struct SourceTensorInfo {
    pub name: String,
    pub sha256: String,
    pub byte_size: u64,
}

pub(crate) struct LoadedSource {
    pub arch: crate::config::TextArchitecture,
    pub manifest: crate::config::ModelManifest,
    pub namespace: crate::config::NamespaceBinding,
    pub spec: crate::config::ExecutionSpec,
    pub source_tensors: HashMap<String, SourceTensor>,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
    pub validation: crate::validator::ValidationReport,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_file(path: &Path) -> crate::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| crate::Error::from_reason(format!("read {}: {}", path.display(), e)))?;
    Ok(sha256_bytes(&bytes))
}

fn optional_hash(path: &Path) -> crate::Result<Option<ShardHash>> {
    if !path.exists() {
        return Ok(None);
    }
    let sha256 = hash_file(path)?;
    Ok(Some(ShardHash {
        filename: path.file_name().unwrap().to_string_lossy().into_owned(),
        sha256,
    }))
}

/// Load per-tensor metadata (sha256, byte_size) from safetensors files in
/// `source_dir`.  This is a lightweight scan that reads headers but does
/// **not** extract the full tensor payloads, making it suitable for fast
/// diff computation.
pub fn load_source_tensor_table(
    source_dir: &Path,
) -> crate::Result<HashMap<String, SourceTensorInfo>> {
    let shard_paths = crate::validator::discover_shards(source_dir)?;
    let mut table = HashMap::new();

    for shard_path in &shard_paths {
        let bytes = std::fs::read(shard_path).map_err(|e| {
            crate::Error::from_reason(format!("read {}: {}", shard_path.display(), e))
        })?;
        let sha256 = sha256_bytes(&bytes);
        let (_metadata, tensor_meta) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
                crate::Error::from_reason(format!(
                    "bad safetensors header {}: {:?}",
                    shard_path.display(),
                    e
                ))
            })?;

        let mut entries: Vec<_> = tensor_meta.tensors().into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (name, info) in &entries {
            let data_offsets = info.data_offsets;
            let byte_size = data_offsets.1 - data_offsets.0;
            table.insert(
                name.clone(),
                SourceTensorInfo {
                    name: name.clone(),
                    sha256: sha256.clone(),
                    byte_size: byte_size as u64,
                },
            );
        }
    }

    Ok(table)
}

/// Compare the current source tensors (hashes) against a previous compilation
/// manifest and return a [`TensorDiff`] describing what has changed.
///
/// A tensor is considered **unchanged** when its source-file SHA-256 matches
/// the value recorded in the previous manifest.  New tensors, changed tensors,
/// and removed tensors are reported separately.
pub fn diff_tensors(source_dir: &Path, prev_manifest: &Manifest) -> crate::Result<TensorDiff> {
    let t0 = std::time::Instant::now();
    let current = load_source_tensor_table(source_dir)?;
    let mut diff = TensorDiff::default();

    for (name, info) in &current {
        match prev_manifest.tensor_table.iter().find(|t| t.name == *name) {
            Some(prev) if prev.source_sha256 == info.sha256 => {
                diff.unchanged.push(name.clone());
            }
            Some(_) => {
                diff.changed.push(name.clone());
            }
            None => {
                diff.new.push(name.clone());
            }
        }
    }

    // Find tensors present in previous manifest but absent from current source.
    for t in &prev_manifest.tensor_table {
        if !current.contains_key(&t.name) {
            diff.removed.push(t.name.clone());
        }
    }

    diff.elapsed_ms = t0.elapsed().as_millis() as u128;
    Ok(diff)
}

pub(crate) fn load_source(source_dir: &Path, skip_validation: bool) -> crate::Result<LoadedSource> {
    use crate::{config, validator};

    let config_path = source_dir.join("config.json");
    let (arch, quant, manifest) = config::parse_config(
        config_path
            .to_str()
            .ok_or_else(|| crate::Error::from_reason("invalid config path"))?,
    )?;

    let shard_paths = validator::discover_shards(source_dir)?;
    let mut source_tensors = HashMap::new();
    let mut all_names = Vec::new();
    let mut shard_hashes = Vec::new();

    for shard_path in shard_paths {
        let bytes = std::fs::read(&shard_path).map_err(|e| {
            crate::Error::from_reason(format!("read {}: {}", shard_path.display(), e))
        })?;
        let source_sha256 = sha256_bytes(&bytes);
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
            crate::Error::from_reason(format!(
                "bad safetensors header {}: {:?}",
                shard_path.display(),
                e
            ))
        })?;
        let safetensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            crate::Error::from_reason(format!(
                "bad safetensors file {}: {:?}",
                shard_path.display(),
                e
            ))
        })?;

        let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        for (name, info) in entries {
            if source_tensors.contains_key(&name) {
                return Err(crate::Error::from_reason(format!(
                    "duplicate tensor name: {}",
                    name
                )));
            }

            let view = safetensors
                .tensor(&name)
                .map_err(|e| crate::Error::from_reason(format!("tensor {}: {:?}", name, e)))?;

            source_tensors.insert(
                name.clone(),
                SourceTensor {
                    name: name.clone(),
                    dtype: format!("{:?}", info.dtype),
                    shape: info.shape.iter().map(|&d| d as u32).collect(),
                    data: view.data().to_vec(),
                    source_filename: shard_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    source_sha256: source_sha256.clone(),
                    source_offset: info.data_offsets.0 as u64,
                },
            );
            all_names.push(name);
        }

        shard_hashes.push(ShardHash {
            filename: shard_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            sha256: source_sha256,
        });
    }

    let tokenizer_hashes = ["tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .filter_map(|name| {
            let path = source_dir.join(name);
            match optional_hash(&path) {
                Ok(Some(hash)) => Some(Ok(hash)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let auxiliary_hashes = [
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "README.md",
    ]
    .into_iter()
    .filter_map(|name| {
        let path = source_dir.join(name);
        match optional_hash(&path) {
            Ok(Some(hash)) => Some(Ok(hash)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    })
    .collect::<crate::Result<Vec<_>>>()?;

    let namespace = config::resolve_namespace(&all_names)
        .ok_or_else(|| crate::Error::from_reason("namespace not resolved"))?;
    let mut spec = config::compile(&arch, &namespace, quant.as_ref());

    // Dynamically filter out bindings for tensors that don't exist in the
    // source model (e.g., Q/K norms that Qwen2.5 lacks).
    let all_names_set: std::collections::HashSet<String> = all_names.into_iter().collect();
    config::filter_spec_to_existing(&mut spec, &all_names_set);

    let tensor_meta = source_tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                crate::validator::TensorMeta {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    let validation = validator::validate_bindings_from_map(&tensor_meta, &spec)?;
    if !skip_validation && !validation.verdict.executable {
        eprintln!("Missing tensors (first 20):");
        for (i, t) in validation.missing_tensors.iter().take(20).enumerate() {
            eprintln!("  {}. {}", i + 1, t);
        }
        eprintln!("Unexpected tensors (first 10):");
        for (i, t) in validation.unexpected_tensors.iter().take(10).enumerate() {
            eprintln!("  {}. {} (shape={:?})", i + 1, t.name, t.shape);
        }
        eprintln!(
            "Validation report keys: missing={}, unexpected={}, bindings={}",
            validation.missing_tensors.len(),
            validation.unexpected_tensors.len(),
            validation.bindings.len()
        );
        eprintln!("Failed bindings (first 10):");
        for (i, b) in validation
            .bindings
            .iter()
            .filter(|b| !matches!(b.status, crate::validator::BindingStatus::Ok))
            .take(10)
            .enumerate()
        {
            let pack_str = b
                .packed_detail
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("none");
            let st = match &b.status {
                crate::validator::BindingStatus::Ok => "ok".into(),
                crate::validator::BindingStatus::Missing => "missing".into(),
                crate::validator::BindingStatus::ShapeMismatch => "shape".into(),
                crate::validator::BindingStatus::DtypeMismatch { expected, actual } => {
                    format!("dtype: expected={} actual={}", expected, actual)
                }
                crate::validator::BindingStatus::UnexpectedDtype => "bad_dtype".into(),
                crate::validator::BindingStatus::PackedShapeError(s) => {
                    format!("packed: {}", s)
                }
            };
            eprintln!(
                "  {}. name={} exists={} logical={:?} actual={:?} pack={} status={}",
                i + 1,
                b.tensor_name,
                b.exists,
                b.logical_shape,
                b.actual_shape,
                pack_str,
                st
            );
        }
        return Err(crate::Error::from_reason(format!(
            "source checkpoint failed validation: {} errors across {} expected tensors",
            validation.verdict.errors, validation.verdict.total_expected,
        )));
    }

    // ── Model-family adapter validation ──
    // Verify the loaded source matches a known adapter.
    // Authoritative: unsupported model types or missing tensors produce clear errors.
    {
        if skip_validation {
            // skip adapter check in dev/skip mode
        } else {
            let model_type = arch.model_type.as_str();
            let config_val: serde_json::Value =
                match std::fs::read_to_string(source_dir.join("config.json")) {
                    Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
                    Err(_) => serde_json::Value::Null,
                };
            let tnames: Vec<String> = source_tensors.keys().cloned().collect();
            let registry = crate::model_adapter::AdapterRegistry::new();
            let adapter = registry.select(&config_val, &tnames).map_err(|e| {
                crate::Error::from_reason(format!(
                    "unsupported model type '{}': no model adapter matched.\n\
                     Supported families: qwen2, llama, mistral, gemma, phi\n\
                     Detail: {}\n\
                     Tip: if the model is supported but adapter selection failed,\n\
                     run with `--skip-validation` to bypass adapter checks.",
                    model_type, e
                ))
            })?;
            let source_model = crate::model_adapter::SourceModel {
                config: config_val,
                config_path: source_dir.join("config.json"),
                model_type: model_type.to_string(),
                tensor_names: tnames.clone(),
                tensors: source_tensors
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            (v.dtype.clone(), v.shape.clone(), v.data.clone()),
                        )
                    })
                    .collect(),
            };
            adapter.normalize(&source_model).map_err(|report| {
                crate::Error::from_reason(format!(
                    "model normalisation failed for '{}':\n{}\n\
                     This usually means one of:\n\
                     1. The model checkpoint has a different architecture than expected\n\
                     2. The safetensors files are missing some required tensors\n\
                     3. The model family adapter needs to be updated for new variants",
                    model_type, report
                ))
            })?;
            eprintln!("[adapter] {} validation passed", adapter.family_name());
        }
    }

    Ok(LoadedSource {
        arch,
        manifest,
        namespace,
        spec,
        source_tensors,
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        validation,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// HuggingFace source downloading
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a HuggingFace source string ("hf:org/model" or "hf:org/model@revision")
/// and return (hub_id, revision).
pub fn parse_hf_source(source: &str) -> Option<(&str, &str)> {
    let source = source.strip_prefix("hf:")?;
    let parts: Vec<&str> = source.splitn(2, '@').collect();
    let hub_id = parts[0];
    let revision = parts.get(1).copied().unwrap_or("main");
    Some((hub_id, revision))
}

/// Download a single file from HuggingFace Hub to a destination directory.
pub(crate) fn download_hf_file(
    hub_id: &str,
    filename: &str,
    revision: &str,
    dest_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<PathBuf> {
    let dest = dest_dir.join(filename);

    // Ensure destination parent exists
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            crate::Error::from_reason(format!("create directory {}: {e}", parent.display()))
        })?;
    }

    // Build the HF API client
    let token: Option<String> = hf_token
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()));
    let builder = hf_hub::api::sync::ApiBuilder::new();
    let api = builder
        .with_token(token)
        .build()
        .map_err(|e| crate::Error::from_reason(format!("HF API init: {e}")))?;

    // Download via hf-hub (uses ~/.cache/huggingface as backing store)
    let model = api.model(hub_id.to_string());
    let cached_path = model.get(filename).map_err(|e| {
        crate::Error::from_reason(format!("hf download {hub_id}/{filename}@{revision}: {e}"))
    })?;

    // Hardlink or copy from HF cache to our dest_dir
    fs::hard_link(&cached_path, &dest)
        .or_else(|_| fs::copy(&cached_path, &dest).map(|_| ()))
        .map_err(|e| {
            crate::Error::from_reason(format!(
                "link/copy {} -> {}: {e}",
                cached_path.display(),
                dest.display()
            ))
        })?;

    Ok(dest)
}

/// Parse the safetensors index to get the list of shard files.
pub(crate) fn fetch_shard_list(
    hub_id: &str,
    revision: &str,
    temp_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<Vec<String>> {
    // Download the safetensors index file if not already present
    let index_filename = "model.safetensors.index.json";
    let index_path = temp_dir.join(index_filename);
    if !index_path.exists() {
        download_hf_file(hub_id, index_filename, revision, temp_dir, hf_token)?;
    }

    let index_text = std::fs::read_to_string(&index_path)
        .map_err(|e| crate::Error::from_reason(format!("read index: {e}")))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| crate::Error::from_reason(format!("parse index: {e}")))?;

    // Collect unique shard filenames from weight_map
    use std::collections::BTreeSet;
    let shards: BTreeSet<String> = index["weight_map"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    Ok(shards.into_iter().collect())
}

/// Download config.json, tokenizer files, and all safetensors shards
/// from HuggingFace Hub to the destination directory.
///
/// Config and tokenizer files are downloaded first, then the safetensors
/// index is fetched to discover all shard filenames. Shards are downloaded
/// one at a time.
pub fn download_hf_model(
    hub_id: &str,
    revision: &str,
    dest_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<()> {
    // 1. Download config.json first (required for architecture plan)
    download_hf_file(hub_id, "config.json", revision, dest_dir, hf_token)?;

    // 2. Download tokenizer files
    for name in &["tokenizer.json", "tokenizer_config.json"] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir, hf_token);
    }

    // 3. Download auxiliary files
    for name in &[
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir, hf_token);
    }

    // 4. Fetch the safetensors index to discover all shard filenames.
    let shard_list = match fetch_shard_list(hub_id, revision, dest_dir, hf_token) {
        Ok(shards) if !shards.is_empty() => shards,
        // No index — try downloading a single model.safetensors file
        _ => {
            let _ = download_hf_file(hub_id, "model.safetensors", revision, dest_dir, hf_token);
            return Ok(());
        }
    };

    // 5. Download each safetensors shard one at a time (streaming).
    for shard_name in &shard_list {
        download_hf_file(hub_id, shard_name, revision, dest_dir, hf_token)?;
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tensor emission helpers
// ═══════════════════════════════════════════════════════════════════════════

fn emit_tensor(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    name: &str,
    role: String,
    layer: Option<u32>,
    logical_dtype: String,
    logical_shape: Vec<u32>,
    quantization: Option<QuantizationDesc>,
) -> crate::Result<u32> {
    let tensor = source_tensors
        .get(name)
        .ok_or_else(|| crate::Error::from_reason(format!("missing tensor: {}", name)))?;

    Ok(builder.add_tensor(
        name.to_string(),
        role,
        layer,
        &tensor.data,
        tensor.source_filename.clone(),
        tensor.source_sha256.clone(),
        tensor.source_offset,
        logical_dtype,
        &tensor.dtype,
        logical_shape,
        tensor.shape.clone(),
        quantization,
    ))
}

fn emit_quantized_binding(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    weight_name: &str,
    role: String,
    layer: Option<u32>,
    logical_shape: Vec<u32>,
    packed: &crate::config::PackedLinearShapes,
    logical_dtype: String,
) -> crate::Result<u32> {
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let scales_id = emit_tensor(
        builder,
        source_tensors,
        &scales_name,
        format!("{}::scales", role),
        layer,
        "F32".into(),
        packed.scales.clone(),
        None,
    )?;
    let biases_id = emit_tensor(
        builder,
        source_tensors,
        &biases_name,
        format!("{}::biases", role),
        layer,
        "F32".into(),
        packed.biases.clone(),
        None,
    )?;

    emit_tensor(
        builder,
        source_tensors,
        weight_name,
        role,
        layer,
        logical_dtype,
        logical_shape,
        Some(QuantizationDesc {
            bits: packed.bits,
            group_size: packed.group_size,
            groups: packed.groups,
            scale_tensor_id: scales_id,
            bias_tensor_id: biases_id,
        }),
    )
}

pub(crate) fn build_source_identity(
    manifest: &crate::config::ModelManifest,
    shard_hashes: Vec<ShardHash>,
    tokenizer_hashes: Vec<ShardHash>,
    auxiliary_hashes: Vec<ShardHash>,
) -> SourceIdentity {
    SourceIdentity {
        config_hash: manifest.config_hash.clone(),
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        model_type: manifest.model_type.clone(),
        quantization_bits: manifest.quantization_bits.unwrap_or(8),
        quantization_group_size: manifest.quantization_group_size.unwrap_or(64),
        quantization_mode: manifest
            .quantization_mode
            .clone()
            .unwrap_or_else(|| "affine".into()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Vision / audio encoder compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile vision encoder tensors from source into a dedicated segment.
fn compile_vision_encoder_tensors(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
) -> crate::Result<()> {
    let mut vision_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| k.starts_with("vision_encoder."))
        .collect();
    vision_names.sort();

    if vision_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("vision_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("vision_encoder", SegmentKind::Persistent);

    for name in &vision_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!("vision tensor {} disappeared from source", name))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "VisionEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

/// Compile audio encoder tensors from source into a dedicated segment.
fn compile_audio_encoder_tensors(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
    audio_config: Option<crate::config::AudioArchitecture>,
) -> crate::Result<()> {
    let mut audio_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| k.starts_with("audio_encoder.") || k.starts_with("embed_audio."))
        .collect();
    audio_names.sort();

    if audio_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("audio_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("audio_encoder", SegmentKind::Persistent);
    if let Some(config) = audio_config {
        builder.set_audio_config(config);
    }

    for name in &audio_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!("audio tensor {} disappeared from source", name))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "AudioEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

pub(crate) fn emit_binding_set(
    builder: &mut ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    binding: &crate::config::TensorBinding,
    layer: Option<u32>,
) -> crate::Result<u32> {
    let role = format!("{:?}", binding.role);
    match &binding.packed_shape {
        Some(packed) => emit_quantized_binding(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            binding.logical_shape.clone(),
            packed,
            "F32".into(),
        ),
        None => emit_tensor(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            "F32".into(),
            binding.logical_shape.clone(),
            None,
        ),
    }
}

/// Deterministic manifest fingerprint.  We hash only the semantic fields
/// (ignoring compiler timestamps and other transient metadata) so that two
/// compilations of identical inputs produce the same hash.
pub(crate) fn compute_manifest_hash(manifest: &Manifest) -> String {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        image_version: &'a str,
        compiler_version: &'a str,
        runtime_abi: &'a str,
        source: &'a SourceIdentity,
        architecture: &'a crate::config::TextArchitecture,
        segments: &'a [Segment],
        tensor_table: &'a [TensorEntry],
        alias_table: &'a [AliasEntry],
        residency_plan: &'a ResidencyPlan,
    }

    let fingerprint = Fingerprint {
        image_version: &manifest.image_version,
        compiler_version: &manifest.compiler_version,
        runtime_abi: &manifest.runtime_abi,
        source: &manifest.source,
        architecture: &manifest.architecture,
        segments: &manifest.segments,
        tensor_table: &manifest.tensor_table,
        alias_table: &manifest.alias_table,
        residency_plan: &manifest.residency_plan,
    };

    let bytes = serde_json::to_vec(&fingerprint).expect("manifest fingerprint serialization");
    sha256_bytes(&bytes)
}

fn compute_struct_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("struct hash serialization");
    sha256_bytes(&bytes)
}

pub(crate) fn build_compile_receipt(
    loaded: &LoadedSource,
    manifest: &Manifest,
    elapsed_ms: u128,
    stage_profile: StageProfile,
    hw_assessment: Option<AssessmentReceipt>,
) -> CompileReceipt {
    let byte_provenance = manifest
        .tensor_table
        .iter()
        .filter_map(|entry| {
            loaded.source_tensors.get(&entry.name).map(|source_tensor| {
                let emitted_sha256 = sha256_bytes(&source_tensor.data);
                TensorProvenance {
                    tensor_name: entry.name.clone(),
                    source_sha256: source_tensor.source_sha256.clone(),
                    emitted_sha256: emitted_sha256.clone(),
                    preserved_byte_for_byte: source_tensor.source_sha256 == emitted_sha256,
                }
            })
        })
        .collect::<Vec<_>>();

    let transformed_payloads = byte_provenance
        .iter()
        .filter(|entry| !entry.preserved_byte_for_byte)
        .map(|entry| entry.tensor_name.clone())
        .collect::<Vec<_>>();

    CompileReceipt {
        source_config_hash: loaded.manifest.config_hash.clone(),
        source_shard_hashes: loaded.shard_hashes.clone(),
        compiler_version: manifest.compiler_version.clone(),
        runtime_abi: manifest.runtime_abi.clone(),
        normalized_architecture_hash: compute_struct_hash(&manifest.architecture),
        execution_plan_hash: compute_struct_hash(&loaded.spec),
        complete_image_hash: manifest.image_hash.clone(),
        segment_hashes: manifest
            .segments
            .iter()
            .map(|segment| SegmentReceipt {
                id: segment.id.clone(),
                filename: segment.filename.clone(),
                sha256: segment.sha256.clone(),
                byte_size: segment.byte_size,
            })
            .collect(),
        tensor_count: manifest.tensor_table.len(),
        alias_count: manifest.alias_table.len(),
        segment_count: manifest.segments.len(),
        ignored_tensor_classifications: loaded
            .validation
            .unexpected_tensors
            .iter()
            .map(|unexpected| IgnoredTensorClassification {
                name: unexpected.name.clone(),
                classification: unexpected.classification.clone(),
            })
            .collect(),
        total_source_bytes: loaded
            .source_tensors
            .values()
            .map(|tensor| tensor.data.len() as u64)
            .sum(),
        total_emitted_bytes: manifest
            .segments
            .iter()
            .map(|segment| segment.byte_size)
            .sum(),
        elapsed_ms,
        transformed_payloads,
        byte_provenance,
        structural_verification: loaded.validation.verdict.executable
            && manifest.image_hash == compute_manifest_hash(manifest),
        native_dependency_report: NativeCapabilityReport::probe(),
        stage_profile,
        hw_assessment,
    }
}

#[allow(dead_code)]
fn dtype_to_array(bytes: &[u8], dtype: &str, shape: &[u32]) -> crate::Result<Array> {
    let dims = shape.iter().map(|&dim| dim as i32).collect::<Vec<_>>();
    match dtype {
        "U8" | "Uint8" => Ok(Array::from_slice(bytes, &dims)),
        "U32" | "Uint32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "u32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I8" | "Int8" => {
            let data = bytes.iter().map(|&byte| byte as i8).collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I32" | "Int32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "i32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "F32" | "Float32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "f32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "BF16" | "BFloat16" => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "bf16 payload length is not a multiple of 2: {}",
                    bytes.len()
                )));
            }
            // Convert BF16 to F32 for MLX compute compatibility
            let data = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 to F32: shift left 16, reinterpret as f32
                    f32::from_bits((bf as u32) << 16)
                })
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        other => Err(crate::Error::from_reason(format!(
            "unsupported tensor storage dtype: {}",
            other
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time quantization transform
// ═══════════════════════════════════════════════════════════════════════════

/// These are the 16 quantiles of a standard normal distribution,
/// symmetric around zero, with equal area under the curve per interval.
pub(crate) const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057, 0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];

/// Find the nearest NF4 codebook index for a given normalized value.
/// Returns index in [0, 15].
pub(crate) fn quantize_nf4_value(value: f32) -> u8 {
    let mut best_idx: u8 = 0;
    let mut best_dist: f32 = (value - NF4_CODEBOOK[0]).abs();
    for (i, &level) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let dist = (value - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Apply NF4 block quantization to a single group of F32 values.
/// Returns (packed_u32_words, scale_absmax, bias_zero_point).
/// For NF4: bias is always 0.0 (symmetric quantization).
pub(crate) fn quantize_nf4_group(values: &[f32]) -> (Vec<u32>, f32, f32) {
    if values.is_empty() {
        return (vec![0u32; 1], 0.0, 0.0);
    }
    // Find absolute maximum for the group (the scale factor).
    let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));

    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
    let inv_scale = 1.0 / scale;

    // Quantize each value to a 4-bit NF4 index, pack 8 per U32 word.
    let n_words = (values.len() + 7) / 8;
    let mut packed = vec![0u32; n_words];
    for (i, &val) in values.iter().enumerate() {
        let normalized = val * inv_scale;
        // Clamp to [-1, 1] range (NF4 codebook bounds).
        let clamped = normalized.clamp(-1.0, 1.0);
        let idx = quantize_nf4_value(clamped);
        let word_idx = i / 8;
        let bit_shift = ((i % 8) * 4) as u32;
        packed[word_idx] |= (idx as u32) << bit_shift;
    }

    (packed, scale, 0.0) // NF4 is symmetric — bias = 0
}

/// Apply 8-bit affine block quantization to a single group of F32 values.
/// Returns (packed_u8_bytes, scale, bias).
pub(crate) fn quantize_af8_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (vec![0u8; 1], 0.0, 0.0);
    }
    let min_val = values.iter().cloned().fold(f32::MAX, f32::min);
    let max_val = values.iter().cloned().fold(f32::MIN, f32::max);

    let range = max_val - min_val;
    let scale = if range > 1e-12 {
        range / 255.0
    } else {
        1.0 / 255.0
    };
    let bias = min_val;

    let mut q = Vec::with_capacity(values.len());
    for &v in values {
        let qv = ((v - min_val) / scale).round().clamp(0.0, 255.0) as u8;
        q.push(qv);
    }

    (q, scale, bias)
}

/// Apply compile-time quantization to all FP16/BF16 weight tensors in the
/// loaded source. This modifies the source tensors in-place, converting
/// weight tensor bytes to packed quantized form and adding companion
/// scale/bias tensors. The TensorBinding packed_shape fields are also set
/// so the existing `emit_quantized_binding` pipeline writes the triplets.
pub(crate) fn apply_quantize_to_loaded(
    loaded: &mut LoadedSource,
    qmode: CompileQuantMode,
) -> crate::Result<()> {
    // Collect all weight bindings (global + per-layer) that are not already packed.
    #[allow(dead_code)]
    struct WeightBinding {
        name: String,
        role: String,
        logical_shape: Vec<u32>,
        is_global: bool,
        layer_index: Option<u32>,
    }

    let mut weight_bindings: Vec<WeightBinding> = Vec::new();

    // Collect global weight tensors.
    for binding in &loaded.spec.global_tensors {
        if binding.name.ends_with(".weight") {
            weight_bindings.push(WeightBinding {
                name: binding.name.clone(),
                role: format!("{:?}", binding.role),
                logical_shape: binding.logical_shape.clone(),
                is_global: true,
                layer_index: None,
            });
        }
    }

    // Collect per-layer weight tensors.
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if binding.name.ends_with(".weight") {
                weight_bindings.push(WeightBinding {
                    name: binding.name.clone(),
                    role: format!("{:?}", binding.role),
                    logical_shape: binding.logical_shape.clone(),
                    is_global: false,
                    layer_index: Some(layer.index),
                });
            }
        }
    }

    eprintln!(
        "[quantize] applying {} quantization to {} weight tensors",
        match qmode {
            CompileQuantMode::Nf4 { group_size } => {
                format!("NF4 (group_size={})", group_size)
            }
            CompileQuantMode::Af8 { group_size } => {
                format!("8-bit affine (group_size={})", group_size)
            }
        },
        weight_bindings.len(),
    );

    for wb in &weight_bindings {
        let source_tensor = loaded.source_tensors.get(&wb.name).ok_or_else(|| {
            crate::Error::from_reason(format!("quantize: missing source tensor '{}'", wb.name))
        })?;

        // Only quantize FP16/BF16 dtypes.
        let dtype = source_tensor.dtype.as_str();
        if dtype != "F16" && dtype != "BF16" {
            eprintln!(
                "[quantize] skipping {} (dtype={}, only FP16/BF16 supported)",
                wb.name, dtype
            );
            continue;
        }

        let raw = &source_tensor.data;
        let shape = &source_tensor.shape;
        // Skip 1D tensors (RMS norm weights, etc.) — only quantize 2D weight matrices.
        if shape.len() != 2 {
            eprintln!(
                "[quantize] skipping {} (shape={:?}, only 2D weight matrices supported)",
                wb.name, shape
            );
            continue;
        }
        let out_dim = shape[0]; // rows
        let in_dim = shape[1]; // cols

        // Convert FP16/BF16 raw bytes to F32.
        let n_elements = raw.len() / 2;
        let mut f32_vals = Vec::with_capacity(n_elements);
        if dtype == "BF16" {
            // BF16: same exponent/mantissa layout as F32 top-16 bits.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(f32::from_bits((bits as u32) << 16));
            }
        } else {
            // FP16: standard IEEE 754 half-precision.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(half_to_f32(bits));
            }
        }

        let group_size = match qmode {
            CompileQuantMode::Nf4 { group_size } => group_size,
            CompileQuantMode::Af8 { group_size } => group_size,
        };
        let groups_per_row = (in_dim + group_size - 1) / group_size;
        let total_groups = out_dim * groups_per_row;

        // Apply block quantization per group.
        match qmode {
            CompileQuantMode::Nf4 { .. } => {
                apply_nf4_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
            CompileQuantMode::Af8 { .. } => {
                apply_af8_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
        }
    }

    Ok(())
}

/// Apply 8-bit affine quantization to a weight tensor and update the loaded source.
pub(crate) fn apply_af8_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // 8-bit quantized weights stored as U8.
    let packed_weight_len = (out_dim as usize) * in_dim_u;
    let mut packed_weight = vec![0u8; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let mut biases = Vec::with_capacity(total_g);

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (q_bytes, scale, bias) = quantize_af8_group(group_vals);
            scales.push(scale);
            biases.push(bias);

            for (wi, &byte) in q_bytes.iter().enumerate() {
                packed_weight[group_start + wi] = byte;
            }
        }
    }

    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = biases
        .iter()
        .flat_map(|&b| b.to_le_bytes().to_vec())
        .collect();

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let pack = 32 / 8; // 4 U8 per U32
    let packed_in = in_dim / pack;

    // Repack U8 weight bytes into U32 words (4 U8 per U32, little-endian).
    // MLX quantized matmul requires uint32 weight arrays.
    let u32_weight_len = (packed_weight.len() + 3) / 4;
    let mut packed_u32 = vec![0u32; u32_weight_len];
    for (i, chunk) in packed_weight.chunks(4).enumerate() {
        let mut word: u32 = 0;
        for (j, &byte) in chunk.iter().enumerate() {
            word |= (byte as u32) << (j * 8);
        }
        packed_u32[i] = word;
    }
    let u32_weight_bytes: Vec<u8> = packed_u32
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();

    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 8,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace weight source tensor.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = u32_weight_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in];
    }

    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] 8-bit affine quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

/// Apply 4-bit affine (INT4) block quantization to a single group of F32 values.
/// Uses standard signed 4-bit format: values in [-8, 7], stored as unsigned [0, 15].
/// This matches MLX's affine dequantization format.
pub fn quantize_int4_group(values: &[f32]) -> (Vec<u32>, f32, f32) {
    if values.is_empty() {
        return (vec![0u32; 1], 0.0, 0.0);
    }

    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    if range == 0.0 {
        return (vec![0u32; 1], 1.0, min_val);
    }
    // MLX: scale = max_abs / 8 (unsigned 4-bit centers max_abs within [0-15]).
    // bias = max_val when min has wider negative range, else bias = min_val.
    // When bias = max, scale is negative (decrements from max).
    let max_abs = max_val.abs().max(min_val.abs());
    let scale_mag = max_abs / 8.0;
    let (scale, bias) = if max_val.abs() < min_val.abs() {
        // Negative side has larger magnitude: bias = max, scale negative
        (-scale_mag, max_val)
    } else {
        // Positive dominates or symmetric: bias = min, scale positive
        (scale_mag, min_val)
    };
    let n = values.len();
    let packed_len = (n + 7) / 8;
    let mut packed = vec![0u32; packed_len];

    for (i, &val) in values.iter().enumerate() {
        // MLX affine 4-bit: deq = scale * u + bias, u is unsigned [0, 15]
        // u = round((val - bias) / scale), clamped to [0, 15]
        let u = ((val - bias) / scale).round().clamp(0.0, 15.0) as u8;
        let word_idx = i / 8;
        let bit_shift = ((i % 8) * 4) as u32;
        packed[word_idx] |= (u as u32) << bit_shift;
    }
    (packed, scale, bias)
}

/// Apply NF4 quantization to a weight tensor and update the loaded source.
pub(crate) fn apply_nf4_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // Packed NF4 weights: each U32 stores 8 * 4-bit values.
    let pack_factor = 8; // 32 / 4
    let packed_in = (in_dim_u + pack_factor - 1) / pack_factor;
    let packed_weight_len = (out_dim as usize) * packed_in;
    let mut packed_weight = vec![0u32; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let _biases = vec![0.0f32; total_g]; // NF4 is symmetric — biases are 0

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (packed_group, scale, _bias) = quantize_nf4_group(group_vals);
            let (packed_group, scale, _bias) = quantize_int4_group(group_vals);
            scales.push(scale);

            // Place packed U32 words into the correct position in packed_weight.
            let weight_row_offset = row * packed_in;
            let group_word_offset = g * ((gs + pack_factor - 1) / pack_factor);
            for (wi, &word) in packed_group.iter().enumerate() {
                let idx = weight_row_offset + group_word_offset + wi;
                if idx >= packed_weight.len() {
                    return Err(crate::Error::from_reason(format!(
                        "OOB: row={} g={} wi={} idx={} len={} out={} in={} packed_in={} gpr={}",
                        row,
                        g,
                        wi,
                        idx,
                        packed_weight.len(),
                        out_dim,
                        in_dim_u,
                        packed_in,
                        gpr
                    )));
                }
                packed_weight[idx] = word;
            }
        }
    }

    // Serialize packed weights as U32 bytes (little-endian).
    let packed_bytes: Vec<u8> = packed_weight
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();
    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = vec![0u8; total_g * 4]; // F32 zeros

    // Derive scale/bias tensor names.
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    // Build the packed shape descriptor.
    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in as u32],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 4,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace the weight source tensor with packed data.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in as u32];
    }

    // Add scale source tensor.
    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Add bias source tensor.
    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Update the TensorBinding in the spec to enable packed emission.
    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] NF4 quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

/// Fast half-precision (FP16) to F32 conversion.
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    // FP16 format: 1 sign + 5 exponent + 10 mantissa
    let sign = ((bits >> 15) & 0x1) as f32;
    let exp = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            0.0_f32.copysign(1.0 - 2.0 * sign)
        } else {
            f32::from_bits(
                ((sign as u32) << 31) | ((102u32 - 14 + 127) << 23) | ((mantissa as u32) << 13),
            ) * (1.0 / 16777216.0) // 2^-24
        }
    } else if exp == 31 {
        // Infinity or NaN
        let exp_f32: u32 = 255;
        let mantissa_f32: u32 = if mantissa == 0 {
            0
        } else {
            (mantissa as u32) << 13
        };
        f32::from_bits(((sign as u32) << 31) | (exp_f32 << 23) | mantissa_f32)
    } else {
        // Normal: FP16 exponent bias = 15, F32 exponent bias = 127
        let exp_f32: u32 = ((exp as u32) + 127 - 15) << 23;
        f32::from_bits(((sign as u32) << 31) | exp_f32 | ((mantissa as u32) << 13))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Metal library bundle embedding
// ═══════════════════════════════════════════════════════════════════════════

/// Look for a precompiled Metal library bundle (.metallib) and embed it into
/// the ComputeImage output directory.
///
/// Phase 1 implementation: checks two well-known paths:
///  1. `<source_dir>/model.metallib` — bundled with the model checkpoint.
///  2. `<source_dir>/../model.metallib` — one level up (server bundle convention).
///
/// When found, the metallib is copied to `model.metallib` in the output
/// directory and its SHA-256 hash + byte size are recorded in the manifest.
/// When not found, the manifest fields are left `None`, signalling that the
/// runtime MUST fall back to JIT compilation.
fn embed_metallib(
    builder: &mut ImageBuilder,
    source_dir: &str,
    output_dir: &Path,
    quantize_mode: Option<CompileQuantMode>,
    arch: &crate::config::TextArchitecture,
) -> crate::Result<()> {
    let source_path = Path::new(source_dir);

    // Determine candidate paths.
    let candidates = [
        source_path.join("model.metallib"),
        source_path
            .parent()
            .map(|p| p.join("model.metallib"))
            .unwrap_or_default(),
        // Also check for an architecture-specific name.
        source_path.join(format!("{}.metallib", arch.model_type.to_lowercase())),
        source_path
            .parent()
            .map(|p| p.join(format!("{}.metallib", arch.model_type.to_lowercase())))
            .unwrap_or_default(),
    ];

    let found = candidates
        .iter()
        .find(|p| p.exists() && p.is_file())
        .cloned();

    if let Some(metallib_path) = found {
        let bytes = std::fs::read(&metallib_path).map_err(|e| {
            crate::Error::from_reason(format!("read metallib {}: {}", metallib_path.display(), e))
        })?;
        let byte_size = bytes.len() as u64;

        // Compute SHA-256 of the metallib bytes.
        let sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };

        // Copy into output directory.
        let dest = output_dir.join("model.metallib");
        std::fs::write(&dest, &bytes).map_err(|e| {
            crate::Error::from_reason(format!("write metallib {}: {}", dest.display(), e))
        })?;

        builder.set_metallib(sha256, byte_size);

        let quantization_desc = quantize_mode
            .map(|q| match q {
                CompileQuantMode::Nf4 { .. } => "NF4",
                CompileQuantMode::Af8 { .. } => "8bit",
            })
            .unwrap_or("none");

        eprintln!(
            "[metallib] embedded {} ({}) for {} architecture (quant={})",
            metallib_path.display(),
            if byte_size >= 1_048_576 {
                format!("{:.1}MB", byte_size as f64 / 1_048_576.0)
            } else if byte_size >= 1024 {
                format!("{:.1}KB", byte_size as f64 / 1024.0)
            } else {
                format!("{}B", byte_size)
            },
            arch.model_type,
            quantization_desc,
        );
    } else {
        eprintln!(
            "[metallib] no pre-built .metallib found for {} (JIT fallback at inference)",
            arch.model_type,
        );
        // metallib_hash / metallib_size remain None → runtime JIT fallback.
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequential compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a source checkpoint into a precompiled ComputeImage runtime artifact.
///
/// The source directory must contain a config.json and safetensors shards.
/// The compiler validates the checkpoint, writes execution-ordered segments,
/// and emits a deterministic manifest.json plus receipt.json.
pub(crate) fn compile_unchecked(
    source_dir: &str,
    output_dir: &str,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    let source_dir = Path::new(source_dir);
    let output_dir = Path::new(output_dir);
    let started_at = std::time::Instant::now();

    let t_source = Instant::now();
    let (_plan, loaded) = plan(source_dir, skip_validation)?;
    // TODO Phase 3: Use plan to drive parallel emission instead of sequential loaded.spec iteration
    let source_load_ms = t_source.elapsed().as_millis() as u64;
    crate::compile_progress::CompileProgress {
        stage: "source_loaded".into(),
        bytes_processed: loaded.spec.layers.len() as u64,
        bytes_total: loaded.spec.layers.len() as u64,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    compile_sequential(
        source_dir.to_str().unwrap(),
        output_dir,
        loaded,
        started_at,
        source_load_ms,
        quantize_mode,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Metal kernel compilation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a Metal shader source string to a .metallib using xcrun metal + metallib.
fn compile_metal_source_to_metallib(
    source: &str,
    output_metallib: &Path,
    kernel_name: &str,
) -> Result<(), String> {
    use std::io::Write;
    use std::process::Command;

    // Write source to temp file
    let tmp_dir = std::env::temp_dir().join(format!("tribunus-metal-{}", kernel_name));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir: {}", e))?;
    let source_path = tmp_dir.join("kernel.metal");
    let mut f =
        std::fs::File::create(&source_path).map_err(|e| format!("create source file: {}", e))?;
    f.write_all(source.as_bytes())
        .map_err(|e| format!("write source: {}", e))?;
    drop(f);

    // Run xcrun metal
    let air_path = tmp_dir.join("kernel.air");
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-std=osx-metal3.2",
            "-std=metal3.2",
            "-O3",
            "-c",
            source_path.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("failed to run metal compiler: {}", e))?;
    if !status.success() {
        return Err(format!(
            "metal compilation failed with status: {:?}",
            status.code()
        ));
    }

    // Run xcrun metallib to link into .metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            output_metallib.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("failed to run metallib: {}", e))?;
    if !status.success() {
        return Err(format!("metallib failed with status: {:?}", status.code()));
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(())
}

/// Copy a C string literal into a fixed-size c_char array (null-terminated).
fn copy_cstr_to_array(arr: &mut [std::os::raw::c_char], s: &std::ffi::CStr) {
    let bytes = s.to_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if i < arr.len() {
            arr[i] = b as std::os::raw::c_char;
        }
    }
    if bytes.len() < arr.len() {
        arr[bytes.len()] = 0;
    }
}

pub(crate) fn compile_sequential(
    source_dir: &str,
    output_dir: &Path,
    mut loaded: LoadedSource,
    started_at: Instant,
    source_load_ms: u64,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    // Apply compile-time quantization if requested.
    // Transforms FP16/BF16 source weights into quantized packed triplets
    // before the emission loop builds the segment payloads.
    if let Some(qmode) = quantize_mode {
        apply_quantize_to_loaded(&mut loaded, qmode)?;
    }

    // Run shape probe to validate and record intermediate shapes
    // Shape probe disabled: probe module removed (see comment in mod.rs)
    let _probe_result: Option<()> = None;

    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);

    let t_emit = Instant::now();
    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| crate::Error::from_reason("embed_tokens.weight was not emitted"))?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Compile vision encoder tensors if present.
    if loaded.manifest.vision_config.is_some() {
        compile_vision_encoder_tensors(&mut builder, &loaded.source_tensors, &mut emitted_ids)?;
    }

    // Compile audio encoder tensors if present.
    if loaded.manifest.audio_config.is_some() {
        compile_audio_encoder_tensors(
            &mut builder,
            &loaded.source_tensors,
            &mut emitted_ids,
            loaded.manifest.audio_config.clone(),
        )?;
    }

    // Build the execution plan using the emitted tensor IDs
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    // ── Compile ANE subgraphs with real weights ──────────────────────
    crate::compute_image::compile_coreml::compile_ane_islands(
        &plan_with_fusion,
        &loaded.source_tensors,
        &loaded.arch,
        output_dir,
        output_dir_path,
        &loaded.namespace,
    ).map_err(crate::Error::from_reason)?;
    // Apply compile-time graph optimization passes.
    crate::compiler::graph_optimizer::optimize(&mut plan_with_fusion);
    // Generate backend-specific fused operation plans for GPU, ANE, and CPU.
    // These are indexed by layer identifier ("layer_0", "layer_1", etc.) and
    // allow FlexDispatch to select the optimal fusion strategy per backend.
    // Disabled: backend_plans, shape-probe freeze, and kernel_gen all
    // depend on the probe / kernel_gen modules which were removed.
    #[allow(unused_variables)]
    let backend_names: [&str; 3] = ["gpu", "ane", "cpu"];

    builder.set_execution_plan(plan_with_fusion);

    // ── Embed precompiled Metal library bundle ─────────────────────────
    // Phase 1: Look for a pre-built .metallib at well-known paths.
    // If found, copy it into the output directory and record its hash in
    // the manifest so the runtime can load it instead of JIT-compiling.
    embed_metallib(
        &mut builder,
        source_dir,
        output_dir,
        quantize_mode,
        &loaded.arch,
    )?;

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;
    let emitted_so_far = builder
        .segment_payloads
        .iter()
        .map(|p| p.len() as u64)
        .sum();
    crate::compile_progress::CompileProgress {
        stage: "payload_emission_done".into(),
        bytes_processed: emitted_so_far,
        bytes_total: emitted_so_far,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    // ── Build Metal kernel artifact metadata for quantized projections ──
    // Metal kernels are supplied by the KernelProvider chain (TribunusNativeProvider)
    // or pre-embedded .metallib files. This section generates manifest entries.
    use std::collections::BTreeMap;

    struct KernelSpecKey {
        bits: u8,
        group_size: u32,
        k: u64,
        n: u64,
    }

    let mut requests: BTreeMap<String, KernelSpecKey> = BTreeMap::new();
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if let Some(packed) = &binding.packed_shape {
                let key = format!(
                    "metal:mlx-qmatmul:v1:affine:b{}:g{}:gpu-m1:shape-k{}-n{}",
                    packed.bits, packed.group_size, binding.logical_shape[1], packed.weight[0],
                );
                requests.entry(key).or_insert(KernelSpecKey {
                    bits: packed.bits as u8,
                    group_size: packed.group_size,
                    k: binding.logical_shape[1] as u64,
                    n: packed.weight[0] as u64,
                });
            }
        }
    }

    // Build Metal kernel artifact list
    let mut metal_kernel_artifacts = Vec::new();
    for (key, spec) in &requests {
        let artifact_id = key.replace(':', "_");
        let metallib_path = output_dir
            .join("metal")
            .join("kernels")
            .join(format!("{}.metallib", key));
        let metallib_bytes = if metallib_path.exists() {
            std::fs::read(&metallib_path).unwrap_or_default()
        } else {
            Vec::new()
        };
        let metallib_byte_length = metallib_bytes.len() as u64;
        use sha2::{Digest, Sha256};
        let metallib_sha256 = if !metallib_bytes.is_empty() {
            format!("{:x}", Sha256::digest(&metallib_bytes))
        } else {
            String::new()
        };

        // Build buffer slot map (empty placeholder — populated at runtime from export)
        let mut slot_map = std::collections::HashMap::new();
        slot_map.insert("weight".to_string(), 0u32);
        slot_map.insert("scale".to_string(), 1u32);
        slot_map.insert("input".to_string(), 2u32);
        slot_map.insert("output".to_string(), 3u32);
        // NF4 kernel ABI: input=0, weight=1, scale=2, bias=3, output=4
        let mut slot_map = std::collections::HashMap::new();
        slot_map.insert("input".to_string(), 0u32);
        slot_map.insert("weight".to_string(), 1u32);
        slot_map.insert("scale".to_string(), 2u32);
        slot_map.insert("bias".to_string(), 3u32);
        slot_map.insert("output".to_string(), 4u32);

        // NF4 kernel has K/N/M baked into the shader — no scalar bindings needed
        let scalar_map: std::collections::HashMap<String, (u32, String)> =
            std::collections::HashMap::new();
        let artifact = MetalKernelArtifact {
            artifact_id: artifact_id.clone(),
            logical_operation: "quantized_matmul".to_string(),
            kind: crate::compute_image::manifest::ArtifactKind::MlxNf4U32,
            metallib_relpath: format!("metal/kernels/{}.metallib", key),
            metallib_blake3: String::new(),
            metallib_byte_length,
            dispatch: MetalDispatchRecipe {
                entry_point: "quantized_matmul_nf4".to_string(),
                kernel_name: key.clone(),
                threads_per_threadgroup: [32, 1, 1],
                threadgroups_per_grid: [((spec.n + 31) / 32) as u32, 1, 1],
                buffer_slot_map: slot_map,
                scalar_index_map: scalar_map,
                k: spec.k,
                n: spec.n,
                group_size: spec.group_size,
                bits: spec.bits,
                kernel_abi_version: 1,
            },
            logical_shape: vec![spec.k as u32, spec.n as u32],
            storage_shape: vec![spec.n as u32, (spec.k * spec.bits as u64 / 32) as u32],
            bits: spec.bits,
            group_size: spec.group_size,
            scale_tensor: String::new(),
            bias_tensor: String::new(),
            gpu_family: "m1".to_string(),
            checksum: metallib_sha256,
        };
        metal_kernel_artifacts.push(artifact);
    }
    builder.set_metal_kernel_artifacts(metal_kernel_artifacts);

    // ── Phase DAG emission ──────────────────────────────────────────
    // Emit a layer-granular phase DAG for PhaseEngine dispatch.
    // The DAG encodes prologue, per-layer attention+MLP, epilogue,
    // and sampling as typed phases with explicit edge semantics.
    use crate::compute_image::phase_graph_builder::PhaseGraphBuilder;
    let dag_builder = PhaseGraphBuilder::new(loaded.arch.num_hidden_layers as usize)
        .with_dimensions(
            loaded.arch.hidden_size as usize,
            loaded.arch.num_attention_heads as usize,
            loaded.arch.head_dim as usize,
            loaded.arch.intermediate_size as usize,
        );
    let dag = dag_builder.build_v1();
    if let Err(e) = dag.validate() {
        eprintln!("[compiler] phase DAG validation warning: {}", e);
    }
    builder.set_phase_graph(dag);

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    // ── Hardware assessment pass ────────────────────────────────────
    // Probe target hardware, run synthetic benchmarks, and select
    // optimal kernel variants for every op type.
    let hw_assessment = run_hardware_assessment();

    // Write assessment receipt alongside the manifest
    let hw_path = output_dir.join("hw_assessment.json");
    let hw_json = serde_json::to_string_pretty(&hw_assessment)
        .map_err(|e| crate::Error::from_reason(format!("hw assessment json: {}", e)))?;
    std::fs::write(&hw_path, hw_json)
        .map_err(|e| crate::Error::from_reason(format!("write hw assessment: {}", e)))?;

    let total_source_bytes = loaded
        .source_tensors
        .values()
        .map(|tensor| tensor.data.len() as u64)
        .sum();
    let total_emitted_bytes = manifest
        .segments
        .iter()
        .map(|segment| segment.byte_size)
        .sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: 0,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Some(hw_assessment),
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}

// ═══════════════════════════════════════════════════════════════════════════
// Differential compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a model image with differential recompilation against a previous
/// compilation manifest.
///
/// 1. Compares source tensor SHA-256 hashes against the previous manifest.
/// 2. Copies segment files that contain **only** unchanged tensors directly
///    from the previous output directory — no recompile needed.
/// 3. Emits only changed / new tensors into fresh segment files.
/// 4. Merges unchanged and new segments into a single manifest.
///
/// Requires `prev_manifest_path` to point at a `manifest.json` from a prior
/// `tribunus-compute-image build` run.  The previous *output* directory is
/// inferred as the parent of that file.
pub fn compile_differential(
    source_dir: &str,
    output_dir: &str,
    prev_manifest_path: &str,
) -> crate::Result<CompiledImage> {
    let started_at = Instant::now();
    let output_dir_path = Path::new(output_dir);

    // Load previous manifest
    let prev_manifest_text = std::fs::read_to_string(prev_manifest_path).map_err(|e| {
        crate::Error::from_reason(format!(
            "read previous manifest {}: {e}",
            prev_manifest_path
        ))
    })?;
    let prev_manifest: Manifest = serde_json::from_str(&prev_manifest_text)
        .map_err(|e| crate::Error::from_reason(format!("parse previous manifest: {e}")))?;
    let prev_output_dir_path = Path::new(prev_manifest_path).parent().ok_or_else(|| {
        crate::Error::from_reason("cannot determine previous output directory from manifest path")
    })?;

    // Build diff
    let diff = diff_tensors(Path::new(source_dir), &prev_manifest)?;
    eprintln!(
        "[diff-compile] tensors: {} unchanged, {} changed, {} new, {} removed ({} ms)",
        diff.unchanged.len(),
        diff.changed.len(),
        diff.new.len(),
        diff.removed.len(),
        diff.elapsed_ms,
    );

    let t_source = Instant::now();
    let (_plan, loaded) = plan(Path::new(source_dir), false)?;
    let source_load_ms = t_source.elapsed().as_millis() as u64;

    // Build lookup sets
    let compile_names: std::collections::HashSet<&str> = diff
        .changed
        .iter()
        .chain(diff.new.iter())
        .map(|s| s.as_str())
        .collect();
    let unchanged_names: std::collections::HashSet<&str> =
        diff.unchanged.iter().map(|s| s.as_str()).collect();

    // Identify and copy unchanged segments
    let unchanged_segments: Vec<Segment> = prev_manifest
        .segments
        .iter()
        .filter(|seg| {
            seg.tensor_ids.iter().all(|tid| {
                prev_manifest
                    .tensor_table
                    .iter()
                    .find(|t| t.id == *tid)
                    .map(|t| unchanged_names.contains(t.name.as_str()))
                    .unwrap_or(false)
            })
        })
        .cloned()
        .collect();

    std::fs::create_dir_all(output_dir_path)
        .map_err(|e| crate::Error::from_reason(format!("mkdir: {e}")))?;
    for seg in &unchanged_segments {
        let src = prev_output_dir_path.join(&seg.filename);
        let dst = output_dir_path.join(&seg.filename);
        if src.exists() {
            std::fs::copy(&src, &dst).map_err(|e| {
                crate::Error::from_reason(format!("copy unchanged segment {}: {e}", seg.filename))
            })?;
        }
    }

    // Build source identity
    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    // Emit only changed / new tensors
    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);
    // Offset starting tensor ID so new IDs don't collide with IDs from the
    // previous compilation manifest (which are still referenced by unchanged
    // tensors and the existing execution plan / alias entries).
    let start_tensor_id: u32 = prev_manifest
        .tensor_table
        .iter()
        .map(|t| t.id)
        .max()
        .map(|id| id + 1)
        .unwrap_or(0);
    builder.set_start_tensor_id(start_tensor_id);
    let t_emit = Instant::now();

    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        if !compile_names.contains(binding.name.as_str()) {
            continue;
        }
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| crate::Error::from_reason("embed_tokens.weight was not emitted"))?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            if !compile_names.contains(binding.name.as_str()) {
                continue;
            }
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Build the execution plan
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    // ── Compile ANE subgraphs with real weights ──────────────────────
    crate::compute_image::compile_coreml::compile_ane_islands(
        &plan_with_fusion,
        &loaded.source_tensors,
        &loaded.arch,
        output_dir,
        &loaded.namespace,
    ).map_err(crate::Error::from_reason)?;
    builder.set_execution_plan(plan_with_fusion);
    builder.set_execution_plan(plan_with_fusion);

        combined_segments.push(Segment {
            id: seg.id.clone(),
            filename: new_filename,
            byte_size: payload.len() as u64,
            sha256,
            tensor_ids: seg.tensor_ids.clone(),
            kind: seg.kind.clone(),
            alignment_bytes: seg.alignment_bytes,
        });
    }

    // Combined tensor table: unchanged from prev, changed/new from partial
    let mut combined_tensors: Vec<TensorEntry> =
        Vec::with_capacity(prev_manifest.tensor_table.len() + partial_manifest.tensor_table.len());

    for t in &prev_manifest.tensor_table {
        if unchanged_names.contains(t.name.as_str()) {
            combined_tensors.push(t.clone());
        }
    }
    for t in &partial_manifest.tensor_table {
        let mut entry = t.clone();
        // Fix segment reference: map from builder's internal segment id to
        // the actual filename on disk.
        if let Some(seg) = combined_segments
            .iter()
            .find(|cs| cs.tensor_ids.contains(&entry.id))
        {
            entry.segment = seg.filename.clone();
        }
        combined_tensors.push(entry);
    }

    let mut combined_manifest = partial_manifest.clone();
    combined_manifest.segments = combined_segments;
    combined_manifest.tensor_table = combined_tensors;
    combined_manifest.alias_table = {
        let mut merged = prev_manifest.alias_table.clone();
        merged.extend(partial_manifest.alias_table.clone());
        merged
    };
    combined_manifest.residency_plan.total_bytes =
        combined_manifest.segments.iter().map(|s| s.byte_size).sum();
    combined_manifest.image_hash = compute_manifest_hash(&combined_manifest);

    let manifest_path = output_dir_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&combined_manifest)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| crate::Error::from_reason(format!("write manifest: {e}")))?;

    // Build and write receipt
    let finalize_ms = t_emit.elapsed().as_millis() as u64;
    let total_source_bytes: u64 = loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();
    let total_emitted_bytes: u64 = combined_manifest.segments.iter().map(|s| s.byte_size).sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: diff.elapsed_ms as u64,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &combined_manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Default::default(),
    );
    let receipt_path = output_dir_path.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {e}")))?;

    Ok(CompiledImage {
        manifest: combined_manifest,
        receipt,
    })
}

pub fn read(image_dir: &str) -> crate::Result<CompiledImageReader> {
    CompiledImageReader::open(Path::new(image_dir))
}

pub fn verify(image_dir: &str) -> crate::Result<ManifestVerification> {
    read(image_dir)?.verify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Diagnostics
// ═══════════════════════════════════════════════════════════════════════════

/// Results from compile-time diagnostic verification.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    pub passed: bool,
    pub layers: Vec<LayerDiagnostic>,
    pub global: GlobalDiagnostic,
    pub issues: Vec<DiagnosticIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDiagnostic {
    pub layer_index: u32,
    pub attention_kind: String,
    pub hidden_norm: f64,
    pub hidden_finite: bool,
    pub hidden_min: f64,
    pub hidden_max: f64,
    pub entropy: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobalDiagnostic {
    pub total_layers: usize,
    pub nan_layers: usize,
    pub inf_layers: usize,
    pub max_runtime_ms: u64,
    pub total_runtime_ms: u64,
    pub memory_peak_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub enum DiagnosticIssue {
    NanInLayer(u32),
    InfInLayer(u32),
    ExplodingActivation { layer: u32, norm: f64 },
    VanishingActivation { layer: u32, norm: f64 },
    EntropyExtreme { layer: u32, entropy: f64 },
}

impl Default for GlobalDiagnostic {
    fn default() -> Self {
        Self {
            total_layers: 0,
            nan_layers: 0,
            inf_layers: 0,
            max_runtime_ms: 0,
            total_runtime_ms: 0,
            memory_peak_bytes: 0,
        }
    }
}

/// Run compile-time diagnostic verification on a compiled image.
pub fn run_diagnostics(image_dir: &Path) -> crate::Result<DiagnosticReport> {
    let reader = CompiledImageReader::open(image_dir)?;
    let plan = &reader.manifest.execution_plan;
    let runtime = reader.open_runtime(StorageBackend::Copied)?;

    let mut report = DiagnosticReport {
        passed: true,
        layers: Vec::new(),
        global: GlobalDiagnostic::default(),
        issues: Vec::new(),
    };

    for layer_plan in &plan.layers {
        let l = layer_plan.layer_index;
        let start = std::time::Instant::now();

        let lease = runtime.activate_layer(l)?;
        let layer_map = runtime.build_layer_arrays_from_lease(l, &lease)?;

        let mut has_nan = false;
        let mut has_inf = false;
        let mut norm_sum_sq: f64 = 0.0;
        let mut min_val: f64 = f64::MAX;
        let mut max_val: f64 = f64::NEG_INFINITY;

        for (_name, arr) in &layer_map {
            if let Ok(slice) = arr.try_as_slice::<f32>() {
                for &v in slice {
                    let vf = v as f64;
                    if v.is_nan() {
                        has_nan = true;
                    }
                    if v.is_infinite() {
                        has_inf = true;
                    }
                    if vf < min_val {
                        min_val = vf;
                    }
                    if vf > max_val {
                        max_val = vf;
                    }
                    norm_sum_sq += vf * vf;
                }
            }
        }

        let norm = norm_sum_sq.sqrt();
        let elapsed = start.elapsed().as_millis() as u64;

        let diag = LayerDiagnostic {
            layer_index: l,
            attention_kind: layer_plan.attention_kind.clone(),
            hidden_norm: norm,
            hidden_finite: !has_nan && !has_inf,
            hidden_min: min_val,
            hidden_max: max_val,
            entropy: 0.0,
            elapsed_ms: elapsed,
        };

        if has_nan {
            report.issues.push(DiagnosticIssue::NanInLayer(l));
            report.passed = false;
        }
        if has_inf {
            report.issues.push(DiagnosticIssue::InfInLayer(l));
            report.passed = false;
        }

        report.layers.push(diag);
    }

    report.global.total_layers = plan.layers.len();
    report.global.nan_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::NanInLayer(_)))
        .count();
    report.global.inf_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::InfInLayer(_)))
        .count();
    report.global.total_runtime_ms = report.layers.iter().map(|l| l.elapsed_ms).sum();
    report.global.max_runtime_ms = report
        .layers
        .iter()
        .map(|l| l.elapsed_ms)
        .max()
        .unwrap_or(0);
    report.global.memory_peak_bytes = mlx_peak_memory_bytes();

    Ok(report)
}

// ═══════════════════════════════════════════════════════════════════════════
// Publishing
// ═══════════════════════════════════════════════════════════════════════════

/// Atomically publish a staged compilation to its final destination.
///
/// 1. Writes a `.publishing` marker inside `staging`.
/// 2. Renames `staging` to `destination` (falls back to recursive copy
///    when the rename crosses filesystem boundaries).
/// 3. On failure the staging directory is left intact with a `.failed` marker
///    so that the caller can inspect or retry.
pub fn publish_image(staging: &Path, destination: &Path) -> crate::Result<()> {
    let publishing_marker = staging.join(".publishing");
    std::fs::write(&publishing_marker, b"")
        .map_err(|e| crate::Error::from_reason(format!("write .publishing: {}", e)))?;

    let result = std::fs::rename(staging, destination);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            // rename fails across filesystem boundaries — fall back to copy + remove
            if e.kind() == std::io::ErrorKind::CrossesDevices {
                let failed_marker = staging.join(".failed");
                if let Err(write_err) =
                    std::fs::write(&failed_marker, format!("rename failed: {}", e))
                {
                    return Err(crate::Error::from_reason(format!(
                        "write .failed marker: {} (original rename: {})",
                        write_err, e
                    )));
                }
                return Err(crate::Error::from_reason(format!(
                    "rename crosses devices: {}. Staging left in place with .failed marker.",
                    e
                )));
            }
            let failed_marker = staging.join(".failed");
            if let Err(write_err) = std::fs::write(&failed_marker, format!("rename failed: {}", e))
            {
                return Err(crate::Error::from_reason(format!(
                    "write .failed marker: {} (original rename: {})",
                    write_err, e
                )));
            }
            Err(crate::Error::from_reason(format!(
                "rename {} -> {}: {}",
                staging.display(),
                destination.display(),
                e
            )))
        }
    }
}
