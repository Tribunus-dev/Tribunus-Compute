//! ComputeImage — deterministic, validated, execution-ordered model image.
//!
//! A ComputeImage is a precompiled runtime artifact containing:
//!   manifest.json     — architecture, tensor table, aliases, residency plan
//!   segment_000.bin   — aligned, execution-ordered tensor bytes
//!   segment_001.bin
//!   ...
//!
//! v0 is the copied, runtime-ready image. It proves canonicalization,
//! bounded residency, and output parity. No-copy Metal buffers remain v2.

pub mod compile;
pub mod quant;
pub mod diag;
pub mod hf;
pub mod manifest;
pub mod kv_plan;
pub mod plan;
pub mod segment;
pub mod pipeline;
pub mod verify;
pub mod source;
pub mod hw_assessment;
pub mod hw_bench_suite;
pub mod compile_hw;
pub mod subgraph_mil;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod compile_coreml;
#[cfg(feature = "tensix")]
pub mod tensix;
#[cfg(feature = "tensix")]
pub mod layout_tensix;
#[cfg(feature = "tensix")]
pub mod fusion_tensix;

#[cfg(feature = "tensix")]
pub mod compile_tensix;

pub use manifest::{
    AliasEntry,
    build_tensor_catalog,
    clear_mlx_cache,
    CompilationAuthority,
    CompileReceipt,
    CompiledImage,
    CompiledImageReader,
    CopyClassification,
    ImageBuilder,
    is_valid_storage_abi,
    LeaseState,
    Manifest,
    ManifestVerification,
    mlx_active_memory_bytes,
    mlx_cache_memory_bytes,
    mlx_get_memory_limit,
    mlx_peak_memory_bytes,
    NativeCapabilityReport,
    QuantizationDesc,
    read,
    RepresentationAdmissionEstimate,
    representation_aware_admission_estimate,
    ResidencyPlan,
    resolve_tensor_name,
    ResolvedTensorBinding,
    Segment,
    SegmentKind,
    SegmentLease,
    SegmentReceipt,
    set_mlx_cache_limit,
    set_mlx_memory_limit,
    ShardHash,
    StageProfile,
    StorageAbiSpec,
    StorageBackend,
    STORAGE_ABI_COPIED_V0,
    STORAGE_ABI_MAPPED_NO_COPY_V1,
    TensorEntry,
    TensorLease,
    validate_manifest_for_abi,
    validate_physical_dtype,
    validate_tensor_for_mapped_abi,
    validate_tensor_layout,
};

pub use kv_plan::{
    KvCachePlan,
    KVDtype,
    KvLayout,
    PrefixCompatibilityKey,
};

pub use compile::{
    compile_differential,
    compile_with_authority,
    compile_with_authority_speculative,
    diff_tensors,
    download_hf_model,
    image_build_attestation,
    load_source_tensor_table,
    parse_hf_source,
    SourceTensorInfo,
};

pub use compile_hw::{
    run_hardware_assessment,
};

pub use segment::{
    ImageRuntime,
    LayerLease,
};

pub use verify::{
    DiagnosticIssue, DiagnosticReport, GlobalDiagnostic, LayerDiagnostic, publish_image,
    run_diagnostics, verify,
};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorLookup;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "tribunus-compute-image-{}-{}-{}",
            std::process::id(),
            label,
            stamp
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
        Box::leak(bytes.into_boxed_slice())
    }

    fn u32_tensor(name: &str, shape: &[usize], seed: u32) -> (String, TensorView<'static>) {
        let len = shape.iter().product::<usize>();
        let mut bytes = Vec::with_capacity(len * std::mem::size_of::<u32>());
        for index in 0..len {
            let value = seed.wrapping_add(index as u32);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor =
            TensorView::new(Dtype::U32, shape.to_vec(), leak_bytes(bytes)).expect("tensor");
        (name.to_string(), tensor)
    }

    fn f32_tensor(name: &str, shape: &[usize], seed: f32) -> (String, TensorView<'static>) {
        let len = shape.iter().product::<usize>();
        let mut bytes = Vec::with_capacity(len * std::mem::size_of::<f32>());
        for index in 0..len {
            let value = seed + (index as f32 * 0.03125);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor =
            TensorView::new(Dtype::F32, shape.to_vec(), leak_bytes(bytes)).expect("tensor");
        (name.to_string(), tensor)
    }

    fn write_fixture_model(source_dir: &Path) {
        let config = serde_json::json!({
            "model_type": "tiny_gemma_like",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "sliding_window": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "attention_k_eq_v": true,
                "final_logit_softcapping": null,
                "hidden_size_per_layer_input": 0,
                "layer_types": ["sliding_attention"],
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    },
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    }
                },
                "model_type": "tiny_gemma_like"
            },
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }
        });

        fs::write(
            source_dir.join("config.json"),
            serde_json::to_string_pretty(&config).expect("config json"),
        )
        .expect("write config");

        let root = "language_model.model";
        let mut tensors = vec![
            u32_tensor(&format!("{}.embed_tokens.weight", root), &[64, 16], 1),
            f32_tensor(&format!("{}.embed_tokens.scales", root), &[64, 1], 0.5),
            f32_tensor(&format!("{}.embed_tokens.biases", root), &[64, 1], 1.5),
            f32_tensor(&format!("{}.norm.weight", root), &[64], 2.0),
            f32_tensor(
                &format!("{}.layers.0.input_layernorm.weight", root),
                &[64],
                3.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.post_attention_layernorm.weight", root),
                &[64],
                4.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_norm.weight", root),
                &[16],
                5.0,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_norm.weight", root),
                &[16],
                6.0,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.weight", root),
                &[64, 16],
                7,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.scales", root),
                &[64, 1],
                7.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.q_proj.biases", root),
                &[64, 1],
                7.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.weight", root),
                &[16, 16],
                8,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.scales", root),
                &[16, 1],
                8.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.k_proj.biases", root),
                &[16, 1],
                8.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.weight", root),
                &[16, 16],
                9,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.scales", root),
                &[16, 1],
                9.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.v_proj.biases", root),
                &[16, 1],
                9.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.weight", root),
                &[64, 16],
                10,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.scales", root),
                &[64, 1],
                10.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.self_attn.o_proj.biases", root),
                &[64, 1],
                10.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.weight", root),
                &[128, 16],
                11,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.scales", root),
                &[128, 1],
                11.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.gate_proj.biases", root),
                &[128, 1],
                11.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.up_proj.weight", root),
                &[128, 16],
                12,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.up_proj.scales", root),
                &[128, 1],
                12.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.up_proj.biases", root),
                &[128, 1],
                12.75,
            ),
            u32_tensor(
                &format!("{}.layers.0.mlp.down_proj.weight", root),
                &[64, 32],
                13,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.down_proj.scales", root),
                &[64, 2],
                13.5,
            ),
            f32_tensor(
                &format!("{}.layers.0.mlp.down_proj.biases", root),
                &[64, 2],
                13.75,
            ),
        ];

        tensors.sort_by(|left, right| left.0.cmp(&right.0));
        serialize_to_file(tensors, &None, &source_dir.join("model.safetensors"))
            .expect("write safetensors");
    }

    /// Build a synthetic model with N layers driven by `layer_types`
    /// ("sliding_attention" or "full_attention"). Full-attention layers
    /// omit v_proj (K-equals-V).
    fn write_two_layer_fixture_model(source_dir: &Path, layer_types: &[&str]) {
        let num_layers = layer_types.len();
        let config = serde_json::json!({
            "model_type": "tiny_gemma_like",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "num_hidden_layers": num_layers,
                "vocab_size": 64,
                "sliding_window": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "attention_k_eq_v": true,
                "final_logit_softcapping": null,
                "hidden_size_per_layer_input": 0,
                "layer_types": layer_types,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    },
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    }
                },
                "model_type": "tiny_gemma_like"
            },
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }
        });

        fs::write(
            source_dir.join("config.json"),
            serde_json::to_string_pretty(&config).expect("config json"),
        )
        .expect("write config");

        let root = "language_model.model";
        let mut tensors = vec![
            u32_tensor(&format!("{}.embed_tokens.weight", root), &[64, 16], 1),
            f32_tensor(&format!("{}.embed_tokens.scales", root), &[64, 1], 0.5),
            f32_tensor(&format!("{}.embed_tokens.biases", root), &[64, 1], 1.5),
            f32_tensor(&format!("{}.norm.weight", root), &[64], 2.0),
        ];

        for (i, lt) in layer_types.iter().enumerate() {
            let layer = i as u32;
            let is_full = *lt == "full_attention";

            // Norms
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.input_layernorm.weight", root, layer),
                &[64],
                3.0 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.post_attention_layernorm.weight", root, layer),
                &[64],
                4.0 + layer as f32 * 10.0,
            ));

            // Q/K norms
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_norm.weight", root, layer),
                &[16],
                5.0 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_norm.weight", root, layer),
                &[16],
                6.0 + layer as f32 * 10.0,
            ));

            // Q projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.weight", root, layer),
                &[64, 16],
                7 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.scales", root, layer),
                &[64, 1],
                7.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.q_proj.biases", root, layer),
                &[64, 1],
                7.75 + layer as f32 * 10.0,
            ));

            // K projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.weight", root, layer),
                &[16, 16],
                8 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.scales", root, layer),
                &[16, 1],
                8.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.k_proj.biases", root, layer),
                &[16, 1],
                8.75 + layer as f32 * 10.0,
            ));

            // V projection: only for sliding attention layers
            if !is_full {
                tensors.push(u32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.weight", root, layer),
                    &[16, 16],
                    9 + layer * 100,
                ));
                tensors.push(f32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.scales", root, layer),
                    &[16, 1],
                    9.5 + layer as f32 * 10.0,
                ));
                tensors.push(f32_tensor(
                    &format!("{}.layers.{}.self_attn.v_proj.biases", root, layer),
                    &[16, 1],
                    9.75 + layer as f32 * 10.0,
                ));
            }

            // O projection
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.weight", root, layer),
                &[64, 16],
                10 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.scales", root, layer),
                &[64, 1],
                10.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.self_attn.o_proj.biases", root, layer),
                &[64, 1],
                10.75 + layer as f32 * 10.0,
            ));

            // MLP gate/up/down
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.weight", root, layer),
                &[128, 16],
                11 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.scales", root, layer),
                &[128, 1],
                11.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.gate_proj.biases", root, layer),
                &[128, 1],
                11.75 + layer as f32 * 10.0,
            ));
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.weight", root, layer),
                &[128, 16],
                12 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.scales", root, layer),
                &[128, 1],
                12.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.up_proj.biases", root, layer),
                &[128, 1],
                12.75 + layer as f32 * 10.0,
            ));
            tensors.push(u32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.weight", root, layer),
                &[64, 32],
                13 + layer * 100,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.scales", root, layer),
                &[64, 2],
                13.5 + layer as f32 * 10.0,
            ));
            tensors.push(f32_tensor(
                &format!("{}.layers.{}.mlp.down_proj.biases", root, layer),
                &[64, 2],
                13.75 + layer as f32 * 10.0,
            ));
        }

        tensors.sort_by(|left, right| left.0.cmp(&right.0));
        serialize_to_file(tensors, &None, &source_dir.join("model.safetensors"))
            .expect("write safetensors");
    }

    #[derive(Debug)]
    struct TensorComparison {
        shape_matches: bool,
        dtype_matches: bool,
        source_finite: bool,
        runtime_finite: bool,
        max_abs_diff: f32,
        mean_abs_diff: f32,
        cosine_similarity: f32,
    }

    fn compare_tensors(source: &Array, runtime: &Array) -> TensorComparison {
        let source_slice = source.try_as_slice::<f32>().expect("source slice");
        let runtime_slice = runtime.try_as_slice::<f32>().expect("runtime slice");
        let len = usize::min(source_slice.len(), runtime_slice.len());
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut dot = 0.0f32;
        let mut source_norm = 0.0f32;
        let mut runtime_norm = 0.0f32;

        for i in 0..len {
            let left = source_slice[i];
            let right = runtime_slice[i];
            let diff = (left - right).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            sum_abs_diff += diff;
            dot += left * right;
            source_norm += left * left;
            runtime_norm += right * right;
        }

        let cosine_similarity = if source_norm == 0.0 || runtime_norm == 0.0 {
            0.0
        } else {
            dot / (source_norm.sqrt() * runtime_norm.sqrt())
        };

        TensorComparison {
            shape_matches: source.shape() == runtime.shape(),
            dtype_matches: format!("{:?}", source.dtype()) == format!("{:?}", runtime.dtype()),
            source_finite: source_slice.iter().all(|value| value.is_finite()),
            runtime_finite: runtime_slice.iter().all(|value| value.is_finite()),
            max_abs_diff,
            mean_abs_diff: if len == 0 {
                0.0
            } else {
                sum_abs_diff / len as f32
            },
            cosine_similarity,
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct RealCheckpointReference {
        shape: Vec<i32>,
        values: Vec<f32>,
    }

    fn real_checkpoint_env(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    fn real_checkpoint_run_child(
        phase: &str,
        source_dir: &Path,
        output_dir: &Path,
        reference_path: &Path,
    ) {
        let current_exe = std::env::current_exe().expect("current exe");
        let status = std::process::Command::new(current_exe)
            .arg("compute_image::tests::real_checkpoint_six_layer_prefix_round_trip")
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .env("TRIBUNUS_REAL_CHECKPOINT_PHASE", phase)
            .env("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR", source_dir)
            .env("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR", output_dir)
            .env("TRIBUNUS_REAL_CHECKPOINT_REFERENCE", reference_path)
            .status()
            .expect("spawn real checkpoint child");
        assert!(
            status.success(),
            "real checkpoint child failed in phase {}",
            phase
        );
    }

    fn real_checkpoint_source_phase(source_dir: &Path, reference_path: &Path) {
        let source = crate::model::Shard::load(
            source_dir
                .join("model-00001-of-00003.safetensors")
                .to_str()
                .expect("source shard 1"),
        );
        let source_2 = crate::model::Shard::load(
            source_dir
                .join("model-00002-of-00003.safetensors")
                .to_str()
                .expect("source shard 2"),
        );
        let source_3 = crate::model::Shard::load(
            source_dir
                .join("model-00003-of-00003.safetensors")
                .to_str()
                .expect("source shard 3"),
        );
        let (arch, _, _) = crate::config::parse_config(
            source_dir
                .join("config.json")
                .to_str()
                .expect("config path"),
        )
        .expect("parse config");
        let output = crate::model::run_six_layer_prefix(&[&source, &source_2, &source_3], &arch)
            .expect("source prefix");
        output.eval().expect("source eval");
        let reference = RealCheckpointReference {
            shape: output.shape().to_vec(),
            values: output.try_as_slice::<f32>().expect("source slice").to_vec(),
        };
        std::fs::write(
            reference_path,
            serde_json::to_string_pretty(&reference).expect("reference json"),
        )
        .expect("write reference");
        crate::bridge::ARRAY_REGISTRY.write().drain();
    }

    fn real_checkpoint_compile_phase(source_dir: &Path, output_dir: &Path) {
        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile real checkpoint");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);
        assert_eq!(
            verification.verified_segment_count,
            compiled.manifest.segments.len()
        );
    }

    fn real_checkpoint_runtime_phase(source_dir: &Path, output_dir: &Path, reference_path: &Path) {
        let source_exists = source_dir.exists();
        assert!(
            !source_exists,
            "source checkpoint should not be accessible during runtime"
        );

        let reference: RealCheckpointReference =
            serde_json::from_str(&std::fs::read_to_string(reference_path).expect("read reference"))
                .expect("parse reference");
        let expected = Array::from_slice(&reference.values, &reference.shape);

        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);

        let baseline_handles = crate::bridge::handle_count();
        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        let runtime_prefix = runtime.run_six_layer_prefix().expect("runtime prefix");
        runtime_prefix.eval().expect("runtime eval");

        let comparison = compare_tensors(&expected, &runtime_prefix);
        assert!(comparison.shape_matches, "shape mismatch");
        assert!(comparison.dtype_matches, "dtype mismatch");
        assert!(
            comparison.source_finite,
            "reference output contains non-finite values"
        );
        assert!(
            comparison.runtime_finite,
            "runtime output contains non-finite values"
        );
        assert!(
            comparison.max_abs_diff <= 1e-4,
            "max abs diff too large: {}",
            comparison.max_abs_diff
        );
        assert!(
            comparison.mean_abs_diff <= 1e-5,
            "mean abs diff too large: {}",
            comparison.mean_abs_diff
        );
        assert!(
            comparison.cosine_similarity >= 0.999_999,
            "cosine similarity too low: {}",
            comparison.cosine_similarity
        );
        assert_eq!(crate::bridge::handle_count(), baseline_handles);
    }

    #[test]
    fn compile_source_dir_writes_deterministic_image() {
        let source_dir = temp_dir("source");
        let output_dir_a = temp_dir("out-a");
        let output_dir_b = temp_dir("out-b");

        write_fixture_model(&source_dir);

        let first = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir_a.to_str().expect("output dir a"), CompilationAuthority::TestFixture, false, None, None)
        .expect("second compile");
        let second = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir_b.to_str().expect("output dir b"), CompilationAuthority::TestFixture, false, None, None)
        .expect("first compile");

        assert_eq!(first.manifest.image_hash, second.manifest.image_hash);
        assert_eq!(first.receipt.complete_image_hash, first.manifest.image_hash);
        assert_eq!(first.manifest.segments.len(), 2);
        assert_eq!(
            first.manifest.segments.len(),
            first.manifest.residency_plan.persistent_segments.len()
                + first.manifest.residency_plan.layer_segments.len()
        );
        assert_eq!(first.manifest.alias_table.len(), 1);
        assert_eq!(first.manifest.alias_table[0].logical_name, "lm_head.weight");
        assert!(first.receipt.structural_verification);

        let manifest_path = output_dir_a.join("manifest.json");
        assert!(manifest_path.exists());
        let receipt_path = output_dir_a.join("receipt.json");
        assert!(receipt_path.exists());

        let persisted = fs::read(output_dir_a.join("segment_000.bin")).expect("segment 0");
        assert_eq!(persisted.len() as u64, first.manifest.segments[0].byte_size);

        let reloaded_manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(manifest_path).expect("manifest json"))
                .expect("manifest parse");
        assert_eq!(reloaded_manifest.image_hash, first.manifest.image_hash);
        assert_eq!(
            reloaded_manifest.segments.len(),
            first.manifest.segments.len()
        );
    }

    #[test]
    fn compiled_image_reader_round_trip_matches_source_prefix() {
        let source_dir = temp_dir("source-round-trip");
        let output_dir = temp_dir("out-round-trip");

        write_fixture_model(&source_dir);

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let verification = reader.verify().expect("verification");
        assert!(verification.manifest_hash_matches);
        assert!(verification.segment_hashes_match);
        assert_eq!(
            verification.verified_segment_count,
            compiled.manifest.segments.len()
        );

        let source = crate::model::Shard::load(
            source_dir
                .join("model.safetensors")
                .to_str()
                .expect("source shard"),
        );

        for name in [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.scales",
            "language_model.model.layers.0.self_attn.q_proj.biases",
        ] {
            let left = source.tensor(name).expect("source tensor");
            let right = reader.tensor(name).expect("reader tensor");
            assert_eq!(left.shape(), right.shape());
            let left_dtype = format!("{:?}", left.dtype());
            let right_dtype = format!("{:?}", right.dtype());
            assert_eq!(left_dtype, right_dtype);
            match left_dtype.as_str() {
                "Uint32" | "U32" => {
                    assert_eq!(
                        left.try_as_slice::<u32>().expect("source u32"),
                        right.try_as_slice::<u32>().expect("reader u32")
                    );
                }
                "Float32" | "F32" => {
                    assert_eq!(
                        left.try_as_slice::<f32>().expect("source f32"),
                        right.try_as_slice::<f32>().expect("reader f32")
                    );
                }
                other => panic!("unexpected dtype for {}: {}", name, other),
            }
        }
    }

    #[ignore = "requires compiled modelc fixture on disk"]
    #[test]
    fn compiled_image_runtime_copied_round_trip_matches_source_prefix() {
        let source_dir = temp_dir("source-runtime");
        let output_dir = temp_dir("out-runtime");

        write_fixture_model(&source_dir);

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");
        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");
        let baseline_handles = crate::bridge::handle_count();
        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        assert!(runtime.quantized_binding_count() > 0);
        assert!(crate::bridge::handle_count() > baseline_handles);

        // After open_runtime: only persistent segment bytes loaded, not layer segments.
        let persistent_bytes: u64 = compiled
            .manifest
            .segments
            .iter()
            .filter(|s| matches!(s.kind, SegmentKind::Persistent | SegmentKind::Final))
            .map(|s| s.byte_size)
            .sum();
        assert_eq!(runtime.total_bytes_activated(), persistent_bytes);

        let source = crate::model::Shard::load(
            source_dir
                .join("model.safetensors")
                .to_str()
                .expect("source shard"),
        );
        let source_prefix =
            crate::model::run_six_layer_prefix(&[&source], &compiled.manifest.architecture)
                .expect("source prefix");
        let runtime_prefix = runtime.run_six_layer_prefix().expect("runtime prefix");

        assert_eq!(source_prefix.shape(), runtime_prefix.shape());
        assert_eq!(
            source_prefix.try_as_slice::<f32>().expect("source slice"),
            runtime_prefix.try_as_slice::<f32>().expect("runtime slice")
        );
        assert_eq!(crate::bridge::handle_count(), baseline_handles);
    }

    #[test]
    #[ignore = "real checkpoint smoke test; run manually when you want to pay the 12G cost"]
    fn real_checkpoint_six_layer_prefix_round_trip() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-out");

        if let Some(phase) = real_checkpoint_env("TRIBUNUS_REAL_CHECKPOINT_PHASE") {
            let source = std::env::var("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR")
                .expect("TRIBUNUS_REAL_CHECKPOINT_SOURCE_DIR");
            let output = std::env::var("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR")
                .expect("TRIBUNUS_REAL_CHECKPOINT_OUTPUT_DIR");
            let reference = std::env::var("TRIBUNUS_REAL_CHECKPOINT_REFERENCE")
                .expect("TRIBUNUS_REAL_CHECKPOINT_REFERENCE");
            let source = Path::new(&source);
            let output = Path::new(&output);
            let reference = Path::new(&reference);

            match phase.as_str() {
                "source" => real_checkpoint_source_phase(source, reference),
                "compile" => real_checkpoint_compile_phase(source, output),
                "runtime" => real_checkpoint_runtime_phase(source, output, reference),
                other => panic!("unknown checkpoint phase: {}", other),
            }
            return;
        }

        let reference_path = temp_dir("real-reference").join("reference.json");
        let hidden_source_dir = source_dir.with_extension("hidden-for-runtime");
        struct RestoreSourceDir {
            hidden: PathBuf,
            original: PathBuf,
        }
        impl Drop for RestoreSourceDir {
            fn drop(&mut self) {
                if self.hidden.exists() {
                    let _ = std::fs::rename(&self.hidden, &self.original);
                }
            }
        }

        real_checkpoint_run_child("source", &source_dir, &output_dir, &reference_path);
        real_checkpoint_run_child("compile", &source_dir, &output_dir, &reference_path);

        std::fs::rename(&source_dir, &hidden_source_dir).expect("hide source checkpoint");
        let _restore_source_dir = RestoreSourceDir {
            hidden: hidden_source_dir.clone(),
            original: source_dir.clone(),
        };
        assert!(
            !source_dir.exists(),
            "source checkpoint should be hidden before runtime"
        );

        real_checkpoint_run_child("runtime", &source_dir, &output_dir, &reference_path);
    }

    #[test]
    fn compiled_image_rejects_corruption_and_missing_segment() {
        let source_dir = temp_dir("source-corruption");
        write_fixture_model(&source_dir);

        let corrupted_dir = temp_dir("out-corrupted");
        compile_with_authority(source_dir.to_str().expect("source dir"), corrupted_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile corrupted fixture");
        let segment_path = corrupted_dir.join("segment_000.bin");
        let mut bytes = fs::read(&segment_path).expect("segment bytes");
        bytes[0] ^= 0xFF;
        fs::write(&segment_path, bytes).expect("rewrite corrupted segment");
        let err = match read(corrupted_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected corruption error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("segment hash mismatch"),
            "unexpected corruption error: {}",
            err
        );

        let missing_dir = temp_dir("out-missing");
        compile_with_authority(source_dir.to_str().expect("source dir"), missing_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile missing fixture");
        fs::remove_file(missing_dir.join("segment_000.bin")).expect("remove segment");
        let err = match read(missing_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected missing-segment error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("read segment") || err.to_string().contains("missing segment"),
            "unexpected missing-segment error: {}",
            err
        );

        let abi_dir = temp_dir("out-abi");
        compile_with_authority(source_dir.to_str().expect("source dir"), abi_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile abi fixture");
        let manifest_path = abi_dir.join("manifest.json");
        let manifest = fs::read_to_string(&manifest_path).expect("read manifest");
        let mutated = manifest.replace(
            "\"runtime_abi\": \"mlx-rs/0.21.0 core/",
            "\"runtime_abi\": \"mlx-rs/0.21.0 core-mutated/",
        );
        fs::write(&manifest_path, mutated).expect("rewrite manifest");
        let err = match read(abi_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected abi-mismatch error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("manifest hash mismatch"),
            "unexpected abi-mismatch error: {}",
            err
        );
    }
    #[test]
    fn test_storage_abi_matching() {
        let source = SourceIdentity {
            config_hash: "abc".into(),
            shard_hashes: vec![],
            tokenizer_hashes: vec![],
            auxiliary_hashes: vec![],
            model_type: "test".into(),
            quantization_bits: 8,
            quantization_group_size: 64,
            quantization_mode: "affine".into(),
        };

        let defaults = Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "test".into(),
            runtime_abi: "test".into(),
            source: source,
            architecture: crate::config::TextArchitecture {
                hidden_size: 64,
                intermediate_size: 128,
                num_attention_heads: 4,
                num_key_value_heads: 1,
                head_dim: 16,
                global_head_dim: Some(16),
                num_global_key_value_heads: Some(1),
                num_hidden_layers: 1,
                vocab_size: 64,
                sliding_window: 8,
                max_position_embeddings: 16,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                attention_k_eq_v: true,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: 0,
                layer_types: vec![crate::config::AttentionKind::SlidingAttention],
                rope_local: crate::config::RopeSpec {
                    theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                rope_global: None,
                model_type: "test".into(),
            },
            vision_config: None,
            segments: vec![],
            tensor_table: vec![],
            alias_table: vec![],
            residency_plan: ResidencyPlan {
                persistent_segments: vec![],
                layer_segments: vec![],
                layer_window_size: 2,
                total_bytes: 0,
            },
            image_hash: "dummy".into(),
            required_storage_abi: STORAGE_ABI_COPIED_V0.into(),
            required_capabilities: vec![],
            prepacked_layout: "none".into(),
            execution_plan: crate::config::ModelExecutionPlan::default(),
        };

        assert!(defaults.storage_abi_matches(&StorageBackend::Copied));
        assert!(!defaults.storage_abi_matches(&StorageBackend::MappedNoCopy));

        // Check constants
        assert_eq!(STORAGE_ABI_COPIED_V0, "copied-v0");
        assert_eq!(STORAGE_ABI_MAPPED_NO_COPY_V1, "mapped-no-copy-v1");
    }

    #[test]
    fn test_alignment_validation() {
        // Build a manifest manually with mapped-no-copy-v1 and proper alignment
        let segment = Segment {
            id: "test_seg".into(),
            filename: "segment_000.bin".into(),
            byte_size: 4096,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            tensor_ids: vec![0],
            kind: SegmentKind::Persistent,
            alignment_bytes: 4096,
        };
        let tensor = TensorEntry {
            id: 0,
            name: "weight".into(),
            role: "embed".into(),
            layer: None,
            segment: "test_seg".into(),
            source_filename: "x.safetensors".into(),
            source_sha256: "0000".into(),
            source_offset: 0,
            offset: 0,
            byte_length: 256,
            logical_dtype: "F32".into(),
            storage_dtype: "F32".into(),
            logical_shape: vec![16, 16],
            physical_shape: vec![16, 16],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
        };
        let manifest = Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "test".into(),
            runtime_abi: "test".into(),
            source: SourceIdentity {
                config_hash: "abc".into(),
                shard_hashes: vec![],
                tokenizer_hashes: vec![],
                auxiliary_hashes: vec![],
                model_type: "test".into(),
                quantization_bits: 8,
                quantization_group_size: 64,
                quantization_mode: "affine".into(),
            },
            architecture: crate::config::TextArchitecture {
                hidden_size: 64,
                intermediate_size: 128,
                num_attention_heads: 4,
                num_key_value_heads: 1,
                head_dim: 16,
                global_head_dim: Some(16),
                num_global_key_value_heads: Some(1),
                num_hidden_layers: 1,
                vocab_size: 64,
                sliding_window: 8,
                max_position_embeddings: 16,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                attention_k_eq_v: true,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: 0,
                layer_types: vec![crate::config::AttentionKind::SlidingAttention],
                rope_local: crate::config::RopeSpec {
                    theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                rope_global: None,
                model_type: "test".into(),
            },
            segments: vec![segment],
            tensor_table: vec![tensor],
            alias_table: vec![],
            residency_plan: ResidencyPlan {
                persistent_segments: vec!["test_seg".into()],
                layer_segments: vec![],
                layer_window_size: 2,
                total_bytes: 4096,
            },
            image_hash: "dummy".into(),
            required_storage_abi: STORAGE_ABI_MAPPED_NO_COPY_V1.into(),
            vision_config: None,
            required_capabilities: vec![],
            prepacked_layout: "none".into(),
            execution_plan: crate::config::ModelExecutionPlan::default(),
        };

        assert!(manifest.storage_abi_matches(&StorageBackend::MappedNoCopy));
        assert!(!manifest.storage_abi_matches(&StorageBackend::Copied));
    }

    #[test]
    fn segment_corruption_rejected() {
        let source_dir = temp_dir("source-seg-corr");
        write_fixture_model(&source_dir);

        let output_dir = temp_dir("out-seg-corr");
        compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile corrupted fixture");

        // segment_000.bin = persistent (embed + final), segment_001.bin = layer 0
        let segment_path = output_dir.join("segment_001.bin");
        let mut bytes = fs::read(&segment_path).expect("layer segment bytes");
        bytes[100] ^= 0xFF;
        fs::write(&segment_path, bytes).expect("rewrite corrupted layer segment");

        let err = match read(output_dir.to_str().expect("output dir")) {
            Ok(_) => panic!("expected segment corruption error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("segment hash mismatch"),
            "unexpected segment corruption error: {}",
            err
        );
    }

    #[test]
    fn synthetic_plan_driven_execution() {
        let source_dir = temp_dir("source-plan");
        let output_dir = temp_dir("out-plan");

        write_two_layer_fixture_model(&source_dir, &["sliding_attention", "full_attention"]);

        let baseline_handles = crate::bridge::handle_count();
        {
            let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
            .expect("compile");

            let reader = read(output_dir.to_str().expect("output dir")).expect("reader");

            // Verify execution plan from manifest
            let plan = &compiled.manifest.execution_plan;
            assert_eq!(plan.layers.len(), 2);

            assert_eq!(plan.layers[0].attention_kind, "sliding_attention");
            assert_eq!(plan.layers[0].layer_index, 0);
            assert!(plan.layers[0].global_head_dim.is_none());
            assert!(
                plan.layers[0].v_proj_tensor_id != 0,
                "sliding layer needs v_proj"
            );

            assert_eq!(plan.layers[1].attention_kind, "full_attention");
            assert_eq!(plan.layers[1].layer_index, 1);
            assert_eq!(plan.layers[1].global_head_dim, Some(16));
            // K-equals-V: v_proj aliases k_proj
            assert_eq!(
                plan.layers[1].v_proj_tensor_id,
                plan.layers[1].k_proj_tensor_id
            );

            // Validate the plan
            plan.validate().expect("execution plan should validate");

            // Open runtime and verify handle lifecycle
            let mut runtime = reader
                .open_runtime(StorageBackend::Copied)
                .expect("runtime");

            // Handle count after persistent activation
            let after_persistent = crate::bridge::handle_count();
            assert!(after_persistent > baseline_handles);

            // Run full model - this activates layers, runs inference, then retires them
            let token = runtime.run_full_model(&[2i32]).expect("run full model");
            assert!(token < 64, "token {} should be in [0, 64)", token);
        }

        // After all model-owned values are dropped, handles should return to baseline
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count should return to baseline after runtime teardown; {} != {}",
            after_run, baseline_handles
        );
    }

    #[test]
    fn test_synthetic_prefill_decode_parity() {
        unsafe { std::env::set_var("TRIBUNUS_COMPUTE_ALLOW_HIGH_MEMORY", "1"); }
        let source_dir = temp_dir("source-parity");
        let output_dir = temp_dir("out-parity");

        write_two_layer_fixture_model(&source_dir, &["sliding_attention", "full_attention"]);

        let _compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile");

        let reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        let profiled_model =
            crate::profiled_executor::LoadedProfiledModel::new(&output_dir).expect("load bindings");

        let build_kv_caches = || -> Vec<crate::kv_cache::KvCache> {
            profiled_model
                .reader
                .manifest
                .execution_plan
                .layers
                .iter()
                .map(|layer| {
                    let capacity = if layer.attention_kind == "sliding_attention" {
                        layer.sliding_window
                    } else {
                        16
                    };
                    let n_kv_heads = layer.n_global_kv_heads.unwrap_or(layer.n_kv_heads);
                    let head_dim = layer.global_head_dim.unwrap_or(layer.head_dim);
                    crate::kv_cache::KvCache::new(
                        capacity,
                        n_kv_heads,
                        head_dim,
                        layer.attention_kind == "sliding_attention",
                    )
                })
                .collect()
        };

        let mut session = crate::profiled_executor::ProfiledInferenceSession::new(
            "test-parity-session".to_string(),
            build_kv_caches(),
        );

        // Prefill parity
        let prompt = vec![2u32, 10u32, 15u32];
        let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

        let t1_cached = session.prefill(&prompt, &profiled_model).expect("prefill");
        let t1_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&prompt_i32)
            .expect("run_full_model");
        assert_eq!(t1_cached, t1_uncached, "Prefill token parity mismatch");

        // Decode Step 1 parity
        let mut history = prompt.clone();
        history.push(t1_cached);
        let history_i32: Vec<i32> = history.iter().map(|&t| t as i32).collect();

        let t2_cached = session
            .decode_one(t1_cached, &profiled_model)
            .expect("decode_one step 1");
        let t2_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&history_i32)
            .expect("run_full_model step 1");
        assert_eq!(
            t2_cached, t2_uncached,
            "Decode step 1 token parity mismatch"
        );

        // Decode Step 2 parity
        history.push(t2_cached);
        let history_i32_2: Vec<i32> = history.iter().map(|&t| t as i32).collect();

        let t3_cached = session
            .decode_one(t2_cached, &profiled_model)
            .expect("decode_one step 2");
        let t3_uncached = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime")
            .run_full_model(&history_i32_2)
            .expect("run_full_model step 2");
        assert_eq!(
            t3_cached, t3_uncached,
            "Decode step 2 token parity mismatch"
        );
    }

    #[test]
    #[ignore = "requires sealed image at TRIBUNUS_COMPILED_IMAGE"]
    fn real_checkpoint_full_model_gate() {
        let image_dir =
            std::env::var("TRIBUNUS_COMPILED_IMAGE").expect("set TRIBUNUS_COMPILED_IMAGE");
        let image_path = std::path::Path::new(&image_dir);
        assert!(image_path.join("manifest.json").exists());
        assert!(image_path.join("seal.json").exists());

        eprintln!("Opening sealed image: {}", image_dir);
        let baseline_handles = crate::bridge::handle_count();
        let reader = read(&image_dir).expect("reader");
        let plan = &reader.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48);
        plan.validate().expect("plan validation");

        let mut runtime = reader
            .open_runtime(StorageBackend::Copied)
            .expect("runtime");
        eprintln!("Running 48-layer forward pass...");
        let started = std::time::Instant::now();
        let token = runtime.run_full_model(&[2i32]).expect("run_full_model");
        let elapsed = started.elapsed().as_secs_f64();

        let after_run = crate::bridge::handle_count();
        eprintln!(
            "GATE PASSED: token={} elapsed={:.1}s handles={}->{}",
            token, elapsed, baseline_handles, after_run
        );
        assert_eq!(after_run, baseline_handles);
    }

    #[test]
    fn test_storage_abi_validation_rejects_unknown() {
        // Verify that is_valid_storage_abi rejects unknown identifiers
        assert!(is_valid_storage_abi(STORAGE_ABI_COPIED_V0));
        assert!(is_valid_storage_abi(STORAGE_ABI_MAPPED_NO_COPY_V1));
        assert!(!is_valid_storage_abi("copied-v2"));
        assert!(!is_valid_storage_abi("mapped-no-copy-v0"));
        assert!(!is_valid_storage_abi(""));
        assert!(!is_valid_storage_abi("unknown-abi"));
    }

    #[test]
    fn test_tensor_layout_offset_oob() {
        // A tensor whose offset + byte_length exceeds its segment should fail.
        let entry = TensorEntry {
            id: 0,
            name: "oob_tensor".into(),
            role: "test".into(),
            layer: None,
            segment: "seg".into(),
            source_filename: "x.safetensors".into(),
            source_sha256: "0000".into(),
            source_offset: 0,
            offset: 100,
            byte_length: 200,
            logical_dtype: "F32".into(),
            storage_dtype: "F32".into(),
            logical_shape: vec![10, 5],
            physical_shape: vec![10, 5],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
        };

        // Segment is only 250 bytes, tensor ends at 300 -> OOB
        let result = validate_tensor_layout(&entry, 250);
        assert!(result.is_err(), "expected OOB error");
        assert!(
            result.unwrap_err().contains("exceeds segment size"),
            "unexpected error message"
        );

        // With enough space it should succeed
        let result = validate_tensor_layout(&entry, 301);
        assert!(result.is_ok(), "expected OK for large enough segment");

        // Zero byte_length should be rejected
        let zero_entry = TensorEntry {
            byte_length: 0,
            ..entry.clone()
        };
        let result = validate_tensor_layout(&zero_entry, 100);
        assert!(result.is_err(), "expected error for zero byte_length");
        assert!(
            result.unwrap_err().contains("zero byte_length"),
            "unexpected error message"
        );
    }

    #[test]
    fn test_physical_dtype_byte_count() {
        // f32: 4 * (2*3*4) = 96
        let r = validate_physical_dtype("f32", 96, &[2, 3, 4]);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), 96);

        // bf16: 2 * (8*4) = 64
        let r = validate_physical_dtype("BF16", 64, &[8, 4]);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), 64);

        // f16: 2 * 128 = 256
        let r = validate_physical_dtype("f16", 256, &[128]);
        assert!(r.is_ok());

        // u8: 1 * (4*8) = 32
        let r = validate_physical_dtype("U8", 32, &[4, 8]);
        assert!(r.is_ok());

        // i8: same as u8
        let r = validate_physical_dtype("I8", 32, &[4, 8]);
        assert!(r.is_ok());

        // u32: 4 * 50 = 200
        let r = validate_physical_dtype("U32", 200, &[50]);
        assert!(r.is_ok());

        // Wrong byte count
        let r = validate_physical_dtype("f32", 100, &[2, 3, 4]);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected 96 bytes"));

        // Unknown dtype
        let r = validate_physical_dtype("f64", 8, &[1]);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("unsupported"));
    }

    #[test]
    #[ignore = "real checkpoint prefill+decode_one; requires ~12GB quantized model at models/gemma4-12b-8bit"]
    fn real_checkpoint_decode_one_token_after_prefill() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-decode-1-out");

        if !source_dir.join("config.json").exists() {
            eprintln!("SKIP: no model at {}", source_dir.display());
            return;
        }

        eprintln!("Compiling quantized Gemma 4 12B...");
        let started = std::time::Instant::now();

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile model");

        let compile_secs = started.elapsed().as_secs_f64();
        eprintln!(
            "Compiled in {:.1}s: {} segments, {} tensors, {:?}",
            compile_secs,
            compiled.manifest.segments.len(),
            compiled.manifest.tensor_table.len(),
            compiled.manifest.image_hash
        );

        let plan = &compiled.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48, "expected 48 layers");
        plan.validate().expect("execution plan should validate");
        eprintln!("image hash: {}", compiled.manifest.image_hash);

        eprintln!("Opening runtime...");
        let baseline_handles = crate::bridge::handle_count();
        let _reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        eprintln!("Loading profiled model...");
        let profiled_model = crate::profiled_executor::LoadedProfiledModel::new(&output_dir)
            .expect("load profiled model");

        let kv_caches: Vec<crate::kv_cache::KvCache> = plan
            .layers
            .iter()
            .map(|lp| {
                let is_sliding = lp.attention_kind == "sliding_attention";
                let (capacity, n_kv_heads, head_dim) = if is_sliding {
                    (lp.sliding_window, lp.n_kv_heads, lp.head_dim)
                } else {
                    let g_kv = lp.n_global_kv_heads.unwrap_or(lp.n_kv_heads);
                    let g_hd = lp.global_head_dim.unwrap_or(lp.head_dim);
                    (32768u32, g_kv, g_hd)
                };
                crate::kv_cache::KvCache::new(capacity, n_kv_heads, head_dim, is_sliding)
            })
            .collect();

        let mut session =
            crate::profiled_executor::ProfiledInferenceSession::new("decode-1".into(), kv_caches);

        eprintln!("Prefilling with [2, 42, 100, 500]...");
        let first_token = session
            .prefill(&[2, 42, 100, 500], &profiled_model)
            .expect("prefill");
        eprintln!("Prefill token: {}", first_token);
        assert!(
            first_token < 262144,
            "first token {} out of vocab range",
            first_token
        );
        assert!(first_token != 0, "first token must not be padding token 0");

        eprintln!("Decoding one token after prefill...");
        let second_token = session
            .decode_one(first_token, &profiled_model)
            .expect("decode_one");
        eprintln!("Decode token: {}", second_token);
        assert!(
            second_token < 262144,
            "second token {} out of vocab range",
            second_token
        );
        assert!(
            second_token != 0,
            "second token must not be padding token 0"
        );

        drop(session);
        drop(profiled_model);
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count must return to baseline after decode; {} != {}",
            after_run, baseline_handles
        );

        eprintln!(
            "[decode-1] PASSED: first={} second={}",
            first_token, second_token
        );
    }

    #[test]
    #[ignore = "real checkpoint 8-token decode; requires ~12GB quantized model at models/gemma4-12b-8bit"]
    fn real_checkpoint_decode_eight_tokens() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit");
        let output_dir = temp_dir("real-decode-8-out");

        if !source_dir.join("config.json").exists() {
            eprintln!("SKIP: no model at {}", source_dir.display());
            return;
        }

        eprintln!("Compiling quantized Gemma 4 12B...");
        let started = std::time::Instant::now();

        let compiled = compile_with_authority(source_dir.to_str().expect("source dir"), output_dir.to_str().expect("output dir"), CompilationAuthority::TestFixture, false, None, None)
        .expect("compile model");

        let compile_secs = started.elapsed().as_secs_f64();
        eprintln!(
            "Compiled in {:.1}s: {} segments, {} tensors, {:?}",
            compile_secs,
            compiled.manifest.segments.len(),
            compiled.manifest.tensor_table.len(),
            compiled.manifest.image_hash
        );

        let plan = &compiled.manifest.execution_plan;
        assert_eq!(plan.layers.len(), 48, "expected 48 layers");
        eprintln!("image hash: {}", compiled.manifest.image_hash);

        plan.validate().expect("execution plan should validate");

        eprintln!("Opening runtime...");
        let baseline_handles = crate::bridge::handle_count();
        let _reader = read(output_dir.to_str().expect("output dir")).expect("reader");

        eprintln!("Loading profiled model...");
        let profiled_model = crate::profiled_executor::LoadedProfiledModel::new(&output_dir)
            .expect("load profiled model");

        let kv_caches: Vec<crate::kv_cache::KvCache> = plan
            .layers
            .iter()
            .map(|lp| {
                let is_sliding = lp.attention_kind == "sliding_attention";
                let (capacity, n_kv_heads, head_dim) = if is_sliding {
                    (lp.sliding_window, lp.n_kv_heads, lp.head_dim)
                } else {
                    let g_kv = lp.n_global_kv_heads.unwrap_or(lp.n_kv_heads);
                    let g_hd = lp.global_head_dim.unwrap_or(lp.head_dim);
                    (32768u32, g_kv, g_hd)
                };
                crate::kv_cache::KvCache::new(capacity, n_kv_heads, head_dim, is_sliding)
            })
            .collect();

        let mut session =
            crate::profiled_executor::ProfiledInferenceSession::new("decode-8".into(), kv_caches);

        eprintln!("Prefilling with BOS token [2]...");
        let first_token = session.prefill(&[2u32], &profiled_model).expect("prefill");
        assert!(
            first_token < 262144,
            "first token {} out of vocab range",
            first_token
        );
        assert!(first_token != 0, "first token must not be 0");

        let mut tokens: Vec<u32> = Vec::with_capacity(9);
        tokens.push(first_token);

        eprintln!("Decoding 8 tokens...");
        let mut prev = first_token;
        for i in 0..8 {
            let next = session
                .decode_one(prev, &profiled_model)
                .expect("decode_one");
            assert!(
                next < 262144,
                "token {} out of vocab range at step {}",
                next,
                i
            );
            assert!(next != 0, "token must not be 0 at step {}", i);
            tokens.push(next);
            prev = next;
        }

        eprintln!("Tokens: {:?}", tokens);
        assert_eq!(tokens.len(), 9, "expected 9 tokens (1 prefill + 8 decode)");

        drop(session);
        drop(profiled_model);
        let after_run = crate::bridge::handle_count();
        assert_eq!(
            after_run, baseline_handles,
            "handle count must return to baseline after 8 decode steps; {} != {}",
            after_run, baseline_handles
        );

        eprintln!("[decode-8] PASSED: {} tokens", tokens.len());
    }
}

