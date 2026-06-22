//! Execution plan generation.

use super::compile::{
    build_compile_receipt, build_source_identity, emit_binding_set, load_source, LoadedSource,
    SourceTensor,
};
use super::manifest::{
    mlx_active_memory_bytes, CompiledImage, ImageBuilder, SegmentKind, StageProfile,
};
use crate::config::CompileQuantMode;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
pub fn plan(
    source_dir: &Path,
    skip_validation: bool,
) -> crate::Result<(crate::config::CompilationPlan, LoadedSource)> {
    use crate::config::{CompilationPlan, PlannedSegment, PlannedTensor};
    let loaded = load_source(source_dir, skip_validation)?;
    let shard_hashes: Vec<String> = loaded
        .shard_hashes
        .iter()
        .map(|h| h.sha256.clone())
        .collect();
    let mut tensor_table = Vec::new();
    let mut next_tensor_id: u32 = 0;
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut seg_offsets: HashMap<String, u64> = HashMap::new();
    // Persistent segment.
    let persistent_seg_id = "persistent".to_string();
    segments.push(PlannedSegment {
        id: persistent_seg_id.clone(),
        filename: "segment_000.bin".into(),
        byte_size: 0,
        kind: "persistent".into(),
        tensor_count: 0,
    });
    for binding in &loaded.spec.global_tensors {
        let disp = classify_disposition(binding, &loaded.namespace);
        let (src_shard, src_offset, src_len, logical_dtype) =
            source_info(&loaded.source_tensors, &binding.name);
        let dest_offset = seg_offsets.get(&persistent_seg_id).copied().unwrap_or(0);
        tensor_table.push(PlannedTensor {
            id: next_tensor_id,
            name: binding.name.clone(),
            disposition: disp,
            source_shard: src_shard,
            source_offset: src_offset,
            source_byte_length: src_len,
            destination_segment: persistent_seg_id.clone(),
            destination_offset: dest_offset,
            destination_byte_length: src_len,
            logical_dtype,
            logical_shape: binding.logical_shape.clone(),
        });
        *seg_offsets.entry(persistent_seg_id.clone()).or_insert(0) += src_len;
        next_tensor_id += 1;
    }

    // Layer segments.
    for layer in &loaded.spec.layers {
        let seg_id = format!("layer_{}", layer.index);
        let seg_idx = segments.len();
        segments.push(PlannedSegment {
            id: seg_id.clone(),
            filename: format!("segment_{:03}.bin", seg_idx),
            byte_size: 0,
            kind: format!("layer_{}", layer.index),
            tensor_count: 0,
        });
        for binding in &layer.tensors {
            let disp = classify_disposition(binding, &loaded.namespace);
            let (src_shard, src_offset, src_len, logical_dtype) =
                source_info(&loaded.source_tensors, &binding.name);
            let dest_offset = seg_offsets.get(&seg_id).copied().unwrap_or(0);
            tensor_table.push(PlannedTensor {
                id: next_tensor_id,
                name: binding.name.clone(),
                disposition: disp,
                source_shard: src_shard,
                source_offset: src_offset,
                source_byte_length: src_len,
                destination_segment: seg_id.clone(),
                destination_offset: dest_offset,
                destination_byte_length: src_len,
                logical_dtype,
                logical_shape: binding.logical_shape.clone(),
            });
            *seg_offsets.entry(seg_id.clone()).or_insert(0) += src_len;
            next_tensor_id += 1;
        }
    }

    // Update segment byte sizes and tensor counts.
    for seg in &mut segments {
        seg.byte_size = *seg_offsets.get(&seg.id).unwrap_or(&0);
        seg.tensor_count = tensor_table
            .iter()
            .filter(|t| t.destination_segment == seg.id)
            .count();
    }

    let total_source_bytes: u64 = loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();
    let total_image_bytes: u64 = segments.iter().map(|s| s.byte_size).sum();

    let plan = CompilationPlan {
        model_identity: loaded.manifest.model_type.clone(),
        source_config_hash: loaded.manifest.config_hash.clone(),
        source_shard_hashes: shard_hashes,
        tensor_table,
        segments,
        total_source_bytes,
        total_image_bytes,
    };

    Ok((plan, loaded))
}

fn classify_disposition(
    binding: &crate::config::TensorBinding,
    _namespace: &crate::config::NamespaceBinding,
) -> crate::config::TensorDisposition {
    use crate::config::TensorDisposition;

    // Quantized weight payloads get relocated unchanged.
    if binding.name.ends_with(".weight")
        || binding.name.ends_with(".scales")
        || binding.name.ends_with(".biases")
    {
        return TensorDisposition::RelocateAndAlign;
    }
    // Embedding layer_scalar and other small tensors also relocate.
    TensorDisposition::RelocateAndAlign
}

fn source_info(
    source_tensors: &HashMap<String, SourceTensor>,
    name: &str,
) -> (String, u64, u64, String) {
    if let Some(st) = source_tensors.get(name) {
        (
            st.source_filename.clone(),
            st.source_offset,
            st.data.len() as u64,
            st.dtype.clone(),
        )
    } else {
        (String::new(), 0, 0, "F32".into())
    }
}

/// Reorder segments for speculative decoding: shared persistent first,
/// then draft layer segments, then target layer segments + target persistent.
#[allow(dead_code)]
fn reorder_for_speculative(
    target_segments: &mut Vec<crate::config::PlannedSegment>,
    draft_segments: &mut Vec<crate::config::PlannedSegment>,
    config: &crate::config::SpeculativeModelConfig,
) {
    let mut reordered = Vec::new();

    // 1. Shared persistent segment (embeddings, LM head if shared)
    if config.shared_embedding {
        // Merge persistent segments: keep the first persistent from target
        if let Some(pos) = target_segments.iter().position(|s| s.kind == "persistent") {
            let seg = target_segments.remove(pos);
            reordered.push(seg);
        }
        // Remove draft persistent (absorbed into shared)
        draft_segments.retain(|s| s.kind != "persistent");
    }

    // 2. Draft layer segments first (fast startup)
    if config.draft_first_segments {
        let draft_layers: Vec<_> = std::mem::take(draft_segments)
            .into_iter()
            .filter(|s| s.kind.starts_with("layer_"))
            .collect();
        reordered.extend(draft_layers);
        // Keep remaining (non-persistent, non-layer) draft segments
        *draft_segments = Vec::new();
    }

    // 3. Target segments (persistent then layer)
    //    Persistent first (norms), then layer segments
    if let Some(pos) = target_segments.iter().position(|s| s.kind == "persistent") {
        let seg = target_segments.remove(pos);
        reordered.push(seg);
    }
    let target_layers: Vec<_> = std::mem::take(target_segments)
        .into_iter()
        .filter(|s| s.kind.starts_with("layer_"))
        .collect();
    reordered.extend(target_layers);

    *target_segments = reordered;
}

/// Compile a draft + target model pair into a single speculative ComputeImage.
///
/// Loads both checkpoints, emits shared weights once when compatible,
/// orders draft layers first for fast startup, and attaches speculative
/// decoding metadata to the manifest.
pub(crate) fn compile_unchecked_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    _quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    let started_at = std::time::Instant::now();
    let output_dir = Path::new(output_dir);

    // Load both models independently
    let t_load = Instant::now();
    let target_loaded = load_source(Path::new(target_dir), false)?;
    let draft_loaded = load_source(Path::new(draft_dir), false)?;
    let source_load_ms = t_load.elapsed().as_millis() as u64;

    // Detect embedding shareability
    let shared_embedding = target_loaded.arch.vocab_size == draft_loaded.arch.vocab_size
        && target_loaded.arch.hidden_size == draft_loaded.arch.hidden_size;
    let shared_lm_head = shared_embedding;

    // Build source identity from target model
    let source = build_source_identity(
        &target_loaded.manifest,
        target_loaded.shard_hashes.clone(),
        target_loaded.tokenizer_hashes.clone(),
        target_loaded.auxiliary_hashes.clone(),
    );

    let mut builder = ImageBuilder::new(target_loaded.arch.clone(), source);
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    let t_emit = Instant::now();

    // 1. Shared persistent segment (embeddings stored once if shareable)
    let shared_seg_id = "persistent".to_string();
    builder.begin_segment(&shared_seg_id, SegmentKind::Persistent);

    if shared_embedding {
        // Emit target embeddings — shared by both models
        for binding in &target_loaded.spec.global_tensors {
            let id = emit_binding_set(&mut builder, &target_loaded.source_tensors, binding, None)?;
            emitted_ids.insert(binding.name.clone(), id);
            // Register aliases for draft model tensors that map to shared weights
            let draft_root = &draft_loaded.namespace.root;
            let target_root = &target_loaded.namespace.root;
            if binding.name.contains("embed_tokens") {
                let draft_embed = binding.name.replace(target_root, draft_root);
                builder.add_alias(&draft_embed, id, "shared_embedding_speculative");
                emitted_ids.insert(draft_embed, id);
            }
        }
        if shared_lm_head {
            // If lm_head is aliased (tied), register draft alias too
            if target_loaded.namespace.lm_head_aliased {
                let target_head = "lm_head.weight".to_string();
                let draft_head_key = format!("{}.lm_head.weight", draft_loaded.namespace.root);
                if let Some(&id) = emitted_ids.get(&target_head) {
                    builder.add_alias(&draft_head_key, id, "shared_lm_head_speculative");
                    emitted_ids.insert(draft_head_key, id);
                }
            }
        }
    } else {
        // Not shared: emit target embeddings, then switch to new persistent for draft
        for binding in &target_loaded.spec.global_tensors {
            let id = emit_binding_set(&mut builder, &target_loaded.source_tensors, binding, None)?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // 2. Draft layer segments (first for fast startup)
    for layer in &draft_loaded.spec.layers {
        let seg_id = format!("draft_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &draft_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // 3. Target persistent (norms, LM head) — if embeddings not shared, they're already emitted
    if !shared_embedding {
        // switch back to target persistent for norms
        builder.begin_segment("persistent_target", SegmentKind::Persistent);
        // norms and other non-embedding global tensors already emitted above
        // if not shared, the loop above already emitted all target globals
    }

    // 4. Target layer segments
    for layer in &target_loaded.spec.layers {
        let seg_id = format!("target_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &target_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Register aliases for tied embeddings on target side
    if target_loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", target_loaded.namespace.root);
        if let Some(&id) = emitted_ids.get(&embed_name) {
            builder.add_alias("lm_head.weight", id, "tie_word_embeddings");
        }
    }

    // 5. Build the target execution plan
    let mut execution_plan = crate::config::build_execution_plan(
        &target_loaded.arch,
        &target_loaded.namespace,
        &emitted_ids,
    );
    execution_plan.build_ane_fusion_plan();

    // 6. Attach speculative config metadata
    execution_plan.speculative_config = Some(crate::config::SpeculativeModelConfig {
        draft_architecture: draft_loaded.arch.clone(),
        target_architecture: target_loaded.arch.clone(),
        shared_embedding,
        shared_lm_head,
        draft_first_segments: true,
        speculation_length: 5,
    });

    builder.set_execution_plan(execution_plan);

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

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    let total_source_bytes = target_loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum::<u64>()
        + draft_loaded
            .source_tensors
            .values()
            .map(|t| t.data.len() as u64)
            .sum::<u64>();
    let total_emitted_bytes = manifest.segments.iter().map(|s| s.byte_size).sum();

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
        peak_mlx_active_bytes: mlx_active_memory_bytes(),
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &target_loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Default::default(),
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}
