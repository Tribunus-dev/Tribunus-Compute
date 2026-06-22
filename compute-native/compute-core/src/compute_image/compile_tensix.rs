//! Tensix compiler — maps compute_ir ops to Metal2 ProgramSpec.
//! Each compute_ir operation becomes a KernelSpec with LLK math template instantiation.
//!
//! The compiler iterates over all scheduling regions and their ops, producing
//! a TensixComputeImage with per-core kernel configs and DRAM buffer bindings.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::compute_image::tensix::{
    CardCoord, DataFormat, GoldenPath, KernelConfig, KernelType, MathFidelity, TensixArch,
    TensixComputeImage, TensorBinding,
};
use crate::compute_ir::{ComputeExecutionIR, IROp, IrTensor};
use crate::Result;

/// Compile a compute execution IR into a Tensix compute image.
///
/// Each `IROp` in every scheduling region is mapped to a `KernelConfig`
/// describing the LLK math template instantiation. Output tensors are
/// registered as DRAM buffer bindings with standard (32, 32) tile shapes.
///
/// `interconnect_map` lists all card coordinates in the mesh.
/// `golden_path` describes the predetermined dataflow route through cards.
/// When both are empty/zero-card, a single-card topology is assumed.
pub fn compile_tensix(
    ir: &ComputeExecutionIR,
    arch: TensixArch,
    interconnect_map: &[CardCoord],
    golden_path: &GoldenPath,
) -> Result<TensixComputeImage> {
    let mut kernel_configs = Vec::new();
    let mut tensor_bindings = Vec::new();
    let mut core_count: u32 = 0;
    let mut total_cycles: u64 = 0;
    let mut dram_bytes: u64 = 0;
    let mut op_index: u32 = 0;

    let is_tensor_parallel = false; // Placeholder: will be determined by topology when available
    let tp_degree: u32 = 1;

    // Flatten all ops across all scheduling regions into a linear kernel stream.
    for region in &ir.regions {
        for op in &region.ops {
            let (kernel, cycles, cores) = map_op_to_tensix_kernel(op, arch)?;
            kernel_configs.push(kernel);
            core_count += cores;
            total_cycles += cycles;

            // Register output tensors as DRAM buffer bindings.
            for output_id in &op.output_tensors {
                if let Some(tensor) = ir.tensor_by_id(output_id) {
                    let byte_size = tensor_byte_size(tensor);

                    // Deduplicate: skip if this tensor was already bound by an earlier op.
                    if tensor_bindings
                        .iter()
                        .any(|b: &TensorBinding| b.tensor_name == tensor.name)
                    {
                        continue;
                    }

                    let binding = TensorBinding {
                        tensor_name: tensor.name.clone(),
                        buffer_slot: op_index,
                        byte_offset: dram_bytes,
                        byte_size,
                        tile_shape: (32, 32), // Tensix standard tile
                    };
                    dram_bytes += byte_size;
                    tensor_bindings.push(binding);
                }
            }
            op_index += 1;
        }
    }

    let program_spec_json = build_program_spec_json(&kernel_configs);
    let sram_per_core = estimate_sram_per_core(&kernel_configs);
    let program_hash = compute_ir_hash(ir);
    let card_count = interconnect_map.len() as u32;

    Ok(TensixComputeImage {
        program_hash,
        core_count,
        dram_bytes,
        sram_per_core,
        kernel_configs,
        tensor_bindings,
        estimated_cycles: total_cycles,
        target_arch: arch,
        program_spec_json,
        card_count,
        interconnect_map: interconnect_map.to_vec(),
        golden_path: golden_path.clone(),
    })
}

/// Map a single `IROp` to the corresponding Tensix kernel configuration.
///
/// Returns the `KernelConfig`, estimated cycle count, and number of Tensix cores.
fn map_op_to_tensix_kernel(op: &IROp, _arch: TensixArch) -> Result<(KernelConfig, u64, u32)> {
    let (kernel_type, math_fidelity, cycles, cores) = match op.kind.as_str() {
        "matmul" => {
            // Parse optional dimension hints from metadata.
            let m: u64 = op
                .metadata
                .get("m")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let n: u64 = op
                .metadata
                .get("n")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let k: u64 = op
                .metadata
                .get("k")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);

            // Small matmuls need fewer cores; large matmuls distribute across DRAM banks.
            let cores = if m >= 64 {
                8
            } else if m >= 16 {
                4
            } else {
                1
            };
            let cycles = if m > 0 && n > 0 && k > 0 {
                (m * n * k) / 8
            } else {
                100 // fallback when dimensions are unknown
            };
            (KernelType::Math, MathFidelity::HiFi3, cycles, cores)
        }
        "rms_norm" | "rmsnorm" => {
            let dim: u64 = op
                .metadata
                .get("dim")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            (KernelType::Math, MathFidelity::HiFi4, dim * 2, 1)
        }
        "softmax" => {
            let dim: u64 = op
                .metadata
                .get("dim")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            (KernelType::Math, MathFidelity::LoFi, dim, 1)
        }
        "silu" | "silu_activation" => (KernelType::Math, MathFidelity::LoFi, 10, 1),
        "add" | "residual_add" | "residual" => (KernelType::Relu, MathFidelity::LoFi, 2, 1),
        "rope" | "rotary_embedding" => (KernelType::Math, MathFidelity::HiFi4, 20, 1),
        "sdpa" | "scaled_dot_product_attention" => (KernelType::Math, MathFidelity::HiFi3, 200, 4),
        "kv_cache" | "kvcache" => (KernelType::Relu, MathFidelity::LoFi, 5, 1),
        other => {
            return Err(crate::Error::from_reason(format!(
                "unsupported op for Tensix: {other}"
            )));
        }
    };

    let name = if let Some(label) = op.metadata.get("name") {
        format!("{}_{}", label, op.kind)
    } else {
        // Derive a stable name from the first input and output tensor IDs.
        let in_part = op.input_tensors.first().map(|s| s.as_str()).unwrap_or("in");
        let out_part = op
            .output_tensors
            .first()
            .map(|s| s.as_str())
            .unwrap_or("out");
        format!("{}_{}_{}", in_part, op.kind, out_part)
    };

    Ok((
        KernelConfig {
            name,
            kernel_type,
            math_fidelity,
            tile_dims: (32, 32),
            data_format: DataFormat::BFloat16,
        },
        cycles,
        cores,
    ))
}

/// Compute total byte size of an `IrTensor` from its physical shape and dtype.
fn tensor_byte_size(tensor: &IrTensor) -> u64 {
    let element_size: u64 = match tensor.physical_dtype.as_str() {
        "float32" | "uint32" | "int32" => 4,
        "float16" | "bfloat16" | "bf16" | "fp16" => 2,
        "int8" | "uint8" | "i8" | "u8" => 1,
        _ => 2, // conservative default (bfloat16)
    };
    let elem_count: u64 = tensor.physical_shape.iter().map(|&d| d as u64).product();
    elem_count * element_size
}

/// Compute a stable 64-bit hash over the IR's ops and tensor signatures.
fn compute_ir_hash(ir: &ComputeExecutionIR) -> u64 {
    let mut hasher = DefaultHasher::new();
    for region in &ir.regions {
        for op in &region.ops {
            op.kind.hash(&mut hasher);
            for id in &op.input_tensors {
                id.hash(&mut hasher);
                if let Some(t) = ir.tensor_by_id(id) {
                    t.physical_dtype.hash(&mut hasher);
                    t.physical_shape.hash(&mut hasher);
                }
            }
            for id in &op.output_tensors {
                id.hash(&mut hasher);
                if let Some(t) = ir.tensor_by_id(id) {
                    t.physical_dtype.hash(&mut hasher);
                    t.physical_shape.hash(&mut hasher);
                }
            }
            // Hash metadata key-value pairs in sorted order for stability.
            let mut meta_pairs: Vec<(&String, &String)> = op.metadata.iter().collect();
            meta_pairs.sort_by(|a, b| a.0.cmp(b.0));
            for (k, v) in meta_pairs {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Estimate SRAM per core based on kernel tile sizes and CB requirements.
///
/// Each tile is 32x32 = 1024 elements, each element is 2 bytes (bf16).
/// Each core needs 2 input CBs + 1 output CB = roughly 6 KB per tile pair,
/// with a minimum allocation of 64 KB (standard Tensix SRAM CB region).
fn estimate_sram_per_core(_kernels: &[KernelConfig]) -> u64 {
    let tile_bytes: u64 = 32 * 32 * 2; // bf16 tile
    let cb_slots: u64 = 6; // 2 input + 1 output + 3 scratch
    (tile_bytes * cb_slots).max(64 * 1024) // at least 64 KB
}

/// Build a Metal2-compatible ProgramSpec JSON string from kernel configs.
/// This is consumed by the C++ bridge (tensix_compile_program).
fn build_program_spec_json(kernels: &[KernelConfig]) -> String {
    let mut json = String::from(r#"["#);
    for (i, kc) in kernels.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            r#"{{"name":"{}","kernel_type":"{}","math_fidelity":"{}","tile_dims":[{},{}],"data_format":"{}"}}"#,
            kc.name,
            kernel_type_str(kc.kernel_type),
            math_fidelity_str(kc.math_fidelity),
            kc.tile_dims.0,
            kc.tile_dims.1,
            data_format_str(kc.data_format),
        ));
    }
    json.push(']');
    json
}

fn kernel_type_str(kt: KernelType) -> &'static str {
    match kt {
        KernelType::Math => "Math",
        KernelType::Unpack => "Unpack",
        KernelType::Pack => "Pack",
        KernelType::Relu => "Relu",
    }
}

fn math_fidelity_str(mf: MathFidelity) -> &'static str {
    match mf {
        MathFidelity::LoFi => "LoFi",
        MathFidelity::HiFi2 => "HiFi2",
        MathFidelity::HiFi3 => "HiFi3",
        MathFidelity::HiFi4 => "HiFi4",
    }
}

fn data_format_str(df: DataFormat) -> &'static str {
    match df {
        DataFormat::Float32 => "Float32",
        DataFormat::Float16 => "Float16",
        DataFormat::BFloat16 => "BFloat16",
        DataFormat::Int8 => "Int8",
        DataFormat::UInt8 => "UInt8",
        DataFormat::Int32 => "Int32",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_ir::{ComputeExecutionIR, IROp, IrRegion, IrTensor, TensorDisposition};
    use std::collections::HashMap;

    fn single_card_topology() -> (Vec<CardCoord>, GoldenPath) {
        (
            vec![CardCoord {
                card_id: 0,
                noc_x: 0,
                noc_y: 0,
            }],
            GoldenPath {
                ordered_cards: vec![0],
                interconnect: crate::compute_image::tensix::InterconnectType::Noc,
            },
        )
    }

    fn make_test_tensor(id: &str, name: &str, shape: Vec<u32>, dtype: &str) -> IrTensor {
        IrTensor {
            id: id.to_string(),
            name: name.to_string(),
            logical_dtype: dtype.to_string(),
            logical_shape: shape.clone(),
            physical_dtype: dtype.to_string(),
            physical_shape: shape,
            strides: vec![],
            quant_mode: crate::compute_ir::QuantMode::None,
            disposition: TensorDisposition::Transient,
        }
    }

    fn make_test_op(
        kind: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        meta: Vec<(&str, &str)>,
    ) -> IROp {
        IROp {
            kind: kind.to_string(),
            input_tensors: inputs.into_iter().map(String::from).collect(),
            output_tensors: outputs.into_iter().map(String::from).collect(),
            metadata: meta
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_matmul_mapping() {
        let op = make_test_op(
            "matmul",
            vec!["a", "b"],
            vec!["c"],
            vec![("m", "64"), ("n", "64"), ("k", "64")],
        );
        let (cfg, cycles, cores) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::HiFi3);
        assert_eq!(cores, 8);
        assert!(cycles > 0);
    }

    #[test]
    fn test_small_matmul_uses_fewer_cores() {
        let op = make_test_op(
            "matmul",
            vec!["a", "b"],
            vec!["c"],
            vec![("m", "8"), ("n", "8"), ("k", "8")],
        );
        let (_, _, cores) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cores, 1);
    }

    #[test]
    fn test_rms_norm_mapping() {
        let op = make_test_op("rms_norm", vec!["x"], vec!["y"], vec![("dim", "4096")]);
        let (cfg, cycles, cores) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::HiFi4);
        assert_eq!(cores, 1);
        assert_eq!(cycles, 8192);
    }

    #[test]
    fn test_softmax_mapping() {
        let op = make_test_op("softmax", vec!["x"], vec!["y"], vec![("dim", "4096")]);
        let (cfg, cycles, cores) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::LoFi);
        assert_eq!(cores, 1);
    }

    #[test]
    fn test_silu_mapping() {
        let op = make_test_op("silu", vec!["x"], vec!["y"], vec![]);
        let (cfg, cycles, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::LoFi);
        assert_eq!(cycles, 10);
    }

    #[test]
    fn test_add_mapping() {
        let op = make_test_op("add", vec!["a", "b"], vec!["c"], vec![]);
        let (cfg, cycles, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Relu);
        assert_eq!(cfg.math_fidelity, MathFidelity::LoFi);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn test_rope_mapping() {
        let op = make_test_op("rope", vec!["x"], vec!["y"], vec![]);
        let (cfg, cycles, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::HiFi4);
        assert_eq!(cycles, 20);
    }

    #[test]
    fn test_sdpa_mapping() {
        let op = make_test_op("sdpa", vec!["q", "k", "v"], vec!["o"], vec![]);
        let (cfg, cycles, cores) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Math);
        assert_eq!(cfg.math_fidelity, MathFidelity::HiFi3);
        assert_eq!(cores, 4);
        assert!(cycles >= 200);
    }

    #[test]
    fn test_kv_cache_mapping() {
        let op = make_test_op("kv_cache", vec!["k", "v"], vec!["k_out", "v_out"], vec![]);
        let (cfg, cycles, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.kernel_type, KernelType::Relu);
        assert_eq!(cfg.math_fidelity, MathFidelity::LoFi);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn test_unsupported_op_returns_error() {
        let op = make_test_op("conv2d", vec!["x"], vec!["y"], vec![]);
        assert!(map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).is_err());
    }

    #[test]
    fn test_tensor_byte_size_bf16() {
        let tensor = make_test_tensor("t0", "weight", vec![1024, 4096], "bfloat16");
        assert_eq!(tensor_byte_size(&tensor), 1024 * 4096 * 2);
    }

    #[test]
    fn test_tensor_byte_size_f32() {
        let tensor = make_test_tensor("t0", "bias", vec![4096], "float32");
        assert_eq!(tensor_byte_size(&tensor), 4096 * 4);
    }

    #[test]
    fn test_tensor_byte_size_int8() {
        let tensor = make_test_tensor("t0", "quant", vec![1024], "int8");
        assert_eq!(tensor_byte_size(&tensor), 1024);
    }

    #[test]
    fn test_compile_tensix_empty_ir() {
        let ir = ComputeExecutionIR::new();
        let (map, path) = single_card_topology();
        let image = compile_tensix(&ir, TensixArch::WormholeB0, &map, &path).unwrap();
        assert_eq!(image.kernel_configs.len(), 0);
        assert_eq!(image.tensor_bindings.len(), 0);
        assert_eq!(image.core_count, 0);
        assert_eq!(image.dram_bytes, 0);
        assert_eq!(image.card_count, 1);
        assert_eq!(image.interconnect_map.len(), 1);
    }

    #[test]
    fn test_compile_tensix_single_op() {
        let tensor = make_test_tensor("y", "output", vec![64, 64], "bfloat16");
        let op = make_test_op("silu", vec!["x"], vec!["y"], vec![]);
        let ir = ComputeExecutionIR {
            tensors: vec![tensor],
            aliases: vec![],
            layers: vec![],
            regions: vec![IrRegion {
                ops: vec![op],
                dependencies: vec![],
                candidates: vec![],
                state_effects: vec![],
            }],
            metadata: HashMap::new(),
        };
        let (map, path) = single_card_topology();
        let image = compile_tensix(&ir, TensixArch::WormholeB0, &map, &path).unwrap();
        assert_eq!(image.kernel_configs.len(), 1);
        assert_eq!(image.tensor_bindings.len(), 1);
        assert_eq!(image.tensor_bindings[0].tensor_name, "output");
        assert_eq!(image.tensor_bindings[0].byte_size, 64 * 64 * 2);
        assert!(image.program_hash != 0);
        assert_eq!(image.card_count, 1);
    }

    #[test]
    fn test_estimate_sram_minimum() {
        let kernels = vec![KernelConfig {
            name: "test".into(),
            kernel_type: KernelType::Math,
            math_fidelity: MathFidelity::HiFi3,
            tile_dims: (32, 32),
            data_format: DataFormat::BFloat16,
        }];
        let sram = estimate_sram_per_core(&kernels);
        assert!(sram >= 64 * 1024);
    }

    #[test]
    fn test_kernel_name_from_metadata_label() {
        let mut meta = HashMap::new();
        meta.insert("name".to_string(), "layer0_attn".to_string());
        let op = IROp {
            kind: "matmul".to_string(),
            input_tensors: vec!["q".into(), "k".into()],
            output_tensors: vec!["scores".into()],
            metadata: meta,
        };
        let (cfg, _, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert_eq!(cfg.name, "layer0_attn_matmul");
    }

    #[test]
    fn test_kernel_name_fallback() {
        let op = IROp {
            kind: "silu".to_string(),
            input_tensors: vec!["x".into()],
            output_tensors: vec!["y".into()],
            metadata: HashMap::new(),
        };
        let (cfg, _, _) = map_op_to_tensix_kernel(&op, TensixArch::WormholeB0).unwrap();
        assert!(cfg.name.contains("silu"));
    }

    #[test]
    fn test_dedup_tensor_bindings() {
        // Two ops producing the same output tensor should only register one binding.
        let tensor = make_test_tensor("y", "shared_out", vec![32], "bfloat16");
        let op1 = make_test_op("silu", vec!["x"], vec!["y"], vec![]);
        let op2 = make_test_op("silu", vec!["x"], vec!["y"], vec![]);
        let ir = ComputeExecutionIR {
            tensors: vec![tensor],
            aliases: vec![],
            layers: vec![],
            regions: vec![IrRegion {
                ops: vec![op1, op2],
                dependencies: vec![],
                candidates: vec![],
                state_effects: vec![],
            }],
            metadata: HashMap::new(),
        };
        let (map, path) = single_card_topology();
        let image = compile_tensix(&ir, TensixArch::WormholeB0, &map, &path).unwrap();
        assert_eq!(image.tensor_bindings.len(), 1);
    }
}
