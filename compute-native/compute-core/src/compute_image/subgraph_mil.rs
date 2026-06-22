//! MIL program builders for Core ML subgraphs.
//!
//! Each function produces a [`mil_spec::Program`] suitable for
//! [`mlpackage::write_mlpackage()`] serialization.  Callers then compile
//! via `coremlcompiler` and load the resulting `.mlmodelc`.
//!
//! # Subgraphs
//!
//! | Builder                    | Inputs        | Outputs                   | Ops                         |
//! |----------------------------|---------------|---------------------------|-----------------------------|
//! | `build_matmul_mil`         | hidden [1, M] | out [1, N]                | matmul                      |
//! | `build_mlp_block_mil`      | hidden [1, D] | out [1, D]                | 2× matmul, SiLU, mul        |
//! | `build_rmsnorm_qkv_mil`    | hidden [1, D] | Q, K, V                   | pow, reduce-sum, rsqrt, 3× matmul |
//! | `build_output_proj_mil`    | hidden [1, D] | logits [1, V]             | matmul                      |
//! | `build_ffn_output_mil`     | hidden [1, D] | logits [1, V]             | MLP block + lm_head matmul  |
//! | `build_qkv_bundle_mil`     | hidden [1, D] | Q, K, V                   | 3× matmul                   |

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value};
use std::collections::HashMap;

use crate::mil_builder::MilBuilder;

// ═══════════════════════════════════════════════════════════════════════════
// Attribute / value helpers
// ═══════════════════════════════════════════════════════════════════════════

fn named_arg(name: &str) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

fn float_attr(val: f32) -> mil_spec::Value {
    let float_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Float32)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(float_tensor)),
        })),
    }
}

fn bool_attr(val: bool) -> mil_spec::Value {
    let bool_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Bool)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(bool_tensor)),
        })),
    }
}

fn int32s_attr(vals: &[i32]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
            values: vals.to_vec(),
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Int32)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

fn string_attr(val: &str) -> mil_spec::Value {
    let string_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Strings(
            tensor_value::RepeatedStrings {
                values: vec![val.to_string()],
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::String)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(string_tensor)),
        })),
    }
}

fn scalar_value_type(dtype: mil_spec::DataType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(
            mil_spec::TensorType {
                data_type: dtype as i32,
                rank: 0,
                dimensions: vec![],
                attributes: HashMap::new(),
            },
        )),
    }
}

fn tensor_type(dtype: mil_spec::DataType, shape: &[i64]) -> mil_spec::TensorType {
    let dims: Vec<mil_spec::Dimension> = shape
        .iter()
        .map(|&s| mil_spec::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: s as u64 },
            )),
        })
        .collect();
    mil_spec::TensorType {
        data_type: dtype as i32,
        rank: shape.len() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    }
}

fn value_type_tensor(tt: mil_spec::TensorType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(tt)),
    }
}

fn float32_tensor_type_2d(rows: i64, cols: i64) -> mil_spec::TensorType {
    tensor_type(mil_spec::DataType::Float32, &[rows, cols])
}

fn float32_value_type_2d(rows: i64, cols: i64) -> mil_spec::ValueType {
    value_type_tensor(float32_tensor_type_2d(rows, cols))
}

// Build a manual MIL operation with a "name" attribute.
fn make_operation(
    op_type: &str,
    out_name: &str,
    inputs: HashMap<String, mil_spec::Argument>,
    out_vt: &mil_spec::ValueType,
    extra_attrs: HashMap<String, mil_spec::Value>,
) -> mil_spec::Operation {
    let mut attrs = extra_attrs;
    attrs.insert("name".to_string(), string_attr(out_name));
    mil_spec::Operation {
        r#type: op_type.to_string(),
        inputs,
        outputs: vec![mil_spec::NamedValueType {
            name: out_name.to_string(),
            r#type: Some(out_vt.clone()),
        }],
        blocks: vec![],
        attributes: attrs,
    }
}

/// Resolve the shape of a named value from the builder's internal type map.
fn resolve_shape(builder: &MilBuilder, name: &str) -> Vec<i64> {
    builder
        .value_shapes()
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![1, 1])
}

// ═══════════════════════════════════════════════════════════════════════════
// Manual MIL op helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Add `pow(x, alpha)` — element-wise exponentiation.
fn op_pow(builder: MilBuilder, input: &str, alpha: f32) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_name = format!("pow_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(shape[0], shape[1]));

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));

    let mut attrs = HashMap::new();
    attrs.insert("alpha".to_string(), float_attr(alpha));

    let op = make_operation("pow", &out_name, inputs, &vt, attrs);
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

/// Add `reduce_sum(x, axes)` over a single axis, keeping dimensions.
fn op_reduce_sum(builder: MilBuilder, input: &str, axis: i32) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_rows = if axis == 1 || axis == -1 { shape[0] } else { 1 };
    let out_name = format!("reduce_sum_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(out_rows, 1));

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));

    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), int32s_attr(&[axis]));
    attrs.insert("keep_dims".to_string(), bool_attr(true));

    let op = make_operation("reduce_sum", &out_name, inputs, &vt, attrs);
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

/// Add `rsqrt(x)` — element-wise reciprocal square root (1 / sqrt(x)).
fn op_rsqrt(builder: MilBuilder, input: &str) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_name = format!("rsqrt_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(shape[0], shape[1]));

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));

    let op = make_operation("rsqrt", &out_name, inputs, &vt, HashMap::new());
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

/// Add composite SiLU(x) = x * sigmoid(x).
/// Returns the mul output name (the SiLU result).
fn op_composite_silu(builder: MilBuilder, input: &str) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let vt = float32_value_type_2d(shape[0], shape[1]);

    // sigmoid(x)
    let sig_name = format!("sig_{}", builder.ops().len());
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let sig_op = make_operation("sigmoid", &sig_name, inputs, &vt, HashMap::new());
    let builder = builder.operation(sig_op, Some((sig_name.as_str(), vt.clone())));

    // mul(x, sigmoid(x)) = SiLU
    let mul_name = format!("silu_mul_{}", builder.ops().len());
    let mut mul_inputs = HashMap::new();
    mul_inputs.insert("x".to_string(), named_arg(input));
    mul_inputs.insert("y".to_string(), named_arg(&sig_name));
    let mul_op = make_operation("mul", &mul_name, mul_inputs, &vt, HashMap::new());
    let builder = builder.operation(mul_op, Some((mul_name.as_str(), vt)));
    (builder, mul_name)
}

/// Add a native `silu` op (MIL-level, if the compiler accepts it).
#[allow(dead_code)]
fn op_native_silu(builder: MilBuilder, input: &str) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_name = format!("silu_{}", builder.ops().len());
    let vt = float32_value_type_2d(shape[0], shape[1]);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));

    let op = make_operation("silu", &out_name, inputs, &vt, HashMap::new());
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

// ═══════════════════════════════════════════════════════════════════════════
// Canonical subgraph output SSA names
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical output SSA names for each subgraph type.
pub fn subgraph_output_names(kind: &str) -> &[&str] {
    match kind {
        "matmul" => &["matmul_1"],
        "mlp_block" => &["matmul_down"],
        "rmsnorm_qkv" => &["matmul_q", "matmul_k", "matmul_v"],
        "output_proj" => &["matmul_lm_head"],
        "ffn_output" => &["matmul_lm_head"],
        "qkv_bundle" => &["matmul_q", "matmul_k", "matmul_v"],
        other => panic!("unknown subgraph kind '{other}'"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build a matmul MIL program for a single projection.
///
/// Produces:  input [1, K] × weight [K, N] = output [1, N]
///
/// SSA trace (counter 0 start):
///   {weight_name}_0 — const_f32
///   matmul_1        — matmul(input, {weight_name}_0)
/// Output: "matmul_1"
pub fn build_matmul_mil(
    input_name: &str,
    weight_name: &str,
    _output_name: &str,
    m: u32,
    k: u32,
    n: u32,
) -> Result<mil_spec::Program, String> {
    let builder = MilBuilder::new("main")
        .input(
            input_name,
            mil_spec::DataType::Float32,
            &[m as i64, k as i64],
        )
        .const_f32(weight_name, &[], &[k as i64, n as i64])
        .matmul(input_name, &format!("{}_0", weight_name));
    let builder = builder.output("matmul_1");
    builder
        .build()
        .map_err(|e| format!("matmul MIL build failed: {e}"))
}

/// Build an MLP block subgraph: gate_proj → SiLU → up_proj → mul → down_proj.
///
/// Input:  [1, hidden_dim]
/// Output: [1, hidden_dim]
///
/// SSA trace (counter 0 start):
///   w_gate_0    — const_f32(gate weight)
///   matmul_1    — matmul(x, w_gate_0)
///   sig_1       — sigmoid (manual op, no counter advance)
///   silu_mul_2  — mul(x, sig) = SiLU (manual op, no counter advance)
///   w_up_2      — const_f32(up weight)  — counter is 2 after matmul_1
///   matmul_3    — matmul(x, w_up_2)
///   mul_4       — mul(silu, up_proj)    — from MilBuilder
///   w_down_5    — const_f32(down weight)
///   matmul_6    — matmul(mul_4, w_down_5)
/// Output: "matmul_6"
pub fn build_mlp_block_mil(
    input_name: &str,
    hidden_dim: u32,
    intermediate_dim: u32,
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let f = intermediate_dim as i64;

    // Start with input
    let builder = MilBuilder::new("main").input(input_name, mil_spec::DataType::Float32, &[1, h]);

    // Gate projection: matmul(x, w_gate) → [1, f]
    let builder = builder
        .const_f32("w_gate", &[], &[h, f])
        .matmul(input_name, "w_gate_0");

    // SiLU(gate): sigmoid + mul composite
    let (builder, gate_silu) = op_composite_silu(builder, "matmul_1");

    // Up projection: matmul(x, w_up) → [1, f]
    let builder = builder
        .const_f32("w_up", &[], &[h, f])
        .matmul(input_name, "w_up_2");

    // Element-wise mul(gate_silu, up_proj)
    let builder = builder.mul(&gate_silu, "matmul_3");

    // Down projection: matmul(mul_result, w_down) → [1, h]
    let builder = builder
        .const_f32("w_down", &[], &[f, h])
        .matmul("mul_4", "w_down_5");
    // Counter: 6
    // mul_4 is output from mul(gate_silu, matmul_3). MilBuilder fresh_name("mul")
    // at counter=4 gives "mul_4".
    // w_down uses fresh_name("w_down") at counter=5 → "w_down_5"
    // matmul uses fresh_name("matmul") at counter=6 → "matmul_6"

    let builder = builder.output("matmul_6");
    builder
        .build()
        .map_err(|e| format!("MLP block MIL build failed: {e}"))
}

/// Build RMSNorm + QKV projection subgraph.
///
/// Input:  [1, hidden_dim]
/// Output: Q [1, n_heads * head_dim], K [1, n_kv_heads * head_dim],
///         V [1, n_kv_heads * head_dim]
///
/// RMSNorm: pow(x, 2) → reduce_sum(axis=1) → add(eps) → rsqrt → mul(x, factor) → mul(w_rms)
/// Then Q = matmul(rmsnorm_out, w_q), K = matmul(rmsnorm_out, w_k), V = matmul(rmsnorm_out, w_v)
pub fn build_rmsnorm_qkv_mil(
    input_name: &str,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let q_dim = (n_heads * head_dim) as i64;
    let kv_dim = (n_kv_heads * head_dim) as i64;
    let eps: f32 = 1e-5;

    // ── RMSNorm ──────────────────────────────────────────────────────
    let builder = MilBuilder::new("main").input(input_name, mil_spec::DataType::Float32, &[1, h]);

    // pow(x, 2.0)
    let (builder, pow_out) = op_pow(builder, input_name, 2.0);

    // reduce_sum(pow_out, axis=1) → [1, 1]
    let (builder, sum_out) = op_reduce_sum(builder, &pow_out, 1);

    // const eps [1, 1] → "eps_0"
    let builder = builder.const_f32("eps", &[eps], &[1, 1]);

    // add(sum_out, eps_0) → "add_1"
    let builder = builder.add(&sum_out, "eps_0");

    // rsqrt(add_1) → [1, 1]
    let (builder, rsqrt_out) = op_rsqrt(builder, "add_1");

    // mul(input_name, rsqrt_out) → normed input [1, h]
    let builder = builder.mul(input_name, &rsqrt_out);

    // const w_rms [1, h] → "w_rms_3"
    let builder = builder.const_f32("w_rms", &[], &[1, h]);

    // mul(normed, w_rms_3) → fully normed [1, h]
    // normed is "mul_2" — mul(input, rsqrt) with fresh_name("mul") at counter=2
    let builder = builder.mul("mul_2", "w_rms_3");
    // mul_4 = RMSNorm output (fresh_name("mul") at counter=4)

    // ── QKV projections ──────────────────────────────────────────────
    // Q: [1, h] × [h, q_dim] → [1, q_dim]
    let builder = builder
        .const_f32("w_q", &[], &[h, q_dim])
        .matmul("mul_4", "w_q_5");
    // w_q: fresh_name("w_q") at counter=5 → "w_q_5"
    // matmul: fresh_name("matmul") at counter=6 → "matmul_6"

    // K: [1, h] × [h, kv_dim] → [1, kv_dim]
    let builder = builder
        .const_f32("w_k", &[], &[h, kv_dim])
        .matmul("mul_4", "w_k_7");
    // w_k: fresh_name("w_k") at counter=7 → "w_k_7"
    // matmul: fresh_name("matmul") at counter=8 → "matmul_8"

    // V: [1, h] × [h, kv_dim] → [1, kv_dim]
    let builder = builder
        .const_f32("w_v", &[], &[h, kv_dim])
        .matmul("mul_4", "w_v_9");
    // w_v: fresh_name("w_v") at counter=9 → "w_v_9"
    // matmul: fresh_name("matmul") at counter=10 → "matmul_10"

    let builder = builder
        .output("matmul_6")
        .output("matmul_8")
        .output("matmul_10");

    builder
        .build()
        .map_err(|e| format!("RMSNorm+QKV MIL build failed: {e}"))
}

/// Build an output projection (lm_head) subgraph.
///
/// Input:  [1, hidden_dim]
/// Output: [1, vocab_dim]
///
/// SSA trace: w_lm_head_0, matmul_1
pub fn build_output_proj_mil(
    input_name: &str,
    hidden_dim: u32,
    vocab_dim: u32,
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let v = vocab_dim as i64;

    let builder = MilBuilder::new("main")
        .input(input_name, mil_spec::DataType::Float32, &[1, h])
        .const_f32("w_lm_head", &[], &[h, v])
        .matmul(input_name, "w_lm_head_0")
        .output("matmul_1");

    builder
        .build()
        .map_err(|e| format!("output proj MIL build failed: {e}"))
}

/// Build a full FFN + output projection (lm_head) subgraph.
///
/// Composes `build_mlp_block_mil` followed by lm_head matmul.
/// Input:  [1, hidden_dim]
/// Output: [1, vocab_dim]
///
/// SSA trace (counter 0 start):
///   w_gate_0, matmul_1, sig_1, silu_mul_2 (manual), w_up_2, matmul_3,
///   mul_4, w_down_5, matmul_6,                    ← MLP block
///   w_lm_head_7, matmul_8                          ← lm_head
/// Output: "matmul_8"
pub fn build_ffn_output_mil(
    input_name: &str,
    hidden_dim: u32,
    intermediate_dim: u32,
    vocab_dim: u32,
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let f = intermediate_dim as i64;
    let v = vocab_dim as i64;

    // ── MLP block ────────────────────────────────────────────────────
    let builder = MilBuilder::new("main").input(input_name, mil_spec::DataType::Float32, &[1, h]);

    // Gate projection
    let builder = builder
        .const_f32("w_gate", &[], &[h, f])
        .matmul(input_name, "w_gate_0");

    // SiLU(gate)
    let (builder, gate_silu) = op_composite_silu(builder, "matmul_1");

    // Up projection
    let builder = builder
        .const_f32("w_up", &[], &[h, f])
        .matmul(input_name, "w_up_2");

    // Gate × up
    let builder = builder.mul(&gate_silu, "matmul_3");

    // Down projection
    let builder = builder
        .const_f32("w_down", &[], &[f, h])
        .matmul("mul_4", "w_down_5");
    // matmul_6 = MLP output

    // ── LM head ──────────────────────────────────────────────────────
    let builder = builder
        .const_f32("w_lm_head", &[], &[h, v])
        .matmul("matmul_6", "w_lm_head_7");
    // matmul_8 = final logits

    let builder = builder.output("matmul_8");
    builder
        .build()
        .map_err(|e| format!("FFN+output MIL build failed: {e}"))
}

/// Build a QKV projection bundle (Q, K, V matmuls only, no RMSNorm).
///
/// Input:  [1, hidden_dim]
/// Output: Q [1, n_heads * head_dim], K [1, n_kv_heads * head_dim],
///         V [1, n_kv_heads * head_dim]
///
/// SSA trace:
///   w_q_0, matmul_1, w_k_2, matmul_3, w_v_4, matmul_5
/// Outputs: "matmul_1" (Q), "matmul_3" (K), "matmul_5" (V)
pub fn build_qkv_bundle_mil(
    input_name: &str,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let q_dim = (n_heads * head_dim) as i64;
    let kv_dim = (n_kv_heads * head_dim) as i64;

    let builder = MilBuilder::new("main").input(input_name, mil_spec::DataType::Float32, &[1, h]);

    // Q
    let builder = builder
        .const_f32("w_q", &[], &[h, q_dim])
        .matmul(input_name, "w_q_0");

    // K
    let builder = builder
        .const_f32("w_k", &[], &[h, kv_dim])
        .matmul(input_name, "w_k_2");

    // V
    let builder = builder
        .const_f32("w_v", &[], &[h, kv_dim])
        .matmul(input_name, "w_v_4");

    let builder = builder
        .output("matmul_1")
        .output("matmul_3")
        .output("matmul_5");

    builder
        .build()
        .map_err(|e| format!("QKV bundle MIL build failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    // ── matmul ───────────────────────────────────────────────────────

    #[test]
    fn build_matmul_acceptance() {
        let prog = build_matmul_mil("x", "w", "out", 1, 64, 32).unwrap();
        assert_eq!(prog.version, 1);
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.operations.len(), 2); // const + matmul
        assert_eq!(block.operations[0].r#type, "const");
        assert_eq!(block.operations[1].r#type, "matmul");
        assert_eq!(block.outputs, vec!["matmul_1"]);
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    // ── MLP block ────────────────────────────────────────────────────

    #[test]
    fn build_mlp_block_acceptance() {
        let prog = build_mlp_block_mil("x", 64, 256).unwrap();
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        // const_gate, matmul(gate), sig, mul(silu), const_up, matmul(up),
        // mul(gate_silu×up), const_down, matmul(down)
        assert!(
            block.operations.len() >= 6,
            "expected >=6 ops, got {}",
            block.operations.len()
        );
        assert_eq!(block.outputs, vec!["matmul_6"]);
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    // ── RMSNorm+QKV ──────────────────────────────────────────────────

    #[test]
    fn build_rmsnorm_qkv_acceptance() {
        let prog = build_rmsnorm_qkv_mil("x", 64, 4, 2, 32).unwrap();
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.outputs.len(), 3, "expected 3 outputs");
        assert!(block.outputs.contains(&"matmul_6".to_string()));
        assert!(block.outputs.contains(&"matmul_8".to_string()));
        assert!(block.outputs.contains(&"matmul_10".to_string()));
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    // ── Output proj ──────────────────────────────────────────────────

    #[test]
    fn build_output_proj_acceptance() {
        let prog = build_output_proj_mil("x", 64, 32768).unwrap();
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.outputs, vec!["matmul_1"]);
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    // ── FFN+output ───────────────────────────────────────────────────

    #[test]
    fn build_ffn_output_acceptance() {
        let prog = build_ffn_output_mil("x", 64, 256, 32768).unwrap();
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.outputs, vec!["matmul_8"]);
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    // ── QKV bundle ───────────────────────────────────────────────────

    #[test]
    fn build_qkv_bundle_acceptance() {
        let prog = build_qkv_bundle_mil("x", 64, 4, 2, 32).unwrap();
        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.outputs.len(), 3, "expected 3 outputs");
        assert!(block.outputs.contains(&"matmul_1".to_string()));
        assert!(block.outputs.contains(&"matmul_3".to_string()));
        assert!(block.outputs.contains(&"matmul_5".to_string()));
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }
}
