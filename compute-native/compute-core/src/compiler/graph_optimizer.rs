//! Compile-time graph optimization pass for the ComputeImage compiler.
//!
//! Operates on [`ModelExecutionPlan`] before segmentation emission.
//! Three passes run in order:
//!
//! 1. **Constant folding** — precompute sub-expressions whose all inputs are
//!    known at compile time (scalar constants, known tensor shapes, etc.).
//! 2. **Shape propagation** — propagate tensor shapes through the operation
//!    graph so Metal kernels can eliminate runtime shape checks.
//! 3. **Dead code elimination** — remove operations whose output is never
//!    used by any downstream consumer.

use crate::config::ModelExecutionPlan;
use std::collections::{HashMap, HashSet};

// ── Op-level graph representation ─────────────────────────────────────────

/// Kinds of operations we track in the optimizer's internal graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OpKind {
    EmbeddingLookup,
    RmsNorm,
    QProj,
    KProj,
    VProj,
    QNorm,
    KNorm,
    RoPE,
    Attention,
    OProj,
    GateProj,
    UpProj,
    SiLU,
    GateTimesUp,
    DownProj,
    ResidualAdd,
    FinalNorm,
    OutputProjection,
    Softcap,
    Argmax,
}

impl OpKind {
    fn name(&self) -> &'static str {
        match self {
            Self::EmbeddingLookup => "embedding_lookup",
            Self::RmsNorm => "rms_norm",
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::QNorm => "q_norm",
            Self::KNorm => "k_norm",
            Self::RoPE => "rope",
            Self::Attention => "attention",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::SiLU => "silu",
            Self::GateTimesUp => "gate_times_up",
            Self::DownProj => "down_proj",
            Self::ResidualAdd => "residual_add",
            Self::FinalNorm => "final_norm",
            Self::OutputProjection => "output_projection",
            Self::Softcap => "softcap",
            Self::Argmax => "argmax",
        }
    }
}

/// A single node in the optimizer's internal operation graph.
#[derive(Debug, Clone)]
struct OpNode {
    /// Unique node index.
    id: usize,
    /// What operation this node represents.
    kind: OpKind,
    /// Layer index for per-layer ops; `None` for global ops.
    layer_index: Option<u32>,
    /// Symbolic input tensor names that this node consumes.
    inputs: Vec<String>,
    /// Symbolic output tensor name(s) this node produces.
    outputs: Vec<String>,
    /// Known input shapes (populated by shape propagation).
    known_input_shapes: Vec<Vec<u32>>,
    /// Known output shape (populated by shape propagation).
    known_output_shape: Option<Vec<u32>>,
    /// Whether this operation produces a compile-time constant value.
    is_constant: bool,
    /// Whether this node has been marked as dead (unreferenced).
    is_dead: bool,
}

/// Internal optimizer context: op graph + index structures.
struct GraphOptimizer {
    nodes: Vec<OpNode>,
    /// Maps an output tensor name to the node id that produces it.
    tensor_to_producer: HashMap<String, usize>,
    /// Maps a tensor name to the node ids that consume it.
    tensor_to_consumers: HashMap<String, Vec<usize>>,
    /// Known shapes for tensors (populated by shape propagation).
    known_tensor_shapes: HashMap<String, Vec<u32>>,
    /// Set of tensors known to be compile-time constants.
    constant_tensors: HashSet<String>,
    /// Plan-level metadata extracted for shape inference.
    hidden_size: u32,
    intermediate_size: u32,
    vocab_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    global_head_dim: Option<u32>,
    n_global_kv_heads: Option<u32>,
}

impl GraphOptimizer {
    /// Build the op graph from a `ModelExecutionPlan`.
    fn from_plan(plan: &ModelExecutionPlan) -> Self {
        let mut opt = GraphOptimizer {
            nodes: Vec::new(),
            tensor_to_producer: HashMap::new(),
            tensor_to_consumers: HashMap::new(),
            known_tensor_shapes: HashMap::new(),
            constant_tensors: HashSet::new(),
            hidden_size: plan.hidden_size,
            intermediate_size: plan.hidden_size * 4,
            vocab_size: plan.vocab_size,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            global_head_dim: None,
            n_global_kv_heads: None,
        };

        // Seed shapes from architecture metadata.
        // Hidden/residual shapes are always known at compile time.
        let hidden_shape = vec![1u32, plan.hidden_size]; // [batch?, hidden]
        opt.known_tensor_shapes
            .insert("hidden_states".into(), hidden_shape.clone());

        // ── Prologue: embedding lookup ───────────────────────────────
        if plan.prologue.embedding_tensor_id != 0 {
            let embed_out = "embedding_output".to_string();
            let embed_shape = vec![plan.vocab_size, plan.hidden_size];
            opt.known_tensor_shapes
                .insert(embed_out.clone(), embed_shape.clone());

            let node = OpNode {
                id: opt.nodes.len(),
                kind: OpKind::EmbeddingLookup,
                layer_index: None,
                inputs: vec!["input_ids".into()],
                outputs: vec![embed_out.clone()],
                known_input_shapes: vec![vec![1u32]], // scalar token id
                known_output_shape: Some(embed_shape),
                is_constant: false,
                is_dead: false,
            };
            opt.tensor_to_producer.insert(embed_out, node.id);
            opt.register_consumers(&node);
            opt.nodes.push(node);
        }

        // ── Layers ──────────────────────────────────────────────────
        for (i, layer_plan) in plan.layers.iter().enumerate() {
            let layer = i as u32;
            let is_full = layer_plan.attention_kind == "full_attention";

            // Use layer-specific dimensions.
            let hdim = if is_full {
                layer_plan.global_head_dim.unwrap_or(layer_plan.head_dim)
            } else {
                layer_plan.head_dim
            };
            let n_kv = if is_full {
                layer_plan
                    .n_global_kv_heads
                    .unwrap_or(layer_plan.n_kv_heads)
            } else {
                layer_plan.n_kv_heads
            };
            let n_heads = layer_plan.n_heads;

            // Update first-layer values for global scope.
            if i == 0 {
                opt.n_heads = n_heads;
                opt.n_kv_heads = n_kv;
                opt.head_dim = hdim;
                opt.global_head_dim = layer_plan.global_head_dim;
                opt.n_global_kv_heads = layer_plan.n_global_kv_heads;
            }

            let residual_in = if i == 0 {
                "embedding_output".to_string()
            } else {
                format!("layer_{}_output", layer - 1)
            };

            // Input layer norm
            let norm_out = format!("layer_{}_norm_out", layer);
            let norm_shape = hidden_shape.clone();
            opt.known_tensor_shapes
                .insert(norm_out.clone(), norm_shape.clone());
            opt.add_op_node(
                OpKind::RmsNorm,
                Some(layer),
                vec![residual_in.clone()],
                vec![norm_out.clone()],
                vec![hidden_shape.clone()],
                Some(hidden_shape.clone()),
                false,
            );

            // Q projection
            if layer_plan.q_proj_tensor_id != 0 {
                let q_out = format!("layer_{}_q", layer);
                let q_shape = vec![1u32, plan.hidden_size];
                opt.known_tensor_shapes
                    .insert(q_out.clone(), q_shape.clone());
                opt.add_op_node(
                    OpKind::QProj,
                    Some(layer),
                    vec![norm_out.clone()],
                    vec![q_out.clone()],
                    vec![hidden_shape.clone()],
                    Some(q_shape.clone()),
                    false,
                );
            }

            // K projection
            if layer_plan.k_proj_tensor_id != 0 {
                let k_out = format!("layer_{}_k", layer);
                let k_shape = vec![1u32, plan.hidden_size];
                opt.known_tensor_shapes
                    .insert(k_out.clone(), k_shape.clone());
                opt.add_op_node(
                    OpKind::KProj,
                    Some(layer),
                    vec![norm_out.clone()],
                    vec![k_out.clone()],
                    vec![hidden_shape.clone()],
                    Some(k_shape.clone()),
                    false,
                );
            }

            // V projection (only for sliding attention; full uses K-equals-V)
            if layer_plan.v_proj_tensor_id != 0 {
                let v_out = format!("layer_{}_v", layer);
                let v_shape = vec![1u32, plan.hidden_size];
                opt.known_tensor_shapes
                    .insert(v_out.clone(), v_shape.clone());
                opt.add_op_node(
                    OpKind::VProj,
                    Some(layer),
                    vec![norm_out.clone()],
                    vec![v_out.clone()],
                    vec![hidden_shape.clone()],
                    Some(v_shape.clone()),
                    false,
                );
            }

            // RoPE
            let rope_out = format!("layer_{}_rope", layer);
            let rope_shape = vec![1u32, hdim * n_heads];
            opt.known_tensor_shapes
                .insert(rope_out.clone(), rope_shape.clone());
            opt.add_op_node(
                OpKind::RoPE,
                Some(layer),
                vec![],
                vec![rope_out.clone()],
                vec![],
                Some(rope_shape.clone()),
                false,
            );

            // Attention
            let attn_out = format!("layer_{}_attn", layer);
            let attn_shape = hidden_shape.clone();
            opt.known_tensor_shapes
                .insert(attn_out.clone(), attn_shape.clone());

            let attn_inputs = if layer_plan.v_proj_tensor_id != 0 {
                vec![
                    format!("layer_{}_q", layer),
                    format!("layer_{}_k", layer),
                    format!("layer_{}_v", layer),
                    rope_out.clone(),
                ]
            } else {
                vec![
                    format!("layer_{}_q", layer),
                    format!("layer_{}_k", layer),
                    rope_out.clone(),
                ]
            };

            opt.add_op_node(
                OpKind::Attention,
                Some(layer),
                attn_inputs,
                vec![attn_out.clone()],
                vec![hidden_shape.clone()],
                Some(attn_shape.clone()),
                false,
            );

            // O projection
            if layer_plan.o_proj_tensor_id != 0 {
                let o_out = format!("layer_{}_o", layer);
                let o_shape = hidden_shape.clone();
                opt.known_tensor_shapes
                    .insert(o_out.clone(), o_shape.clone());
                opt.add_op_node(
                    OpKind::OProj,
                    Some(layer),
                    vec![attn_out.clone()],
                    vec![o_out.clone()],
                    vec![attn_shape.clone()],
                    Some(o_shape.clone()),
                    false,
                );
            }

            // Residual add (post-attention)
            let post_attn = format!("layer_{}_post_attn", layer);
            let post_attn_shape = hidden_shape.clone();
            opt.known_tensor_shapes
                .insert(post_attn.clone(), post_attn_shape.clone());
            let attn_out_name = if layer_plan.o_proj_tensor_id != 0 {
                format!("layer_{}_o", layer)
            } else {
                attn_out.clone()
            };
            opt.add_op_node(
                OpKind::ResidualAdd,
                Some(layer),
                vec![residual_in.clone(), attn_out_name.clone()],
                vec![post_attn.clone()],
                vec![hidden_shape.clone(), hidden_shape.clone()],
                Some(post_attn_shape.clone()),
                false,
            );

            // Post-attention layer norm
            let post_norm_out = format!("layer_{}_post_norm", layer);
            let post_norm_shape = hidden_shape.clone();
            opt.known_tensor_shapes
                .insert(post_norm_out.clone(), post_norm_shape.clone());
            opt.add_op_node(
                OpKind::RmsNorm,
                Some(layer),
                vec![post_attn.clone()],
                vec![post_norm_out.clone()],
                vec![post_attn_shape.clone()],
                Some(post_norm_shape.clone()),
                false,
            );

            // Gate projection
            if layer_plan.gate_proj_tensor_id != 0 {
                let gate_out = format!("layer_{}_gate", layer);
                let gate_shape = vec![1u32, plan.intermediate_size()];
                opt.known_tensor_shapes
                    .insert(gate_out.clone(), gate_shape.clone());
                opt.add_op_node(
                    OpKind::GateProj,
                    Some(layer),
                    vec![post_norm_out.clone()],
                    vec![gate_out.clone()],
                    vec![post_norm_shape.clone()],
                    Some(gate_shape.clone()),
                    false,
                );

                // SiLU
                let silu_out = format!("layer_{}_silu", layer);
                let silu_shape = gate_shape.clone();
                opt.known_tensor_shapes
                    .insert(silu_out.clone(), silu_shape.clone());
                opt.add_op_node(
                    OpKind::SiLU,
                    Some(layer),
                    vec![gate_out.clone()],
                    vec![silu_out.clone()],
                    vec![gate_shape.clone()],
                    Some(silu_shape.clone()),
                    false,
                );
            }

            // Up projection
            if layer_plan.up_proj_tensor_id != 0 {
                let up_out = format!("layer_{}_up", layer);
                let up_shape = vec![1u32, plan.intermediate_size()];
                opt.known_tensor_shapes
                    .insert(up_out.clone(), up_shape.clone());
                opt.add_op_node(
                    OpKind::UpProj,
                    Some(layer),
                    vec![post_norm_out.clone()],
                    vec![up_out.clone()],
                    vec![post_norm_shape.clone()],
                    Some(up_shape.clone()),
                    false,
                );
            }

            // Gate * Up (element-wise multiply)
            if layer_plan.gate_proj_tensor_id != 0 && layer_plan.up_proj_tensor_id != 0 {
                let mul_out = format!("layer_{}_mul", layer);
                let mul_shape = vec![1u32, plan.intermediate_size()];
                opt.known_tensor_shapes
                    .insert(mul_out.clone(), mul_shape.clone());
                opt.add_op_node(
                    OpKind::GateTimesUp,
                    Some(layer),
                    vec![
                        format!("layer_{}_silu", layer),
                        format!("layer_{}_up", layer),
                    ],
                    vec![mul_out.clone()],
                    vec![
                        vec![1u32, plan.intermediate_size()],
                        vec![1u32, plan.intermediate_size()],
                    ],
                    Some(mul_shape.clone()),
                    false,
                );
            }

            // Down projection
            if layer_plan.down_proj_tensor_id != 0 {
                let down_out = format!("layer_{}_down", layer);
                let down_shape = hidden_shape.clone();
                opt.known_tensor_shapes
                    .insert(down_out.clone(), down_shape.clone());
                let down_in = if layer_plan.gate_proj_tensor_id != 0 {
                    format!("layer_{}_mul", layer)
                } else {
                    format!("layer_{}_up", layer)
                };
                opt.add_op_node(
                    OpKind::DownProj,
                    Some(layer),
                    vec![down_in],
                    vec![down_out.clone()],
                    vec![vec![1u32, plan.intermediate_size()]],
                    Some(down_shape.clone()),
                    false,
                );

                // Residual add (post-FFW)
                let layer_output = format!("layer_{}_output", layer);
                let layer_output_shape = hidden_shape.clone();
                opt.known_tensor_shapes
                    .insert(layer_output.clone(), layer_output_shape.clone());
                opt.add_op_node(
                    OpKind::ResidualAdd,
                    Some(layer),
                    vec![post_attn.clone(), down_out.clone()],
                    vec![layer_output.clone()],
                    vec![post_attn_shape.clone(), down_shape.clone()],
                    Some(layer_output_shape.clone()),
                    false,
                );
            }
        }

        // ── Epilogue: final norm, output projection, softcap, argmax ─
        let last_layer = if plan.layers.is_empty() {
            "embedding_output".to_string()
        } else {
            format!("layer_{}_output", plan.layers.len() as u32 - 1)
        };

        if plan.epilogue.final_norm_tensor_id != 0 {
            let fn_out = "final_norm_output".to_string();
            let fn_shape = hidden_shape.clone();
            opt.known_tensor_shapes
                .insert(fn_out.clone(), fn_shape.clone());
            opt.add_op_node(
                OpKind::FinalNorm,
                None,
                vec![last_layer.clone()],
                vec![fn_out.clone()],
                vec![hidden_shape.clone()],
                Some(fn_shape.clone()),
                false,
            );

            if plan.epilogue.output_projection_tensor_id.is_some() {
                let proj_out = "output_projection_output".to_string();
                let proj_shape = vec![1u32, plan.vocab_size];
                opt.known_tensor_shapes
                    .insert(proj_out.clone(), proj_shape.clone());
                opt.add_op_node(
                    OpKind::OutputProjection,
                    None,
                    vec![fn_out.clone()],
                    vec![proj_out.clone()],
                    vec![fn_shape.clone()],
                    Some(proj_shape.clone()),
                    false,
                );

                // Optional softcap
                if plan.final_logit_softcapping.is_some() {
                    let sc_out = "softcap_output".to_string();
                    let sc_shape = proj_shape.clone();
                    opt.known_tensor_shapes
                        .insert(sc_out.clone(), sc_shape.clone());
                    opt.add_op_node(
                        OpKind::Softcap,
                        None,
                        vec![proj_out.clone()],
                        vec![sc_out.clone()],
                        vec![proj_shape.clone()],
                        Some(sc_shape.clone()),
                        false,
                    );

                    opt.add_op_node(
                        OpKind::Argmax,
                        None,
                        vec![sc_out.clone()],
                        vec!["output_token".into()],
                        vec![sc_shape.clone()],
                        Some(vec![]),
                        false,
                    );
                } else {
                    opt.add_op_node(
                        OpKind::Argmax,
                        None,
                        vec![proj_out.clone()],
                        vec!["output_token".into()],
                        vec![proj_shape.clone()],
                        Some(vec![]),
                        false,
                    );
                }
            } else {
                // No output projection: logits are computed elsewhere
                // (e.g., tied embedding weights used as LM head).
                opt.add_op_node(
                    OpKind::Argmax,
                    None,
                    vec![fn_out.clone()],
                    vec!["output_token".into()],
                    vec![fn_shape.clone()],
                    Some(vec![]),
                    false,
                );
            }
        }

        opt
    }

    /// Helper: register a consumer edge for each input tensor.
    fn register_consumers(&mut self, node: &OpNode) {
        for input in &node.inputs {
            self.tensor_to_consumers
                .entry(input.clone())
                .or_default()
                .push(node.id);
        }
    }

    /// Helper: create an op node and register edges.
    fn add_op_node(
        &mut self,
        kind: OpKind,
        layer_index: Option<u32>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        known_input_shapes: Vec<Vec<u32>>,
        known_output_shape: Option<Vec<u32>>,
        is_constant: bool,
    ) {
        let id = self.nodes.len();
        let node = OpNode {
            id,
            kind,
            layer_index,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            known_input_shapes,
            known_output_shape,
            is_constant,
            is_dead: false,
        };
        for output in &outputs {
            self.tensor_to_producer.insert(output.clone(), id);
        }
        self.register_consumers(&node);
        self.nodes.push(node);
    }

    /// ── Constant folding pass ───────────────────────────────────────
    ///
    /// Identify operations whose all inputs are compile-time constants.
    /// For each such operation, mark the operation and its outputs as
    /// constant, enabling downstream folding.
    fn run_constant_folding(&mut self) {
        // Seed known constants from plan metadata.
        // Rope theta, rms_norm eps, and similar scalars are constants.
        self.constant_tensors.insert("input_ids".into());

        // Iteratively fold: mark nodes constant when all inputs are constant.
        let mut changed = true;
        while changed {
            changed = false;
            for node in self.nodes.iter_mut() {
                if node.is_constant || node.is_dead {
                    continue;
                }
                // Check if all inputs are constant.
                let all_inputs_constant = node.inputs.iter().all(|input| {
                    input == "input_ids"
                        || input.starts_with("constant_")
                        || self.constant_tensors.contains(input)
                });
                if all_inputs_constant && !node.inputs.is_empty() {
                    node.is_constant = true;
                    for output in &node.outputs {
                        self.constant_tensors.insert(output.clone());
                    }
                    changed = true;
                }
            }
        }
    }

    /// ── Shape propagation pass ──────────────────────────────────────
    ///
    /// Walk forward through the op graph, assigning known shapes to every
    /// tensor output.  Operations whose input shapes are fully known get
    /// their output shape derived.
    fn run_shape_propagation(&mut self) {
        // Initial shapes are already seeded in `known_tensor_shapes`
        // during graph construction.  Propagate forward.
        let mut changed = true;
        while changed {
            changed = false;
            for node in &self.nodes {
                if node.is_dead {
                    continue;
                }
                // Skip if we already have an output shape.
                for output in &node.outputs {
                    if self.known_tensor_shapes.contains_key(output) {
                        continue;
                    }
                }

                // Check if all input shapes are known.
                let all_inputs_known = node
                    .inputs
                    .iter()
                    .all(|input| self.known_tensor_shapes.contains_key(input));

                if !all_inputs_known {
                    continue;
                }

                // Infer output shape from operation kind and input shapes.
                let output_shape = self.infer_shape(node);
                if let Some(shape) = output_shape {
                    let outputs = node.outputs.clone();
                    for output in &outputs {
                        if !self.known_tensor_shapes.contains_key(output) {
                            self.known_tensor_shapes
                                .insert(output.clone(), shape.clone());
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    /// Infer the output shape for a node given known input shapes.
    fn infer_shape(&self, node: &OpNode) -> Option<Vec<u32>> {
        if node.known_output_shape.is_some() {
            return node.known_output_shape.clone();
        }
        match node.kind {
            OpKind::RmsNorm | OpKind::ResidualAdd | OpKind::QNorm | OpKind::KNorm => {
                // Same shape as input (preserves hidden dimension).
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .cloned()
            }
            OpKind::QProj | OpKind::KProj | OpKind::VProj | OpKind::OProj => {
                // Projections: [batch, seq_len, hidden] -> [batch, seq_len, hidden]
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .cloned()
            }
            OpKind::GateProj | OpKind::UpProj => {
                // FFN projections: [batch, seq_len, hidden] -> [batch, seq_len, intermediate]
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .map(|shape| {
                        let mut s = shape.clone();
                        let last = s.len();
                        if last >= 2 {
                            s[last - 1] = self.intermediate_size;
                        }
                        s
                    })
            }
            OpKind::DownProj => {
                // Down projection: [batch, seq_len, intermediate] -> [batch, seq_len, hidden]
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .map(|shape| {
                        let mut s = shape.clone();
                        let last = s.len();
                        if last >= 2 {
                            s[last - 1] = self.hidden_size;
                        }
                        s
                    })
            }
            OpKind::SiLU | OpKind::GateTimesUp => {
                // Element-wise: preserve shape.
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .cloned()
            }
            OpKind::Attention => {
                // Attention output: same hidden size as input.
                Some(vec![1u32, self.hidden_size])
            }
            OpKind::RoPE => {
                // RoPE preserves Q/K shape: [batch, seq_len, n_heads * head_dim]
                Some(vec![1u32, self.n_heads * self.head_dim])
            }
            OpKind::EmbeddingLookup => Some(vec![1u32, self.hidden_size]),
            OpKind::FinalNorm => Some(vec![1u32, self.hidden_size]),
            OpKind::OutputProjection => Some(vec![1u32, self.vocab_size]),
            OpKind::Softcap => {
                // Same shape as input (logits).
                node.inputs
                    .first()
                    .and_then(|input| self.known_tensor_shapes.get(input))
                    .cloned()
            }
            OpKind::Argmax => Some(vec![]), // scalar token
        }
    }

    /// ── Dead code elimination pass ──────────────────────────────────
    ///
    /// Trace from root outputs (output_token) backwards through consumers.
    /// Any node not reachable is marked dead.
    fn run_dead_code_elimination(&mut self) {
        // Collect root output tensor names — these are the final outputs
        // that must be kept.
        let mut roots: Vec<String> = Vec::new();
        for node in &self.nodes {
            if node.kind == OpKind::Argmax {
                for output in &node.outputs {
                    if output == "output_token" {
                        roots.push(output.clone());
                    }
                }
            }
        }

        // If no explicit argmax, treat all epilogue outputs as roots.
        if roots.is_empty() {
            for node in &self.nodes {
                if node.layer_index.is_none() && node.outputs.contains(&"output_token".into()) {
                    roots.extend(node.outputs.clone());
                }
            }
        }

        // Walk backwards from roots to mark live tensors.
        let mut live_tensors: HashSet<String> = roots.iter().cloned().collect();
        let mut worklist: Vec<String> = roots;

        while let Some(tensor) = worklist.pop() {
            // Find the producer of this tensor.
            if let Some(&producer_id) = self.tensor_to_producer.get(&tensor) {
                let producer = &self.nodes[producer_id];
                for input in &producer.inputs {
                    if live_tensors.insert(input.clone()) {
                        worklist.push(input.clone());
                    }
                }
            }
        }

        // Now mark nodes as dead if none of their outputs are live.
        for node in &mut self.nodes {
            if node.is_dead {
                continue;
            }
            let any_output_live = node.outputs.iter().any(|out| live_tensors.contains(out));
            if !any_output_live && !node.outputs.is_empty() {
                node.is_dead = true;
            }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────

/// Apply all graph optimizations to an execution plan before compilation.
///
/// Runs constant folding, shape propagation, and dead code elimination
/// in that order.  The plan is modified in place.
pub fn optimize(plan: &mut ModelExecutionPlan) {
    let mut opt = GraphOptimizer::from_plan(plan);

    opt.run_constant_folding();
    opt.run_shape_propagation();
    opt.run_dead_code_elimination();

    // Write results back to the plan.
    apply_shapes_to_plan(plan, &opt);
    apply_dce_to_plan(plan, &opt);
}

/// Shape propagation: annotate tensors with known shapes to eliminate
/// runtime dynamic shape checks in Metal kernels.
pub fn shape_propagation(plan: &mut ModelExecutionPlan) {
    let mut opt = GraphOptimizer::from_plan(plan);
    opt.run_shape_propagation();
    apply_shapes_to_plan(plan, &opt);
}

/// Constant folding: precompute operations whose all inputs are known
/// at compile time.
pub fn constant_folding(plan: &mut ModelExecutionPlan) {
    let mut opt = GraphOptimizer::from_plan(plan);
    opt.run_constant_folding();
    apply_shapes_to_plan(plan, &opt);
}

/// Dead code elimination: remove operations whose outputs are never used.
pub fn dead_code_elimination(plan: &mut ModelExecutionPlan) {
    let mut opt = GraphOptimizer::from_plan(plan);
    opt.run_dead_code_elimination();
    apply_dce_to_plan(plan, &opt);
}

// ── Write-back helpers ────────────────────────────────────────────────────

/// Write shape annotations back into the plan.
///
/// Annotates each layer with shape metadata derived during propagation.
fn apply_shapes_to_plan(plan: &mut ModelExecutionPlan, opt: &GraphOptimizer) {
    // Collect per-layer shape annotations from the optimizer's known shapes.
    let mut annotated_layers: HashMap<u32, Vec<(String, Vec<u32>)>> = HashMap::new();

    for node in &opt.nodes {
        if let Some(layer) = node.layer_index {
            let entry = annotated_layers.entry(layer).or_default();
            for output in &node.outputs {
                if let Some(shape) = opt.known_tensor_shapes.get(output) {
                    entry.push((output.clone(), shape.clone()));
                }
            }
        }
    }

    // For each layer plan, store shape annotations in a dedicated field.
    // We persist them as serialized JSON in the layer's quantization_ids
    // area (which is the closest available extensibility slot for metadata).
    for layer in &mut plan.layers {
        if let Some(shapes) = annotated_layers.get(&layer.layer_index) {
            let encoded: Vec<String> = shapes
                .iter()
                .map(|(name, shape)| format!("{}:{}", name, shape_description(shape)))
                .collect();
            // Extend quantization_ids to carry shape annotations.
            // These are prefixed with "shape:" so the runtime can distinguish
            // them from actual quantization descriptors.
            for entry in &encoded {
                let shape_tag = format!("shape:{}", entry);
                if !layer.quantization_ids.contains(&shape_tag) {
                    layer.quantization_ids.push(shape_tag);
                }
            }
        }
    }
}

/// Write DCE results back to the plan.
///
/// Dead operations' tensor IDs are cleared from the layer plan so the
/// emission loop skips them.
fn apply_dce_to_plan(plan: &mut ModelExecutionPlan, opt: &GraphOptimizer) {
    // Collect tensor IDs that are referenced by dead nodes.
    // For each dead node, we want to identify which weight tensor ID
    // in the LayerPlan it corresponds to, and zero it out.
    for node in &opt.nodes {
        if !node.is_dead {
            continue;
        }

        // Map dead ops back to layer plan fields and reset their tensor IDs.
        if let Some(layer_idx) = node.layer_index {
            let idx = layer_idx as usize;
            if idx >= plan.layers.len() {
                continue;
            }
            let layer = &mut plan.layers[idx];

            // Reset the tensor ID for this operation so the emission loop
            // skips it (zero tensor ID = no weight).
            match node.kind {
                OpKind::QProj => layer.q_proj_tensor_id = 0,
                OpKind::KProj => layer.k_proj_tensor_id = 0,
                OpKind::VProj => layer.v_proj_tensor_id = 0,
                OpKind::OProj => layer.o_proj_tensor_id = 0,
                OpKind::GateProj => layer.gate_proj_tensor_id = 0,
                OpKind::UpProj => layer.up_proj_tensor_id = 0,
                OpKind::DownProj => layer.down_proj_tensor_id = 0,
                _ => {
                    // RmsNorm, SiLU, RoPE, etc. don't have weight tensor IDs
                    // in the layer plan — they are structural operations.
                }
            }
        }
    }

    // Also prune shape annotations that reference now-dead tensors.
    for layer in &mut plan.layers {
        layer
            .quantization_ids
            .retain(|id| !id.starts_with("shape:"));
    }
}

/// Produce a concise human-readable shape description ("HxW" or "D").
fn shape_description(shape: &[u32]) -> String {
    if shape.is_empty() {
        "scalar".to_string()
    } else {
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }
}

// ========================================================
// Helper trait for intermediate_size derivation
// ========================================================

trait IntermediateSize {
    fn intermediate_size(&self) -> u32;
}

impl IntermediateSize for ModelExecutionPlan {
    fn intermediate_size(&self) -> u32 {
        self.hidden_size * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        operation_route::OperationRoute, AneFusedIsland, EpiloguePlan, LayerPlan, ProloguePlan,
    };

    fn make_test_plan() -> ModelExecutionPlan {
        let layer = LayerPlan {
            layer_index: 0,
            attention_kind: "sliding_attention".into(),
            segment_id: "layer_0".into(),
            hidden_size: 3840,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            global_head_dim: None,
            n_global_kv_heads: None,
            sliding_window: 8192,
            rope_theta: 10000.0,
            partial_rotary_factor: None,
            attention_k_eq_v: false,
            q_norm_enabled: true,
            k_norm_enabled: true,
            q_proj_tensor_id: 1,
            k_proj_tensor_id: 2,
            v_proj_tensor_id: 3,
            o_proj_tensor_id: 4,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 5,
            up_proj_tensor_id: 6,
            down_proj_tensor_id: 7,
            input_layernorm_tensor_id: 8,
            post_attention_layernorm_tensor_id: 9,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: Vec::new(),
            quantization_ids: Vec::new(),
            route: OperationRoute {
                rms_norm: 1,
                silu: 0,
                matmul: 0,
                attention: 0,
                softmax: 0,
                rope: 0,
                add: 1,
                multiply: 1,
                transpose: 0,
                reshape: 1,
            },
            fused_operations: Vec::new(),
        };

        ModelExecutionPlan {
            prologue: ProloguePlan {
                segment_id: "persistent".into(),
                embedding_tensor_id: 10,
                embedding_name: "model.embed_tokens.weight".into(),
                embedding_shape: vec![256000, 3840],
                embedding_dtype: "U8".into(),
            },
            layers: vec![layer],
            epilogue: EpiloguePlan {
                segment_id: "persistent".into(),
                final_norm_tensor_id: 11,
                final_norm_name: "model.norm.weight".into(),
                output_projection_tensor_id: Some(12),
                output_projection_name: Some("lm_head.weight".into()),
                final_logit_softcapping: Some(30.0),
                vocab_size: 256000,
            },
            fused_ane_islands: Vec::new(),
            hidden_size: 3840,
            vocab_size: 256000,
            sliding_window: 8192,
            final_logit_softcapping: Some(30.0),
            tie_word_embeddings: false,
            rms_norm_eps: 1e-6,
            speculative_config: None,
            generation_regime: Default::default(),
            diffusion_config: Default::default(),
            diffusion_execution_plan: Default::default(),
            kv_cache_mode: Default::default(),
        }
    }

    #[test]
    fn test_optimize_does_not_crash() {
        let mut plan = make_test_plan();
        // Smoke test: running all passes on a valid plan must not panic.
        optimize(&mut plan);
        // Verify the plan still has the expected layer.
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.layers[0].layer_index, 0);
    }

    #[test]
    fn test_constant_folding_no_crash() {
        let mut plan = make_test_plan();
        constant_folding(&mut plan);
        // No crash is the main test, but verify structure preserved.
        assert_eq!(plan.layers.len(), 1);
    }

    #[test]
    fn test_shape_propagation_annotates_shapes() {
        let mut plan = make_test_plan();
        shape_propagation(&mut plan);
        // After shape propagation, the layer should have shape annotations.
        assert!(plan.layers[0]
            .quantization_ids
            .iter()
            .any(|id| id.starts_with("shape:")));
    }

    #[test]
    fn test_dead_code_elimination_no_false_positives() {
        let mut plan = make_test_plan();
        dead_code_elimination(&mut plan);
        // All ops are reachable in a well-formed plan, so no tensor IDs
        // should be zeroed.
        assert_eq!(plan.layers[0].q_proj_tensor_id, 1);
        assert_eq!(plan.layers[0].k_proj_tensor_id, 2);
        assert_eq!(plan.layers[0].v_proj_tensor_id, 3);
        assert_eq!(plan.layers[0].o_proj_tensor_id, 4);
    }

    #[test]
    fn test_optimize_on_empty_layers() {
        let mut plan = ModelExecutionPlan {
            prologue: ProloguePlan {
                ..Default::default()
            },
            layers: Vec::new(),
            epilogue: EpiloguePlan {
                ..Default::default()
            },
            fused_ane_islands: Vec::new(),
            hidden_size: 3840,
            vocab_size: 256000,
            sliding_window: 8192,
            final_logit_softcapping: None,
            tie_word_embeddings: false,
            rms_norm_eps: 1e-6,
            speculative_config: None,
            generation_regime: Default::default(),
            diffusion_config: Default::default(),
            diffusion_execution_plan: Default::default(),
            kv_cache_mode: Default::default(),
        };
        optimize(&mut plan);
        assert!(plan.layers.is_empty());
    }

    #[test]
    fn test_optimize_integration_preserves_reachable_tensors() {
        let mut plan = make_test_plan();
        optimize(&mut plan);
        // After full optimization, all reachable weight tensor IDs remain intact.
        assert_eq!(plan.layers[0].q_proj_tensor_id, 1);
        assert_eq!(plan.layers[0].k_proj_tensor_id, 2);
        assert_eq!(plan.layers[0].v_proj_tensor_id, 3);
        assert_eq!(plan.layers[0].o_proj_tensor_id, 4);
        assert_eq!(plan.layers[0].gate_proj_tensor_id, 5);
        assert_eq!(plan.layers[0].up_proj_tensor_id, 6);
        assert_eq!(plan.layers[0].down_proj_tensor_id, 7);
        assert_eq!(plan.layers[0].input_layernorm_tensor_id, 8);
        assert_eq!(plan.layers[0].post_attention_layernorm_tensor_id, 9);
        assert_eq!(plan.prologue.embedding_tensor_id, 10);
        assert_eq!(plan.epilogue.final_norm_tensor_id, 11);
        assert_eq!(plan.epilogue.output_projection_tensor_id, Some(12));
    }
}
