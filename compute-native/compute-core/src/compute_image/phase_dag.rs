//! PhaseDAG — compiler-emitted typed, acyclic execution plan.
//!
//! These types represent the final output of the Tribunus ComputeImage
//! compiler. The scheduler consumes this plan directly — no runtime
//! dependency reconstruction is required.
//!
//! Each [`EmittedPhaseGraph`] encodes phases, edges, arena layout, and
//! a concurrency hint. The graph is guaranteed acyclic at emission time
//! (see [`EmittedPhaseGraph::validate`]).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Target compute lane for a phase.
///
/// Determines which execution backend dispatches the phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeLane {
    Metal,
    Accelerate,
    CoreMl,
    Arena,
}

/// Semantic meaning of a directed edge between phases.
///
/// Each variant represents a *non-overlapping* dependency reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticKind {
    /// Data tensor dependency — `from_phase` produces the tensor `to_phase` consumes.
    /// Data tensor dependency — `from_phase` produces the tensor `to_phase` consumes.
    Data,
    /// Arena slot ownership transfer.
    ArenaOwnership,
    /// State-epoch ordering (e.g. KV-cache epoch boundaries).
    StateEpoch,
    /// Completion of a transfer (load / device-to-device) operation.
    TransferCompletion,
    /// Request-ordering constraint (e.g. token order).
    RequestOrdering,
    /// Decomposition of a fallback variant into its higher-ranked primitives.
    FallbackDecomposition,
}

/// Kind of work performed by a phase.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhaseKind {
    MlxDecode,
    MetalFusedKernel,
    CoreMlGraph,
    AccelMatMul,
    AccelElementWise,
    ArenaAlloc,
    SyncBarrier,
    Transfer,
    ResidualRmsNorm,
    /// Migrate weight residency for required layers.
    WeightResidency,
    /// Legacy MLX prologue runner — embedding lookup.
    LegacyMlxPrologue,
    /// Legacy MLX epilogue runner — final norm + lm_head.
    LegacyMlxEpilogue,
    /// Token sampling runner — argmax from logits.
    Sampling,
}

/// Completion status of a phase — used for observability and fallback
/// bookkeeping.
///
/// `Failed` and `FallbackUsed` carry a human-readable reason string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason")]
pub enum PhaseCompletionStatus {
    Pending,
    Complete,
    Failed(String),
    FallbackUsed(String),
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Reference to a single arena allocation slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaSlotRef {
    /// Unique slot identifier within the arena plan.
    pub slot_id: String,
    /// Size of the allocation in bytes.
    pub byte_size: u64,
    /// Required alignment in bytes.
    pub alignment: u64,
    /// Lane that owns / manages this slot.
    pub lane: ComputeLane,
}

/// A single emitted phase — one schedulable unit of work.
///
/// Phases are the vertices of the emitted DAG. Each phase runs on a single
/// [`ComputeLane`] and may reference multiple arena slots and tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedPhase {
    /// Unique phase identifier.
    pub phase_id: String,
    /// Kind of work this phase performs.
    pub kind: PhaseKind,
    /// Target compute lane.
    pub lane: ComputeLane,
    /// Logical operation names (e.g. `["q_proj", "k_proj", "v_proj"]`).
    pub ops: Vec<String>,
    /// Arena slots claimed by this phase.
    pub arena_slots: Vec<ArenaSlotRef>,
    /// Names of tensors this phase reads.
    pub tensor_reads: Vec<String>,
    /// Names of tensors this phase writes.
    pub tensor_writes: Vec<String>,
    /// Estimated floating-point operations (used for scheduling heuristics).
    pub estimated_ops: u64,
    /// Compiler-extended metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// A directed edge between two emitted phases.
///
/// Edges form the dependency structure of the phase DAG. Each edge carries a
/// single, non-overlapping [`SemanticKind`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedPhaseEdge {
    /// Source phase identifier.
    pub from_phase: String,
    /// Destination phase identifier.
    pub to_phase: String,
    /// Semantic kind of this dependency.
    pub semantic_kind: SemanticKind,
    /// Human-readable label (e.g. `"hidden_states"` for Data edges).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Compiler-extended metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Compiled arena layout for an emitted phase graph.
///
/// Declares all arena slots, their sizes, and alignments in a single
/// plan that the scheduler can materialise ahead of execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedArenaPlan {
    /// Total arena bytes required.
    pub total_bytes: u64,
    /// All arena slots in declaration order.
    pub slots: Vec<ArenaSlotRef>,
}

/// Concurrency hint emitted by the compiler.
///
/// Each inner `Vec<String>` names a set of phase IDs that *may* run in
/// parallel. This is a hint, not a constraint — the scheduler remains
/// free to serialise if resources are constrained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedConcurrencyPlan {
    /// Sets of phase IDs that may execute concurrently.
    pub independent_sets: Vec<Vec<String>>,
}

/// Compiler-emitted typed, acyclic execution plan.
///
/// This is the top-level container that the scheduler consumes
/// directly. It is guaranteed acyclic and well-formed after
/// [`EmittedPhaseGraph::validate`] succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedPhaseGraph {
    /// All phases in the graph.
    pub phases: Vec<EmittedPhase>,
    /// All dependency edges.
    pub edges: Vec<EmittedPhaseEdge>,
    /// Arena layout plan.
    pub arena_plan: EmittedArenaPlan,
    /// Concurrency hints.
    pub concurrency_plan: EmittedConcurrencyPlan,
    /// Compiler version string.
    pub compiler_version: String,
}

impl Default for EmittedPhaseGraph {
    fn default() -> Self {
        Self {
            phases: Vec::new(),
            edges: Vec::new(),
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: Vec::new(),
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: Vec::new(),
            },
            compiler_version: "tribunus-phase-dag-v1".to_string(),
        }
    }
}

impl EmittedPhaseGraph {
    /// Validate structural and semantic invariants of the graph.
    ///
    /// Checks performed:
    /// 1. No duplicate `phase_id` values.
    /// 2. Every edge's `from_phase` and `to_phase` exist in [`Self::phases`].
    /// 3. Arena plan contains no duplicate `slot_id` values.
    /// 4. The graph is acyclic (DFS cycle detection).
    /// 5. Every phase has at least one `tensor_write`.
    /// 6. Every `tensor_read` is produced by a transitive predecessor.
    /// 7. [`SemanticKind::Data`] edges carry a label; [`SemanticKind::StateEpoch`]
    ///    edges reference KV-cache state.
    pub fn validate(&self) -> Result<(), String> {
        // --- 1. Duplicate phase ids ---
        {
            let mut seen = HashSet::new();
            for p in &self.phases {
                if !seen.insert(&p.phase_id) {
                    return Err(format!("Duplicate phase_id: '{}'", p.phase_id));
                }
            }
        }

        let phase_ids: HashSet<&str> = self.phases.iter().map(|p| p.phase_id.as_str()).collect();

        // --- 2. Edge endpoints exist ---
        for e in &self.edges {
            if !phase_ids.contains(e.from_phase.as_str()) {
                return Err(format!(
                    "Edge references unknown from_phase '{}'",
                    e.from_phase
                ));
            }
            if !phase_ids.contains(e.to_phase.as_str()) {
                return Err(format!("Edge references unknown to_phase '{}'", e.to_phase));
            }
        }

        // --- 3. No duplicate arena slot ids ---
        {
            let mut seen = HashSet::new();
            for slot in &self.arena_plan.slots {
                if !seen.insert(&slot.slot_id) {
                    return Err(format!("Duplicate arena slot_id: '{}'", slot.slot_id));
                }
            }
        }

        // --- 4. Acyclic (DFS) ---
        {
            let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
            for pid in &phase_ids {
                adj.entry(pid).or_default();
            }
            for e in &self.edges {
                adj.entry(e.from_phase.as_str())
                    .or_default()
                    .push(e.to_phase.as_str());
            }

            // Iterative DFS cycle detection using a manual stack.
            // Each entry: (node, iterator_index, visited_children)
            enum VisitState {
                Enter,
                Exit,
            }

            let mut visited: HashSet<&str> = HashSet::new();
            let mut in_stack: HashSet<&str> = HashSet::new();
            let mut stack: Vec<(&str, VisitState)> = Vec::new();

            let all_nodes: Vec<&str> = phase_ids.iter().copied().collect();

            for &start in &all_nodes {
                if visited.contains(start) {
                    continue;
                }
                stack.push((start, VisitState::Enter));

                while let Some((node, state)) = stack.pop() {
                    match state {
                        VisitState::Enter => {
                            if in_stack.contains(node) {
                                return Err(format!(
                                    "Cycle detected: phase '{}' is part of a cycle",
                                    node
                                ));
                            }
                            if visited.contains(node) {
                                continue;
                            }
                            in_stack.insert(node);
                            visited.insert(node);
                            // Push exit marker then children.
                            stack.push((node, VisitState::Exit));
                            if let Some(neighbours) = adj.get(node) {
                                for &next in neighbours.iter().rev() {
                                    stack.push((next, VisitState::Enter));
                                }
                            }
                        }
                        VisitState::Exit => {
                            in_stack.remove(node);
                        }
                    }
                }
            }
        }

        // --- 5. Every phase must write at least one tensor ---
        for p in &self.phases {
            if p.tensor_writes.is_empty() {
                return Err(format!(
                    "Phase '{}' has no tensor_writes (phases must produce something)",
                    p.phase_id
                ));
            }
        }

        // --- 6. Every tensor_read is produced by some transitive predecessor ---
        {
            // Compute transitive closure of written tensors for each phase.
            // We do a forward topological propagation: for each phase in
            // topological order, its reachable writes = its own writes ∪
            // union of predecessors' reachable writes.
            let topo = self.topological_order()?;
            let mut reachable_writes: HashMap<&str, HashSet<&str>> = HashMap::new();
            for p in &self.phases {
                reachable_writes
                    .entry(p.phase_id.as_str())
                    .or_insert_with(HashSet::new)
                    .extend(p.tensor_writes.iter().map(|s| s.as_str()));
            }

            // Build reverse adjacency (predecessors).
            let mut rev_adj: HashMap<&str, Vec<&str>> = HashMap::new();
            for pid in &phase_ids {
                rev_adj.entry(pid).or_default();
            }
            for e in &self.edges {
                rev_adj
                    .get_mut(e.to_phase.as_str())
                    .expect("edge to_phase already validated")
                    .push(e.from_phase.as_str());
            }

            // Propagate writes forward along topological order.
            for phase in &topo {
                let pid = phase.phase_id.as_str();
                let predecessors = rev_adj.get(pid).map(|v| v.as_slice()).unwrap_or(&[]);
                let pred_writes: HashSet<&str> = predecessors
                    .iter()
                    .flat_map(|pred| reachable_writes.get(pred).into_iter().flatten().copied())
                    .collect();
                if !pred_writes.is_empty() {
                    reachable_writes
                        .get_mut(pid)
                        .expect("phase in reachable_writes")
                        .extend(pred_writes);
                }
            }

            for p in &self.phases {
                let available = reachable_writes.get(p.phase_id.as_str());
                for read in &p.tensor_reads {
                    let found = available
                        .map(|set| set.contains(read.as_str()))
                        .unwrap_or(false)
                        // Also check own writes (a phase may read a tensor it
                        // produced earlier in the same phase).
                        || p.tensor_writes.contains(read);
                    if !found {
                        return Err(format!(
                            "Phase '{}' reads tensor '{}' which is not produced \
                             by any transitive predecessor",
                            p.phase_id, read
                        ));
                    }
                }
            }
        }

        // --- 7. Semantic edge kind consistency ---
        for e in &self.edges {
            match e.semantic_kind {
                SemanticKind::Data => {
                    if e.label.is_none() {
                        return Err(format!(
                            "Data edge from '{}' to '{}' has no label",
                            e.from_phase, e.to_phase
                        ));
                    }
                }
                SemanticKind::StateEpoch => {
                    // StateEpoch edges should reference KV-cache related state.
                    let label = e.label.as_deref().unwrap_or("");
                    if label.is_empty() || !label.to_lowercase().contains("kv") {
                        return Err(format!(
                            "StateEpoch edge from '{}' to '{}' must reference \
                             KV state (got label: {:?})",
                            e.from_phase, e.to_phase, e.label
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Return phases that have a direct edge pointing *to* `phase_id`
    /// (i.e. phases that must complete before `phase_id` may start).
    pub fn predecessors(&self, phase_id: &str) -> Vec<&EmittedPhase> {
        let ids: HashSet<&str> = self
            .edges
            .iter()
            .filter(|e| e.to_phase == phase_id)
            .map(|e| e.from_phase.as_str())
            .collect();
        self.phases
            .iter()
            .filter(|p| ids.contains(p.phase_id.as_str()))
            .collect()
    }

    /// Return phases that have a direct edge *from* `phase_id`
    /// (i.e. phases that depend on `phase_id` completing).
    pub fn successors(&self, phase_id: &str) -> Vec<&EmittedPhase> {
        let ids: HashSet<&str> = self
            .edges
            .iter()
            .filter(|e| e.from_phase == phase_id)
            .map(|e| e.to_phase.as_str())
            .collect();
        self.phases
            .iter()
            .filter(|p| ids.contains(p.phase_id.as_str()))
            .collect()
    }

    /// Compute a topological ordering of phases using Kahn's algorithm.
    ///
    /// Returns an error if the graph contains a cycle (which should not
    /// happen after [`Self::validate`] passes).
    pub fn topological_order(&self) -> Result<Vec<&EmittedPhase>, String> {
        // Map phase_id -> index in self.phases for O(1) lookup.
        let index_of: HashMap<&str, usize> = self
            .phases
            .iter()
            .enumerate()
            .map(|(i, p)| (p.phase_id.as_str(), i))
            .collect();

        // In-degree: how many edges point *to* each phase.
        let mut in_degree = vec![0u32; self.phases.len()];
        for e in &self.edges {
            if let Some(&idx) = index_of.get(e.to_phase.as_str()) {
                in_degree[idx] += 1;
            }
        }

        // Seed the queue with zero-in-degree phases.
        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| i)
            .collect();

        let mut result: Vec<&EmittedPhase> = Vec::with_capacity(self.phases.len());

        while let Some(idx) = queue.pop() {
            result.push(&self.phases[idx]);

            // Decrease in-degree of each successor.
            for e in &self.edges {
                if e.from_phase == self.phases[idx].phase_id {
                    if let Some(&succ) = index_of.get(e.to_phase.as_str()) {
                        in_degree[succ] = in_degree[succ].checked_sub(1).ok_or_else(|| {
                            "Underflow in in-degree during Kahn's algorithm".to_string()
                        })?;
                        if in_degree[succ] == 0 {
                            queue.push(succ);
                        }
                    }
                }
            }
        }

        if result.len() != self.phases.len() {
            return Err(format!(
                "Graph contains a cycle: emitted {} of {} phases",
                result.len(),
                self.phases.len()
            ));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for EmittedPhaseGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhaseGraph({} phases, {} edges, Arena: {} bytes)",
            self.phases.len(),
            self.edges.len(),
            self.arena_plan.total_bytes,
        )
    }
}
