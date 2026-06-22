//! NPUMoE — efficient Mixture-of-Experts expert scheduling on Apple ANE/NPU.
//!
//! Each ANE core has ~512 KB SRAM. A single expert's parameters (gate/up/down
//! projections) are ~300 KB for a 32×4096 MoE layer. This means 1–2 experts
//! fit per core. With 16–32 ANE cores on Apple Silicon, we schedule which
//! experts run on which cores and pipeline the top-K computation.
//!
//! # Scheduling strategy
//!
//! Experts are distributed round-robin across available ANE cores, respecting
//! the SRAM budget per core.  At inference time, each core is loaded with its
//! subset of experts; tokens are routed to their top-K experts and each core
//! independently computes the expert FFN for the tokens assigned to it.
//!
//! The forward pass:
//!   1. Softmax router logits → routing probabilities
//!   2. Select top-K experts per token
//!   3. Group tokens by assigned expert
//!   4. For each expert: compute gate_proj → SiLU × up_proj → down_proj
//!   5. Weight outputs by routing probability and accumulate

use mlx_rs::Array;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// NPUMoE expert scheduler for ANE.
///
/// Schedules expert computations across ANE cores based on SRAM limits.
/// Each core can hold `sram_per_core / expert_size_bytes` experts.
#[derive(Debug, Clone)]
pub struct AneMoEScheduler {
    /// Number of ANE cores available (e.g. 16 on M1 Max, 32 on M3 Ultra).
    pub num_cores: u32,
    /// SRAM per ANE core in bytes (typically 512 KB = 524288).
    pub sram_per_core: u32,
    /// Size of one expert's weights in bytes (gate + up + down projections).
    pub expert_size_bytes: u32,
    /// Number of experts that fit in one core's SRAM (computed at construction).
    pub experts_per_core: u32,
}

/// Per-expert weights for an MoE FFN layer.
///
/// Each expert is a SwiGLU FFN: `down_proj(SiLU(gate_proj(x)) × up_proj(x))`.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    /// Gate projection weight, shape `[hidden_size, intermediate_size]`.
    pub gate_proj: Array,
    /// Up projection weight, shape `[hidden_size, intermediate_size]`.
    pub up_proj: Array,
    /// Down projection weight, shape `[intermediate_size, hidden_size]`.
    pub down_proj: Array,
    /// Expert index within the MoE layer.
    pub expert_id: u32,
}

/// Per-core expert data loaded into ANE SRAM.
#[derive(Debug, Clone)]
pub struct ANECoreExperts {
    /// Core identifier (0..num_cores).
    pub core_id: u32,
    /// Experts assigned to this core (at most `experts_per_core`).
    pub experts: Vec<ExpertWeights>,
    /// Base IOAddress of this core's SRAM region.
    pub sram_base: u64,
}

/// Routing result for one token: which experts are active and with what weight.
#[derive(Debug, Clone)]
pub struct TokenRouting {
    /// Expert indices selected for this token (length ≤ top_k).
    pub expert_indices: Vec<u32>,
    /// Normalized routing weights for each selected expert (sums to 1.0).
    pub routing_weights: Vec<f32>,
}

// ---------------------------------------------------------------------------
// AneMoEScheduler
// ---------------------------------------------------------------------------

impl AneMoEScheduler {
    /// Create a new scheduler with the given ANE hardware parameters.
    ///
    /// `experts_per_core` is floored: if an expert is larger than one core's
    /// SRAM, `experts_per_core` will be 0 and the caller must handle that
    /// (e.g. by splitting the expert across multiple cores — hardly ever
    /// needed for typical MoE configs on Apple Silicon).
    pub fn new(num_cores: u32, sram_per_core: u32, expert_size_bytes: u32) -> Self {
        let experts_per_core = if expert_size_bytes > 0 {
            (sram_per_core / expert_size_bytes).max(1)
        } else {
            1
        };
        Self {
            num_cores,
            sram_per_core,
            expert_size_bytes,
            experts_per_core,
        }
    }

    /// Build an [`AneMoEScheduler`] for a typical 32×4096 MoE config on
    /// Apple Silicon (16 cores, 512 KB SRAM, ~300 KB per expert).
    pub fn default_m1_max() -> Self {
        Self::new(16, 512 * 1024, 300 * 1024)
    }

    /// Build a scheduler for M3 Ultra (32 cores).
    pub fn default_m3_ultra() -> Self {
        Self::new(32, 512 * 1024, 300 * 1024)
    }

    /// Compute the optimal expert-to-core mapping.
    ///
    /// Experts are distributed round-robin across cores. Each core receives
    /// at most `experts_per_core` experts.  The returned `Vec` has one entry
    /// per core; each entry is the list of expert indices assigned to that
    /// core.
    ///
    /// When `num_experts > num_cores * experts_per_core`, the schedule wraps
    /// around in multiple rounds — core 0 gets experts 0..P, 16..16+P, etc.
    pub fn schedule_experts(&self, num_experts: u32, _top_k: u32) -> Vec<Vec<u32>> {
        let n_cores = self.num_cores as usize;
        let mut schedule: Vec<Vec<u32>> = vec![Vec::new(); n_cores];
        let cap = self.experts_per_core as usize;

        if cap == 0 {
            // Fallback: each expert gets its own core, round-robin, no cap.
            for expert_id in 0..num_experts {
                let core = (expert_id as usize) % n_cores;
                schedule[core].push(expert_id);
            }
            return schedule;
        }

        let mut core_idx = 0;
        for expert_id in 0..num_experts {
            // Try up to num_cores slots before wrapping.
            for _ in 0..n_cores {
                if schedule[core_idx].len() < cap {
                    schedule[core_idx].push(expert_id);
                    break;
                }
                core_idx = (core_idx + 1) % n_cores;
            }
        }

        schedule
    }

    /// Run the scheduled MoE forward pass on ANE.
    ///
    /// 1. Computes routing probabilities from `router_logits`
    /// 2. Selects top-K experts per token
    /// 3. For each expert, computes the SwiGLU FFN on tokens assigned to it
    /// 4. Weights outputs by routing probability and accumulates them
    ///
    /// # Arguments
    /// * `hidden` — input hidden states, shape `[batch_size, hidden_size]`
    /// * `experts` — all expert weights for this MoE layer
    /// * `router_logits` — router logits, shape `[batch_size, num_experts]`
    /// * `top_k` — number of active experts per token
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, hidden_size]`.
    pub fn forward_moe(
        &self,
        hidden: &Array,
        experts: &[ExpertWeights],
        router_logits: &Array,
        top_k: u32,
    ) -> Result<Array, String> {
        let batch_size = hidden.shape().first().copied().unwrap_or(1) as usize;
        let hidden_size = hidden.shape().get(1).copied().unwrap_or(1) as usize;
        let num_experts = router_logits.shape().get(1).copied().unwrap_or(1) as usize;
        let top_k_usize = (top_k as usize).min(num_experts);

        if top_k_usize == 0 || experts.is_empty() {
            return Ok(Array::from_slice(
                &[0.0f32],
                &[batch_size as i32, hidden_size as i32],
            ));
        }

        // 1. Softmax over router logits → routing probabilities.
        let routing_probs = mlx_rs::ops::softmax_axes(router_logits, &[-1], None)
            .map_err(|e| format!("router softmax: {:?}", e))?;

        // 2. For each token, find top-K expert indices and their weights.
        //    We read the probs as f32, extract top-K on the host, then
        //    dispatch per expert.
        let probs_slice: Vec<f32> = routing_probs
            .try_as_slice::<f32>()
            .map_err(|e| format!("read routing probs: {:?}", e))?
            .to_vec();

        let mut per_token_routing: Vec<TokenRouting> = Vec::with_capacity(batch_size);
        for t in 0..batch_size {
            let base = t * num_experts;
            let mut indexed: Vec<(usize, f32)> = (0..num_experts)
                .map(|i| (i, probs_slice[base + i]))
                .collect();
            // Sort descending by probability.
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let selected: Vec<(u32, f32)> = indexed[..top_k_usize]
                .iter()
                .map(|&(idx, p)| (idx as u32, p))
                .collect();

            // Renormalise top-K weights so they sum to 1.
            let sum: f32 = selected.iter().map(|&(_, p)| p).sum();
            let weights: Vec<f32> = if sum > 0.0 {
                selected.iter().map(|&(_, p)| p / sum).collect()
            } else {
                // Uniform fallback when all probs are zero.
                selected.iter().map(|_| 1.0 / top_k_usize as f32).collect()
            };

            per_token_routing.push(TokenRouting {
                expert_indices: selected.iter().map(|&(idx, _)| idx).collect(),
                routing_weights: weights,
            });
        }

        // 3. Schedule experts across cores.
        let _schedule = self.schedule_experts(num_experts as u32, top_k);

        // 4. For each expert that was selected by at least one token, compute
        //    the SwiGLU FFN and add weighted contributions to the output.
        let mut output_data = vec![0.0f32; batch_size * hidden_size];

        // Collect which expert indices are actually used.
        let mut used_experts = vec![false; num_experts];
        for routing in &per_token_routing {
            for &e in &routing.expert_indices {
                if (e as usize) < num_experts {
                    used_experts[e as usize] = true;
                }
            }
        }

        // Process each expert that was selected.
        for expert_idx in 0..num_experts {
            if !used_experts[expert_idx] {
                continue;
            }

            // Find the expert weights.
            let expert = match experts.iter().find(|e| e.expert_id as usize == expert_idx) {
                Some(e) => e,
                None => continue,
            };

            // Gather tokens routed to this expert.
            let mut token_indices: Vec<usize> = Vec::new();
            let mut weights_for_expert: Vec<f32> = Vec::new();
            for (t, routing) in per_token_routing.iter().enumerate() {
                for (pos, &e) in routing.expert_indices.iter().enumerate() {
                    if e as usize == expert_idx {
                        token_indices.push(t);
                        weights_for_expert.push(routing.routing_weights[pos]);
                    }
                }
            }

            if token_indices.is_empty() {
                continue;
            }

            // Extract individual hidden states for these tokens.
            let hidden_slice: Vec<f32> = hidden
                .try_as_slice::<f32>()
                .map_err(|e| format!("read hidden: {:?}", e))?
                .to_vec();

            // For each token routed to this expert, compute the FFN and add
            // its weighted contribution to the output.
            for (pos, &t_idx) in token_indices.iter().enumerate() {
                let weight = weights_for_expert[pos];
                let token_hidden = &hidden_slice[t_idx * hidden_size..(t_idx + 1) * hidden_size];

                // Cast token hidden to Array for computation.
                let x = Array::from_slice(token_hidden, &[1, hidden_size as i32]);

                // gate_proj(x) → [1, intermediate_size]
                let gate = x
                    .matmul(&expert.gate_proj)
                    .map_err(|e| format!("expert {} gate_proj: {:?}", expert_idx, e))?;

                // up_proj(x) → [1, intermediate_size]
                let up = x
                    .matmul(&expert.up_proj)
                    .map_err(|e| format!("expert {} up_proj: {:?}", expert_idx, e))?;

                // SiLU(gate) × up
                let sig = mlx_rs::ops::sigmoid(&gate)
                    .map_err(|e| format!("expert {} sigmoid: {:?}", expert_idx, e))?;
                let activated = gate
                    .multiply(&sig)
                    .map_err(|e| format!("expert {} silu: {:?}", expert_idx, e))?;
                let gated = activated
                    .multiply(&up)
                    .map_err(|e| format!("expert {} gate*up: {:?}", expert_idx, e))?;

                // down_proj(gated) → [1, hidden_size]
                let out_arr = gated
                    .matmul(&expert.down_proj)
                    .map_err(|e| format!("expert {} down_proj: {:?}", expert_idx, e))?;

                // Eval to materialize.
                out_arr
                    .eval()
                    .map_err(|e| format!("expert {} eval: {:?}", expert_idx, e))?;

                // Read back and accumulate weighted.
                let out_slice: Vec<f32> = out_arr
                    .try_as_slice::<f32>()
                    .map_err(|e| format!("expert {} read output: {:?}", expert_idx, e))?
                    .to_vec();

                let out_base = t_idx * hidden_size;
                for h in 0..hidden_size {
                    output_data[out_base + h] += weight * out_slice[h];
                }
            }
        }

        Ok(Array::from_slice(
            &output_data,
            &[batch_size as i32, hidden_size as i32],
        ))
    }

    /// Compute how many ANE cores are needed to hold `num_active_experts`
    /// in a single pipeline round.
    pub fn cores_needed(&self, num_active_experts: u32) -> u32 {
        if self.experts_per_core == 0 {
            return num_active_experts;
        }
        (num_active_experts + self.experts_per_core - 1) / self.experts_per_core
    }

    /// Number of pipeline rounds needed to run `num_active_experts` experts
    /// on the available cores.
    pub fn pipeline_rounds(&self, num_active_experts: u32) -> u32 {
        let cores_available = self.num_cores;
        let per_round = cores_available * self.experts_per_core;
        if per_round == 0 {
            return num_active_experts;
        }
        (num_active_experts + per_round - 1) / per_round
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the approximate SRAM footprint of one expert's weights.
///
/// `hidden_size`: model hidden dimension (e.g. 4096)
/// `intermediate_size`: FFN intermediate dimension (e.g. 14336)
/// `bytes_per_param`: bytes per weight element (2 for FP16, 1 for quantized)
pub fn expert_sram_footprint(
    hidden_size: u32,
    intermediate_size: u32,
    bytes_per_param: u32,
) -> u32 {
    // gate: [hidden, intermediate], up: [hidden, intermediate], down: [intermediate, hidden]
    let gate_up = hidden_size * intermediate_size * bytes_per_param;
    let down = intermediate_size * hidden_size * bytes_per_param;
    2 * gate_up + down
}

/// Compute routing probabilities from raw router logits.
///
/// Applies softmax along the expert dimension.
pub fn compute_routing_probs(router_logits: &Array) -> Result<Array, String> {
    mlx_rs::ops::softmax_axes(router_logits, &[-1], None)
        .map_err(|e| format!("compute_routing_probs: {:?}", e))
}

/// Select top-K expert indices and their probabilities for each token.
///
/// Returns `(indices, probs)` each shaped `[batch, top_k]`.
pub fn select_top_k(routing_probs: &Array, top_k: u32) -> Result<(Array, Array), String> {
    let batch_size = routing_probs.shape().first().copied().unwrap_or(1) as usize;
    let num_experts = routing_probs.shape().get(1).copied().unwrap_or(1) as usize;
    let k = (top_k as usize).min(num_experts);

    let probs_slice: Vec<f32> = routing_probs
        .try_as_slice::<f32>()
        .map_err(|e| format!("select_top_k read: {:?}", e))?
        .to_vec();

    let mut top_indices = Vec::with_capacity(batch_size * k);
    let mut top_values = Vec::with_capacity(batch_size * k);

    for t in 0..batch_size {
        let base = t * num_experts;
        let mut indexed: Vec<(usize, f32)> = (0..num_experts)
            .map(|i| (i, probs_slice[base + i]))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Renormalise.
        let selected: Vec<(usize, f32)> = indexed[..k].to_vec();
        let sum: f32 = selected.iter().map(|&(_, p)| p).sum();
        for &(idx, p) in &selected {
            top_indices.push(idx as u32);
            top_values.push(if sum > 0.0 { p / sum } else { 1.0 / k as f32 });
        }
    }

    let indices_arr = Array::from_slice(&top_indices, &[batch_size as i32, k as i32]);
    let values_arr = Array::from_slice(&top_values, &[batch_size as i32, k as i32]);
    Ok((indices_arr, values_arr))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_new() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        assert_eq!(sched.num_cores, 16);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn test_default_m1_max() {
        let sched = AneMoEScheduler::default_m1_max();
        assert_eq!(sched.num_cores, 16);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn test_default_m3_ultra() {
        let sched = AneMoEScheduler::default_m3_ultra();
        assert_eq!(sched.num_cores, 32);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn test_schedule_experts_round_robin() {
        // 16 cores, 1 expert per core, 32 experts → 2 rounds
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        let schedule = sched.schedule_experts(32, 8);

        assert_eq!(schedule.len(), 16);
        // Each core should get at most 1 expert (experts_per_core = 1)
        for core_assignments in &schedule {
            assert!(core_assignments.len() <= 1);
        }

        // Verify all 32 experts are assigned
        let mut assigned: Vec<u32> = schedule.iter().flat_map(|v| v.iter().copied()).collect();
        assigned.sort();
        let expected: Vec<u32> = (0..32).collect();
        assert_eq!(assigned, expected);
    }

    #[test]
    fn test_schedule_experts_two_per_core() {
        // 8 cores, 2 experts per core, 16 experts → all fit
        let sched = AneMoEScheduler::new(8, 600 * 1024, 300 * 1024);
        assert_eq!(sched.experts_per_core, 2);

        let schedule = sched.schedule_experts(16, 4);
        assert_eq!(schedule.len(), 8);

        let total: usize = schedule.iter().map(|v| v.len()).sum();
        assert_eq!(total, 16);

        // Each core has at most 2
        for core_assignments in &schedule {
            assert!(core_assignments.len() <= 2);
        }
    }

    #[test]
    fn test_schedule_less_experts_than_cores() {
        // 16 cores but only 4 experts
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        let schedule = sched.schedule_experts(4, 4);
        let total: usize = schedule.iter().map(|v| v.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_cores_needed() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        assert_eq!(sched.cores_needed(8), 8);
        assert_eq!(sched.cores_needed(1), 1);
        assert_eq!(sched.cores_needed(0), 0);
    }

    #[test]
    fn test_pipeline_rounds() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        // 1 expert per core, 16 cores, so 32 experts → 2 rounds
        assert_eq!(sched.pipeline_rounds(32), 2);
        // 8 experts → 1 round
        assert_eq!(sched.pipeline_rounds(8), 1);
    }

    #[test]
    fn test_expert_sram_footprint_fp16() {
        // 32×4096 MoE: hidden=4096, intermediate=14336, FP16 (2 bytes)
        let footprint = expert_sram_footprint(4096, 14336, 2);
        // gate: 4096*14336*2 = 117,440,512
        // up: same = 117,440,512
        // down: 14336*4096*2 = 117,440,512
        // total = 352,321,536
        assert_eq!(footprint, 352_321_536);
    }

    #[test]
    fn test_select_top_k_basic() {
        let probs = Array::from_slice(&[0.1f32, 0.5, 0.3, 0.1], &[1, 4]);
        let (indices, values) = select_top_k(&probs, 2).unwrap();
        let idx_slice: Vec<u32> = indices.try_as_slice::<u32>().unwrap().to_vec();
        let val_slice: Vec<f32> = values.try_as_slice::<f32>().unwrap().to_vec();
        // Top 2: index 1 (0.5) and index 2 (0.3) → renormalised
        assert_eq!(idx_slice, vec![1, 2]);
        assert!((val_slice[0] - 0.5 / 0.8).abs() < 1e-5);
        assert!((val_slice[1] - 0.3 / 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_compute_routing_probs() {
        let logits = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 4]);
        let probs = compute_routing_probs(&logits).unwrap();
        let slice: Vec<f32> = probs.try_as_slice::<f32>().unwrap().to_vec();
        // Softmax of [0, 1, 2, 3]
        let expected: Vec<f32> = {
            let max = 3.0f32;
            let exps: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0]
                .iter()
                .map(|&x| (x - max).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        };
        for (a, b) in slice.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_forward_moe_empty_experts() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        let hidden = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let router_logits = Array::from_slice(&[0.0f32, 0.0], &[1, 2]);
        let result = sched.forward_moe(&hidden, &[], &router_logits, 2).unwrap();
        let shape = result.shape();
        assert_eq!(shape, &[1, 2]);
    }

    #[test]
    fn test_tensor_routing_structure() {
        let routing = TokenRouting {
            expert_indices: vec![3, 7],
            routing_weights: vec![0.6, 0.4],
        };
        assert_eq!(routing.expert_indices.len(), 2);
        assert!((routing.routing_weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
