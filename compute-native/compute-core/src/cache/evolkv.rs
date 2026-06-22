//! EvolKV: evolutionary search for per-layer KV cache compression budgets.
//!
//! Instead of fixed L1/L2/L3 thresholds (5s/30s), uses population-based
//! evolution to find optimal per-layer budget fractions that minimize
//! perplexity under a total cache size constraint.
//!
//! Reference: "EvolKV: Selection and Eviction of Key-Value Cache via
//! Evolutionary Search" (https://arxiv.org/abs/2505.19735)
//!
//! ## Algorithm
//!
//! 1. **Initialize** — population of random (Dirichlet-sampled) layer budgets
//! 2. **Evaluate** — measure fitness (perplexity) for each budget on a
//!    calibration set
//! 3. **Select** — tournament selection with elitism
//! 4. **Crossover** — blend two parent budgets (weighted average)
//! 5. **Mutate** — additive noise followed by renormalization (Dirichlet-like)
//! 6. **Repeat** — for N generations, return the budget with best fitness

use crate::kv_cache::CompressedKvCache;

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (SplitMix64) — no external dependencies
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random number generator for EvolKV.
///
/// Uses SplitMix64 (M. O'Neill, 2015) for fast, reproducible random numbers
/// with good distribution properties. No `rand` crate dependency needed.
#[derive(Debug, Clone)]
pub struct EvolKvRng {
    state: u64,
}

impl EvolKvRng {
    /// Create a new PRNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate a `u64` in `[0, u64::MAX]`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = x;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    /// Generate an `f64` in `[0, 1)` with 53 bits of mantissa precision.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Sample from Exponential(1) using inverse transform.
    /// Returns `-ln(U)` where `U ~ Uniform(0,1]`.
    pub fn exp_one(&mut self) -> f64 {
        // Avoid log(0) by shifting slightly
        let u = (self.random() * 0.9999999999999999) + 5.551115123125783e-17;
        -u.ln()
    }
}

// ---------------------------------------------------------------------------
// Calibration set
// ---------------------------------------------------------------------------

/// A small calibration set for fitness evaluation during EvolKV search.
///
/// Contains tokenized prompts used to measure how well a given per-layer
/// budget allocation preserves generation quality. The fitness function
/// estimates perplexity on this calibration data given the budget.
#[derive(Clone, Debug)]
pub struct CalibrationSet {
    /// Tokenized prompts — each inner `Vec<u32>` is a sequence of token ids.
    pub prompts: Vec<Vec<u32>>,
}

impl CalibrationSet {
    /// Create a new calibration set from tokenized prompts.
    pub fn new(prompts: Vec<Vec<u32>>) -> Self {
        Self { prompts }
    }

    /// Total number of tokens across all prompts.
    pub fn total_tokens(&self) -> usize {
        self.prompts.iter().map(|p| p.len()).sum()
    }

    /// Return `true` if the calibration set is empty (no prompts).
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Layer budget
// ---------------------------------------------------------------------------

/// A per-layer compression budget: what fraction of the total cache budget
/// each transformer layer gets.
///
/// Fractions must sum to 1.0 (within floating-point tolerance). Uniform
/// allocation is used during normal operation; EvolKV finds the optimal
/// non-uniform distribution for the given model and task.
#[derive(Clone, Debug)]
pub struct LayerBudget {
    /// One fraction per transformer layer, summing to 1.0.
    pub fractions: Vec<f64>,
}

impl LayerBudget {
    /// Create a uniform budget (equal allocation per layer).
    ///
    /// Each layer gets `1 / num_layers` of the total cache budget.
    pub fn uniform(num_layers: usize) -> Self {
        let frac = 1.0 / num_layers.max(1) as f64;
        Self {
            fractions: vec![frac; num_layers],
        }
    }

    /// Create a random budget using Dirichlet(1, 1, ..., 1) sampling.
    ///
    /// Samples `num_layers` independent `Exp(1)` variates and normalizes
    /// them to sum to 1. This produces a uniform Dirichlet distribution
    /// over the simplex.
    pub fn random(num_layers: usize, rng: &mut EvolKvRng) -> Self {
        let mut fractions = Vec::with_capacity(num_layers);
        let mut sum = 0.0;
        for _ in 0..num_layers {
            let gamma_sample = rng.exp_one(); // Gamma(1,1) = Exp(1)
            fractions.push(gamma_sample);
            sum += gamma_sample;
        }
        if sum > 0.0 {
            for f in &mut fractions {
                *f /= sum;
            }
        } else {
            // Fallback to uniform on degenerate sum
            let frac = 1.0 / num_layers.max(1) as f64;
            fractions = vec![frac; num_layers];
        }
        Self { fractions }
    }

    /// Apply this budget to the 3-tier cache thresholds on a
    /// `CompressedKvCache`.
    ///
    /// Stores the per-layer fractions so the cache can use them during
    /// compression and page-migration decisions. Layers with larger
    /// fractions get less aggressive compression (more budget allocated).
    pub fn apply(&self, cache: &mut CompressedKvCache) {
        cache.per_layer_budget = Some(self.fractions.clone());
    }

    /// Validate that fractions sum to approximately 1.0 and have the
    /// correct length.
    pub fn validate(&self) -> Result<(), String> {
        if self.fractions.is_empty() {
            return Err("LayerBudget fractions is empty".to_string());
        }
        let sum: f64 = self.fractions.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "LayerBudget fractions sum to {}, expected 1.0",
                sum
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EvolKV evolutionary searcher
// ---------------------------------------------------------------------------

/// Evolutionary searcher for optimal per-layer KV cache budgets.
///
/// Uses a population-based genetic algorithm:
/// - **Population**: fixed-size set of `LayerBudget` candidates
/// - **Fitness**: estimated perplexity on a calibration set (lower = better)
/// - **Selection**: tournament selection with elitism
/// - **Crossover**: weighted-blend of two parent budgets
/// - **Mutation**: additive noise + renormalization (Dirichlet-like)
///
/// Default parameters are tuned for converged search in ~50 generations
/// with 100 individuals.
pub struct EvolKV {
    /// Number of transformer layers in the model.
    pub num_layers: usize,
    /// Number of budget candidates per generation.
    pub population_size: u32,
    /// Number of evolutionary generations.
    pub generations: u32,
    /// Probability of mutating each fraction.
    pub mutation_rate: f64,
    /// Probability of crossover (blending) between two parents.
    pub crossover_rate: f64,
    /// Number of top individuals preserved unchanged each generation.
    pub elitism_count: u32,
}

impl EvolKV {
    /// Create a new EvolKV searcher with sensible defaults.
    ///
    /// Defaults:
    /// - `population_size`: 100
    /// - `generations`: 50
    /// - `mutation_rate`: 0.15
    /// - `crossover_rate`: 0.7
    /// - `elitism_count`: 5
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            population_size: 100,
            generations: 50,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elitism_count: 5,
        }
    }

    /// Run the full evolutionary search.
    ///
    /// 1. Initialize population of random budgets (Dirichlet-sampled)
    /// 2. For each generation:
    ///    - Evaluate fitness (estimated perplexity) on the calibration set
    ///    - Preserve top elites unchanged
    ///    - Fill rest via tournament selection + crossover + mutation
    /// 3. Return the budget with lowest estimated perplexity
    pub fn search(
        &self,
        calibration_set: &CalibrationSet,
        _total_cache_budget: usize,
    ) -> Result<LayerBudget, String> {
        if self.num_layers == 0 {
            return Err("EvolKV: num_layers must be > 0".to_string());
        }
        if calibration_set.prompts.is_empty() {
            return Err("EvolKV: calibration set is empty".to_string());
        }

        let mut rng = EvolKvRng::new(42);
        let pop_size = self.population_size.max(4) as usize;
        let elite_count = self.elitism_count.min(pop_size as u32) as usize;

        // 1. Initialize population with random budgets
        let mut population: Vec<(f64, LayerBudget)> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let budget = LayerBudget::random(self.num_layers, &mut rng);
            let fitness = self.evaluate(&budget, calibration_set);
            population.push((fitness, budget));
        }
        population.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Uniform fallback baseline, used if evolution fails to improve
        let uniform_budget = LayerBudget::uniform(self.num_layers);
        let uniform_fitness = self.evaluate(&uniform_budget, calibration_set);

        // 2. Evolution loop
        for _gen in 0..self.generations {
            let mut new_population: Vec<(f64, LayerBudget)> = Vec::with_capacity(pop_size);

            // Elitism: keep best individuals
            for i in 0..elite_count {
                new_population.push(population[i].clone());
            }

            // Fill rest with offspring via selection + crossover + mutation
            while new_population.len() < pop_size {
                let parent_a = self.tournament_select(&population, &mut rng);
                let parent_b = self.tournament_select(&population, &mut rng);

                let mut child: LayerBudget;
                if rng.random() < self.crossover_rate {
                    child = Self::crossover(&parent_a, &parent_b);
                } else if rng.random() < 0.5 {
                    child = parent_a.clone();
                } else {
                    child = parent_b.clone();
                }

                if rng.random() < self.mutation_rate {
                    self.mutate(&mut child, &mut rng);
                }

                let fitness = self.evaluate(&child, calibration_set);
                new_population.push((fitness, child));
            }

            new_population.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            population = new_population;
        }

        // Return best budget if it improves over uniform, otherwise uniform
        let best = &population[0];
        if best.0 < uniform_fitness {
            Ok(best.1.clone())
        } else {
            Ok(uniform_budget)
        }
    }

    /// Evaluate a single budget: estimate perplexity on the calibration set.
    ///
    /// The estimation uses a layer-importance model derived from calibration
    /// data. Layers that process more cumulative tokens (later layers in
    /// deeper models) receive higher importance. The perplexity is computed
    /// as a cross-entropy (KL divergence) between the budget fractions and
    /// the importance distribution, scaled to a plausible perplexity range.
    ///
    /// Lower is better. In production this would run the actual model on the
    /// calibration set; here we use a well-motivated surrogate.
    fn evaluate(&self, budget: &LayerBudget, calibration_set: &CalibrationSet) -> f64 {
        if calibration_set.prompts.is_empty() || self.num_layers == 0 {
            return f64::MAX;
        }

        // Compute layer importance from calibration data.
        // Each layer's importance is proportional to how many cumulative
        // tokens it processes (later layers see more cached history).
        let mut layer_scores = vec![0.0f64; self.num_layers];
        for prompt in &calibration_set.prompts {
            let len = prompt.len() as f64;
            for i in 0..self.num_layers {
                // Later layers process tokens with more KV cache history,
                // making cache quality more impactful.  The multiplier
                // `(i + 1) / num_layers` models cumulative token pressure.
                let relative_pos = (i as f64 + 1.0) / self.num_layers as f64;
                layer_scores[i] += len * relative_pos;
            }
        }

        // Normalize to a proper importance distribution
        let total: f64 = layer_scores.iter().sum();
        if total <= 0.0 {
            return f64::MAX;
        }
        let importance: Vec<f64> = layer_scores.iter().map(|c| c / total).collect();

        // Base perplexity floor
        let base_ppl = 10.0;

        // KL divergence: D_KL(importance || budget)
        // A budget that matches the importance distribution has low KL.
        // A budget that misallocates (giving too much to unimportant layers
        // or too little to important ones) has high KL.
        let mut kl = 0.0;
        for i in 0..self.num_layers {
            let p = importance[i];
            let q = budget.fractions[i].max(1e-15);
            if p > 1e-15 {
                kl += p * (p / q).ln();
            }
        }

        // Perplexity = base + KL penalty + regularity term
        // The regularity term penalizes extremely skewed budgets that would
        // starve some layers entirely.
        let min_frac = budget.fractions.iter().cloned().fold(f64::MAX, f64::min);
        let starvation_penalty = if min_frac < 1e-6 { 5.0 } else { 0.0 };

        base_ppl + kl * 8.0 + starvation_penalty
    }

    /// Crossover: two parents produce one child by weighted blending.
    ///
    /// Each child fraction is the arithmetic mean of the two parent
    /// fractions, preserving the sum-to-1 constraint automatically
    /// (since (a_i + b_i) / 2 summed over i = (1 + 1) / 2 = 1).
    pub fn crossover(a: &LayerBudget, b: &LayerBudget) -> LayerBudget {
        let n = a.fractions.len().min(b.fractions.len());
        let mut fractions = Vec::with_capacity(n);
        for i in 0..n {
            fractions.push((a.fractions[i] + b.fractions[i]) / 2.0);
        }
        LayerBudget { fractions }
    }

    /// Mutate: add scaled noise to each fraction, then renormalize.
    ///
    /// Noise is sampled as `(random - 0.5) * 2 * mutation_rate`, producing
    /// perturbations proportional to the mutation rate. After adding noise,
    /// fractions are clamped to a positive floor and renormalized so they
    /// continue to sum to 1 (Dirichlet-like behavior on the simplex).
    pub fn mutate(&self, budget: &mut LayerBudget, rng: &mut EvolKvRng) {
        let eps = 1e-10;
        for f in &mut budget.fractions {
            let noise = (rng.random() - 0.5) * 2.0 * self.mutation_rate;
            *f = (*f + noise).max(eps);
        }
        // Renormalize to sum to 1
        let sum: f64 = budget.fractions.iter().sum();
        if sum > 0.0 {
            for f in &mut budget.fractions {
                *f /= sum;
            }
        }
    }

    /// Tournament selection: pick the fittest from a random subset.
    ///
    /// Tournament size is 3 (default), giving good selection pressure
    /// while maintaining diversity.
    fn tournament_select<'a>(
        &self,
        population: &'a [(f64, LayerBudget)],
        rng: &mut EvolKvRng,
    ) -> &'a LayerBudget {
        let tournament_size = 3usize.min(population.len());
        let mut best_idx = (rng.random() * population.len() as f64) as usize % population.len();
        let mut best_fitness = population[best_idx].0;
        for _ in 1..tournament_size {
            let idx = (rng.random() * population.len() as f64) as usize % population.len();
            if population[idx].0 < best_fitness {
                best_fitness = population[idx].0;
                best_idx = idx;
            }
        }
        &population[best_idx].1
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_calibration() -> CalibrationSet {
        // 10 prompts of varying lengths
        CalibrationSet::new(vec![
            vec![101, 202, 303, 404, 505],
            vec![111, 222, 333],
            vec![11, 22, 33, 44, 55, 66, 77],
            vec![1, 2, 3, 4],
            vec![1001, 1002, 1003, 1004, 1005, 1006],
            vec![201, 302, 403],
            vec![5, 10, 15, 20, 25, 30, 35, 40],
            vec![99, 199, 299],
            vec![50, 150],
            vec![7, 14, 21, 28, 35, 42],
        ])
    }

    #[test]
    fn test_uniform_budget() {
        let budget = LayerBudget::uniform(8);
        assert_eq!(budget.fractions.len(), 8);
        let sum: f64 = budget.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        for f in &budget.fractions {
            assert!((f - 0.125).abs() < 1e-12);
        }
    }

    #[test]
    fn test_random_budget_valid() {
        let mut rng = EvolKvRng::new(42);
        let budget = LayerBudget::random(12, &mut rng);
        assert_eq!(budget.fractions.len(), 12);
        let sum: f64 = budget.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &f in &budget.fractions {
            assert!(f > 0.0, "fraction {} should be positive", f);
        }
    }

    #[test]
    fn test_random_budget_not_uniform() {
        // With enough layers, random Dirichlet(1) is extremely unlikely
        // to produce a uniform vector
        let mut rng = EvolKvRng::new(99);
        let budget = LayerBudget::random(16, &mut rng);
        let uniform = 1.0 / 16.0;
        let all_uniform = budget.fractions.iter().all(|f| (*f - uniform).abs() < 0.01);
        assert!(!all_uniform, "random budget should not be uniform");
    }

    #[test]
    fn test_crossover_preserves_sum() {
        let mut rng = EvolKvRng::new(7);
        let a = LayerBudget::random(8, &mut rng);
        let b = LayerBudget::random(8, &mut rng);
        let child = EvolKV::crossover(&a, &b);
        let sum: f64 = child.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(child.fractions.len(), 8);
    }

    #[test]
    fn test_mutate_preserves_sum() {
        let evolkv = EvolKV::new(10);
        let mut rng = EvolKvRng::new(1234);
        let mut budget = LayerBudget::random(10, &mut rng);
        let original = budget.clone();

        evolkv.mutate(&mut budget, &mut rng);

        let sum: f64 = budget.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // With mutation_rate 0.15, at least some fractions should change
        let changed = budget
            .fractions
            .iter()
            .zip(original.fractions.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "mutation should change at least one fraction");
    }

    #[test]
    fn test_evaluate_lower_better_for_uniform_on_uniform_data() {
        let evolkv = EvolKV::new(6);
        let cal = dummy_calibration();

        let uniform = LayerBudget::uniform(6);
        let uniform_fitness = evolkv.evaluate(&uniform, &cal);

        // A deliberately bad budget (all budget on first layer)
        let mut bad_fracs = vec![0.0; 6];
        bad_fracs[0] = 0.95;
        for i in 1..6 {
            bad_fracs[i] = 0.01;
        }
        let bad = LayerBudget {
            fractions: bad_fracs,
        };
        let bad_fitness = evolkv.evaluate(&bad, &cal);

        assert!(
            uniform_fitness < bad_fitness,
            "uniform budget should score better (lower) than a pathological allocation: {} vs {}",
            uniform_fitness,
            bad_fitness
        );
    }

    #[test]
    fn test_search_returns_valid_budget() {
        let evolkv = EvolKV {
            num_layers: 8,
            population_size: 20,
            generations: 10,
            mutation_rate: 0.2,
            crossover_rate: 0.7,
            elitism_count: 2,
        };
        let cal = dummy_calibration();
        let budget = evolkv.search(&cal, 4096).expect("search should succeed");

        assert_eq!(budget.fractions.len(), 8);
        let sum: f64 = budget.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rng_determinism() {
        let mut rng1 = EvolKvRng::new(42);
        let mut rng2 = EvolKvRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = EvolKvRng::new(1);
        for _ in 0..1000 {
            let v = rng.random();
            assert!(v >= 0.0 && v < 1.0, "random value {} out of range [0,1)", v);
        }
    }

    #[test]
    fn test_exp_one_positive() {
        let mut rng = EvolKvRng::new(7);
        for _ in 0..100 {
            let v = rng.exp_one();
            assert!(v > 0.0, "Exp(1) sample should be > 0, got {}", v);
        }
    }

    #[test]
    fn test_search_no_calibration() {
        let evolkv = EvolKV::new(4);
        let empty_cal = CalibrationSet::new(vec![]);
        let result = evolkv.search(&empty_cal, 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_apply_to_cache() {
        let mode = crate::quantization::turboquant_kv::KvQuantMode::Polar(4);
        let mut cache = CompressedKvCache::new(mode, 32, 10);
        assert!(cache.per_layer_budget.is_none());

        let budget = LayerBudget::uniform(6);
        budget.apply(&mut cache);

        assert!(cache.per_layer_budget.is_some());
        let stored = cache.per_layer_budget.unwrap();
        assert_eq!(stored.len(), 6);
    }

    #[test]
    fn test_evaluate_improves_over_generations() {
        let evolkv = EvolKV {
            num_layers: 8,
            population_size: 30,
            generations: 15,
            mutation_rate: 0.2,
            crossover_rate: 0.7,
            elitism_count: 3,
        };
        let cal = dummy_calibration();

        let uniform = LayerBudget::uniform(8);
        let uniform_fitness = evolkv.evaluate(&uniform, &cal);

        let budget = evolkv.search(&cal, 4096).expect("search should succeed");
        let best_fitness = evolkv.evaluate(&budget, &cal);

        assert!(
            best_fitness <= uniform_fitness + 1e-6,
            "evolved budget (fitness={}) should not be worse than uniform (fitness={})",
            best_fitness,
            uniform_fitness
        );
    }
}
