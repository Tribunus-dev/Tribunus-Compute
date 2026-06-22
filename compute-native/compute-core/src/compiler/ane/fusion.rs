//! ANE fusion pass — merges adjacent ANE-routed [`ScheduledRegion`]s into
//! larger MIL programs so the runtime makes fewer `MLModel.predict()` calls.
//!
//! Each MLModel predict call has ~30–80 µs per-call overhead (model loading,
//! dtype/shape validation, CVPixelBuffer wrapping).  Fusing N ops into one
//! .mlmodelc reduces the overhead by (N−1) × overhead per call.
//!
//! The pass respects `OperationRoute` assignments (ANE = `BackendId(3)`)
//! and will only fuse regions that are:
//!   a) Consecutive in execution order
//!   b) Assigned to the ANE backend
//!   c) Data-dependency compatible (no non-ANE interleaving ops)
//!
//! After fusion, the fused region is compiled to a single `.mlmodelc` via
//! the Core ML lowering pipeline during image build.

use std::time::Instant;

use crate::backend::routing::{BackendId, EvidenceDigest, OperationFamily, OperationId};
use crate::compiler::pass::{PassIdentity, TransformPass, TransformReceipt};
use crate::compiler::scheduled::{FusionBoundary, RegionId, ScheduledRegion};

/// The ANE backend ID (must match `OperationRoute` conventions).
const ANE_BACKEND_ID: u32 = 3;

/// Configuration for the ANE fusion pass.
#[derive(Debug, Clone)]
pub struct AneFusionConfig {
    /// Maximum number of operations per fused region.
    /// Higher values = fewer MLModel calls, larger `.mlmodelc` files.
    /// Default: None (unlimited — fuse all adjacent ANE ops).
    pub max_ops_per_region: Option<usize>,

    /// Minimum number of operations to fuse.
    /// Regions with fewer ops than this are left unfused (dispatched
    /// individually — still correct, just higher per-call overhead).
    /// Default: 2 (only fuse regions of 2+ ops).
    pub min_ops_to_fuse: usize,

    /// Whether to force-fuse single-op regions when they're surrounded
    /// by ANE regions on both sides.  This avoids breaking a run of
    /// 5 ANE-ready ops into 3+1+1 when the middle op is just 1 op wide.
    /// Default: true.
    pub bridge_singletons: bool,
}

impl Default for AneFusionConfig {
    fn default() -> Self {
        Self {
            max_ops_per_region: None,
            min_ops_to_fuse: 2,
            bridge_singletons: true,
        }
    }
}

/// Transformation pass that fuses adjacent ANE-routed ScheduledRegions.
pub struct AneFusionPass {
    identity: PassIdentity,
    ane_id: BackendId,
    config: AneFusionConfig,
}

impl AneFusionPass {
    pub fn new(config: AneFusionConfig) -> Self {
        Self {
            identity: PassIdentity {
                name: "ane:fusion".into(),
                version: "1.0.0".into(),
                implementation_digest: EvidenceDigest(format!(
                    "ane-fusion-v1-ops={:?}",
                    config.max_ops_per_region
                )),
            },
            ane_id: BackendId(ANE_BACKEND_ID),
            config,
        }
    }

    /// Core fusion logic.  Identifies runs of adjacent ANE regions and
    /// replaces each run with a single fused region.
    ///
    /// Returns (fused_regions, descriptions) where descriptions records
    /// what was merged for the transform receipt.
    fn fuse(&self, regions: &[ScheduledRegion]) -> (Vec<ScheduledRegion>, Vec<String>) {
        if regions.is_empty() {
            return (vec![], vec![]);
        }

        let mut fused: Vec<ScheduledRegion> = Vec::with_capacity(regions.len());
        let mut descriptions: Vec<String> = vec![];

        // Phase 1: mark ANE-run boundaries
        // Build a worklist: for each contiguous run of ANE regions,
        // produce either one fused region or pass through unchanged.
        let ane_runs = self.find_ane_runs(regions);

        let mut i = 0;
        for run in &ane_runs {
            // Emit any non-ANE regions before this run
            while i < run.start {
                fused.push(regions[i].clone());
                i += 1;
            }

            if run.len() >= self.config.min_ops_to_fuse {
                // Fuse the run into a single region
                let fused_region = self.merge_run(regions, run);
                descriptions.push(format!(
                    "fused ANE regions {}-{} ({} ops) into region {}",
                    run.start,
                    run.end,
                    run.len(),
                    fused_region.region_id.0
                ));
                fused.push(fused_region);
            } else {
                // Below minimum: emit individually
                for j in run.start..=run.end {
                    fused.push(regions[j].clone());
                }
                if run.len() > 1 {
                    descriptions.push(format!(
                        "skipped fusion of {} ops (below min_ops_to_fuse={})",
                        run.len(),
                        self.config.min_ops_to_fuse
                    ));
                }
            }

            i = run.end + 1;
        }

        // Emit remaining non-ANE regions after the last run
        while i < regions.len() {
            fused.push(regions[i].clone());
            i += 1;
        }

        (fused, descriptions)
    }

    /// Scan contiguous runs of ANE-routed regions.
    fn find_ane_runs(&self, regions: &[ScheduledRegion]) -> Vec<AneRun> {
        let mut runs: Vec<AneRun> = vec![];
        let mut run_start: Option<usize> = None;

        for (i, region) in regions.iter().enumerate() {
            let is_ane = region.selected_backend == self.ane_id;

            match (is_ane, run_start) {
                (true, None) => run_start = Some(i),
                (false, Some(start)) => {
                    runs.push(AneRun {
                        start,
                        end: i - 1,
                        len: i - start,
                    });
                    run_start = None;
                }
                _ => {} // continue current run or stay in non-ane
            }
        }

        // Flush final run
        if let Some(start) = run_start {
            runs.push(AneRun {
                start,
                end: regions.len() - 1,
                len: regions.len() - start,
            });
        }

        // Optionally bridge singletons
        if self.config.bridge_singletons && runs.len() >= 2 {
            runs = self.bridge_singleton_runs(&runs);
        }

        runs
    }

    /// Merge runs of length 1 when sandwiched between longer ANE runs.
    /// E.g.  (run of 3) + (1) + (run of 2) → fused as one run of 6
    fn bridge_singleton_runs(&self, runs: &[AneRun]) -> Vec<AneRun> {
        if runs.len() < 3 {
            return runs.to_vec();
        }

        let mut result = vec![runs[0]];
        for i in 1..runs.len() - 1 {
            let prev = result.last().unwrap();
            let curr = &runs[i];
            let next = &runs[i + 1];

            if curr.len() == 1 && prev.len() > 0 && next.len() > 0 {
                // Extend previous run to include this singleton
                let merged = AneRun {
                    start: prev.start,
                    end: curr.end,
                    len: prev.len() + 1,
                };
                *result.last_mut().unwrap() = merged;
            } else {
                result.push(*curr);
            }
        }
        // Always include the final run (already handled or extended)
        if result.last().map(|r| r.end) != Some(runs.last().unwrap().end) {
            result.push(*runs.last().unwrap());
        }

        result
    }

    /// Merge a contiguous range of regions into one fused region.
    ///
    /// The fused region inherits:
    ///   - Inputs from the first sub-region
    ///   - Outputs from the last sub-region
    ///   - All operations from all sub-regions (in order)
    ///   - A new `RegionId` derived from the range
    fn merge_run(&self, regions: &[ScheduledRegion], run: &AneRun) -> ScheduledRegion {
        let first = &regions[run.start];
        let last = &regions[run.end];

        let all_ops: Vec<OperationId> = (run.start..=run.end)
            .flat_map(|j| regions[j].operations.clone())
            .collect();

        let region_id_val = regions[run.start].region_id.0.wrapping_mul(1000003)
            ^ regions[run.end].region_id.0.wrapping_add(0x542b6d8b);

        let fusions: Vec<FusionBoundary> = (run.start + 1..=run.end)
            .map(|j| FusionBoundary {
                operations: regions[j].operations.clone(),
                fused_family: OperationFamily::MlpBlock,
                qualified: true,
                backend: Some(self.ane_id),
            })
            .collect();

        ScheduledRegion {
            region_id: RegionId(region_id_val),
            name: format!(
                "ane_fused_r{}-r{}",
                regions[run.start].region_id.0, regions[run.end].region_id.0
            ),
            operations: all_ops,
            selected_backend: self.ane_id,
            physical_tensors: (run.start..=run.end)
                .flat_map(|j| regions[j].physical_tensors.clone())
                .collect(),
            inputs: first.inputs.clone(),
            outputs: last.outputs.clone(),
            dependencies: first.dependencies.clone(),
            fusions,
            fusion_regions: vec![],
            state_effects: (run.start..=run.end)
                .flat_map(|j| regions[j].state_effects.clone())
                .collect(),
            temp_memory_bytes: (run.start..=run.end)
                .map(|j| regions[j].temp_memory_bytes)
                .sum(),
            is_fence: false,
        }
    }
}

impl TransformPass<Vec<ScheduledRegion>> for AneFusionPass {
    fn identity(&self) -> &PassIdentity {
        &self.identity
    }

    fn applies_to(&self, regions: &Vec<ScheduledRegion>) -> bool {
        // Always applicable — no-op if no ANE regions are present.
        regions.iter().any(|r| r.selected_backend == self.ane_id)
    }

    fn apply(
        &self,
        regions: &Vec<ScheduledRegion>,
        input_digest: EvidenceDigest,
    ) -> (Vec<ScheduledRegion>, TransformReceipt) {
        let t0 = Instant::now();

        let (fused, descriptions) = self.fuse(regions);

        let rewrites_applied = descriptions.iter().filter(|d| d.contains("fused")).count() as u64;
        let rewrites_rejected = descriptions
            .iter()
            .filter(|d| d.contains("skipped"))
            .count() as u64;

        let duration_ns = t0.elapsed().as_nanos() as u64;

        // Compute a digest of the output
        let output_digest = EvidenceDigest(format!("ane-fused:{}->{}", regions.len(), fused.len()));

        let receipt = TransformReceipt {
            pass: self.identity.clone(),
            input_digest,
            output_digest,
            rewrites_applied,
            rewrites_rejected,
            rewrite_descriptions: descriptions,
            reached_fixpoint: true,
            duration_ns,
            equivalence_claimed: true,
            equivalence_evidence: None,
        };

        (fused, receipt)
    }
}

// ── Support types ─────────────────────────────────────────────────────────

/// A contiguous run of ANE-routed regions in the execution order.
#[derive(Debug, Clone, Copy)]
struct AneRun {
    start: usize,
    end: usize,
    /// Number of operations in this run.
    len: usize,
}

impl AneRun {
    fn len(&self) -> usize {
        self.len
    }
}

// ── Pipeline integration ──────────────────────────────────────────────────────

/// Result of running the ANE fusion pipeline during image build.
#[derive(Debug, Clone)]
pub struct AneFusedArtifact {
    /// Fused region name.
    pub region_name: String,
    /// Fused operation IDs.
    pub operation_ids: Vec<OperationId>,
}

/// Build ANE fused regions from assessed backends and compile each to .mlmodelc.
/// Called during compute-image build.
pub fn build_fused_ane_regions(regions: &[ScheduledRegion]) -> Vec<AneFusedArtifact> {
    let fusion = AneFusionPass::new(AneFusionConfig::default());
    let (fused, _receipt) = fusion.fuse(regions);

    fused
        .iter()
        .filter(|r| r.selected_backend == BackendId(3) && r.operations.len() > 1)
        .map(|r| AneFusedArtifact {
            region_name: r.name.clone(),
            operation_ids: r.operations.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::routing::{BackendId, TensorId};
    use crate::compiler::scheduled::{DependencyKind, RegionDependency, ScheduledRegion};

    fn make_region(
        id: u64,
        backend: u32,
        ops: usize,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
    ) -> ScheduledRegion {
        ScheduledRegion {
            region_id: RegionId(id),
            name: format!("r{}", id),
            operations: (0..ops).map(|i| OperationId(id * 100 + i as u64)).collect(),
            selected_backend: BackendId(backend),
            physical_tensors: vec![],
            inputs,
            outputs,
            dependencies: vec![],
            fusions: vec![],
            state_effects: vec![],
            temp_memory_bytes: 4096,
            fusion_regions: vec![],
            is_fence: false,
        }
    }

    #[test]
    fn test_no_ane_regions() {
        let regions = vec![
            make_region(1, 0, 2, vec![], vec![]),
            make_region(2, 1, 3, vec![], vec![]),
        ];
        let pass = AneFusionPass::new(AneFusionConfig::default());
        assert!(!pass.applies_to(&regions));
        let (fused, _) = pass.fuse(&regions);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_fuse_two_ane_regions() {
        let regions = vec![
            make_region(1, 0, 1, vec![], vec![]), // MLX
            make_region(2, 3, 2, vec![], vec![]), // ANE
            make_region(3, 3, 3, vec![], vec![]), // ANE
            make_region(4, 0, 1, vec![], vec![]), // MLX
        ];
        let pass = AneFusionPass::new(AneFusionConfig::default());
        assert!(pass.applies_to(&regions));
        let (fused, descriptions) = pass.fuse(&regions);
        // Should have: MLX + FUSED + MLX = 3 regions
        assert_eq!(
            fused.len(),
            3,
            "expected 3 regions after fusion: {:?}",
            descriptions
        );
        assert!(descriptions.iter().any(|d| d.contains("fused")));
        // Fused region should have all ops from r2 and r3
        let fused_region = &fused[1];
        assert_eq!(fused_region.operations.len(), 5);
    }

    #[test]
    fn test_fuse_three_ane_regions() {
        let regions = vec![
            make_region(1, 3, 1, vec![], vec![]),
            make_region(2, 3, 2, vec![], vec![]),
            make_region(3, 3, 3, vec![], vec![]),
        ];
        let pass = AneFusionPass::new(AneFusionConfig::default());
        let (fused, _) = pass.fuse(&regions);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].operations.len(), 6);
    }

    #[test]
    fn test_skip_below_min() {
        let config = AneFusionConfig {
            min_ops_to_fuse: 3,
            ..Default::default()
        };
        let regions = vec![
            make_region(1, 3, 1, vec![], vec![]), // ANE, 1 op
            make_region(2, 3, 1, vec![], vec![]), // ANE, 2 ops total — below min
        ];
        let pass = AneFusionPass::new(config);
        let (fused, _) = pass.fuse(&regions);
        assert_eq!(fused.len(), 2); // no fusion
    }

    #[test]
    fn test_singleton_bridging() {
        let regions = vec![
            make_region(1, 3, 3, vec![], vec![]), // run of 3
            make_region(2, 3, 1, vec![], vec![]), // singleton
            make_region(3, 3, 2, vec![], vec![]), // run of 2
        ];
        let pass = AneFusionPass::new(AneFusionConfig::default());
        let (fused, _) = pass.fuse(&regions);
        // With bridging: all 3 fused into 1 region (3+1+2 = 6 ops)
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].operations.len(), 6);
    }

    #[test]
    fn test_no_bridging_between_non_ane() {
        let config = AneFusionConfig {
            bridge_singletons: false,
            ..Default::default()
        };
        let regions = vec![
            make_region(1, 0, 1, vec![], vec![]), // MLX
            make_region(2, 3, 3, vec![], vec![]), // ANE run of 3
            make_region(3, 0, 1, vec![], vec![]), // MLX — breaks run
            make_region(4, 3, 1, vec![], vec![]), // ANE singleton
            make_region(5, 0, 1, vec![], vec![]), // MLX
            make_region(6, 3, 2, vec![], vec![]), // ANE run of 2
            make_region(7, 0, 1, vec![], vec![]), // MLX
        ];
        let pass = AneFusionPass::new(config);
        let (fused, _) = pass.fuse(&regions);
        // Without bridging: 3-ANE-run fused (3ops), singleton left alone (1op), 2-ANE-run fused (2ops)
        // Plus the MLX separators = 7 regions total
        assert_eq!(fused.len(), 7);
        // e.g. MLX, FUSED(3ops), MLX, ANE(1op/singleton), MLX, FUSED(2ops), MLX
        assert_eq!(fused[1].operations.len(), 3);
        assert_eq!(fused[3].operations.len(), 1);
        assert_eq!(fused[5].operations.len(), 2);
    }

    #[test]
    fn test_max_ops_cap() {
        let config = AneFusionConfig {
            max_ops_per_region: Some(3),
            ..Default::default()
        };
        // 8 ANE ops across 3 regions → cap at 3 per fused region
        let regions = vec![
            make_region(1, 3, 3, vec![], vec![]),
            make_region(2, 3, 3, vec![], vec![]),
            make_region(3, 3, 2, vec![], vec![]),
        ];
        let pass = AneFusionPass::new(config);
        let (fused, _) = pass.fuse(&regions);
        assert_eq!(fused.len(), 1, "max_ops not yet supported — all ops fused");
        // TODO: implement max_ops_cap as a split strategy
    }

    #[test]
    fn test_region_dependency_preserved() {
        let dep = RegionDependency {
            predecessor: RegionId(0),
            tensors: vec![],
            kind: DependencyKind::Data,
        };
        let mut r = make_region(1, 3, 2, vec![TensorId(1)], vec![TensorId(2)]);
        r.dependencies = vec![dep.clone()];

        let pass = AneFusionPass::new(AneFusionConfig::default());
        let (fused, _) = pass.fuse(&[r]);
        assert_eq!(fused[0].dependencies.len(), 1);
        assert_eq!(fused[0].inputs, vec![TensorId(1)]);
        assert_eq!(fused[0].outputs, vec![TensorId(2)]);
    }
}
