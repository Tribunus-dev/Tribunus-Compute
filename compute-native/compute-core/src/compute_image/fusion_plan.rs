//! Fusion plan selection — matches compiler-emitted region candidates against
//! the model's operation topology and the target hardware qualification status.
//!
//! The compiler calls [`select_fusion_plan`] after building segments.  It
//! consumes [`crate::fusion_region::FusionRegion`] entries, filters for
//! viable (qualified, shape-compatible) regions, and produces a
//! [`FusionSelection`] that the later codegen/compilation pipeline uses.

use crate::fusion_region::{FusionImplBackend, FusionRegion, QualificationStatus};
use serde::{Deserialize, Serialize};

/// The compiler's decision about which fusion regions to realise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSelection {
    /// Regions selected for codegen and compilation.
    pub selected: Vec<SelectedFusionRegion>,
    /// Regions that were considered but rejected, with a reason.
    pub rejected: Vec<RejectedFusionRegion>,
    /// Backend that will host the selected fusions (Metal, Accelerate, CoreML).
    pub primary_backend: FusionImplBackend,
}

/// A single fusion region selected for codegen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedFusionRegion {
    /// Matches `FusionRegion::id`.
    pub region_id: String,
    /// The component ops that will be fused.
    pub ops: Vec<String>,
    /// Which backend implementation was chosen.
    pub backend: FusionImplBackend,
    /// How many intermediate tensors are eliminated.
    pub eliminated_intermediates: u32,
    /// Input tensor size in elements (from the model's actual shape).
    pub input_elements: u64,
    /// Output tensor size in elements.
    pub output_elements: u64,
    /// Model hidden dimension.
    pub hidden_size: u64,
    /// Number of query heads.
    pub num_heads: u64,
    /// Number of key/value heads (GQA).
    pub num_kv_heads: u64,
    /// Dimension per head.
    pub head_dim: u64,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: u64,
}
 
/// A fusion region that was considered but rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedFusionRegion {
    pub region_id: String,
    pub reason: RejectionReason,
}

/// Why a region was rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Model does not contain the required component operations.
    MissingOps(Vec<String>),
    /// No qualified backend implementation exists.
    NoQualifiedBackend,
    /// Shape mismatch between the region contract and the model.
    ShapeMismatch(String),
    /// Explicitly disabled by compiler flag or policy.
    DisabledByPolicy(String),
}

/// Select viable fusion regions for a given set of model operations.
///
/// `model_ops` is the set of operation names present in the model
/// (e.g. `{"q_proj", "k_proj", "v_proj", "gate_proj", ...}`).
/// `regions` comes from [`crate::fusion_region::generate_all_fusion_regions`].
/// `primary_backend` selects which backend's implementation to prefer.
pub fn select_fusion_plan(
    model_ops: &std::collections::HashSet<String>,
    regions: &[FusionRegion],
    primary_backend: FusionImplBackend,
) -> FusionSelection {
    let mut selected = Vec::new();
    let mut rejected = Vec::new();

    for region in regions {
        // Check that all component ops exist in the model.
        let missing: Vec<String> = region
            .component_ops
            .iter()
            .filter(|op| !model_ops.contains(*op))
            .cloned()
            .collect();

        if !missing.is_empty() {
            rejected.push(RejectedFusionRegion {
                region_id: region.id.clone(),
                reason: RejectionReason::MissingOps(missing),
            });
            continue;
        }

        // Find a qualified implementation for the primary backend.
        let qualified: Vec<&crate::fusion_region::FusionImpl> = region
            .implementations
            .iter()
            .filter(|impl_| {
                impl_.backend == primary_backend
                    && impl_.qualification_status == QualificationStatus::Qualified
            })
            .collect();

        if qualified.is_empty() {
            rejected.push(RejectedFusionRegion {
                region_id: region.id.clone(),
                reason: RejectionReason::NoQualifiedBackend,
            });
            continue;
        }

        // Shape/alignment check — skip if the region contract looks
        // incompatible (for now, only check that input layout exists).
        if region.input_layout.dims.is_empty() {
            rejected.push(RejectedFusionRegion {
                region_id: region.id.clone(),
                reason: RejectionReason::ShapeMismatch(
                    "input layout has zero dimensions".into(),
                ),
            });
            continue;
        }

        let input_elements: u64 = region.input_layout.dims.iter().map(|d| *d as u64).product();
        let output_elements: u64 = region
            .output_contract
            .dims
            .iter()
            .map(|d| *d as u64)
            .product();

        // Extract model dimensions from the input layout if available,
        // otherwise fall back to LLaMA-7B-like defaults.
        let hidden_size = if region.input_layout.dims.len() >= 3 {
            region.input_layout.dims[2] as u64
        } else {
            4096u64
        };

        selected.push(SelectedFusionRegion {
            region_id: region.id.clone(),
            ops: region.component_ops.clone(),
            backend: primary_backend,
            eliminated_intermediates: region.expected_eliminated_intermediates,
            input_elements,
            output_elements,
            hidden_size,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
        });
    }

    FusionSelection {
        selected,
        rejected,
        primary_backend,
    }
}

/// Return the default primary backend for fusion on the current platform.
#[cfg(target_os = "macos")]
pub fn default_fusion_backend() -> FusionImplBackend {
    FusionImplBackend::MlxGpu
}

#[cfg(not(target_os = "macos"))]
pub fn default_fusion_backend() -> FusionImplBackend {
    FusionImplBackend::RustNeon
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_region::generate_all_fusion_regions;
    use std::collections::HashSet;

    #[test]
    fn test_select_qkv_proj_gate_up_when_ops_present() {
        let mut ops = HashSet::new();
        ops.insert("q_proj".into());
        ops.insert("k_proj".into());
        ops.insert("v_proj".into());
        ops.insert("gate_proj".into());
        ops.insert("up_proj".into());
        ops.insert("down_proj".into());
        ops.insert("silu".into());
        ops.insert("mul".into());
        ops.insert("rms_norm".into());

        let regions = generate_all_fusion_regions();
        let plan = select_fusion_plan(&ops, &regions, FusionImplBackend::MlxGpu);

        assert!(!plan.selected.is_empty(), "expected at least one selected fusion region");
        assert!(
            plan.selected.iter().any(|r| r.region_id == "qkv_proj"),
            "qkv_proj should be selected"
        );
        assert!(
            plan.selected.iter().any(|r| r.region_id == "gate_up_proj"),
            "gate_up_proj should be selected"
        );
        assert!(
            plan.selected.iter().any(|r| r.region_id == "silu_mul"),
            "silu_mul should be selected"
        );
    }

    #[test]
    fn test_select_rejects_when_ops_missing() {
        let mut ops = HashSet::new();
        ops.insert("q_proj".into());
        // missing k_proj, v_proj

        let regions = generate_all_fusion_regions();
        let plan = select_fusion_plan(&ops, &regions, FusionImplBackend::MlxGpu);

        assert!(
            !plan.selected.iter().any(|r| r.region_id == "qkv_proj"),
            "qkv_proj requires Q/K/V"
        );
        assert!(
            plan.rejected.iter().any(|r| r.region_id == "qkv_proj"),
            "qkv_proj should be rejected"
        );
    }

    #[test]
    fn test_serialize_roundtrip() {
        let sel = FusionSelection {
            selected: vec![SelectedFusionRegion {
                region_id: "qkv_proj".into(),
                ops: vec!["q_proj".into(), "k_proj".into(), "v_proj".into()],
                backend: FusionImplBackend::MlxGpu,
                eliminated_intermediates: 3,
                input_elements: 3840,
                output_elements: 11520,
                hidden_size: 4096,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                intermediate_size: 14336,
            }],
            rejected: vec![],
            primary_backend: FusionImplBackend::MlxGpu,
        };
        let json = serde_json::to_string(&sel).expect("serialize");
        let deser: FusionSelection = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.selected.len(), 1);
        assert_eq!(deser.selected[0].region_id, "qkv_proj");
        assert_eq!(deser.selected[0].eliminated_intermediates, 3);
    }
}
