use crate::compute_image::phase_graph::WeightResidencySetId;
use serde::{Deserialize, Serialize};
use crate::profiled_executor::WorkingSetManager;

/// Status of a weight residency set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyStatus {
    NotResident,
    PartiallyResident,
    FullyResident,
    Evicted,
    Staging,
}

/// Source of residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencySource {
    SessionWarmup,
    OnDemandActivation,
    CompilerDeclared,
    Prefetch,
}

/// A device binding identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceBindingId(pub String);

/// Receipt for a weight residency operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightResidencyReceipt {
    pub set_id: WeightResidencySetId,
    pub status: ResidencyStatus,
    pub bytes_already_resident: u64,
    pub bytes_staged: u64,
    pub staging_duration_us: u64,
    pub device_binding_ids: Vec<DeviceBindingId>,
    pub source: ResidencySource,
}

/// A weight residency set declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightResidencySet {
    pub set_id: WeightResidencySetId,
    pub weight_names: Vec<String>,
    pub total_bytes: u64,
    pub required: bool,
}

/// Residency plan — declares which weight sets must be resident and when.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyPlan {
    pub sets: Vec<WeightResidencySet>,
    pub total_resident_bytes: u64,
    pub staging_window_bytes: u64,
}

/// Runner for weight residency phases.
///
/// The engine calls this before dispatching weight-consuming phases.
/// It ensures the required weight set is active on the target device.
pub struct WeightResidencyRunner;

impl WeightResidencyRunner {
    pub fn new() -> Self {
        Self
    }

    /// Activate a weight set through the session's working set manager.
    pub fn activate(
        &self,
        set_id: &WeightResidencySetId,
        working_set: &mut WorkingSetManager,
        layer_index: u32,
    ) -> Result<WeightResidencyReceipt, String> {
        let start = std::time::Instant::now();
        working_set.weight_streamer.activate(layer_index)?;
        let duration_us = start.elapsed().as_micros() as u64;

        Ok(WeightResidencyReceipt {
            set_id: set_id.clone(),
            status: ResidencyStatus::FullyResident,
            bytes_already_resident: 0,
            bytes_staged: 0,
            staging_duration_us: duration_us,
            device_binding_ids: vec![DeviceBindingId(format!("device_layer_{}", layer_index))],
            source: ResidencySource::OnDemandActivation,
        })
    }

    /// Check whether the working set has the required weights active.
    pub fn check_resident(
        &self,
        _set_id: &WeightResidencySetId,
        _working_set: &mut WorkingSetManager,
    ) -> bool {
        // In a full implementation, check the working set's active weight map.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_residency_receipt_creation() {
        let receipt = WeightResidencyReceipt {
            set_id: WeightResidencySetId("layer_0_weights".to_string()),
            status: ResidencyStatus::FullyResident,
            bytes_already_resident: 1024 * 1024 * 100,
            bytes_staged: 0,
            staging_duration_us: 1500,
            device_binding_ids: vec![DeviceBindingId("gpu0".to_string())],
            source: ResidencySource::SessionWarmup,
        };
        assert_eq!(receipt.set_id.0, "layer_0_weights");
        assert_eq!(receipt.bytes_already_resident, 104857600);
    }

    #[test]
    fn test_residency_set_declaration() {
        let set = WeightResidencySet {
            set_id: WeightResidencySetId("model_weights".to_string()),
            weight_names: vec!["embed_tokens.weight".to_string()],
            total_bytes: 1024 * 1024,
            required: true,
        };
        assert!(set.required);
    }
}
