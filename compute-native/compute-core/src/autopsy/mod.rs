//! ModelAutopsy — self-healing inference pipeline.
//!
//! Detects anomalies during inference (NaN, Inf, perplexity spikes,
//! forbidden tokens, exploding activations), traces them to source
//! weight tensors, runs diagnostic replay to confirm the issue, and
//! patches the ComputeImage segments without full recompilation.

pub mod patch;
pub mod replay;
pub mod tracer;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::backend::routing::BackendId;
use crate::profiled_executor::LoadedProfiledModel;
use crate::runtime_orchestration::{InMemoryCoordinationFabric, RuntimeReceipt};

pub use self::patch::SegmentPatch;
pub use self::tracer::AnomalyTracer;

// ---------------------------------------------------------------------------
// 1. Anomaly types
// ---------------------------------------------------------------------------

/// An anomaly detected during inference.
#[derive(Debug, Clone, Serialize)]
pub enum InferenceAnomaly {
    /// Layer output contains NaN
    NanInLayer { layer: u32, backend: BackendId },
    /// Layer output contains Infinity
    InfInLayer { layer: u32, backend: BackendId },
    /// Exploding activation norm (beyond threshold)
    ExplodingActivation {
        layer: u32,
        norm: f64,
        threshold: f64,
    },
    /// Forbidden token generated
    ForbiddenToken { token: u32, context: String },
    /// Perplexity spike (sudden loss increase)
    PerplexitySpike { step: u32, ppl: f64, prev_ppl: f64 },
}

/// Severity of an anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    /// Might be fine, just log it
    Warning,
    /// Likely a bug, trace it
    Error,
    /// Model is producing garbage, must fix
    Critical,
}

impl InferenceAnomaly {
    /// Return the severity level for this anomaly.
    pub fn severity(&self) -> AnomalySeverity {
        match self {
            InferenceAnomaly::NanInLayer { .. } | InferenceAnomaly::InfInLayer { .. } => {
                AnomalySeverity::Critical
            }
            InferenceAnomaly::ExplodingActivation {
                norm, threshold, ..
            } => {
                if *norm > *threshold * 10.0 {
                    AnomalySeverity::Critical
                } else if *norm > *threshold * 2.0 {
                    AnomalySeverity::Error
                } else {
                    AnomalySeverity::Warning
                }
            }
            InferenceAnomaly::ForbiddenToken { .. } => AnomalySeverity::Error,
            InferenceAnomaly::PerplexitySpike { ppl, prev_ppl, .. } => {
                if *ppl > *prev_ppl * 5.0 || *ppl > 1000.0 {
                    AnomalySeverity::Critical
                } else if *ppl > *prev_ppl * 2.0 {
                    AnomalySeverity::Error
                } else {
                    AnomalySeverity::Warning
                }
            }
        }
    }

    /// Return a human-readable description.
    pub fn description(&self) -> String {
        match self {
            InferenceAnomaly::NanInLayer { layer, backend } => {
                format!("NaN detected in layer {} (backend {})", layer, backend.0)
            }
            InferenceAnomaly::InfInLayer { layer, backend } => {
                format!("Inf detected in layer {} (backend {})", layer, backend.0)
            }
            InferenceAnomaly::ExplodingActivation {
                layer,
                norm,
                threshold,
            } => {
                format!(
                    "Exploding activation in layer {}: norm {:.4} exceeds threshold {:.4}",
                    layer, norm, threshold
                )
            }
            InferenceAnomaly::ForbiddenToken { token, context } => {
                format!("Forbidden token {} generated: {}", token, context)
            }
            InferenceAnomaly::PerplexitySpike {
                step,
                ppl,
                prev_ppl,
            } => {
                format!(
                    "Perplexity spike at step {}: {:.2} (prev {:.2})",
                    step, ppl, prev_ppl
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Anomaly tracing
// ---------------------------------------------------------------------------

/// Trace an anomaly back to the source weight tensor.
#[derive(Debug, Clone)]
pub struct AnomalyTrace {
    pub anomaly: InferenceAnomaly,
    /// Which layer's weights caused the issue
    pub layer: u32,
    /// The specific weight tensor (e.g. "layers.23.self_attn.q_proj.weight")
    pub weight_tensor: Option<String>,
    /// Which segment file contains this tensor
    pub segment_file: Option<String>,
    /// The source tensor's hash in the manifest
    pub source_hash: Option<String>,
    /// The backend that processed this operation
    pub backend: BackendId,
    /// The work receipt from the coordination fabric
    pub receipt: Option<RuntimeReceipt>,
    /// Timestamp
    pub detected_at: Instant,
}

// ---------------------------------------------------------------------------
// 3. ModelAutopsy — the central system
// ---------------------------------------------------------------------------

/// Self-healing model pipeline.
///
/// 1. Monitors inference for anomalies (NaN, Inf, perplexity spikes)
/// 2. When found, traces to the source weight tensor
/// 3. Runs diagnostic replay to confirm the issue
/// 4. Creates a segment patch with corrected weights
/// 5. Applies the patch to the ComputeImage
/// 6. Regenerates the receipt
///
/// The patched model is used for all subsequent inferences.
pub struct ModelAutopsy {
    /// Anomaly tracer
    pub tracer: AnomalyTracer,
    /// Detected anomalies
    pub anomalies: Vec<InferenceAnomaly>,
    /// Applied patches
    pub patches: Vec<SegmentPatch>,
    /// Callbacks for real-time anomaly notification
    pub on_anomaly: Option<Box<dyn Fn(&InferenceAnomaly) + Send>>,
    /// Auto-patch: if true, automatically patch when confirmed
    pub auto_patch: bool,
    /// Patch backup directory
    pub backup_dir: PathBuf,
}

impl ModelAutopsy {
    /// Create a new ModelAutopsy for the given model and coordination fabric.
    pub fn new(model: Arc<LoadedProfiledModel>, fabric: InMemoryCoordinationFabric) -> Self {
        Self {
            tracer: AnomalyTracer::new(model, fabric),
            anomalies: Vec::new(),
            patches: Vec::new(),
            on_anomaly: None,
            auto_patch: false,
            backup_dir: PathBuf::from("."),
        }
    }

    /// Called after each decode step. Checks for anomalies.
    /// If auto_patch is true and a confirmed issue is found, applies the patch.
    ///
    /// `hidden_states` is indexed by layer: `hidden_states[l]` is the hidden
    /// state after running layer `l`.
    pub fn inspect_step(
        &mut self,
        hidden_states: &[mlx_rs::Array],
        generated_tokens: &[u32],
    ) -> Result<Vec<SegmentPatch>, String> {
        let mut patches = Vec::new();

        for (l, hidden) in hidden_states.iter().enumerate() {
            let layer = l as u32;

            // Check for NaN
            if has_nan(hidden) {
                let anomaly = InferenceAnomaly::NanInLayer {
                    layer,
                    backend: BackendId(0),
                };
                self.handle_anomaly(anomaly, &mut patches)?;
            }

            // Check for Inf
            if has_inf(hidden) {
                let anomaly = InferenceAnomaly::InfInLayer {
                    layer,
                    backend: BackendId(0),
                };
                self.handle_anomaly(anomaly, &mut patches)?;
            }
        }

        // Check generated tokens for forbidden values
        for &token in generated_tokens {
            if is_forbidden_token(token) {
                let anomaly = InferenceAnomaly::ForbiddenToken {
                    token,
                    context: format!("token {} is in the forbidden set", token),
                };
                self.anomalies.push(anomaly.clone());
                if let Some(cb) = &self.on_anomaly {
                    cb(&anomaly);
                }
            }
        }

        Ok(patches)
    }

    /// Internal: record an anomaly and optionally trace + patch.
    fn handle_anomaly(
        &mut self,
        anomaly: InferenceAnomaly,
        patches: &mut Vec<SegmentPatch>,
    ) -> Result<(), String> {
        self.anomalies.push(anomaly.clone());
        if let Some(cb) = &self.on_anomaly {
            cb(&anomaly);
        }

        if self.auto_patch {
            let trace = self.tracer.trace(&anomaly)?;
            let replay = self.tracer.replay_weight(&trace)?;
            if !replay.hash_match || replay.reference_mse > 0.01 {
                let patch = SegmentPatch::from_replay(&replay);
                patch.apply(&self.backup_dir)?;
                self.patches.push(patch.clone());
                patches.push(patch);
            }
        }

        Ok(())
    }

    /// Run a full diagnostic scan on the loaded model.
    /// Iterates every layer's projection weights and replays them through
    /// the reference matmul, reporting MSE/cosine for each.
    pub fn full_scan(&self) -> Result<Vec<replay::ReplayResult>, String> {
        let n_layers = self
            .tracer
            .model
            .reader
            .manifest
            .execution_plan
            .layers
            .len();
        let mut results = Vec::new();

        for layer in 0..n_layers {
            let anomaly = InferenceAnomaly::ExplodingActivation {
                layer: layer as u32,
                norm: 0.0,
                threshold: 0.0,
            };
            let trace = self.tracer.trace(&anomaly)?;
            let replay = self.tracer.replay_weight(&trace)?;
            results.push(replay);
        }

        Ok(results)
    }

    /// Get a summary of all anomalies and patches.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("Anomalies detected: {}", self.anomalies.len()));
        for (i, anomaly) in self.anomalies.iter().enumerate() {
            let sev_label = match anomaly.severity() {
                AnomalySeverity::Warning => "WARN",
                AnomalySeverity::Error => "ERROR",
                AnomalySeverity::Critical => "CRITICAL",
            };
            lines.push(format!(
                "  [{}] {:>8}: {}",
                i,
                sev_label,
                anomaly.description()
            ));
        }

        lines.push(format!("\nPatches applied: {}", self.patches.len()));
        for patch in &self.patches {
            lines.push(format!(
                "  {} -> {}: {}",
                patch.segment_filename, patch.tensor_name, patch.reason
            ));
        }

        lines.join("\n")
    }
}

impl std::fmt::Debug for ModelAutopsy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelAutopsy")
            .field("anomalies", &self.anomalies.len())
            .field("patches", &self.patches.len())
            .field("auto_patch", &self.auto_patch)
            .field("backup_dir", &self.backup_dir)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether an `Array` contains any NaN values (when accessible as f32).
fn has_nan(arr: &mlx_rs::Array) -> bool {
    let slice = arr.as_slice::<f32>();
    slice.iter().any(|x| x.is_nan())
}

/// Check whether an `Array` contains any Inf values (when accessible as f32).
fn has_inf(arr: &mlx_rs::Array) -> bool {
    let slice = arr.as_slice::<f32>();
    slice.iter().any(|x| x.is_infinite())
}

/// Check whether `token` is in the forbidden set (e.g. BOS injected mid-sequence).
fn is_forbidden_token(token: u32) -> bool {
    matches!(token, 0 | 1)
}
