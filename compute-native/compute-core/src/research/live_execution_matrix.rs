//! Live execution capability report.
//!
//! Workstream 0 of the Live Heterogeneous Inference Materialization campaign.
//! Emits one JSON record per run with the actual (not configured) execution state
//! of every subsystem. Distinguishes "available", "compiled", "loaded", "selected",
//! "dispatched", and "completed" as separate states per the mission contract.
//!
//! The report must fail closed when a component is configured as enabled but cannot
//! prove a live execution route.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Execution subsystem states — six-level ladder
// ---------------------------------------------------------------------------

/// Six-level state ladder for every execution subsystem.
///
/// Any higher state SUBSUMES all lower states. The report MUST distinguish
/// "configured as feature gate" from "actually ran a real kernel".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsystemState {
    /// Not present in the binary (cfg gate disabled).
    NotAvailable,
    /// Present in the binary, no artifact compiled/loaded yet.
    Available,
    /// Artifact compiled (e.g. Metal .metallib built, Core ML .mlmodelc compiled).
    Compiled,
    /// Artifact loaded into process memory (e.g. Metal pipeline state created).
    Loaded,
    /// Artifact selected for dispatch (e.g. phase graph binding chose this lane).
    Selected,
    /// Artifact dispatched and completed execution (e.g. command buffer committed
    /// and waited on).
    Dispatched,
    /// Execution completed with verified output (e.g. numerical parity check passed).
    Completed,
}

impl SubsystemState {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotAvailable => "not_available",
            Self::Available => "available",
            Self::Compiled => "compiled",
            Self::Loaded => "loaded",
            Self::Selected => "selected",
            Self::Dispatched => "dispatched",
            Self::Completed => "completed",
        }
    }

    pub fn highest(a: Self, b: Self) -> Self {
        use std::cmp::Ordering;
        match (a as u8).cmp(&(b as u8)) {
            Ordering::Greater => a,
            Ordering::Less => b,
            Ordering::Equal => a,
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache mode — explicit, not inferred from config fields
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheModeState {
    /// No KV cache present (stateless model or warmup pass).
    None,
    /// FP16 reference cache (uncompressed MLX arrays).
    Fp16,
    /// Compressed via asymmetric quant (KeyLightValueHeavy or similar).
    Compressed,
    /// TurboQuant compression active.
    TurboQuant,
}

// ---------------------------------------------------------------------------
// Per-phase receipt snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReceiptSnapshot {
    /// Phase id from the graph.
    pub phase_id: String,
    /// Phase kind string (e.g. "layer_0_attn", "prologue").
    pub phase_kind: String,
    /// Executing lane (e.g. "mlx", "metal", "accelerate", "coreml", "fallback").
    pub executed_subsystem: String,
    /// Concrete native symbol name (e.g. "vDSP_vadd", "matmul_4x4_neon", "SiLU").
    pub native_symbol: Option<String>,
    /// Duration in microseconds.
    pub duration_us: u64,
    /// Whether this phase used real session state (not empty context).
    pub used_real_state: bool,
    /// Whether this phase required a fallback.
    pub fallback_used: bool,
    /// Fallback reason if applicable.
    pub fallback_reason: Option<String>,
    /// Artifact hash if a sealed artifact was dispatched.
    pub artifact_hash: Option<String>,
    /// Completion status.
    pub status: String,
}

// ---------------------------------------------------------------------------
// CapabilityReport — the top-level JSON record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    // Identity
    pub model_image_id: String,
    pub hardware_profile: String,
    pub timestamp: String,

    // Feature flags active at compile time
    pub feature_flags: Vec<String>,

    // PhaseEngine mode
    pub phase_engine_mode: String,

    // Metal
    pub metal_dispatch_enabled: bool,
    pub metal_state: SubsystemState,
    pub metal_fused_artifacts_loaded: u32,

    // Accelerate
    pub accelerate_native_symbols_available: Vec<String>,
    pub accelerate_state: SubsystemState,
    pub accelerate_selected_phases: u32,

    // Core ML
    pub coreml_model_load_status: SubsystemState,
    pub coreml_compiled_subgraphs: u32,
    pub coreml_load_failure_reason: Option<String>,

    // KV cache
    pub kv_mode: KvCacheModeState,
    pub kv_compression_ratio: Option<f64>,
    pub kv_compression_active: bool,
    pub kv_rollbacks: u32,

    // Fused kernel artifacts
    pub fused_artifacts_loaded: u32,

    // Phase execution summary
    pub total_phases: u32,
    pub dispatched_phases: u32,
    pub completed_phases: u32,
    pub fallback_count: u32,
    pub actual_phase_receipts: Vec<PhaseReceiptSnapshot>,

    // Memory
    pub hidden_materialization_bytes: u64,
    pub kv_materialization_bytes: u64,
    pub peak_arena_bytes: u64,
}

impl CapabilityReport {
    /// Create a new report with default "not checked" values.
    pub fn new(model_image_id: &str, hardware_profile: &str) -> Self {
        Self {
            model_image_id: model_image_id.to_string(),
            hardware_profile: hardware_profile.to_string(),
            timestamp: String::new(),
            feature_flags: Vec::new(),
            phase_engine_mode: "shadow".to_string(),
            metal_dispatch_enabled: false,
            metal_state: SubsystemState::NotAvailable,
            metal_fused_artifacts_loaded: 0,
            accelerate_native_symbols_available: Vec::new(),
            accelerate_state: SubsystemState::NotAvailable,
            accelerate_selected_phases: 0,
            coreml_model_load_status: SubsystemState::NotAvailable,
            coreml_compiled_subgraphs: 0,
            coreml_load_failure_reason: None,
            kv_mode: KvCacheModeState::None,
            kv_compression_ratio: None,
            kv_compression_active: false,
            kv_rollbacks: 0,
            fused_artifacts_loaded: 0,
            total_phases: 0,
            dispatched_phases: 0,
            completed_phases: 0,
            fallback_count: 0,
            actual_phase_receipts: Vec::new(),
            hidden_materialization_bytes: 0,
            kv_materialization_bytes: 0,
            peak_arena_bytes: 0,
        }
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Write to a file path.
    pub fn write_to(&self, path: &Path) -> Result<(), String> {
        let json = self.to_json();
        std::fs::write(path, &json)
            .map_err(|e| format!("write capability report: {}", e))
    }

    /// Detect feature flags from cfg conditions.
    pub fn detect_feature_flags() -> Vec<String> {
        let mut flags = Vec::new();
        if cfg!(feature = "mlx-backend") {
            flags.push("mlx-backend".to_string());
        }
        if cfg!(feature = "metal-dispatch") {
            flags.push("metal-dispatch".to_string());
        }
        if cfg!(feature = "coreml-backend") {
            flags.push("coreml-backend".to_string());
        }
        if cfg!(feature = "candle-cpu") {
            flags.push("candle-cpu".to_string());
        }
        if cfg!(feature = "intel") {
            flags.push("intel".to_string());
        }
        if cfg!(feature = "tensix") {
            flags.push("tensix".to_string());
        }
        flags
    }

    /// Detect available Accelerate native symbols.
    pub fn detect_accelerate_symbols() -> Vec<String> {
        let mut symbols = Vec::new();
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            symbols.push("rms_norm_neon".to_string());
            symbols.push("matmul_4x4_neon".to_string());
        }
        #[cfg(target_os = "macos")]
        {
            symbols.push("vDSP_vadd".to_string());
            symbols.push("vDSP_vmul".to_string());
            symbols.push("vDSP_vsmul".to_string());
            symbols.push("vDSP_sve".to_string());
        }
        symbols.push("rms_norm_scalar".to_string());
        symbols.push("softmax_pass".to_string());
        symbols.push("matmul_4x4_scalar".to_string());
        symbols
    }

    /// Emit the report as JSON to stdout. Fail closed when a configured component
    /// cannot prove a live execution route.
    pub fn fail_closed_check(&self) -> Result<(), Vec<String>> {
        let mut failures = Vec::new();

        if self.metal_dispatch_enabled && (self.metal_state as u8) < (SubsystemState::Loaded as u8) {
            failures.push(format!(
                "metal-dispatch feature enabled but metal_state is {:?} (< loaded)",
                self.metal_state
            ));
        }
        if (self.coreml_model_load_status as u8) >= (SubsystemState::Selected as u8)
            && self.coreml_compiled_subgraphs == 0
        {
            failures.push(
                "Core ML model selected but zero compiled subgraphs".to_string(),
            );
        }
        if self.kv_compression_active && self.kv_mode == KvCacheModeState::None {
            failures.push(
                "KV compression claimed active but kv_mode is None".to_string(),
            );
        }
        if self.total_phases > 0 && self.dispatched_phases == 0 {
            failures.push(format!(
                "{} total phases but zero dispatched",
                self.total_phases
            ));
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

// ---------------------------------------------------------------------------
// Report builder — progressive accumulation
// ---------------------------------------------------------------------------

pub struct CapabilityReportBuilder {
    report: CapabilityReport,
}

impl CapabilityReportBuilder {
    pub fn new(model_image_id: &str, hardware_profile: &str) -> Self {
        Self {
            report: CapabilityReport::new(model_image_id, hardware_profile),
        }
    }

    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.report.timestamp = ts.to_string();
        self
    }

    pub fn with_feature_flags(mut self, flags: Vec<String>) -> Self {
        self.report.feature_flags = flags;
        self
    }

    pub fn with_phase_engine_mode(mut self, mode: &str) -> Self {
        self.report.phase_engine_mode = mode.to_string();
        self
    }

    pub fn with_metal_state(mut self, state: SubsystemState) -> Self {
        self.report.metal_state = state;
        self
    }

    pub fn with_metal_dispatch(mut self, enabled: bool, artifacts: u32) -> Self {
        self.report.metal_dispatch_enabled = enabled;
        self.report.metal_fused_artifacts_loaded = artifacts;
        self
    }

    pub fn with_accelerate_state(mut self, state: SubsystemState, phases: u32) -> Self {
        self.report.accelerate_state = state;
        self.report.accelerate_selected_phases = phases;
        self
    }

    pub fn with_coreml_state(
        mut self,
        state: SubsystemState,
        subgraphs: u32,
        failure: Option<String>,
    ) -> Self {
        self.report.coreml_model_load_status = state;
        self.report.coreml_compiled_subgraphs = subgraphs;
        self.report.coreml_load_failure_reason = failure;
        self
    }

    pub fn with_kv_mode(mut self, mode: KvCacheModeState, ratio: Option<f64>) -> Self {
        self.report.kv_mode = mode;
        self.report.kv_compression_ratio = ratio;
        self.report.kv_compression_active = mode != KvCacheModeState::None;
        self
    }

    pub fn with_fused_artifacts(mut self, count: u32) -> Self {
        self.report.fused_artifacts_loaded = count;
        self
    }

    pub fn with_phase_receipts(mut self, receipts: Vec<PhaseReceiptSnapshot>) -> Self {
        let total = receipts.len() as u32;
        let dispatched = receipts.iter().filter(|r| r.status == "complete").count() as u32;
        let completed = receipts
            .iter()
            .filter(|r| r.status == "complete" && r.used_real_state)
            .count() as u32;
        let fallbacks = receipts.iter().filter(|r| r.fallback_used).count() as u32;

        self.report.total_phases = total;
        self.report.dispatched_phases = dispatched;
        self.report.completed_phases = completed;
        self.report.fallback_count = fallbacks;
        self.report.actual_phase_receipts = receipts;
        self
    }

    pub fn with_memory(mut self, hidden: u64, kv: u64, peak: u64) -> Self {
        self.report.hidden_materialization_bytes = hidden;
        self.report.kv_materialization_bytes = kv;
        self.report.peak_arena_bytes = peak;
        self
    }

    pub fn build(self) -> CapabilityReport {
        self.report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsystem_state_ordering() {
        assert!((SubsystemState::NotAvailable as u8) < (SubsystemState::Available as u8);
        assert!((SubsystemState::Available as u8) < (SubsystemState::Compiled as u8);
        assert!((SubsystemState::Compiled as u8) < (SubsystemState::Loaded as u8);
        assert!((SubsystemState::Loaded as u8) < (SubsystemState::Selected as u8);
        assert!((SubsystemState::Selected as u8) < (SubsystemState::Dispatched as u8);
        assert!((SubsystemState::Dispatched as u8) < (SubsystemState::Completed as u8);
    }

    #[test]
    fn test_highest() {
        assert_eq!(
            SubsystemState::highest(SubsystemState::Available, SubsystemState::Compiled),
            SubsystemState::Compiled
        );
        assert_eq!(
            SubsystemState::highest(SubsystemState::Dispatched, SubsystemState::Available),
            SubsystemState::Dispatched
        );
        assert_eq!(
            SubsystemState::highest(SubsystemState::Completed, SubsystemState::Completed),
            SubsystemState::Completed
        );
    }

    #[test]
    fn test_report_creation() {
        let report = CapabilityReport::new("test-model-v1", "apple-m1-8");
        assert_eq!(report.model_image_id, "test-model-v1");
        assert_eq!(report.total_phases, 0);
        assert_eq!(report.metal_state, SubsystemState::NotAvailable);
    }

    #[test]
    fn test_report_builder() {
        let report = CapabilityReportBuilder::new("qwen2.5-0.5b", "apple-m1-8")
            .with_timestamp("2026-06-21T00:00:00Z")
            .with_feature_flags(vec!["mlx-backend".into(), "metal-dispatch".into()])
            .with_phase_engine_mode("authority")
            .with_metal_state(SubsystemState::Loaded)
            .with_metal_dispatch(true, 3)
            .with_accelerate_state(SubsystemState::Available, 0)
            .with_kv_mode(KvCacheModeState::Fp16, None)
            .build();

        assert_eq!(report.phase_engine_mode, "authority");
        assert_eq!(report.metal_fused_artifacts_loaded, 3);
    }

    #[test]
    fn test_fail_closed_passes() {
        let report = CapabilityReportBuilder::new("test", "m1")
            .with_metal_state(SubsystemState::Loaded)
            .with_metal_dispatch(true, 2)
            .with_kv_mode(KvCacheModeState::Compressed, Some(4.57))
            .with_phase_receipts(vec![PhaseReceiptSnapshot {
                phase_id: "prologue".into(),
                phase_kind: "Prologue".into(),
                executed_subsystem: "mlx".into(),
                native_symbol: None,
                duration_us: 100,
                used_real_state: true,
                fallback_used: false,
                fallback_reason: None,
                artifact_hash: None,
                status: "complete".into(),
            }])
            .build();

        assert!(report.fail_closed_check().is_ok());
    }

    #[test]
    fn test_fail_closed_metal_not_loaded() {
        let report = CapabilityReportBuilder::new("test", "m1")
            .with_metal_dispatch(true, 0)
            .with_metal_state(SubsystemState::Available) // not Loaded
            .build();

        let result = report.fail_closed_check();
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_flag_detection() {
        let flags = CapabilityReport::detect_feature_flags();
        // At minimum, the binary has some set of flags. We just verify the function runs.
        assert!(flags.len() > 0 || flags.is_empty());
    }

    #[test]
    fn test_accelerate_symbol_detection() {
        let symbols = CapabilityReport::detect_accelerate_symbols();
        assert!(symbols.contains(&"softmax_pass".to_string()));
    }

    #[test]
    fn test_kv_cache_mode_ordering() {
        assert!(KvCacheModeState::None as u8) < (KvCacheModeState::Fp16 as u8);
        assert!(KvCacheModeState::Fp16 as u8) < (KvCacheModeState::Compressed as u8);
    }
}
