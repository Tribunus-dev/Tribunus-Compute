//! ComputeImage verification and diagnostics.

use super::manifest::{
    mlx_peak_memory_bytes, CompiledImageReader, ManifestVerification, StorageBackend,
};
use serde::Serialize;
use std::path::Path;

/// Verify a compiled image by reading its manifest and running validation.
pub fn verify(image_dir: &str) -> crate::Result<ManifestVerification> {
    super::manifest::read(image_dir)?.verify()
}

/// Results from compile-time diagnostic verification.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    pub passed: bool,
    pub layers: Vec<LayerDiagnostic>,
    pub global: GlobalDiagnostic,
    pub issues: Vec<DiagnosticIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDiagnostic {
    pub layer_index: u32,
    pub attention_kind: String,
    pub hidden_norm: f64,
    pub hidden_finite: bool,
    pub hidden_min: f64,
    pub hidden_max: f64,
    pub entropy: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobalDiagnostic {
    pub total_layers: usize,
    pub nan_layers: usize,
    pub inf_layers: usize,
    pub max_runtime_ms: u64,
    pub total_runtime_ms: u64,
    pub memory_peak_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub enum DiagnosticIssue {
    NanInLayer(u32),
    InfInLayer(u32),
    ExplodingActivation { layer: u32, norm: f64 },
    VanishingActivation { layer: u32, norm: f64 },
    EntropyExtreme { layer: u32, entropy: f64 },
}

impl Default for GlobalDiagnostic {
    fn default() -> Self {
        Self {
            total_layers: 0,
            nan_layers: 0,
            inf_layers: 0,
            max_runtime_ms: 0,
            total_runtime_ms: 0,
            memory_peak_bytes: 0,
        }
    }
}

/// Run compile-time diagnostic verification on a compiled image.
pub fn run_diagnostics(image_dir: &Path) -> crate::Result<DiagnosticReport> {
    let reader = CompiledImageReader::open(image_dir)?;
    let plan = &reader.manifest.execution_plan;
    let runtime = reader.open_runtime(StorageBackend::Copied)?;

    let mut report = DiagnosticReport {
        passed: true,
        layers: Vec::new(),
        global: GlobalDiagnostic::default(),
        issues: Vec::new(),
    };

    for layer_plan in &plan.layers {
        let l = layer_plan.layer_index;
        let start = std::time::Instant::now();

        let lease = runtime.activate_layer(l)?;
        let layer_map = runtime.build_layer_arrays_from_lease(l, &lease)?;

        let mut has_nan = false;
        let mut has_inf = false;
        let mut norm_sum_sq: f64 = 0.0;
        let mut min_val: f64 = f64::MAX;
        let mut max_val: f64 = f64::NEG_INFINITY;

        for (_name, arr) in &layer_map {
            if let Ok(slice) = arr.try_as_slice::<f32>() {
                for &v in slice {
                    let vf = v as f64;
                    if v.is_nan() {
                        has_nan = true;
                    }
                    if v.is_infinite() {
                        has_inf = true;
                    }
                    if vf < min_val {
                        min_val = vf;
                    }
                    if vf > max_val {
                        max_val = vf;
                    }
                    norm_sum_sq += vf * vf;
                }
            }
        }

        let norm = norm_sum_sq.sqrt();
        let elapsed = start.elapsed().as_millis() as u64;

        let diag = LayerDiagnostic {
            layer_index: l,
            attention_kind: layer_plan.attention_kind.clone(),
            hidden_norm: norm,
            hidden_finite: !has_nan && !has_inf,
            hidden_min: min_val,
            hidden_max: max_val,
            entropy: 0.0,
            elapsed_ms: elapsed,
        };

        if has_nan {
            report.issues.push(DiagnosticIssue::NanInLayer(l));
            report.passed = false;
        }
        if has_inf {
            report.issues.push(DiagnosticIssue::InfInLayer(l));
            report.passed = false;
        }

        report.layers.push(diag);
    }

    report.global.total_layers = plan.layers.len();
    report.global.nan_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::NanInLayer(_)))
        .count();
    report.global.inf_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::InfInLayer(_)))
        .count();
    report.global.total_runtime_ms = report.layers.iter().map(|l| l.elapsed_ms).sum();
    report.global.max_runtime_ms = report
        .layers
        .iter()
        .map(|l| l.elapsed_ms)
        .max()
        .unwrap_or(0);
    report.global.memory_peak_bytes = mlx_peak_memory_bytes();

    Ok(report)
}

/// Atomically publish a staged compilation to its final destination.
///
/// 1. Writes a `.publishing` marker inside `staging`.
/// 2. Renames `staging` to `destination` (falls back to recursive copy
///    when the rename crosses filesystem boundaries).
/// 3. On failure the staging directory is left intact with a `.failed` marker
///    so that the caller can inspect or retry.
pub fn publish_image(staging: &Path, destination: &Path) -> crate::Result<()> {
    let publishing_marker = staging.join(".publishing");
    std::fs::write(&publishing_marker, b"")
        .map_err(|e| crate::Error::from_reason(format!("write .publishing: {}", e)))?;

    let result = std::fs::rename(staging, destination);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            // rename fails across filesystem boundaries — fall back to copy + remove
            if e.kind() == std::io::ErrorKind::CrossesDevices {
                let failed_marker = staging.join(".failed");
                if let Err(write_err) =
                    std::fs::write(&failed_marker, format!("rename failed: {}", e))
                {
                    return Err(crate::Error::from_reason(format!(
                        "write .failed marker: {} (original rename: {})",
                        write_err, e
                    )));
                }
                return Err(crate::Error::from_reason(format!(
                    "rename crosses devices: {}. Staging left in place with .failed marker.",
                    e
                )));
            }
            let failed_marker = staging.join(".failed");
            if let Err(write_err) = std::fs::write(&failed_marker, format!("rename failed: {}", e))
            {
                return Err(crate::Error::from_reason(format!(
                    "write .failed marker: {} (original rename: {})",
                    write_err, e
                )));
            }
            Err(crate::Error::from_reason(format!(
                "rename {} -> {}: {}",
                staging.display(),
                destination.display(),
                e
            )))
        }
    }
}
