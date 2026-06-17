//! FULL-SUITE-TIER1-DEFECT-CLUSTERING-0001 binary entry point.
//!
//! Reads receipts from a run directory, clusters non-pass rows by root cause,
//! and writes 5 output files:
//!
//! - `tier1_defect_observations.json`
//! - `tier1_defect_clusters.json`
//! - `tier1_cross_backend_correlations.json`
//! - `tier1_defect_summary.md`
//! - `tier1_blockers_for_decode_microphase.json`
//! - `next_fix_gates.json`
//!
//! Usage:
//!   cargo run --bin tribunus-tier1-defect-cluster -- \
//!     --run-dir decode_attribution_runs/TIER1-GATE/DA-0001-193461 \
//!     [--output-dir ./tier1-defect-closure]

use std::path::PathBuf;
use std::time::Instant;

use tribunus_compute_core::decode_attribution::defect_clustering::{
    cluster_defects, load_receipts_from_run, write_clusters_json, write_correlations_json,
    write_next_fix_gates_json, write_observations_json, write_summary_md,
    write_tier2_blockers_json,
};

fn main() {
    let start = Instant::now();

    // ── CLI args ──────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut run_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--run-dir" => {
                i += 1;
                run_dir = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: tribunus-tier1-defect-cluster --run-dir <path> [--output-dir <path>]"
                );
                eprintln!();
                eprintln!("Reads receipts from <run-dir> and writes clustering output.");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!(
                    "Usage: tribunus-tier1-defect-cluster --run-dir <path> [--output-dir <path>]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        tribunus_compute_core::log_error!("Error: --run-dir is required");
        std::process::exit(1);
    });

    let output_dir = output_dir.unwrap_or_else(|| {
        // Default: write alongside the run directory
        run_dir.parent().unwrap_or(&run_dir).to_path_buf()
    });

    if !run_dir.is_dir() {
        tribunus_compute_core::log_error!("Error: run directory not found: {:?}", run_dir);
        std::process::exit(1);
    }

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        tribunus_compute_core::log_error!("Error creating output directory {:?}: {e}", output_dir);
        std::process::exit(1);
    });

    tribunus_compute_core::log_info!("Reading receipts from {:?}...", run_dir);

    // ── Load receipts ─────────────────────────────────────────────────────────
    let receipts = match load_receipts_from_run(&run_dir) {
        Ok(r) => r,
        Err(e) => {
            tribunus_compute_core::log_error!("Error loading receipts: {e}");
            std::process::exit(1);
        }
    };

    tribunus_compute_core::log_info!("  Loaded {} receipts", receipts.len());

    // ── Cluster ───────────────────────────────────────────────────────────────
    let (observations, clusters, correlations, pass_tier0, pass_tier1, total_tier0, total_tier1) =
        cluster_defects(&receipts);

    tribunus_compute_core::log_info!("=== Clustering results ===");
    tribunus_compute_core::log_info!("  Tier 0:    {pass_tier0}/{total_tier0} pass");
    tribunus_compute_core::log_info!(
        "  Tier 1:    {pass_tier1}/{total_tier1} pass ({} non-pass)",
        observations.len()
    );
    tribunus_compute_core::log_info!("  Clusters:  {}", clusters.len());
    tribunus_compute_core::log_info!("  Correlations: {}", correlations.len());

    for cluster in &clusters {
        tribunus_compute_core::log_info!(
            "  {:45} {:25} {:8} {:5} rows",
            cluster.cluster_id,
            cluster.cluster_kind.as_str(),
            cluster.severity.as_str(),
            cluster.representative_rows.len()
        );
    }

    if !correlations.is_empty() {
        tribunus_compute_core::log_info!("Cross-backend correlations:");
        for corr in &correlations {
            tribunus_compute_core::log_info!(
                "  {}: contract '{}' fails on {} backends ({})",
                corr.correlation_id,
                corr.semantic_contract_id,
                corr.backends.len(),
                corr.backends.join(", ")
            );
        }
    }

    // ── Write outputs ─────────────────────────────────────────────────────────
    let obs_path = output_dir.join("tier1_defect_observations.json");
    let clusters_path = output_dir.join("tier1_defect_clusters.json");
    let corr_path = output_dir.join("tier1_cross_backend_correlations.json");
    let summary_path = output_dir.join("tier1_defect_summary.md");
    let blockers_path = output_dir.join("tier1_blockers_for_decode_microphase.json");
    let fix_gates_path = output_dir.join("next_fix_gates.json");

    let mut had_error = false;

    if let Err(e) = write_observations_json(&observations, &obs_path) {
        tribunus_compute_core::log_error!("Error writing observations: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", obs_path);
    }

    if let Err(e) = write_clusters_json(&clusters, &clusters_path) {
        tribunus_compute_core::log_error!("Error writing clusters: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", clusters_path);
    }

    if let Err(e) = write_correlations_json(&correlations, &corr_path) {
        tribunus_compute_core::log_error!("Error writing correlations: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", corr_path);
    }

    if let Err(e) = write_summary_md(
        &clusters,
        &observations,
        &correlations,
        pass_tier0,
        pass_tier1,
        total_tier0,
        total_tier1,
        &summary_path,
    ) {
        tribunus_compute_core::log_error!("Error writing summary: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", summary_path);
    }

    if let Err(e) = write_tier2_blockers_json(&clusters, &blockers_path) {
        tribunus_compute_core::log_error!("Error writing blockers: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", blockers_path);
    }

    if let Err(e) = write_next_fix_gates_json(&clusters, &fix_gates_path) {
        tribunus_compute_core::log_error!("Error writing fix gates: {e}");
        had_error = true;
    } else {
        tribunus_compute_core::log_info!("  Wrote {:?}", fix_gates_path);
    }

    let elapsed = start.elapsed();
    if had_error {
        tribunus_compute_core::log_error!(
            "Clustering completed with errors in {:.1}s",
            elapsed.as_secs_f64()
        );
        std::process::exit(1);
    } else {
        tribunus_compute_core::log_info!(
            "Clustering completed successfully in {:.1}s",
            elapsed.as_secs_f64()
        );
    }
}
