use serde::Serialize;

use crate::projection_identity::RuntimeMode;
use crate::tokenizer::TribunusTokenizer;
#[cfg(feature = "mlx-backend")]
use crate::worker_protocol::StartGenerationPayload;
#[cfg(feature = "mlx-backend")]
use crate::streaming::GenerationEvent;
#[cfg(feature = "mlx-backend")]
use crate::worker_supervisor::WorkerSupervisor;

/// Status of a single readiness gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
}

/// One readiness gate with name, status, timing, and optional detail.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessGate {
    pub name: &'static str,
    pub status: GateStatus,
    pub elapsed_ms: Option<u64>,
    pub detail: Option<String>,
}

/// Aggregate readiness state that gates `/v1/chat/completions` behind a
/// sequence of startup checks.
///
/// Gates are evaluated inline by [`ReadinessGates::run_all`] which runs
/// real one-token inference against the worker (MLX backend) or checks
/// tokenizer and CPU backend availability (Candle CPU backend).
pub struct ReadinessGates {
    gates: Vec<ReadinessGate>,
    ready_for_inference: bool,
}

impl ReadinessGates {
    /// Create a new gate set with every gate in `Pending` status.
    pub fn new() -> Self {
        let gates = vec![
            ReadinessGate {
                name: "worker_health",
                status: GateStatus::Pending,
                elapsed_ms: None,
                detail: None,
            },
            ReadinessGate {
                name: "tokenizer",
                status: GateStatus::Pending,
                elapsed_ms: None,
                detail: None,
            },
            ReadinessGate {
                name: "smoke_prefill",
                status: GateStatus::Pending,
                elapsed_ms: None,
                detail: None,
            },
            ReadinessGate {
                name: "smoke_decode",
                status: GateStatus::Pending,
                elapsed_ms: None,
                detail: None,
            },
        ];
        Self {
            gates,
            ready_for_inference: false,
        }
    }

    /// Human-readable summary of every gate name and status.
    pub fn summary(&self) -> String {
        self.gates
            .iter()
            .map(|g| format!("{}:{:?}", g.name, g.status))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Run all readiness gates: worker health check, tokenizer availability,
    /// and a one-token prefill + decode smoke test.
    ///
    /// Gates are evaluated in order. If any gate fails, later gates are
    /// skipped and `ready_for_inference` stays `false`.
    ///
    /// Available only when the `mlx-backend` feature is enabled.
    #[cfg(feature = "mlx-backend")]
    pub fn run_all(
        &mut self,
        supervisor: Option<&WorkerSupervisor>,
        tokenizer: Option<&TribunusTokenizer>,
        runtime_mode: RuntimeMode,
    ) {
        // Gate 1: Worker Health
        let worker_healthy = supervisor.map_or(false, |s| s.process_ctrl.is_alive());
        self.set_gate("worker_health", worker_healthy);
        if !worker_healthy {
            return;
        }

        // Gate 2: Tokenizer
        let has_tokenizer = tokenizer.is_some() || runtime_mode == RuntimeMode::Experimental;
        self.set_gate("tokenizer", has_tokenizer);
        if !has_tokenizer {
            return;
        }

        // Gate 3: One-token prefill + decode smoke test
        let supervisor = match supervisor {
            Some(s) => s,
            None => {
                self.set_gate("smoke_prefill", false);
                return;
            }
        };
        let tokenizer = match tokenizer {
            Some(t) => t,
            None => {
                self.set_gate("smoke_prefill", false);
                return;
            }
        };

        let encode_result = tokenizer.encode("Hello");
        match encode_result {
            Ok(prompt_token_ids) => {
                let payload = StartGenerationPayload {
                    prompt_token_ids,
                    max_output_tokens: 1,
                    deadline_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64
                        + 30_000,
                    request_id: "readiness-smoke".to_string(),
                    temperature: None,
                    top_k: None,
                    top_p: None,
                    seed: None,
                    stop_token_ids: vec![],
                };
                match supervisor.start_generation(&payload) {
                    Ok(mut handle) => {
                        let mut got_token = false;
                        loop {
                            match handle.stream.recv() {
                                Some(GenerationEvent::Token(tok)) => {
                                    if tok < 128000 {
                                        got_token = true;
                                    }
                                }
                                Some(GenerationEvent::Done) => break,
                                Some(GenerationEvent::Error(msg)) => {
                                    self.set_gate("smoke_prefill", false);
                                    self.set_gate_detail(
                                        "smoke_prefill",
                                        Some(format!("error: {}", msg)),
                                    );
                                    return;
                                }
                                _ => break,
                            }
                        }
                        if got_token {
                            self.set_gate("smoke_prefill", true);
                            self.set_gate("smoke_decode", true);
                        } else {
                            self.set_gate("smoke_prefill", false);
                            self.set_gate_detail(
                                "smoke_prefill",
                                Some("no valid token received".into()),
                            );
                        }
                    }
                    Err(e) => {
                        self.set_gate("smoke_prefill", false);
                        self.set_gate_detail(
                            "smoke_prefill",
                            Some(format!("start_generation failed: {:?}", e)),
                        );
                    }
                }
            }
            Err(e) => {
                self.set_gate("smoke_prefill", false);
                self.set_gate_detail(
                    "smoke_prefill",
                    Some(format!("tokenizer encode failed: {}", e)),
                );
            }
        }
    }

    /// Run readiness gates for backends without a worker subprocess (e.g. Candle CPU).
    ///
    /// Checks tokenizer and CPU backend availability.  Worker health and smoke
    /// tests are skipped since there is no separate worker process.
    #[cfg(not(feature = "mlx-backend"))]
    pub fn run_all(
        &mut self,
        tokenizer: Option<&TribunusTokenizer>,
        runtime_mode: RuntimeMode,
    ) {
        // Gate 1: Worker Health — skipped (no worker process in CPU mode)
        if let Some(g) = self.gates.iter_mut().find(|g| g.name == "worker_health") {
            g.status = GateStatus::Skipped;
        }

        // Gate 2: Tokenizer
        let has_tokenizer = tokenizer.is_some() || runtime_mode == RuntimeMode::Experimental;
        self.set_gate("tokenizer", has_tokenizer);
        if !has_tokenizer {
            self.ready_for_inference = self.gates.iter().all(|g| g.status == GateStatus::Passed);
            return;
        }

        // Gates 3 & 4: Smoke tests — skipped (no worker process in CPU mode)
        for name in &["smoke_prefill", "smoke_decode"] {
            if let Some(g) = self.gates.iter_mut().find(|g| g.name == *name) {
                g.status = GateStatus::Skipped;
            }
        }

        self.ready_for_inference = self.gates.iter().all(|g| g.status == GateStatus::Passed);
    }

    /// Set a single gate to Passed (true) or Failed (false).
    /// Recomputes `ready_for_inference` after the change.
    fn set_gate(&mut self, name: &'static str, passed: bool) {
        if let Some(g) = self.gates.iter_mut().find(|g| g.name == name) {
            g.status = if passed {
                GateStatus::Passed
            } else {
                GateStatus::Failed
            };
        }
        self.ready_for_inference = self.gates.iter().all(|g| g.status == GateStatus::Passed);
    }

    /// Set the detail string for a single gate (e.g. an error message).
    fn set_gate_detail(&mut self, name: &'static str, detail: Option<String>) {
        if let Some(g) = self.gates.iter_mut().find(|g| g.name == name) {
            g.detail = detail;
        }
    }

    /// True when every gate has passed.
    pub fn all_passed(&self) -> bool {
        self.gates.iter().all(|g| g.status == GateStatus::Passed)
    }

    /// True when the server is safe to accept inference requests.
    pub fn ready_for_inference(&self) -> bool {
        self.ready_for_inference
    }

    pub fn gate_states(&self) -> &[ReadinessGate] {
        &self.gates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_all_pending() {
        let rg = ReadinessGates::new();
        for gate in &rg.gates {
            assert_eq!(gate.status, GateStatus::Pending);
        }
        assert!(!rg.ready_for_inference());
        assert!(!rg.all_passed());
    }

    #[test]
    fn summary_empty() {
        let rg = ReadinessGates::new();
        let s = rg.summary();
        assert!(s.contains("worker_health"));
        assert!(s.contains("tokenizer"));
        assert!(s.contains("smoke_prefill"));
        assert!(s.contains("smoke_decode"));
    }

    #[test]
    fn set_gate_passed() {
        let mut rg = ReadinessGates::new();
        rg.set_gate("worker_health", true);
        assert_eq!(rg.gates[0].status, GateStatus::Passed);
        // Not yet all passed — other gates still pending
        assert!(!rg.all_passed());
    }

    #[test]
    fn set_gate_failed() {
        let mut rg = ReadinessGates::new();
        rg.set_gate("worker_health", false);
        assert_eq!(rg.gates[0].status, GateStatus::Failed);
        assert!(!rg.all_passed());
    }

    #[test]
    fn set_gate_detail_round_trip() {
        let mut rg = ReadinessGates::new();
        rg.set_gate_detail("tokenizer", Some("not found".into()));
        assert_eq!(
            rg.gates.iter().find(|g| g.name == "tokenizer").unwrap().detail,
            Some("not found".into())
        );
    }

    #[test]
    fn all_pass_sets_ready() {
        let mut rg = ReadinessGates::new();
        for gate_name in &["worker_health", "tokenizer", "smoke_prefill", "smoke_decode"] {
            rg.set_gate(gate_name, true);
        }
        assert!(rg.all_passed());
        assert!(rg.ready_for_inference());
    }

    #[test]
    fn any_fail_blocks_ready() {
        let mut rg = ReadinessGates::new();
        rg.set_gate("worker_health", true);
        rg.set_gate("tokenizer", true);
        rg.set_gate("smoke_prefill", true);
        rg.set_gate("smoke_decode", false);
        assert!(!rg.all_passed());
        assert!(!rg.ready_for_inference());
    }
}
