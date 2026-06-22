use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

use crate::backend::{MatmulOp, RmsNormOp, RoPEOp, EvaluationReceipt};
use crate::compute_image::tensix::TensixComputeImage;

/// A stage in the composed decode plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecodeStage {
    RMSNorm(RmsNormOp, String), // Op, artifact reference
    QKV(MatmulOp, String),
    RoPE(RoPEOp, String),
    KVAppend(String),
    Attention(String),
    OutputProjection(MatmulOp, String),
    Residual(String),
    MLP(MatmulOp, String),
    Output(String),
}

impl Hash for DecodeStage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Implement custom hash for DecodeStage because ops don't derive Hash
        match self {
            DecodeStage::RMSNorm(op, artifact) => {
                "RMSNorm".hash(state);
                op.epsilon.to_bits().hash(state); // Float hashing workaround
                artifact.hash(state);
            }
            DecodeStage::QKV(op, artifact) => {
                "QKV".hash(state);
                op.m.hash(state);
                op.k.hash(state);
                op.n.hash(state);
                artifact.hash(state);
            }
            DecodeStage::RoPE(op, artifact) => {
                "RoPE".hash(state);
                op.dim.hash(state);
                op.base.to_bits().hash(state); // Float hashing workaround
                artifact.hash(state);
            }
            DecodeStage::KVAppend(artifact) => {
                "KVAppend".hash(state);
                artifact.hash(state);
            }
            DecodeStage::Attention(artifact) => {
                "Attention".hash(state);
                artifact.hash(state);
            }
            DecodeStage::OutputProjection(op, artifact) => {
                "OutputProjection".hash(state);
                op.m.hash(state);
                op.k.hash(state);
                op.n.hash(state);
                artifact.hash(state);
            }
            DecodeStage::Residual(artifact) => {
                "Residual".hash(state);
                artifact.hash(state);
            }
            DecodeStage::MLP(op, artifact) => {
                "MLP".hash(state);
                op.m.hash(state);
                op.k.hash(state);
                op.n.hash(state);
                artifact.hash(state);
            }
            DecodeStage::Output(artifact) => {
                "Output".hash(state);
                artifact.hash(state);
            }
        }
    }
}

/// A plan for one step of decoding on Tensix backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensixDecodePlan {
    pub stages: Vec<DecodeStage>,
    /// Receipts from execution/evaluation
    pub evidence: Vec<EvaluationReceipt>,
}

impl Hash for TensixDecodePlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.stages.hash(state);
        // Evidence is usually ignored for the plan identity/hash, or hashed specifically.
        // We'll ignore it for the core plan hash identity to allow plan memoization.
    }
}

impl TensixDecodePlan {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            evidence: Vec::new(),
        }
    }

    pub fn add_stage(&mut self, stage: DecodeStage) {
        self.stages.push(stage);
    }
    
    pub fn add_evidence(&mut self, receipt: EvaluationReceipt) {
        self.evidence.push(receipt);
    }
}
