use crate::compute_image::phase_graph::PhaseId;
use serde::{Deserialize, Serialize};

/// A cache generation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvGeneration(pub u64);

/// Publication state of a KV transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvPublicationState {
    Tentative,
    Committed,
    RolledBack,
}

/// Record of a single KV write operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWriteRecord {
    pub layer_index: usize,
    pub token_position: usize,
    pub num_new_tokens: usize,
    pub bytes_written: u64,
}

/// A tentative KV cache mutation that can be committed or rolled back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvMutationTransaction {
    pub phase_id: PhaseId,
    pub layer_index: usize,
    pub prior_generation: KvGeneration,
    pub tentative_generation: KvGeneration,
    pub writes: Vec<KvWriteRecord>,
    pub publication_state: KvPublicationState,
}

impl KvMutationTransaction {
    pub fn new(phase_id: PhaseId, layer_index: usize, prior_generation: KvGeneration) -> Self {
        Self {
            phase_id,
            layer_index,
            prior_generation,
            tentative_generation: KvGeneration(prior_generation.0 + 1),
            writes: Vec::new(),
            publication_state: KvPublicationState::Tentative,
        }
    }

    /// Commit the transaction — marks writes as visible.
    pub fn commit(&mut self) -> Result<(), String> {
        match self.publication_state {
            KvPublicationState::Tentative => {
                self.publication_state = KvPublicationState::Committed;
                Ok(())
            }
            KvPublicationState::Committed => {
                Err("transaction already committed".to_string())
            }
            KvPublicationState::RolledBack => {
                Err("cannot commit a rolled-back transaction".to_string())
            }
        }
    }

    /// Roll back the transaction — discards tentative writes.
    pub fn rollback(&mut self) -> Result<(), String> {
        match self.publication_state {
            KvPublicationState::Tentative => {
                self.publication_state = KvPublicationState::RolledBack;
                self.writes.clear();
                Ok(())
            }
            KvPublicationState::Committed => {
                Err("cannot roll back a committed transaction".to_string())
            }
            KvPublicationState::RolledBack => {
                Err("transaction already rolled back".to_string())
            }
        }
    }

    pub fn is_committed(&self) -> bool {
        self.publication_state == KvPublicationState::Committed
    }

    pub fn is_tentative(&self) -> bool {
        self.publication_state == KvPublicationState::Tentative
    }
}

/// Transaction receipt stored in the phase receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvTransactionReceipt {
    pub layer_index: usize,
    pub prior_generation: u64,
    pub new_generation: u64,
    pub num_writes: usize,
    pub committed: bool,
    pub bytes_written: u64,
}

impl From<&KvMutationTransaction> for KvTransactionReceipt {
    fn from(tx: &KvMutationTransaction) -> Self {
        let total_bytes: u64 = tx.writes.iter().map(|w| w.bytes_written).sum();
        KvTransactionReceipt {
            layer_index: tx.layer_index,
            prior_generation: tx.prior_generation.0,
            new_generation: tx.tentative_generation.0,
            num_writes: tx.writes.len(),
            committed: tx.is_committed(),
            bytes_written: total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let phase_id = PhaseId("layer_0_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id.clone(), 0, KvGeneration(1));
        assert!(tx.is_tentative());
        tx.writes.push(KvWriteRecord {
            layer_index: 0,
            token_position: 0,
            num_new_tokens: 1,
            bytes_written: 4096,
        });
        assert!(tx.commit().is_ok());
        assert!(tx.is_committed());
        // Double commit fails
        assert!(tx.commit().is_err());
    }

    #[test]
    fn test_rollback() {
        let phase_id = PhaseId("layer_0_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id, 0, KvGeneration(5));
        assert!(tx.rollback().is_ok());
        assert!(!tx.is_committed());
        assert!(tx.commit().is_err());
    }

    #[test]
    fn test_receipt_conversion() {
        let phase_id = PhaseId("layer_1_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id, 1, KvGeneration(3));
        tx.writes.push(KvWriteRecord {
            layer_index: 1,
            token_position: 10,
            num_new_tokens: 1,
            bytes_written: 2048,
        });
        tx.commit().unwrap();
        let receipt: KvTransactionReceipt = (&tx).into();
        assert_eq!(receipt.layer_index, 1);
        assert_eq!(receipt.new_generation, 4);
        assert!(receipt.committed);
        assert_eq!(receipt.bytes_written, 2048);
    }
}
