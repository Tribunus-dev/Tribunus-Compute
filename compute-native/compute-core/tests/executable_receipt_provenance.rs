//! E5F — Receipt and benchmark provenance integration tests.

#[cfg(test)]
mod tests {
    use tribunus_compute_core::compute_image::executable::schema::CompileTimeReceiptBundle;
    use tribunus_compute_core::compute_image::executable::schema::{
        KernelSelectionReceipt, NumericalVerificationReceipt, PhaseGraphVerificationReceipt,
        ResidencyVerificationReceipt, ResourceFitReceipt,
    };
    use tribunus_compute_core::integration::ContentHash;

    fn make_receipt_bundle() -> CompileTimeReceiptBundle {
        CompileTimeReceiptBundle {
            numerical_receipts: vec![NumericalVerificationReceipt {
                artifact_identity: "matmul_q_proj".into(),
                reference_graph_hash: ContentHash(100),
                max_abs_error: 1e-5,
                max_rel_error: 1e-5,
                cosine_similarity: 0.99999,
                passed: true,
            }],
            resource_fit_receipts: vec![ResourceFitReceipt {
                artifact_identity: "mlp_block".into(),
                resource_fit_ok: true,
                peak_memory_bytes: 4 * 1024 * 1024,
            }],
            phase_graph_receipts: vec![PhaseGraphVerificationReceipt {
                artifact_identity: "decoder_layer_0".into(),
                phase_count: 7,
                edge_count: 6,
                graph_valid: true,
            }],
            residency_receipts: vec![ResidencyVerificationReceipt {
                artifact_identity: "decode1".into(),
                residency_ok: true,
                total_weight_bytes: 1024 * 1024 * 1024,
            }],
            artifact_selection_receipts: vec![KernelSelectionReceipt {
                artifact_identity: "q4_k_m_matmul".into(),
                selected_kernel_id: "metal_q4_k_m_128".into(),
                candidate_count: 3,
                selection_valid: true,
            }],
            bundle_hash: ContentHash(0xDEAD),
        }
    }

    #[test]
    fn test_receipt_bundle_constructs() {
        let bundle = make_receipt_bundle();
        assert_eq!(bundle.numerical_receipts.len(), 1);
        assert_eq!(bundle.resource_fit_receipts.len(), 1);
        assert_eq!(bundle.phase_graph_receipts.len(), 1);
        assert_eq!(bundle.residency_receipts.len(), 1);
        assert_eq!(bundle.artifact_selection_receipts.len(), 1);
    }

    #[test]
    fn test_numerical_receipt_thresholds() {
        let r = &make_receipt_bundle().numerical_receipts[0];
        assert!(r.passed);
        assert!(r.max_abs_error < 1e-4);
        assert!(r.max_rel_error < 1e-4);
        assert!(r.cosine_similarity > 0.9999);
    }

    #[test]
    fn test_kernel_selection_receipt() {
        let r = &make_receipt_bundle().artifact_selection_receipts[0];
        assert_eq!(r.selected_kernel_id, "metal_q4_k_m_128");
        assert!(r.selection_valid);
    }
}
