//! Profile proof seal — compile-time artifact selection receipts with
//! numerical and resource-fit verification evidence.
//!
//! A [`ProfileProofSeal`] records which kernel variants were selected at
//! compile time for a given target profile, along with the numerical
//! verification and resource-fit proof receipts that justify each selection.
//! Multiple profile seals can be assembled into a
//! [`ProfileProofSealBundle`] for batch verification.

use serde::{Deserialize, Serialize};
use crate::compute_image::kernel_selection::selection::PreselectedKernelVariant;
use crate::compute_image::verification::numerical::NumericalVerificationReceipt;
use crate::compute_image::verification::resource_fit::ResourceFitReceipt;

/// Compile-time proof seal for a single target profile.
///
/// Records the identity of the profile, the selected kernel variants, and
/// the numerical and resource-fit receipts that certify each selection.
/// The `seal_hash` binds the entire profile seal into a single digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProofSeal {
    pub profile_id: String,
    pub profile_hash: String,
    pub variant_selections: Vec<PreselectedKernelVariant>,
    pub numerical_receipts: Vec<NumericalVerificationReceipt>,
    pub resource_fit_receipts: Vec<ResourceFitReceipt>,
    pub seal_hash: String,
}

/// A bundle of profile proof seals, aggregated for batch attestation.
///
/// Multiple profiles may target different hardware or runtime constraints
/// (e.g. Metal vs Core ML, low-power vs high-throughput). The
/// `bundle_hash` binds the set of profiles into a single attestable
/// digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProofSealBundle {
    pub profiles: Vec<ProfileProofSeal>,
    pub bundle_hash: String,
}
