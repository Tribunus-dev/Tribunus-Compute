//! Runtime executable profile selector — chooses the best-fitting
//! precompiled target profile from a sealed executable for the current
//! hardware environment.
//!
//! The selector **never** creates or modifies profiles.  It evaluates
//! profiles declared at compile time, filters to those compatible with
//! the runtime hardware, and picks the most specific match.

use crate::compute_image::executable::admission::ExecutableAdmissionError;
use crate::compute_image::executable::profile::ExecutableTargetProfile;

/// Snapshot of the runtime's hardware and OS capabilities.
///
/// Captured once at process start (or per-session for environment
/// changes) and compared against each profile's compiled hardware
/// and runtime contracts.
#[derive(Debug, Clone)]
pub struct RuntimeHardwareCaps {
    /// Canonical hardware family identifier (e.g. "apple-m1", "apple-m2").
    pub hardware_family: String,
    /// Number of GPU cores available on the device.
    pub gpu_core_count: u32,
    /// Number of Apple Neural Engine (ANE) cores available.
    pub ane_count: u32,
    /// Whether the device has unified memory (CPU + GPU pool).
    pub has_unified_memory: bool,
    /// Total unified RAM in gigabytes.
    pub unified_ram_gb: u64,
    /// OS version string for runtime contract comparison (e.g. "14.0").
    pub os_version: String,
    /// Whether Core ML framework is available.
    pub coreml_available: bool,
    /// Whether Metal framework is available.
    pub metal_available: bool,
}

/// Selects the best-matching [`ExecutableTargetProfile`] for the current
/// hardware environment.
///
/// Selection is purely analytical — the selector filters and ranks
/// precompiled profiles, never creating or modifying them.
pub struct ProfileSelector;

impl ProfileSelector {
    pub fn new() -> Self {
        Self
    }

    /// Select the best matching target profile from the executable's
    /// available profiles based on the current hardware environment.
    ///
    /// Returns `Ok(profile)` when at least one compatible profile is
    /// found, choosing the most specific match via:
    ///
    /// 1. Exact hardware family match.
    /// 2. Closest (non-exceeding) GPU core count.
    /// 3. Closest (non-exceeding) ANE count.
    /// 4. Highest compatible `min_os_version` (most specific).
    /// 5. Deterministic tiebreak: first declared.
    ///
    /// Returns [`ExecutableAdmissionError::MissingTargetProfile`] when
    /// no profile passes compatibility checks.
    pub fn select_profile<'a>(
        &self,
        profiles: &'a [ExecutableTargetProfile],
        hw_caps: &RuntimeHardwareCaps,
    ) -> Result<&'a ExecutableTargetProfile, ExecutableAdmissionError> {
        // Phase 1: filter to compatible profiles.
        let compatible: Vec<&'a ExecutableTargetProfile> = profiles
            .iter()
            .filter(|p| self.is_profile_compatible(p, hw_caps))
            .collect();

        if compatible.is_empty() {
            return Err(ExecutableAdmissionError::MissingTargetProfile);
        }

        // Phase 2: rank by specificity — pick the profile that best fits
        // the hardware without over-specifying requirements.
        let best = compatible
            .into_iter()
            .max_by(|a, b| {
                // Primary: prefer exact hardware family match.
                let a_family = a.hardware_contract.hardware_family.as_str();
                let b_family = b.hardware_contract.hardware_family.as_str();
                let a_exact = a_family == hw_caps.hardware_family;
                let b_exact = b_family == hw_caps.hardware_family;
                match a_exact.cmp(&b_exact) {
                    std::cmp::Ordering::Equal => {}
                    other => return other,
                }

                // Secondary: prefer closest GPU core count (higher is better
                // as long as it is ≤ runtime core count, which is guaranteed
                // by is_profile_compatible).
                let a_gpu = a.hardware_contract.gpu_core_count;
                let b_gpu = b.hardware_contract.gpu_core_count;
                match a_gpu.cmp(&b_gpu) {
                    std::cmp::Ordering::Equal => {}
                    other => return other,
                }

                // Tertiary: prefer closest ANE count.
                let a_ane = a.hardware_contract.ane_count;
                let b_ane = b.hardware_contract.ane_count;
                match a_ane.cmp(&b_ane) {
                    std::cmp::Ordering::Equal => {}
                    other => return other,
                }

                // Quaternary: prefer higher min_os_version (more specific
                // to the current OS).
                let a_os = &a.runtime_contract.min_os_version;
                let b_os = &b.runtime_contract.min_os_version;
                compare_semver(a_os, b_os).reverse()
            })
            .expect("compatible is non-empty");

        Ok(best)
    }

    /// Check if a specific profile is compatible with the hardware.
    ///
    /// Compatibility is true when all of the following hold:
    ///
    /// - The profile's hardware family exactly matches the runtime
    ///   hardware family.
    /// - The profile's required GPU core count does not exceed the
    ///   runtime's available GPU cores.
    /// - The profile's required ANE count does not exceed the runtime's
    ///   available ANE cores.
    /// - If the profile requires unified memory, the runtime must have it.
    /// - The runtime OS version is >= the profile's `min_os_version`
    ///   (semantic version comparison).
    /// - Every feature flag declared in the profile's runtime contract
    ///   is satisfied by the runtime capabilities.
    pub fn is_profile_compatible(
        &self,
        profile: &ExecutableTargetProfile,
        hw_caps: &RuntimeHardwareCaps,
    ) -> bool {
        let hw = &profile.hardware_contract;
        let rt = &profile.runtime_contract;

        // 1. Hardware family must match exactly.
        if hw.hardware_family != hw_caps.hardware_family {
            return false;
        }

        // 2. GPU core count must be sufficient.
        if hw.gpu_core_count > hw_caps.gpu_core_count {
            return false;
        }

        // 3. ANE count must be sufficient.
        if hw.ane_count > hw_caps.ane_count {
            return false;
        }

        // 4. Unified memory requirement.
        if hw.has_unified_memory && !hw_caps.has_unified_memory {
            return false;
        }

        // 5. OS version — runtime version must be >= min_os_version.
        if !is_runtime_os_compatible(&hw_caps.os_version, &rt.min_os_version) {
            return false;
        }

        // 6. Feature flags — every required feature must be satisfied.
        if !check_feature_flags(&rt.feature_flags, hw_caps) {
            return false;
        }

        true
    }
}

impl Default for ProfileSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Compare two semver-like version strings (e.g. "14.0", "15.2.1").
///
/// Returns `Ordering::Less` when `a < b`, `Equal` when equal, `Greater`
/// when `a > b`.  Components are compared numerically; missing trailing
/// components are treated as zero.
fn compare_semver(a: &str, b: &str) -> std::cmp::Ordering {
    let a_parts: Vec<u32> = a
        .trim()
        .split('.')
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();
    let b_parts: Vec<u32> = b
        .trim()
        .split('.')
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();

    let max_len = a_parts.len().max(b_parts.len());
    for i in 0..max_len {
        let a_val = a_parts.get(i).copied().unwrap_or(0);
        let b_val = b_parts.get(i).copied().unwrap_or(0);
        match a_val.cmp(&b_val) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

/// Returns `true` when the runtime OS version satisfies the profile's
/// minimum OS version requirement.
fn is_runtime_os_compatible(runtime_os: &str, min_os: &str) -> bool {
    compare_semver(runtime_os, min_os) != std::cmp::Ordering::Less
}

/// Check that every feature flag declared in the profile is available
/// on the runtime hardware.
///
/// Known feature names (case-sensitive):
/// - `"coreml"`          → hw_caps.coreml_available
/// - `"metal"`           → hw_caps.metal_available
/// - `"ane"`             → hw_caps.ane_count > 0
/// - `"unified_memory"`  → hw_caps.has_unified_memory
///
/// Unknown feature names are conservatively rejected (assumed
/// unavailable).
fn check_feature_flags(feature_flags: &[String], hw_caps: &RuntimeHardwareCaps) -> bool {
    for flag in feature_flags {
        let available = match flag.as_str() {
            "coreml" => hw_caps.coreml_available,
            "metal" => hw_caps.metal_available,
            "ane" => hw_caps.ane_count > 0,
            "unified_memory" => hw_caps.has_unified_memory,
            // Unknown feature — conservatively treat as unavailable.
            _ => return false,
        };
        if !available {
            return false;
        }
    }
    true
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(name: &str) -> ExecutableTargetProfile {
        ExecutableTargetProfile {
            profile_id: name.into(),
            profile_hash: Default::default(),
            hardware_contract: crate::compute_image::executable::profile::HardwareTargetContract {
                hardware_family: "apple-m1".into(),
                gpu_core_count: 8,
                ane_count: 1,
                has_unified_memory: true,
                max_threadgroup_size: 256,
            },
            runtime_contract: crate::compute_image::executable::profile::RuntimeTargetContract {
                min_os_version: "14.0".into(),
                feature_flags: vec![],
            },
            shape_variants: vec![],
            residency_plans: vec![],
            default_variant_selection:
                crate::compute_image::executable::profile::DefaultVariantSelection {
                    decode_variant_id: "decode1".into(),
                    prefill_variant_id: "prefill_small".into(),
                },
        }
    }

    fn make_hw_caps(
        family: &str,
        gpu: u32,
        ane: u32,
        unified: bool,
        ram_gb: u64,
        os: &str,
        coreml: bool,
        metal: bool,
    ) -> RuntimeHardwareCaps {
        RuntimeHardwareCaps {
            hardware_family: family.into(),
            gpu_core_count: gpu,
            ane_count: ane,
            has_unified_memory: unified,
            unified_ram_gb: ram_gb,
            os_version: os.into(),
            coreml_available: coreml,
            metal_available: metal,
        }
    }

    #[test]
    fn test_select_matching_profile() {
        let selector = ProfileSelector::new();
        let profiles = vec![make_profile("m1")];
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().profile_id, "m1");
    }

    #[test]
    fn test_select_no_compatible_profile() {
        let selector = ProfileSelector::new();
        // Profile requires apple-m1, runtime has apple-m2.
        let profiles = vec![make_profile("m1_profile")];
        let caps = make_hw_caps("apple-m2", 10, 2, true, 24, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutableAdmissionError::MissingTargetProfile
        ));
    }

    #[test]
    fn test_is_profile_compatible_exact_match() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1");
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        assert!(selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_hardware_family_mismatch() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1");
        let caps = make_hw_caps("apple-m2", 10, 2, true, 24, "15.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_insufficient_gpu() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1");
        let caps = make_hw_caps("apple-m1", 4, 1, true, 16, "15.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_insufficient_ane() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1");
        let caps = make_hw_caps("apple-m1", 8, 0, true, 16, "15.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_no_unified_memory() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1"); // has_unified_memory: true
        let caps = make_hw_caps("apple-m1", 8, 1, false, 0, "15.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_os_version_too_low() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1"); // min_os_version: "14.0"
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "13.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_exact_os_version() {
        let selector = ProfileSelector::new();
        let profile = make_profile("m1"); // min_os_version: "14.0"
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "14.0", true, true);
        assert!(selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_missing_coreml() {
        let selector = ProfileSelector::new();
        let mut profile = make_profile("m1");
        profile
            .runtime_contract
            .feature_flags
            .push("coreml".into());
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", false, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_is_profile_compatible_all_feature_flags_present() {
        let selector = ProfileSelector::new();
        let mut profile = make_profile("m1");
        profile
            .runtime_contract
            .feature_flags
            .push("coreml".into());
        profile
            .runtime_contract
            .feature_flags
            .push("metal".into());
        profile.runtime_contract.feature_flags.push("ane".into());
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        assert!(selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_select_best_profile_ranks_gpu_core_count() {
        let selector = ProfileSelector::new();
        let mut profile_4core = make_profile("m1_low");
        profile_4core.hardware_contract.gpu_core_count = 4;
        let mut profile_8core = make_profile("m1");
        profile_8core.hardware_contract.gpu_core_count = 8;

        let profiles = vec![profile_4core, profile_8core];

        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_ok());
        // Should prefer the 8-core profile (more specific to this hardware).
        assert_eq!(result.unwrap().profile_id, "m1");
    }

    #[test]
    fn test_select_best_profile_ranks_ane_count() {
        let selector = ProfileSelector::new();
        let mut profile_1ane = make_profile("m1_1ane");
        profile_1ane.hardware_contract.ane_count = 1;
        let mut profile_0ane = make_profile("m1_0ane");
        profile_0ane.hardware_contract.ane_count = 0;

        let profiles = vec![profile_0ane, profile_1ane];

        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_ok());
        // Should prefer the 1-ANE profile (more specific).
        assert_eq!(result.unwrap().profile_id, "m1_1ane");
    }

    #[test]
    fn test_select_best_profile_ranks_os_version() {
        let selector = ProfileSelector::new();
        let mut profile_legacy = make_profile("legacy");
        profile_legacy.runtime_contract.min_os_version = "13.0".into();
        let mut profile_modern = make_profile("modern");
        profile_modern.runtime_contract.min_os_version = "15.0".into();

        let profiles = vec![profile_legacy, profile_modern];

        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.5", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_ok());
        // Should prefer the higher min_os_version profile (more specific).
        assert_eq!(result.unwrap().profile_id, "modern");
    }

    #[test]
    fn test_select_tiebreak_first_declared() {
        let selector = ProfileSelector::new();
        // Two profiles that are identical in every ranking dimension.
        let p1 = make_profile("first");
        let p2 = make_profile("second");

        let profiles = vec![p1, p2];

        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_ok());
        // Deterministic tiebreak: first declared.
        assert_eq!(result.unwrap().profile_id, "first");
    }

    // ── compare_semver tests ────────────────────────────────────────────

    #[test]
    fn test_compare_semver_equal() {
        assert_eq!(compare_semver("14.0", "14.0"), std::cmp::Ordering::Equal);
        assert_eq!(
            compare_semver("15.2.1", "15.2.1"),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn test_compare_semver_less() {
        assert_eq!(compare_semver("13.0", "14.0"), std::cmp::Ordering::Less);
        assert_eq!(
            compare_semver("14.0", "15.2"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_semver("14.5", "14.10"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_compare_semver_greater() {
        assert_eq!(
            compare_semver("15.0", "14.0"),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            compare_semver("15.2.1", "15.2.0"),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn test_compare_semver_missing_components() {
        // "15" is treated as "15.0.0", "15.0" as "15.0.0"
        assert_eq!(
            compare_semver("15", "15.0"),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_semver("14", "15.0"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_semver("16", "15.0"),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn test_compare_semver_malformed() {
        // Non-numeric components are ignored; empty -> all zeros.
        assert_eq!(
            compare_semver("", "0"),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_semver("abc", "0.0"),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_semver("15.alpha", "15.0"),
            std::cmp::Ordering::Equal
        );
    }

    // ── Feature flag tests ──────────────────────────────────────────────

    #[test]
    fn test_unknown_feature_flag_rejected() {
        let selector = ProfileSelector::new();
        let mut profile = make_profile("test");
        profile
            .runtime_contract
            .feature_flags
            .push("unknown_feature".into());
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        assert!(!selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_all_known_feature_flags() {
        let selector = ProfileSelector::new();
        let mut profile = make_profile("test");
        for flag in &["coreml", "metal", "ane", "unified_memory"] {
            profile
                .runtime_contract
                .feature_flags
                .push((*flag).into());
        }
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        assert!(selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_profile_does_not_require_unified_memory_on_no_um_hardware() {
        let selector = ProfileSelector::new();
        let mut profile = make_profile("test");
        profile.hardware_contract.has_unified_memory = false;
        let caps = make_hw_caps("apple-m1", 8, 1, false, 0, "15.0", true, true);
        // Profile doesn't require unified memory, so non-unified hw is ok.
        assert!(selector.is_profile_compatible(&profile, &caps));
    }

    #[test]
    fn test_empty_profiles_returns_missing() {
        let selector = ProfileSelector::new();
        let profiles: Vec<ExecutableTargetProfile> = vec![];
        let caps = make_hw_caps("apple-m1", 8, 1, true, 16, "15.0", true, true);
        let result = selector.select_profile(&profiles, &caps);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutableAdmissionError::MissingTargetProfile
        ));
    }
}
