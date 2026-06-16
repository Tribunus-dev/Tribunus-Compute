//! Accelerate backend capability discovery.
//!
//! This module provides platform, architecture, and Accelerate availability detection,
//! including enabled subsystems, supported dtypes, supported layouts, and threading policy.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{dtype::AccelerateDType, layout::AccelerateLayout, subsystem::AccelerateSubsystem, subsystem::ThreadingPolicy};

/// Architecture types for Apple platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleArchitecture {
    /// x86_64 (Intel Macs)
    X86_64,
    /// arm64 (Apple Silicon)
    Arm64,
    /// arm64e (Apple Silicon with pointer authentication)
    Arm64e,
    /// Unknown architecture
    Unknown,
}

impl fmt::Display for AppleArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppleArchitecture::X86_64 => write!(f, "x86_64"),
            AppleArchitecture::Arm64 => write!(f, "arm64"),
            AppleArchitecture::Arm64e => write!(f, "arm64e"),
            AppleArchitecture::Unknown => write!(f, "unknown"),
        }
    }
}

impl AppleArchitecture {
    /// Returns true if this is an Apple Silicon architecture.
    pub fn is_apple_silicon(&self) -> bool {
        matches!(self, AppleArchitecture::Arm64 | AppleArchitecture::Arm64e)
    }

    /// Returns true if this is an Intel architecture.
    pub fn is_intel(&self) -> bool {
        matches!(self, AppleArchitecture::X86_64)
    }

    /// Detects the current architecture.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            AppleArchitecture::X86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Check if we're on Apple Silicon specifically
            // For now, assume aarch64 on macOS is Apple Silicon
            if cfg!(target_os = "macos") {
                AppleArchitecture::Arm64
            } else {
                AppleArchitecture::Arm64
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            AppleArchitecture::Unknown
        }
    }
}

impl Default for AppleArchitecture {
    fn default() -> Self {
        AppleArchitecture::Unknown
    }
}

/// Platform types for Accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplePlatform {
    /// macOS
    Macos,
    /// iOS
    Ios,
    /// tvOS
    Tvos,
    /// watchOS
    Watchos,
    /// Unknown platform
    Unknown,
}

impl fmt::Display for ApplePlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplePlatform::Macos => write!(f, "macos"),
            ApplePlatform::Ios => write!(f, "ios"),
            ApplePlatform::Tvos => write!(f, "tvos"),
            ApplePlatform::Watchos => write!(f, "watchos"),
            ApplePlatform::Unknown => write!(f, "unknown"),
        }
    }
}

impl ApplePlatform {
    /// Detects the current platform.
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            ApplePlatform::Macos
        }
        #[cfg(target_os = "ios")]
        {
            ApplePlatform::Ios
        }
        #[cfg(target_os = "tvos")]
        {
            ApplePlatform::Tvos
        }
        #[cfg(target_os = "watchos")]
        {
            ApplePlatform::Watchos
        }
        #[cfg(not(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos"
        )))]
        {
            ApplePlatform::Unknown
        }
    }
}

impl Default for ApplePlatform {
    fn default() -> Self {
        ApplePlatform::Unknown
    }
}

/// Accelerate framework availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerateAvailability {
    /// Accelerate is fully available and functional.
    Available,
    /// Accelerate is available but some subsystems may be limited.
    Limited,
    /// Accelerate is not available on this platform.
    Unavailable,
}

impl fmt::Display for AccelerateAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateAvailability::Available => write!(f, "available"),
            AccelerateAvailability::Limited => write!(f, "limited"),
            AccelerateAvailability::Unavailable => write!(f, "unavailable"),
        }
    }
}

impl Default for AccelerateAvailability {
    fn default() -> Self {
        AccelerateAvailability::Unavailable
    }
}

/// Accelerate backend capabilities.
///
/// This struct records platform, architecture, Accelerate availability,
/// enabled subsystems, supported dtypes, supported layouts, threading policy,
/// and whether BNNSGraph is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateBackendCapabilities {
    /// Mission identifier for this capabilities detection.
    pub mission: String,
    /// The detected platform.
    pub platform: ApplePlatform,
    /// The detected architecture.
    pub architecture: AppleArchitecture,
    /// Whether Accelerate framework is available.
    pub accelerate_available: AccelerateAvailability,
    /// Set of enabled subsystems.
    pub enabled_subsystems: Vec<AccelerateSubsystem>,
    /// Set of supported dtypes.
    pub supported_dtypes: Vec<AccelerateDType>,
    /// Set of supported layouts.
    pub supported_layouts: Vec<AccelerateLayout>,
    /// Default threading policy.
    pub threading_policy: ThreadingPolicy,
    /// Whether BNNSGraph is available.
    pub bnns_graph_available: bool,
    /// Whether real-time characteristics are supported (no runtime allocation, single-threaded).
    pub realtime_characteristics_supported: bool,
    /// Version information for Accelerate framework (if available).
    pub accelerate_version: Option<String>,
    /// Version information for the operating system.
    pub os_version: String,
    /// Timestamp of capability detection.
    pub detection_timestamp: String,
}

impl AccelerateBackendCapabilities {
    /// Creates a new capabilities struct with default values.
    pub fn new() -> Self {
        Self {
            mission: super::MISSION_ACCELERATE_INFERENCE_PIPELINE_V0.to_string(),
            platform: ApplePlatform::detect(),
            architecture: AppleArchitecture::detect(),
            accelerate_available: AccelerateAvailability::detect(),
            enabled_subsystems: Vec::new(),
            supported_dtypes: Vec::new(),
            supported_layouts: Vec::new(),
            threading_policy: ThreadingPolicy::default(),
            bnns_graph_available: false,
            realtime_characteristics_supported: false,
            accelerate_version: None,
            os_version: Self::detect_os_version(),
            detection_timestamp: Self::current_timestamp(),
        }
    }

    /// Detects Accelerate availability based on platform.
    pub fn detect() -> Self {
        let mut caps = Self::new();

        // On macOS, Accelerate should be available
        if caps.platform != ApplePlatform::Unknown {
            caps.accelerate_available = AccelerateAvailability::Available;
            
            // Enable all subsystems on macOS
            caps.enabled_subsystems = AccelerateSubsystem::all().to_vec();
            
            // Enable all dtypes
            caps.supported_dtypes = AccelerateDType::all().to_vec();
            
            // Enable all layouts
            caps.supported_layouts = AccelerateLayout::all().to_vec();
            
            // BNNSGraph availability depends on platform and architecture
            caps.bnns_graph_available = caps.platform == ApplePlatform::Macos && 
                                        caps.architecture.is_apple_silicon();
            
            // Real-time characteristics are available on Apple Silicon
            caps.realtime_characteristics_supported = caps.architecture.is_apple_silicon();
            
            // Set threading policy based on platform
            caps.threading_policy = if caps.bnns_graph_available {
                ThreadingPolicy::SingleThreaded // BNNSGraph supports single-threaded for real-time
            } else {
                ThreadingPolicy::MultiThreaded
            };
        } else {
            // On non-Apple platforms, Accelerate is unavailable
            caps.accelerate_available = AccelerateAvailability::Unavailable;
            caps.enabled_subsystems = vec![AccelerateSubsystem::Reference];
            caps.supported_dtypes = vec![AccelerateDType::F32]; // Only basic f32 support
            caps.supported_layouts = vec![AccelerateLayout::RowMajor];
            caps.threading_policy = ThreadingPolicy::MultiThreaded;
            caps.bnns_graph_available = false;
            caps.realtime_characteristics_supported = false;
        }

        caps
    }

    /// Returns true if Accelerate is available on this platform.
    pub fn is_available(&self) -> bool {
        matches!(
            self.accelerate_available,
            AccelerateAvailability::Available | AccelerateAvailability::Limited
        )
    }

    /// Returns true if the given subsystem is enabled.
    pub fn is_subsystem_enabled(&self, subsystem: AccelerateSubsystem) -> bool {
        self.enabled_subsystems.contains(&subsystem)
    }

    /// Returns true if the given dtype is supported.
    pub fn is_dtype_supported(&self, dtype: AccelerateDType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Returns true if the given layout is supported.
    pub fn is_layout_supported(&self, layout: AccelerateLayout) -> bool {
        self.supported_layouts.contains(&layout)
    }

    /// Returns the v0 capabilities (portable subset).
    pub fn v0_portable() -> Self {
        let mut caps = Self::new();
        caps.accelerate_available = AccelerateAvailability::Unavailable;
        caps.enabled_subsystems = vec![AccelerateSubsystem::Reference];
        caps.supported_dtypes = AccelerateDType::v0_supported().to_vec();
        caps.supported_layouts = vec![AccelerateLayout::RowMajor, AccelerateLayout::Contiguous];
        caps.threading_policy = ThreadingPolicy::MultiThreaded;
        caps.bnns_graph_available = false;
        caps.realtime_characteristics_supported = false;
        caps
    }

    /// Detects the OS version string.
    fn detect_os_version() -> String {
        #[cfg(target_os = "macos")]
        {
            // Try to get macOS version - for now return a placeholder
            // In a real implementation, this would use sysctl or similar
            "15.0.0".to_string()
        }
        #[cfg(not(target_os = "macos"))]
        {
            "unknown".to_string()
        }
    }

    /// Returns the current timestamp.
    fn current_timestamp() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    }

    /// Returns a summary of the capabilities as a string.
    pub fn summary(&self) -> String {
        format!(
            "Accelerate Backend Capabilities: platform={}, arch={}, available={}, subsystems={}, dtypes={}, layouts={}, bnns_graph={}, realtime={}",
            self.platform,
            self.architecture,
            self.accelerate_available,
            self.enabled_subsystems.len(),
            self.supported_dtypes.len(),
            self.supported_layouts.len(),
            self.bnns_graph_available,
            self.realtime_characteristics_supported
        )
    }
}

impl AccelerateAvailability {
    /// Detects Accelerate availability based on platform.
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            AccelerateAvailability::Available
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateAvailability::Unavailable
        }
    }
}

impl Default for AccelerateBackendCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_display() {
        assert_eq!(AppleArchitecture::X86_64.to_string(), "x86_64");
        assert_eq!(AppleArchitecture::Arm64.to_string(), "arm64");
        assert_eq!(AppleArchitecture::Arm64e.to_string(), "arm64e");
        assert_eq!(AppleArchitecture::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_architecture_predicates() {
        assert!(AppleArchitecture::Arm64.is_apple_silicon());
        assert!(AppleArchitecture::Arm64e.is_apple_silicon());
        assert!(!AppleArchitecture::X86_64.is_apple_silicon());

        assert!(AppleArchitecture::X86_64.is_intel());
        assert!(!AppleArchitecture::Arm64.is_intel());
    }

    #[test]
    fn test_platform_display() {
        assert_eq!(ApplePlatform::Macos.to_string(), "macos");
        assert_eq!(ApplePlatform::Ios.to_string(), "ios");
        assert_eq!(ApplePlatform::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_availability_display() {
        assert_eq!(AccelerateAvailability::Available.to_string(), "available");
        assert_eq!(AccelerateAvailability::Limited.to_string(), "limited");
        assert_eq!(AccelerateAvailability::Unavailable.to_string(), "unavailable");
    }

    #[test]
    fn test_capabilities_new() {
        let caps = AccelerateBackendCapabilities::new();
        assert!(!caps.mission.is_empty());
        assert!(!caps.os_version.is_empty());
        assert!(!caps.detection_timestamp.is_empty());
    }

    #[test]
    fn test_capabilities_detect() {
        let caps = AccelerateBackendCapabilities::detect();
        
        // On macOS, should be available
        if cfg!(target_os = "macos") {
            assert!(caps.is_available());
            assert!(caps.accelerate_available == AccelerateAvailability::Available);
        } else {
            assert!(!caps.is_available());
            assert!(caps.accelerate_available == AccelerateAvailability::Unavailable);
        }
    }

    #[test]
    fn test_capabilities_v0_portable() {
        let caps = AccelerateBackendCapabilities::v0_portable();
        assert!(!caps.is_available());
        assert_eq!(caps.accelerate_available, AccelerateAvailability::Unavailable);
        assert_eq!(caps.enabled_subsystems.len(), 1);
        assert!(caps.enabled_subsystems.contains(&AccelerateSubsystem::Reference));
    }

    #[test]
    fn test_capabilities_subsystem_check() {
        let caps = AccelerateBackendCapabilities::detect();
        
        if caps.is_available() {
            assert!(caps.is_subsystem_enabled(AccelerateSubsystem::Blas));
            assert!(caps.is_subsystem_enabled(AccelerateSubsystem::Vdsp));
        } else {
            assert!(!caps.is_subsystem_enabled(AccelerateSubsystem::Blas));
            assert!(caps.is_subsystem_enabled(AccelerateSubsystem::Reference));
        }
    }

    #[test]
    fn test_capabilities_dtype_check() {
        let caps = AccelerateBackendCapabilities::detect();
        
        if caps.is_available() {
            assert!(caps.is_dtype_supported(AccelerateDType::F32));
            assert!(caps.is_dtype_supported(AccelerateDType::F64));
        } else {
            // Portable mode should still support f32
            assert!(caps.is_dtype_supported(AccelerateDType::F32));
        }
    }

    #[test]
    fn test_capabilities_summary() {
        let caps = AccelerateBackendCapabilities::detect();
        let summary = caps.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Accelerate Backend Capabilities"));
    }

    #[test]
    fn test_availability_detect() {
        let avail = AccelerateAvailability::detect();
        
        #[cfg(target_os = "macos")]
        {
            assert_eq!(avail, AccelerateAvailability::Available);
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(avail, AccelerateAvailability::Unavailable);
        }
    }
}
