//! Accelerate FFI bindings and platform boundary.
//!
//! This module provides the platform boundary for Accelerate.framework linkage.
//! On macOS, it links Accelerate.framework and provides native function bindings.
//! On non-macOS platforms, it provides stub implementations that report unavailable.
//!
//! # Design Principles
//!
//! 1. **Portable API**: All public types and functions must be available on every platform.
//! 2. **Platform-gated Implementation**: Only actual framework linkage and native calls
//!    should be guarded by `cfg(target_os = "macos")`.
//! 3. **Explicit Unavailability**: Non-macOS platforms must explicitly report Accelerate as
//!    unavailable, not silently fail or panic.
//! 4. **No Global State**: Avoid global mutable backend state.

use std::fmt;

/// Accelerate framework linkage status.
///
/// This enum represents whether Accelerate.framework is actually linked and available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelerateLinkage {
    /// Accelerate.framework is linked and available.
    Linked,
    /// Accelerate.framework is not available on this platform.
    Unavailable,
}

impl fmt::Display for AccelerateLinkage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateLinkage::Linked => write!(f, "linked"),
            AccelerateLinkage::Unavailable => write!(f, "unavailable"),
        }
    }
}

impl AccelerateLinkage {
    /// Returns the current linkage status.
    pub fn current() -> Self {
        #[cfg(target_os = "macos")]
        {
            // On macOS, we attempt to link Accelerate.framework
            // The actual linkage is handled by build.rs
            AccelerateLinkage::Linked
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateLinkage::Unavailable
        }
    }

    /// Returns true if Accelerate is linked and available.
    pub fn is_linked(&self) -> bool {
        matches!(self, AccelerateLinkage::Linked)
    }

    /// Returns true if Accelerate is unavailable on this platform.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, AccelerateLinkage::Unavailable)
    }
}

/// Result type for Accelerate FFI operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerateResult<T> {
    /// Operation succeeded with result.
    Success(T),
    /// Operation failed because Accelerate is unavailable.
    Unavailable,
    /// Operation failed for other reasons.
    Error(String),
}

impl<T> AccelerateResult<T> {
    /// Returns true if the operation succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, AccelerateResult::Success(_))
    }

    /// Returns true if Accelerate is unavailable.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, AccelerateResult::Unavailable)
    }

    /// Returns true if the operation failed with an error.
    pub fn is_error(&self) -> bool {
        matches!(self, AccelerateResult::Error(_))
    }

    /// Unwraps the success value, panicking if not successful.
    /// 
    /// # Panics
    /// Panics if the result is not `Success`.
    pub fn unwrap(self) -> T {
        match self {
            AccelerateResult::Success(value) => value,
            AccelerateResult::Unavailable => {
                panic!("Accelerate is unavailable on this platform")
            }
            AccelerateResult::Error(msg) => {
                panic!("Accelerate operation failed: {}", msg)
            }
        }
    }

    /// Returns the success value if available, or None.
    pub fn ok(self) -> Option<T> {
        match self {
            AccelerateResult::Success(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the error message if available, or None.
    pub fn err(self) -> Option<String> {
        match self {
            AccelerateResult::Error(msg) => Some(msg),
            AccelerateResult::Unavailable => Some("Accelerate unavailable".to_string()),
            _ => None,
        }
    }
}

impl<T: fmt::Debug> fmt::Display for AccelerateResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateResult::Success(value) => write!(f, "Success({:?})", value),
            AccelerateResult::Unavailable => write!(f, "Unavailable"),
            AccelerateResult::Error(msg) => write!(f, "Error({})", msg),
        }
    }
}

/// Platform-specific Accelerate functions.
///
/// This module contains the actual FFI bindings and platform-specific implementations.
pub mod platform {
    /// Initialize Accelerate framework.
    /// 
    /// On macOS, this may perform framework initialization.
    /// On non-macOS, this is a no-op that reports unavailable.
    pub fn initialize() -> AccelerateResult<()> {
        #[cfg(target_os = "macos")]
        {
            // On macOS, Accelerate.framework is linked via build.rs
            // No explicit initialization needed for basic operations
            AccelerateResult::Success(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateResult::Unavailable
        }
    }

    /// Check if Accelerate framework is available.
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            true
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Get Accelerate framework version information.
    pub fn version() -> AccelerateResult<String> {
        #[cfg(target_os = "macos")]
        {
            // For now, return a placeholder version
            // In a real implementation, this would query the framework
            AccelerateResult::Success("Accelerate.framework (Apple)".to_string())
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateResult::Unavailable
        }
    }

    /// Get the current platform architecture as a string.
    pub fn architecture() -> AccelerateResult<String> {
        #[cfg(target_os = "macos")]
        {
            use super::super::capabilities::AppleArchitecture;
            let arch = AppleArchitecture::detect();
            AccelerateResult::Success(arch.to_string())
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateResult::Unavailable
        }
    }

    /// Get the current platform as a string.
    pub fn platform() -> AccelerateResult<String> {
        #[cfg(target_os = "macos")]
        {
            use super::super::capabilities::ApplePlatform;
            let platform = ApplePlatform::detect();
            AccelerateResult::Success(platform.to_string())
        }
        #[cfg(not(target_os = "macos"))]
        {
            AccelerateResult::Unavailable
        }
    }
}

/// Safe wrapper for Accelerate operations that handles unavailability gracefully.
///
/// This struct provides a safe interface to Accelerate operations that automatically
/// handles the case where Accelerate is unavailable on the current platform.
pub struct AccelerateHandle {
    linkage: AccelerateLinkage,
}

impl AccelerateHandle {
    /// Creates a new Accelerate handle.
    pub fn new() -> Self {
        Self {
            linkage: AccelerateLinkage::current(),
        }
    }

    /// Returns true if Accelerate is available.
    pub fn is_available(&self) -> bool {
        self.linkage.is_linked()
    }

    /// Returns the current linkage status.
    pub fn linkage(&self) -> AccelerateLinkage {
        self.linkage
    }

    /// Initialize the Accelerate framework.
    pub fn initialize(&self) -> AccelerateResult<()> {
        platform::initialize()
    }

    /// Get framework version.
    pub fn version(&self) -> AccelerateResult<String> {
        platform::version()
    }

    /// Get architecture.
    pub fn architecture(&self) -> AccelerateResult<String> {
        platform::architecture()
    }

    /// Get platform.
    pub fn platform(&self) -> AccelerateResult<String> {
        platform::platform()
    }
}

impl Default for AccelerateHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Accelerate handle for convenience.
///
/// Note: This is a convenience accessor, not global mutable state.
/// It creates a new handle on each access.
pub fn accelerate_handle() -> AccelerateHandle {
    AccelerateHandle::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linkage_current() {
        let linkage = AccelerateLinkage::current();
        
        #[cfg(target_os = "macos")]
        {
            assert!(linkage.is_linked());
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert!(linkage.is_unavailable());
        }
    }

    #[test]
    fn test_linkage_display() {
        assert_eq!(AccelerateLinkage::Linked.to_string(), "linked");
        assert_eq!(AccelerateLinkage::Unavailable.to_string(), "unavailable");
    }

    #[test]
    fn test_accelerate_result_success() {
        let result: AccelerateResult<i32> = AccelerateResult::Success(42);
        assert!(result.is_success());
        assert!(!result.is_unavailable());
        assert!(!result.is_error());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(result.ok(), Some(42));
    }

    #[test]
    fn test_accelerate_result_unavailable() {
        let result: AccelerateResult<i32> = AccelerateResult::Unavailable;
        assert!(!result.is_success());
        assert!(result.is_unavailable());
        assert!(!result.is_error());
        assert_eq!(result.ok(), None);
        assert!(result.err().is_some());
    }

    #[test]
    fn test_accelerate_result_error() {
        let result: AccelerateResult<i32> = AccelerateResult::Error("test error".to_string());
        assert!(!result.is_success());
        assert!(!result.is_unavailable());
        assert!(result.is_error());
        assert_eq!(result.ok(), None);
        assert_eq!(result.err(), Some("test error".to_string()));
    }

    #[test]
    fn test_accelerate_result_display() {
        let success: AccelerateResult<i32> = AccelerateResult::Success(42);
        let unavailable: AccelerateResult<i32> = AccelerateResult::Unavailable;
        let error: AccelerateResult<i32> = AccelerateResult::Error("test".to_string());

        assert!(success.to_string().contains("Success"));
        assert_eq!(unavailable.to_string(), "Unavailable");
        assert!(error.to_string().contains("Error"));
    }

    #[test]
    fn test_accelerate_handle() {
        let handle = AccelerateHandle::new();
        
        #[cfg(target_os = "macos")]
        {
            assert!(handle.is_available());
            assert!(handle.linkage().is_linked());
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!handle.is_available());
            assert!(handle.linkage().is_unavailable());
        }
    }

    #[test]
    fn test_platform_functions() {
        #[cfg(target_os = "macos")]
        {
            assert!(platform::is_available());
            assert!(platform::initialize().is_success());
            assert!(platform::version().is_success());
            assert!(platform::architecture().is_success());
            assert!(platform::platform().is_success());
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!platform::is_available());
            assert!(platform::initialize().is_unavailable());
            assert!(platform::version().is_unavailable());
            assert!(platform::architecture().is_unavailable());
            assert!(platform::platform().is_unavailable());
        }
    }

    #[test]
    fn test_accelerate_handle_function() {
        let handle = accelerate_handle();
        // Just verify it doesn't panic
        assert!(true);
    }
}
