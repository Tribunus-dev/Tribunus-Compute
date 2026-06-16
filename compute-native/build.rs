//! Build script for Accelerate backend.
//!
//! This build script handles platform-specific configuration for the Accelerate backend.
//! On macOS, it links Accelerate.framework. On other platforms, it provides stub implementations.

use std::env;
use std::path::Path;

fn main() {
    // Detect the target OS
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/backend/accelerate/");

    if target_os == "macos" {
        // On macOS, link Accelerate.framework
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Accelerate");
        
        // Also link other related frameworks that Accelerate might depend on
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Foundation");
        
        // Set environment variable to indicate macOS availability
        println!("cargo:rustc-env=ACCELERATE_AVAILABLE=1");
        
        // Detect architecture for capability reporting
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
        println!("cargo:rustc-env=ACCELERATE_TARGET_ARCH={}", target_arch);
        
        println!("Building Accelerate backend for macOS ({})", target_arch);
    } else {
        // On non-macOS platforms, set environment variable to indicate unavailability
        println!("cargo:rustc-env=ACCELERATE_AVAILABLE=0");
        println!("cargo:rustc-env=ACCELERATE_TARGET_ARCH=unknown");
        
        println!("Building Accelerate backend stubs for {}", target_os);
    }
    
    // Always emit a marker that the build script ran
    println!("cargo:rustc-env=ACCELERATE_BUILD_SCRIPT_RAN=1");
}
