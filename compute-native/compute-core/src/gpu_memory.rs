/// macOS GPU memory ceiling unlock utility.
///
/// macOS limits GPU-accessible wired memory to ~75% of total RAM by default.
/// On a 16GB M1, that's ~11.4GB.  The sysctl `iogpu.wired_lwm_mb` raises this
/// ceiling without requiring SIP bypass.
///
/// # Safety
///
/// `increase_limit` requires root (sudo).  All other operations are read-only
/// and safe.
use std::process::Command;

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// GPU memory configuration (macOS-specific).
///
/// Manages the `iogpu.wired_lwm_mb` sysctl that controls the maximum
/// amount of wired memory the GPU can use.  Default is ~75% of total RAM.
pub struct GpuMemoryConfig {
    /// Current wired limit in MB.
    pub current_limit_mb: u32,
    /// Total physical RAM in MB.
    pub total_ram_mb: u32,
}

impl GpuMemoryConfig {
    /// Read the current GPU wired memory limit and total physical RAM.
    pub fn new() -> Result<Self, String> {
        let current_limit_mb = get_current_wired_limit_mb()?;
        let total_ram_mb = total_physical_ram_mb();
        Ok(Self {
            current_limit_mb,
            total_ram_mb,
        })
    }

    /// Increase the GPU wired memory limit.
    ///
    /// `target_mb` — desired limit in MB.  Pass `0` to use the max reasonable
    /// value (total RAM minus 2.5 GB for OS overhead).
    ///
    /// Requires root (`sudo`) on macOS.
    pub fn increase_limit(target_mb: u32) -> Result<u32, String> {
        let total_ram_mb = total_physical_ram_mb();
        let actual = if target_mb == 0 {
            Self::max_recommended_mb(total_ram_mb)
        } else {
            target_mb
        };
        increase_wired_limit_mb(actual)
    }

    /// Get the max recommended limit: total RAM minus 2.5 GB (for OS).
    pub fn max_recommended_mb(total_ram_mb: u32) -> u32 {
        total_ram_mb.saturating_sub(2560)
    }

    /// Check if the current wired limit is sufficient for a given model.
    ///
    /// `model_required_mb` — approximate memory needed for model + KV cache,
    /// in MB.
    pub fn is_sufficient(&self, model_required_mb: u32) -> bool {
        self.current_limit_mb >= model_required_mb
    }

    /// Print system memory diagnostics.
    pub fn diagnostics(&self) -> String {
        format!(
            "GPU wired limit: {} MB\nTotal RAM: {} MB\nMax recommended: {} MB\nSufficient for 12 GB model: {}",
            self.current_limit_mb,
            self.total_ram_mb,
            Self::max_recommended_mb(self.total_ram_mb),
            self.is_sufficient(12_000),
        )
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Read the current `iogpu.wired_lwm_mb` sysctl value.
pub fn get_current_wired_limit_mb() -> Result<u32, String> {
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("iogpu.wired_lwm_mb")
        .output()
        .map_err(|e| format!("Failed to execute sysctl: {}", e))?;

    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        return Err("sysctl returned empty output for iogpu.wired_lwm_mb".into());
    }
    s.parse::<u32>()
        .map_err(|e| format!("Failed to parse wired limit '{}': {}", s, e))
}

/// Increase the GPU wired memory limit via `sudo sysctl`.
///
/// Returns the new limit on success, or an error describing why it failed.
/// The error message includes a hint to run the command manually when sudo
/// is unavailable.
pub fn increase_wired_limit_mb(target_mb: u32) -> Result<u32, String> {
    let output = Command::new("sudo")
        .arg("sysctl")
        .arg(format!("iogpu.wired_lwm_mb={}", target_mb))
        .output()
        .map_err(|e| format!("Failed to execute sudo sysctl: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "sysctl failed (try running manually with sudo): {}\n\
             Manually: sudo sysctl iogpu.wired_lwm_mb={}",
            stderr.trim(),
            target_mb,
        ));
    }

    get_current_wired_limit_mb()
}

/// Get total physical RAM in MB.
pub fn total_physical_ram_mb() -> u32 {
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok();
    output
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            s.parse::<u64>().ok()
        })
        .map(|b| (b / 1_048_576) as u32)
        .unwrap_or(16384) // fallback for 16 GB M1
}

/// Maximum recommended wired limit: total RAM minus 2.5 GB for OS overhead.
pub fn max_recommended_mb() -> u32 {
    total_physical_ram_mb().saturating_sub(2560)
}
