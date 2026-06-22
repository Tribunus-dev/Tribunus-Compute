//! ComputeImage compile pipeline — backend-agnostic compilation trait.
//! Each backend (MLX, Candle, Tensix) implements CompileTarget to produce
//! its native compute image format from a common model source.

/// The compiled output artifact for a specific backend target.
pub trait CompileTarget: Send + Sync {
    /// The target identifier (e.g. "mlx", "candle-cpu", "tensix-wormhole").
    fn name(&self) -> &str;

    /// Compile a loaded model source into a compute image.
    fn compile(
        &self,
        source: &super::compile::LoadedSource,
        config: &CompileConfig,
    ) -> crate::Result<Box<dyn std::any::Any + Send>>;

    /// Backend capabilities that affect compilation (e.g., tile size, memory limits).
    fn capabilities(&self) -> TargetCapabilities;
}

/// Configuration for a compile run.
#[derive(Clone, Debug)]
pub struct CompileConfig {
    pub quantize_mode: Option<CompileQuantMode>,
    pub hardware_target: Option<HardwareTarget>,
    pub skip_validation: bool,
    pub output_dir: String,
}

/// Declared capabilities of a compilation target.
#[derive(Clone, Debug)]
pub struct TargetCapabilities {
    pub name: String,
    pub target_type: TargetType,
    pub native_tile_size: u32,
    pub requires_qk_norm: bool,
    pub supports_bias: bool,
    pub max_dram_bytes: u64,
    pub max_sram_per_core: u64,
}

/// Kind of compilation target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetType {
    MlxMetal,
    CandleCpu,
    TensixTensix,
    IntelLevelZero,
}

/// Quantization mode selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileQuantMode {
    Nf4,
    Nf4_128,
    Af8,
}

/// Hardware target specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareTarget {
    AppleM1,
    AppleM1Pro,
    AppleM2,
    AppleM2Ultra,
    AppleM3Ultra,
    LinuxX86,
    TensixWormhole,
    TensixBlackhole,
    IntelLevelZero,
}

impl TargetCapabilities {
    /// Capabilities for the MLX Metal backend.
    pub fn mlx_metal() -> Self {
        TargetCapabilities {
            name: "mlx-metal".into(),
            target_type: TargetType::MlxMetal,
            native_tile_size: 32,
            requires_qk_norm: true, // Qwen2-style models
            supports_bias: true,
            max_dram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB unified
            max_sram_per_core: 0,                    // unified memory, no SRAM
        }
    }

    /// Capabilities for the Candle CPU backend.
    pub fn candle_cpu() -> Self {
        TargetCapabilities {
            name: "candle-cpu".into(),
            target_type: TargetType::CandleCpu,
            native_tile_size: 1, // no native tiling, CPU does arbitrary shapes
            requires_qk_norm: false,
            supports_bias: true,
            max_dram_bytes: 64 * 1024 * 1024 * 1024, // system RAM
            max_sram_per_core: 0,
        }
    }

    /// Capabilities for the Tenstorrent Tensix backend.
    pub fn tensix(arch: &str) -> Self {
        TargetCapabilities {
            name: format!("tensix-{}", arch),
            target_type: TargetType::TensixTensix,
            native_tile_size: 32,
            requires_qk_norm: false,
            supports_bias: false,
            max_dram_bytes: 128 * 1024 * 1024 * 1024,
            max_sram_per_core: 128 * 1024,
        }
    }

    /// Capabilities for the Intel Level Zero backend.
    pub fn intel_level_zero() -> Self {
        TargetCapabilities {
            name: "intel-level-zero".into(),
            target_type: TargetType::IntelLevelZero,
            native_tile_size: 32,
            requires_qk_norm: false,
            supports_bias: true,
            max_dram_bytes: 16 * 1024 * 1024 * 1024,
            max_sram_per_core: 64 * 1024,
        }
    }
}

/// Import common types used by the pipeline.
pub use crate::config::HardwareTarget as TargetHardware; // compat alias
