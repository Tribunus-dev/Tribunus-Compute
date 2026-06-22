//! Backend-neutral kernel source providers.
//!
//! The compiler asks a chain of `KernelProvider` implementations for a compiled
//! Metal kernel artifact. Each provider tries to satisfy the request; the first
//! that succeeds wins.
//!
//! Current providers (in priority order):
//!
//! 1. `TribunusNativeProvider` — generates kernels from Tribunus-owned Metal
//!    source (e.g., NF4 NormalFloat4 codebook). Source: custom .metal files
//!    compiled offline via `xcrun metal`.
//!
//! 2. (Future) `LlamaMetalProvider` — generates Metal source from cherry-picked
//!    llama.cpp kernel templates for GGUF quantizations MLX does not support.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use crate::compute_image::manifest::MetalKernelArtifact;
use std::fs;
use std::io::Write;
use std::process::Command;

// ── Provider trait ──────────────────────────────────────────

/// A request the compiler makes of a kernel provider.
#[derive(Debug, Clone)]
pub struct KernelRequest {
    /// Logical operation name (e.g., "qmatmul", "rms_norm").
    pub operation: String,
    /// Logical shapes of the input tensors.
    pub logical_shapes: Vec<Vec<u32>>,
    /// Storage dtype string (e.g., "U32", "F32", "U8").
    pub storage_dtype: String,
    /// Quantization mode description.
    pub quant_mode: QuantModeDescription,
    /// Block quantization group size (e.g., 128 for NF4).
    pub group_size: u32,
    /// Quantization bits (4 for NF4, 8 for AF8).
    pub bits: u8,
    /// Target GPU family string (e.g., "m1", "m2", "m3").
    pub gpu_family: String,
}

/// Description of the quantization format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantModeDescription {
    /// MLX affine signed-int4: deq = scale * sign_ext(nibble) + bias
    Affine,
    /// NormalFloat4 codebook: deq = scale * nf4_codebook[nibble] + bias
    Nf4,
    /// 8-bit affine: deq = scale * int8 + bias
    Af8,
    /// MLX NVFP4: NVIDIA FP4 e2m1 decode
    Nvfp4,
    /// MLX MXFP4: 32-bit block FP4
    Mxfp4,
    /// MLX MXFP8: 32-bit block FP8
    Mxfp8,
    /// Fallback: unknown or generic
    Unknown(String),
}

/// A compiled kernel artifact from a provider.
#[derive(Debug, Clone)]
pub struct ProvidedKernel {
    /// Raw .metallib bytes.
    pub metallib_bytes: Vec<u8>,
    /// BLAKE3 hash of the metallib bytes.
    pub metallib_blake3: String,
    /// Entry point function name.
    pub entry_point: String,
    /// Buffer slot map: name → buffer index.
    pub buffer_slot_map: HashMap<String, u32>,
    /// Scalar slot map: name → (buffer index, type string).
    pub scalar_slot_map: HashMap<String, (u32, String)>,
    /// Threadgroup grid dimensions.
    pub threadgroups_per_grid: [u32; 3],
    /// Threads per threadgroup.
    pub threads_per_threadgroup: [u32; 3],
    /// Kernel ABI version.
    pub kernel_abi_version: u32,
}

/// A backend-neutral kernel source provider.
pub trait KernelProvider {
    /// Try to compile a kernel for the given request.
    /// Returns `Ok(Some(artifact))` on success,
    /// `Ok(None)` if this provider cannot handle the request,
    /// or `Err(e)` on compilation failure.
    fn compile(
        &self,
        request: &KernelRequest,
        output_dir: &Path,
        artifact_id: &str,
        metallib_relpath: &Path,
    ) -> Result<Option<ProvidedKernel>, Box<dyn Error>>;

    /// Human-readable provider name for diagnostics.
    fn name(&self) -> &'static str;

    /// Priority: lower runs first. Default priority order:
    /// 10 = TribunusNativeProvider, 20 = (future) LlamaMetalProvider.
    fn priority(&self) -> u32 {
        50
    }
}

/// Apply a chain of providers. Returns the first successful result.
pub fn compile_with_providers(
    request: &KernelRequest,
    output_dir: &Path,
    artifact_id: &str,
    metallib_relpath: &Path,
    providers: &[&dyn KernelProvider],
) -> Result<ProvidedKernel, Box<dyn Error>> {
    let mut sorted: Vec<&dyn KernelProvider> = providers.to_vec();
    sorted.sort_by_key(|p| p.priority());

    for provider in &sorted {
        eprintln!(
            "[kernel-provider] trying {} (priority {}) for {}",
            provider.name(),
            provider.priority(),
            artifact_id
        );
        match provider.compile(request, output_dir, artifact_id, metallib_relpath) {
            Ok(Some(artifact)) => {
                eprintln!(
                    "[kernel-provider] {} produced artifact: {}  ({} bytes)",
                    provider.name(),
                    artifact.entry_point,
                    artifact.metallib_bytes.len(),
                );
                return Ok(artifact);
            }
            Ok(None) => {
                // Provider declined — try next.
                continue;
            }
            Err(e) => {
                eprintln!("[kernel-provider] {} failed: {}", provider.name(), e);
                // Try next provider on compilation failure too.
                continue;
            }
        }
    }

    Err(format!(
        "no provider could compile kernel '{}' (op={}, mode={:?}, bits={}, gs={})",
        artifact_id, request.operation, request.quant_mode, request.bits, request.group_size,
    )
    .into())
}

// ── MLX Source Provider ───────────────────────────────────────────
// ── Tribunus Native Provider ──────────────────────────────────────
/// Generates kernels from Tribunus-owned Metal source.
/// Handles NF4 (NormalFloat4 codebook) — the MLX provider cannot.
pub struct TribunusNativeProvider;

/// NF4 codebook constant (must match nf4_contract.h).
const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057, 0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];

fn generate_nf4_kernel_source(k: u32, n: u32, group_size: u32) -> String {
    let codebook_str = NF4_CODEBOOK
        .iter()
        .map(|v| format!("{:.6}f", v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

constant float nf4_codebook[16] = {{ {} }};

kernel void quantized_matmul_nf4(
  device const float* input [[buffer(0)]],
  device const uint* weights [[buffer(1)]],
  device const float* scales [[buffer(2)]],
  device const float* biases [[buffer(3)]],
  device float* output [[buffer(4)]],
  uint3 gid [[threadgroup_position_in_grid]],
  uint3 lid [[thread_position_in_threadgroup]])
{{
  uint K = {};
  uint N = {};
  uint group_size = {};
  uint group_stride = K / group_size;
  uint packed_per_col = K / 8;
  uint row = gid.x * 32 + lid.x;
  if (row >= N) return;
  float sum = 0.0;
  for (uint g = 0; g < group_stride; g++) {{
    float scale = scales[row * group_stride + g];
    float bias = biases[row * group_stride + g];
    for (uint v = 0; v < group_size / 8; v++) {{
      uint word_idx = row * packed_per_col + g * (group_size / 8) + v;
      uint packed = weights[word_idx];
      for (uint nibble = 0; nibble < 8; nibble++) {{
        uint idx = (packed >> (nibble * 4)) & 0xF;
        float deq = scale * nf4_codebook[idx] + bias;
        uint in_idx = g * group_size + v * 8 + nibble;
        if (in_idx < K) {{
          sum += input[in_idx] * deq;
        }}
      }}
    }}
  }}
  output[row] = sum;
}}"#,
        codebook_str, k, n, group_size
    )
}

fn compile_metal_source_to_file(
    source: &str,
    output_metallib: &Path,
    kernel_name: &str,
) -> Result<(), Box<dyn Error>> {
    let tmp_dir = std::env::temp_dir().join(format!("tribunus-metal-{}", kernel_name));
    fs::create_dir_all(&tmp_dir)?;
    let source_path = tmp_dir.join("kernel.metal");
    let mut f = fs::File::create(&source_path)?;
    f.write_all(source.as_bytes())?;
    drop(f);

    let air_path = tmp_dir.join("kernel.air");
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-std=osx-metal3.2",
            "-std=metal3.2",
            "-O3",
            "-c",
            source_path.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err(format!("metal compilation failed: {:?}", status.code()).into());
    }

    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            output_metallib.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err(format!("metallib failed: {:?}", status.code()).into());
    }
    Ok(())
}

impl KernelProvider for TribunusNativeProvider {
    fn name(&self) -> &'static str {
        "tribunus-native"
    }
    fn priority(&self) -> u32 {
        20
    }

    fn compile(
        &self,
        request: &KernelRequest,
        output_dir: &Path,
        artifact_id: &str,
        metallib_relpath: &Path,
    ) -> Result<Option<ProvidedKernel>, Box<dyn Error>> {
        if request.operation != "qmatmul" {
            return Ok(None);
        }
        let k = request
            .logical_shapes
            .get(0)
            .and_then(|s| s.get(0))
            .copied()
            .unwrap_or(0);
        let n = request
            .logical_shapes
            .get(0)
            .and_then(|s| s.get(1))
            .copied()
            .unwrap_or(0);
        if k == 0 || n == 0 {
            return Ok(None);
        }

        match request.quant_mode {
            QuantModeDescription::Nf4 => {}
            _ => return Ok(None),
        }

        let metallib_path = output_dir.join(metallib_relpath);
        if let Some(parent) = metallib_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let source = generate_nf4_kernel_source(k, n, request.group_size);
        let entry_point = "quantized_matmul_nf4";
        compile_metal_source_to_file(&source, &metallib_path, entry_point)?;
        let metallib_bytes = fs::read(&metallib_path)?;
        let blake3 = blake3::hash(&metallib_bytes).to_hex().to_string();

        let mut buf_map = HashMap::new();
        buf_map.insert("input".to_string(), 0u32);
        buf_map.insert("weight".to_string(), 1u32);
        buf_map.insert("scale".to_string(), 2u32);
        buf_map.insert("bias".to_string(), 3u32);
        buf_map.insert("output".to_string(), 4u32);

        let scalar_map = HashMap::new();

        let grid_x = (n + 31) / 32;

        Ok(Some(ProvidedKernel {
            metallib_bytes,
            metallib_blake3: blake3,
            entry_point: entry_point.to_string(),
            buffer_slot_map: buf_map,
            scalar_slot_map: scalar_map,
            threadgroups_per_grid: [grid_x, 1, 1],
            threads_per_threadgroup: [32, 1, 1],
            kernel_abi_version: 1,
        }))
    }
}

/// Build a MetalKernelArtifact from a ProvidedKernel + metadata.
pub fn provided_to_artifact(
    provided: ProvidedKernel,
    artifact_id: String,
    logical_operation: String,
    logical_shape: Vec<u32>,
    storage_shape: Vec<u32>,
    bits: u8,
    group_size: u32,
    scale_tensor: String,
    bias_tensor: String,
) -> MetalKernelArtifact {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(artifact_id.as_bytes());
    hasher.update(
        &logical_shape
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    hasher.update(
        &storage_shape
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    hasher.update(&[bits, group_size as u8]);
    hasher.update(&provided.metallib_bytes);
    let checksum = format!("{:x}", hasher.finalize());

    MetalKernelArtifact {
        artifact_id,
        logical_operation,
        kind: crate::compute_image::manifest::ArtifactKind::MlxNf4U32,
        metallib_relpath: String::new(), // caller sets this
        metallib_blake3: provided.metallib_blake3,
        metallib_byte_length: provided.metallib_bytes.len() as u64,
        dispatch: crate::compute_image::manifest::MetalDispatchRecipe {
            entry_point: provided.entry_point,
            kernel_name: String::new(),
            threads_per_threadgroup: provided.threads_per_threadgroup,
            threadgroups_per_grid: provided.threadgroups_per_grid,
            buffer_slot_map: provided.buffer_slot_map,
            scalar_index_map: provided.scalar_slot_map,
            k: logical_shape.get(0).copied().unwrap_or(0) as u64,
            n: logical_shape.get(1).copied().unwrap_or(0) as u64,
            group_size,
            bits,
            kernel_abi_version: provided.kernel_abi_version,
        },
        logical_shape,
        storage_shape,
        bits,
        group_size,
        scale_tensor,
        bias_tensor,
        gpu_family: String::new(),
        checksum,
    }
}
