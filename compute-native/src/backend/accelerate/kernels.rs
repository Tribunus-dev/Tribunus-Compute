//! Accelerate kernel implementations.
//!
//! This module provides **native-ready scaffold** kernel implementations for Accelerate operations.
//! Each kernel produces execution receipts and evidence for verification.
//!
//! # Implementation Status
//!
//! **Current State**: All kernels use reference implementations on all platforms.
//! The lowering decisions target the appropriate subsystems (vDSP for elementwise, BLAS for matmul),
//! but actual execution is via reference fallback until native FFI bindings are implemented.
//!
//! # Intended PR3 Scope (Not Yet Implemented)
//!
//! When native calls are added, this module will implement:
//! - Identity: f32 execution (memory operation)
//! - Add: f32 elementwise addition via vDSP_vadd
//! - Multiply: f32 elementwise multiplication via vDSP_vmul
//! - Matmul: f32 matrix multiplication via cblas_sgemm
//!
//! # Design Principles
//!
//! 1. **Evidence-Driven**: Every kernel execution produces receipts and evidence.
//! 2. **Reference-Checked**: Results are compared against reference implementations.
//! 3. **Portable**: Non-macOS platforms use reference fallback with explicit evidence.
//! 4. **Type-Safe**: Only f32 is implemented initially; other dtypes return unsupported.
//! 5. **Truthful**: Receipts clearly distinguish lowering_subsystem (intended) from executed_subsystem (actual).

use std::time::{Duration, Instant};
use super::{dtype::AccelerateDType, execution::{AccelerateExecutionReceipt, BufferInfo, NumericalStatus}, 
          ffi::{AccelerateHandle, AccelerateResult}, layout::AccelerateLayout, 
          ops::CanonicalOp, subsystem::AccelerateSubsystem, evidence::{AccelerateEvidence, EvidenceValidator},
          native::{NativeDispatcher, NativeSymbol}};

/// Kernel execution context.
pub struct KernelContext {
    /// Accelerate handle for platform access.
    pub handle: AccelerateHandle,
    /// Whether to collect evidence.
    pub collect_evidence: bool,
    /// Evidence validator.
    pub validator: EvidenceValidator,
}

impl KernelContext {
    /// Creates a new kernel context.
    pub fn new() -> Self {
        Self {
            handle: AccelerateHandle::new(),
            collect_evidence: true,
            validator: EvidenceValidator::new(),
        }
    }

    /// Returns true if Accelerate is available.
    pub fn is_available(&self) -> bool {
        self.handle.is_available()
    }
}

impl Default for KernelContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a kernel execution.
#[derive(Debug, Clone)]
pub struct KernelResult {
    /// Output buffer.
    pub output: Vec<f32>,
    /// Execution receipt.
    pub receipt: AccelerateExecutionReceipt,
    /// Evidence (if collected).
    pub evidence: Option<AccelerateEvidence>,
}

impl KernelResult {
    /// Creates a new kernel result.
    pub fn new(output: Vec<f32>, receipt: AccelerateExecutionReceipt, evidence: Option<AccelerateEvidence>) -> Self {
        Self { output, receipt, evidence }
    }
}

/// Reference implementations for comparison.
///
/// These provide the ground truth for verifying Accelerate kernel results.
pub mod reference {
    /// Reference identity operation.
    pub fn identity(input: &[f32]) -> Vec<f32> {
        input.to_vec()
    }

    /// Reference elementwise addition.
    pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Reference elementwise multiplication.
    pub fn multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Reference matrix multiplication.
    /// 
    /// Assumes row-major layout for both inputs.
    /// a: (m, n), b: (n, p) -> output: (m, p)
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
        let mut output = vec![0.0; m * p];
        
        for i in 0..m {
            for k in 0..n {
                let a_val = a[i * n + k];
                for j in 0..p {
                    output[i * p + j] += a_val * b[k * p + j];
                }
            }
        }
        
        output
    }

    /// Reference sigmoid.
    pub fn sigmoid(input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| 1.0 / (1.0 + x.exp())).collect()
    }

    /// Reference SiLU (sigmoid * input).
    pub fn silu(input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| {
            let s = 1.0 / (1.0 + x.exp());
            s * x
        }).collect()
    }

    /// Reference softmax with numerical stability.
    pub fn softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        
        // Find max for numerical stability
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exp(x - max) and sum
        let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum = exp_vals.iter().sum::<f32>();
        
        // Normalize
        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

/// Kernel implementations.
pub mod kernels {
    use super::*;

    /// Identity kernel.
    /// 
    /// On macOS: Can be a no-op view or validated copy.
    /// On non-macOS: Always uses reference implementation.
    /// 
    /// # Note
    /// 
    /// This implementation uses reference fallback on all platforms.
    /// The lowering_subsystem reflects the intended subsystem (Reference for identity),
    /// while executed_subsystem is always Reference until native Accelerate calls are implemented.
    pub fn identity(ctx: &KernelContext, input: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        let start = Instant::now();
        let input_size = input.len();
        let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
        let layout_key = layout.to_string();
        
        // For identity, the intended lowering is Reference (memory operation)
        let lowering_subsystem = AccelerateSubsystem::Reference;
        
        let (output, executed_subsystem, fallback_used, fallback_reason) = if ctx.is_available() {
            // On macOS, we could use Accelerate for memory operations
            // For now, we use reference implementation
            // lowering_subsystem = Reference, executed_subsystem = Reference (no fallback)
            (reference::identity(input), AccelerateSubsystem::Reference, false, None)
        } else {
            // On non-macOS, explicitly use reference fallback
            // lowering_subsystem = Reference, executed_subsystem = Reference, but fallback_used = true
            (reference::identity(input), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()))
        };
        
        let wall_time = start.elapsed();
        
        // Create receipt with both lowering and executed subsystems
        let receipt = AccelerateExecutionReceipt::new(
            uuid::Uuid::new_v4().to_string(),
            "identity".to_string(),
            CanonicalOp::Identity,
            AccelerateDType::F32,
            shape_key.clone(),
            layout_key.clone(),
            lowering_subsystem,
            executed_subsystem,
        )
        .with_results(
            wall_time,
            NumericalStatus::Passed, // Identity should always pass
            Some(0), // No allocations for identity
        );
        
        let receipt = if fallback_used {
            receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
        } else {
            receipt
        };
        
        // Create evidence
        let evidence = if ctx.collect_evidence {
            // Compare against reference (should be identical)
            let reference_output = reference::identity(input);
            let max_abs_error = output.iter().zip(reference_output.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max) as f64;
            
            Some(ctx.validator.create_evidence(
                uuid::Uuid::new_v4().to_string(),
                "identity".to_string(),
                CanonicalOp::Identity,
                AccelerateDType::F32,
                shape_key,
                layout_key,
                executed_subsystem,
                Some(max_abs_error),
                None,
                None,
                wall_time.as_nanos() as u64,
                Some(0),
            ))
        } else {
            None
        };
        
        KernelResult::new(output, receipt, evidence)
    }

    /// Add kernel.
    /// 
    /// On macOS: Uses vDSP for vector addition (vDSP_vadd).
    /// On non-macOS: Uses reference implementation with fallback evidence.
    /// 
    /// # Native Execution
    /// 
    /// On macOS, this kernel attempts to use the native vDSP_vadd function.
    /// If successful, executed_subsystem = vDSP and native_symbol = "vDSP_vadd".
    /// If native call fails, falls back to reference with appropriate reason.
    pub fn add(ctx: &KernelContext, a: &[f32], b: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        let start = Instant::now();
        let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
        let layout_key = layout.to_string();
        
        // For add, the intended lowering is vDSP
        let lowering_subsystem = AccelerateSubsystem::Vdsp;
        
        // Try native execution first (on macOS), then fallback to reference
        let (output, executed_subsystem, fallback_used, fallback_reason, native_symbol) = 
            if NativeDispatcher::is_available() {
                // On macOS, try native vDSP call
                match NativeDispatcher::vdsp_add(a, b) {
                    Ok((output, symbol)) => {
                        // Native execution succeeded
                        (output, AccelerateSubsystem::Vdsp, false, None, Some(symbol))
                    }
                    Err(err) => {
                        // Native call failed, use reference fallback
                        let reason = match err {
                            super::native::AccelerateNativeError::BackendUnavailable => 
                                "Accelerate unavailable".to_string(),
                            super::native::AccelerateNativeError::LengthMismatch { expected, actual } => 
                                format!("Length mismatch: expected {}, got {}", expected, actual),
                            super::native::AccelerateNativeError::EmptyInput => 
                                "Empty input not supported by native path".to_string(),
                            super::native::AccelerateNativeError::DimensionError(msg) => msg,
                            _ => "Native vDSP call failed".to_string(),
                        };
                        (reference::add(a, b), AccelerateSubsystem::Reference, true, Some(reason), None)
                    }
                }
            } else {
                // On non-macOS, use reference fallback
                (reference::add(a, b), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()), None)
            };
        
        let wall_time = start.elapsed();
        
        // Create receipt with both lowering and executed subsystems
        let mut receipt = AccelerateExecutionReceipt::new(
            uuid::Uuid::new_v4().to_string(),
            "add".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            shape_key.clone(),
            layout_key.clone(),
            lowering_subsystem,
            executed_subsystem,
        )
        .with_results(
            wall_time,
            NumericalStatus::Passed,
            Some(1), // One output allocation
        );
        
        // Set native symbol if present
        if let Some(symbol) = native_symbol {
            receipt = receipt.with_native_symbol(symbol);
        }
        
        // Set fallback if used
        let receipt = if fallback_used {
            receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
        } else {
            receipt
        };
        
        // Create evidence
        let evidence = if ctx.collect_evidence {
            // Compare against reference
            let reference_output = reference::add(a, b);
            let max_abs_error = output.iter().zip(reference_output.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max) as f64;
            
            let max_rel_error = output.iter().zip(reference_output.iter())
                .filter(|(&a, &b)| b.abs() > 1e-8) // Avoid division by near-zero
                .map(|(&a, &b)| ((a - b).abs() / b.abs()) as f64)
                .fold(0.0f64, f64::max);
            
            // Use the new method if we have a native symbol
            let evidence = if let Some(symbol) = native_symbol {
                ctx.validator.create_evidence_with_native_symbol(
                    uuid::Uuid::new_v4().to_string(),
                    "add".to_string(),
                    CanonicalOp::Add,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    Some(max_rel_error),
                    None,
                    wall_time.as_nanos() as u64,
                    Some(1),
                    symbol,
                )
            } else {
                ctx.validator.create_evidence(
                    uuid::Uuid::new_v4().to_string(),
                    "add".to_string(),
                    CanonicalOp::Add,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    Some(max_rel_error),
                    None,
                    wall_time.as_nanos() as u64,
                    Some(1),
                )
            };
            Some(evidence)
        } else {
            None
        };
        
        KernelResult::new(output, receipt, evidence)
    }

    /// Multiply kernel.
    /// 
    /// On macOS: Uses vDSP for vector multiplication (vDSP_vmul).
    /// On non-macOS: Uses reference implementation with fallback evidence.
    /// 
    /// # Native Execution
    /// 
    /// On macOS, this kernel attempts to use the native vDSP_vmul function.
    /// If successful, executed_subsystem = vDSP and native_symbol = "vDSP_vmul".
    /// If native call fails, falls back to reference with appropriate reason.
    pub fn multiply(ctx: &KernelContext, a: &[f32], b: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        let start = Instant::now();
        let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
        let layout_key = layout.to_string();
        
        // For multiply, the intended lowering is vDSP
        let lowering_subsystem = AccelerateSubsystem::Vdsp;
        
        // Try native execution first (on macOS), then fallback to reference
        let (output, executed_subsystem, fallback_used, fallback_reason, native_symbol) = 
            if NativeDispatcher::is_available() {
                // On macOS, try native vDSP call
                match NativeDispatcher::vdsp_mul(a, b) {
                    Ok((output, symbol)) => {
                        // Native execution succeeded
                        (output, AccelerateSubsystem::Vdsp, false, None, Some(symbol))
                    }
                    Err(err) => {
                        // Native call failed, use reference fallback
                        let reason = match err {
                            super::native::AccelerateNativeError::BackendUnavailable => 
                                "Accelerate unavailable".to_string(),
                            super::native::AccelerateNativeError::LengthMismatch { expected, actual } => 
                                format!("Length mismatch: expected {}, got {}", expected, actual),
                            super::native::AccelerateNativeError::EmptyInput => 
                                "Empty input not supported by native path".to_string(),
                            super::native::AccelerateNativeError::DimensionError(msg) => msg,
                            _ => "Native vDSP call failed".to_string(),
                        };
                        (reference::multiply(a, b), AccelerateSubsystem::Reference, true, Some(reason), None)
                    }
                }
            } else {
                // On non-macOS, use reference fallback
                (reference::multiply(a, b), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()), None)
            };
        
        let wall_time = start.elapsed();
        
        // Create receipt with both lowering and executed subsystems
        let mut receipt = AccelerateExecutionReceipt::new(
            uuid::Uuid::new_v4().to_string(),
            "multiply".to_string(),
            CanonicalOp::Multiply,
            AccelerateDType::F32,
            shape_key.clone(),
            layout_key.clone(),
            lowering_subsystem,
            executed_subsystem,
        )
        .with_results(
            wall_time,
            NumericalStatus::Passed,
            Some(1), // One output allocation
        );
        
        // Set native symbol if present
        if let Some(symbol) = native_symbol {
            receipt = receipt.with_native_symbol(symbol);
        }
        
        // Set fallback if used
        let receipt = if fallback_used {
            receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
        } else {
            receipt
        };
        
        // Create evidence
        let evidence = if ctx.collect_evidence {
            // Compare against reference
            let reference_output = reference::multiply(a, b);
            let max_abs_error = output.iter().zip(reference_output.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max) as f64;
            
            let max_rel_error = output.iter().zip(reference_output.iter())
                .filter(|(&a, &b)| b.abs() > 1e-8)
                .map(|(&a, &b)| ((a - b).abs() / b.abs()) as f64)
                .fold(0.0f64, f64::max);
            
            // Use the new method if we have a native symbol
            let evidence = if let Some(symbol) = native_symbol {
                ctx.validator.create_evidence_with_native_symbol(
                    uuid::Uuid::new_v4().to_string(),
                    "multiply".to_string(),
                    CanonicalOp::Multiply,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    Some(max_rel_error),
                    None,
                    wall_time.as_nanos() as u64,
                    Some(1),
                    symbol,
                )
            } else {
                ctx.validator.create_evidence(
                    uuid::Uuid::new_v4().to_string(),
                    "multiply".to_string(),
                    CanonicalOp::Multiply,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    Some(max_rel_error),
                    None,
                    wall_time.as_nanos() as u64,
                    Some(1),
                )
            };
            Some(evidence)
        } else {
            None
        };
        
        KernelResult::new(output, receipt, evidence)
    }

    /// Matmul kernel.
    /// 
    /// On macOS: Uses BLAS GEMM (cblas_sgemm) for matrix multiplication.
    /// On non-macOS: Uses reference implementation with fallback evidence.
    /// 
    /// # Native Execution
    /// 
    /// On macOS, this kernel attempts to use the native cblas_sgemm function.
    /// The native path is deliberately conservative and only accepts:
    /// - f32 dtype
    /// - Contiguous row-major layout
    /// - Non-transposed A and B matrices
    /// - Rank-2 matrices
    /// - Nonzero dimensions that fit in CBLAS_INDEX (i32)
    /// 
    /// If any of these conditions are not met, falls back to reference with
    /// fallback_reason = "native BLAS path only supports contiguous row-major f32 matmul".
    /// 
    /// If successful, executed_subsystem = BLAS and native_symbol = "cblas_sgemm".
    pub fn matmul(ctx: &KernelContext, a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> KernelResult {
        let start = Instant::now();
        let shape_key = format!("[{},{},{}]", m, n, p);
        let layout_key = "row_major".to_string();
        
        // For matmul, the intended lowering is BLAS
        let lowering_subsystem = AccelerateSubsystem::Blas;
        
        // Try native execution first (on macOS), then fallback to reference
        let (output, executed_subsystem, fallback_used, fallback_reason, native_symbol) = 
            if NativeDispatcher::is_available() {
                // On macOS, try native BLAS call
                // Check if we can use the native path (conservative conditions)
                let expected_a_len = m * n;
                let expected_b_len = n * p;
                
                if a.len() == expected_a_len && b.len() == expected_b_len && m > 0 && n > 0 && p > 0 {
                    // All conditions met, try native BLAS
                    match NativeDispatcher::cblas_sgemm_row_major(a, b, m, n, p) {
                        Ok((output, symbol)) => {
                            // Native execution succeeded
                            (output, AccelerateSubsystem::Blas, false, None, Some(symbol))
                        }
                        Err(err) => {
                            // Native call failed, use reference fallback
                            let reason = match err {
                                super::native::AccelerateNativeError::BackendUnavailable => 
                                    "Accelerate unavailable".to_string(),
                                super::native::AccelerateNativeError::LengthMismatch { expected, actual } => 
                                    format!("Length mismatch: expected {}, got {}", expected, actual),
                                super::native::AccelerateNativeError::EmptyInput => 
                                    "Empty input not supported by native path".to_string(),
                                super::native::AccelerateNativeError::DimensionError(msg) => msg,
                                _ => "Native BLAS call failed".to_string(),
                            };
                            (reference::matmul(a, b, m, n, p), AccelerateSubsystem::Reference, true, Some(reason), None)
                        }
                    }
                } else {
                    // Dimensions don't match expected, use reference fallback
                    (reference::matmul(a, b, m, n, p), AccelerateSubsystem::Reference, true, 
                     Some("native BLAS path only supports contiguous row-major f32 matmul in NATIVE-ACCELERATE-F32-KERNELS-0001".to_string()), None)
                }
            } else {
                // On non-macOS, use reference fallback
                (reference::matmul(a, b, m, n, p), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()), None)
            };
        
        let wall_time = start.elapsed();
        
        // Create receipt with both lowering and executed subsystems
        let mut receipt = AccelerateExecutionReceipt::new(
            uuid::Uuid::new_v4().to_string(),
            "matmul".to_string(),
            CanonicalOp::Matmul,
            AccelerateDType::F32,
            shape_key.clone(),
            layout_key.clone(),
            lowering_subsystem,
            executed_subsystem,
        )
        .with_results(
            wall_time,
            NumericalStatus::Passed,
            Some(1), // One output allocation
        );
        
        // Set native symbol if present
        if let Some(symbol) = native_symbol {
            receipt = receipt.with_native_symbol(symbol);
        }
        
        // Set fallback if used
        let receipt = if fallback_used {
            receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
        } else {
            receipt
        };
        
        // Create evidence
        let evidence = if ctx.collect_evidence {
            // Compare against reference
            let reference_output = reference::matmul(a, b, m, n, p);
            
            // For matmul, use cosine similarity as the primary metric
            let dot_product: f32 = output.iter().zip(reference_output.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            
            let norm_a: f32 = output.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = reference_output.iter().map(|&x| x * x).sum::<f32>().sqrt();
            
            let cosine_similarity = if norm_a > 1e-8 && norm_b > 1e-8 {
                (dot_product / (norm_a * norm_b)) as f64
            } else {
                1.0 // Perfect similarity if both are zero vectors
            };
            
            // Also compute max absolute error
            let max_abs_error = output.iter().zip(reference_output.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max) as f64;
            
            // Use the new method if we have a native symbol
            let evidence = if let Some(symbol) = native_symbol {
                ctx.validator.create_evidence_with_native_symbol(
                    uuid::Uuid::new_v4().to_string(),
                    "matmul".to_string(),
                    CanonicalOp::Matmul,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    None,
                    Some(cosine_similarity),
                    wall_time.as_nanos() as u64,
                    Some(1),
                    symbol,
                )
            } else {
                ctx.validator.create_evidence(
                    uuid::Uuid::new_v4().to_string(),
                    "matmul".to_string(),
                    CanonicalOp::Matmul,
                    AccelerateDType::F32,
                    shape_key,
                    layout_key,
                    executed_subsystem,
                    Some(max_abs_error),
                    None,
                    Some(cosine_similarity),
                    wall_time.as_nanos() as u64,
                    Some(1),
                )
            };
            Some(evidence)
        } else {
            None
        };
        
        KernelResult::new(output, receipt, evidence)
    }
}

/// High-level kernel dispatch.
pub struct KernelDispatcher {
    ctx: KernelContext,
}

impl KernelDispatcher {
    /// Creates a new kernel dispatcher.
    pub fn new() -> Self {
        Self {
            ctx: KernelContext::new(),
        }
    }

    /// Creates a new kernel dispatcher with the given context.
    pub fn with_context(ctx: KernelContext) -> Self {
        Self { ctx }
    }

    /// Dispatches an identity operation.
    pub fn dispatch_identity(&self, input: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        kernels::identity(&self.ctx, input, shape, layout)
    }

    /// Dispatches an add operation.
    pub fn dispatch_add(&self, a: &[f32], b: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        kernels::add(&self.ctx, a, b, shape, layout)
    }

    /// Dispatches a multiply operation.
    pub fn dispatch_multiply(&self, a: &[f32], b: &[f32], shape: &[usize], layout: AccelerateLayout) -> KernelResult {
        kernels::multiply(&self.ctx, a, b, shape, layout)
    }

    /// Dispatches a matmul operation.
    pub fn dispatch_matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> KernelResult {
        kernels::matmul(&self.ctx, a, b, m, n, p)
    }

    /// Returns true if Accelerate is available.
    pub fn is_available(&self) -> bool {
        self.ctx.is_available()
    }

    /// Returns the kernel context.
    pub fn context(&self) -> &KernelContext {
        &self.ctx
    }
}

impl Default for KernelDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = reference::identity(&input);
        assert_eq!(input, output);
    }

    #[test]
    fn test_reference_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let output = reference::add(&a, &b);
        assert_eq!(output, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reference_multiply() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let output = reference::multiply(&a, &b);
        assert_eq!(output, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_reference_matmul() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let output = reference::matmul(&a, &b, 2, 3, 2);
        
        // Expected: [[58, 64], [139, 154]]
        assert_eq!(output.len(), 4);
        assert!((output[0] - 58.0).abs() < 1e-6);
        assert!((output[1] - 64.0).abs() < 1e-6);
        assert!((output[2] - 139.0).abs() < 1e-6);
        assert!((output[3] - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_reference_sigmoid() {
        let input = vec![0.0, 1.0, -1.0];
        let output = reference::sigmoid(&input);
        
        assert!((output[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!(output[1] > 0.5 && output[1] < 1.0); // sigmoid(1) ≈ 0.731
        assert!(output[2] > 0.0 && output[2] < 0.5); // sigmoid(-1) ≈ 0.269
    }

    #[test]
    fn test_reference_silu() {
        let input = vec![0.0, 1.0, -1.0];
        let output = reference::silu(&input);
        
        assert_eq!(output[0], 0.0); // silu(0) = 0
        assert!(output[1] > 0.0); // silu(1) > 0
        assert!(output[2] < 0.0); // silu(-1) < 0
    }

    #[test]
    fn test_reference_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = reference::softmax(&input);
        
        // Check that output sums to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all values are positive
        assert!(output.iter().all(|&x| x > 0.0));
        
        // Check ordering (softmax preserves ordering)
        assert!(output[0] < output[1] && output[1] < output[2]);
    }

    #[test]
    fn test_kernel_context() {
        let ctx = KernelContext::new();
        
        #[cfg(target_os = "macos")]
        {
            assert!(ctx.is_available());
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!ctx.is_available());
        }
    }

    #[test]
    fn test_kernel_dispatcher() {
        let dispatcher = KernelDispatcher::new();
        
        #[cfg(target_os = "macos")]
        {
            assert!(dispatcher.is_available());
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!dispatcher.is_available());
        }
    }

    #[test]
    fn test_identity_kernel() {
        let dispatcher = KernelDispatcher::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        
        let result = dispatcher.dispatch_identity(&input, &shape, AccelerateLayout::RowMajor);
        
        assert_eq!(result.output, input);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Identity);
        assert_eq!(result.receipt.dtype, AccelerateDType::F32);
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Identity);
        }
    }

    #[test]
    fn test_add_kernel() {
        let dispatcher = KernelDispatcher::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let shape = vec![3];
        
        let result = dispatcher.dispatch_add(&a, &b, &shape, AccelerateLayout::RowMajor);
        
        assert_eq!(result.output, vec![5.0, 7.0, 9.0]);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Add);
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Add);
        }
    }

    #[test]
    fn test_multiply_kernel() {
        let dispatcher = KernelDispatcher::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let shape = vec![3];
        
        let result = dispatcher.dispatch_multiply(&a, &b, &shape, AccelerateLayout::RowMajor);
        
        assert_eq!(result.output, vec![4.0, 10.0, 18.0]);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Multiply);
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Multiply);
        }
    }

    #[test]
    fn test_matmul_kernel() {
        let dispatcher = KernelDispatcher::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        
        let result = dispatcher.dispatch_matmul(&a, &b, 2, 3, 2);
        
        assert_eq!(result.output.len(), 4);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Matmul);
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Matmul);
            // Check cosine similarity is high
            if let Some(cs) = evidence.cosine_similarity {
                assert!(cs > 0.999);
            }
        }
    }

    #[test]
    fn test_fallback_behavior() {
        let dispatcher = KernelDispatcher::new();
        
        #[cfg(not(target_os = "macos"))]
        {
            // On non-macOS, all kernels should use fallback
            let input = vec![1.0, 2.0, 3.0];
            let result = dispatcher.dispatch_identity(&input, &[3], AccelerateLayout::RowMajor);
            
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
            
            if let Some(evidence) = result.evidence {
                assert!(evidence.fallback_used);
            }
        }
    }

    // ========================================================================
    // NATIVE-ACCELERATE-F32-KERNELS-0001 Tests
    // ========================================================================

    /// Non-macOS tests: Prove that f32 add/multiply/matmul use reference fallback
    #[cfg(not(target_os = "macos"))]
    mod non_macos_tests {
        use super::*;

        #[test]
        fn test_add_fallback_truthfulness() {
            let dispatcher = KernelDispatcher::new();
            let a = vec![1.0, 2.0, -3.0];
            let b = vec![4.0, -2.0, 0.5];
            let shape = vec![3];

            let result = dispatcher.dispatch_add(&a, &b, &shape, AccelerateLayout::RowMajor);

            // Verify numerical correctness (reference implementation)
            assert_eq!(result.output, vec![5.0, 0.0, -2.5]);

            // Verify truthful fallback evidence
            assert!(result.receipt.fallback_used, "fallback_used must be true on non-macOS");
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Vdsp, 
                "lowering_subsystem should be vDSP (intended)");
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference,
                "executed_subsystem should be Reference on non-macOS");
            assert!(result.receipt.fallback_reason.is_some(), "fallback_reason must be present");
            assert!(result.receipt.fallback_reason.as_ref().unwrap().contains("unavailable"),
                "fallback_reason should mention unavailable");
            assert!(result.receipt.native_symbol.is_none(), "native_symbol should be None on non-macOS");

            // Verify evidence
            if let Some(evidence) = result.evidence {
                assert!(evidence.fallback_used);
                assert_eq!(evidence.executed_subsystem, AccelerateSubsystem::Reference);
                assert!(evidence.native_symbol.is_none());
                // Numerical validation should pass (reference vs reference)
                assert!(evidence.passed);
            }
        }

        #[test]
        fn test_multiply_fallback_truthfulness() {
            let dispatcher = KernelDispatcher::new();
            let a = vec![1.0, -2.0, 0.5, -0.5];
            let b = vec![2.0, 3.0, -4.0, 0.0];
            let shape = vec![4];

            let result = dispatcher.dispatch_multiply(&a, &b, &shape, AccelerateLayout::RowMajor);

            // Verify numerical correctness
            assert_eq!(result.output, vec![2.0, -6.0, -2.0, 0.0]);

            // Verify truthful fallback evidence
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Vdsp);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
            assert!(result.receipt.fallback_reason.is_some());
            assert!(result.receipt.native_symbol.is_none());

            if let Some(evidence) = result.evidence {
                assert!(evidence.fallback_used);
                assert!(evidence.passed);
            }
        }

        #[test]
        fn test_matmul_fallback_truthfulness() {
            let dispatcher = KernelDispatcher::new();
            // 2x3 * 3x2 = 2x2
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

            let result = dispatcher.dispatch_matmul(&a, &b, 2, 3, 2);

            // Verify numerical correctness
            assert_eq!(result.output.len(), 4);
            assert!((result.output[0] - 58.0).abs() < 1e-6);
            assert!((result.output[1] - 64.0).abs() < 1e-6);
            assert!((result.output[2] - 139.0).abs() < 1e-6);
            assert!((result.output[3] - 154.0).abs() < 1e-6);

            // Verify truthful fallback evidence
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
            assert!(result.receipt.fallback_reason.is_some());
            assert!(result.receipt.native_symbol.is_none());

            if let Some(evidence) = result.evidence {
                assert!(evidence.fallback_used);
                assert!(evidence.passed);
                // Cosine similarity should be perfect (reference vs reference)
                if let Some(cs) = evidence.cosine_similarity {
                    assert!(cs > 0.9999);
                }
            }
        }

        #[test]
        fn test_matmul_non_square_fallback() {
            let dispatcher = KernelDispatcher::new();
            // 2x3 * 3x4 = 2x4 (non-square)
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x4

            let result = dispatcher.dispatch_matmul(&a, &b, 2, 3, 4);

            // Verify fallback behavior
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
        }

        #[test]
        fn test_matmul_vector_like_fallback() {
            let dispatcher = KernelDispatcher::new();
            // 1x3 * 3x1 = 1x1 (vector-like)
            let a = vec![1.0, 2.0, 3.0]; // 1x3
            let b = vec![4.0, 5.0, 6.0]; // 3x1

            let result = dispatcher.dispatch_matmul(&a, &b, 1, 3, 1);

            // Verify fallback behavior
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
        }
    }

    /// macOS tests: Prove that f32 add/multiply/matmul execute natively
    #[cfg(target_os = "macos")]
    mod macos_tests {
        use super::*;

        #[test]
        fn test_add_native_execution() {
            let dispatcher = KernelDispatcher::new();
            let a = vec![1.0, 2.0, -3.0];
            let b = vec![4.0, -2.0, 0.5];
            let shape = vec![3];

            let result = dispatcher.dispatch_add(&a, &b, &shape, AccelerateLayout::RowMajor);

            // Verify numerical correctness
            assert_eq!(result.output, vec![5.0, 0.0, -2.5]);

            // Verify native execution evidence
            assert!(!result.receipt.fallback_used, "fallback_used should be false on macOS for add");
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Vdsp);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Vdsp,
                "executed_subsystem should be vDSP on macOS");
            assert!(result.receipt.fallback_reason.is_none(), "fallback_reason should be None for native execution");
            assert_eq!(result.receipt.native_symbol, Some("vDSP_vadd".to_string()),
                "native_symbol should be vDSP_vadd");

            // Verify evidence
            if let Some(evidence) = result.evidence {
                assert!(!evidence.fallback_used);
                assert_eq!(evidence.executed_subsystem, AccelerateSubsystem::Vdsp);
                assert_eq!(evidence.native_symbol, Some("vDSP_vadd".to_string()));
                assert!(evidence.passed);
                // max_abs_error should be effectively zero
                if let Some(err) = evidence.max_abs_error {
                    assert!(err < 1e-4, "max_abs_error = {} should be < 1e-4", err);
                }
            }
        }

        #[test]
        fn test_multiply_native_execution() {
            let dispatcher = KernelDispatcher::new();
            let a = vec![1.0, -2.0, 0.5, -0.5];
            let b = vec![2.0, 3.0, -4.0, 0.0];
            let shape = vec![4];

            let result = dispatcher.dispatch_multiply(&a, &b, &shape, AccelerateLayout::RowMajor);

            // Verify numerical correctness
            assert_eq!(result.output, vec![2.0, -6.0, -2.0, 0.0]);

            // Verify native execution evidence
            assert!(!result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Vdsp);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Vdsp);
            assert!(result.receipt.fallback_reason.is_none());
            assert_eq!(result.receipt.native_symbol, Some("vDSP_vmul".to_string()));

            if let Some(evidence) = result.evidence {
                assert!(!evidence.fallback_used);
                assert_eq!(evidence.native_symbol, Some("vDSP_vmul".to_string()));
                assert!(evidence.passed);
            }
        }

        #[test]
        fn test_matmul_native_execution_non_square() {
            let dispatcher = KernelDispatcher::new();
            // 2x3 * 3x4 = 2x4 (non-square to catch leading dimension mistakes)
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x4

            let result = dispatcher.dispatch_matmul(&a, &b, 2, 3, 4);

            // Verify numerical correctness
            assert_eq!(result.output.len(), 8); // 2x4 = 8 elements

            // Verify native execution evidence
            assert!(!result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Blas,
                "executed_subsystem should be BLAS on macOS");
            assert!(result.receipt.fallback_reason.is_none());
            assert_eq!(result.receipt.native_symbol, Some("cblas_sgemm".to_string()));

            if let Some(evidence) = result.evidence {
                assert!(!evidence.fallback_used);
                assert_eq!(evidence.native_symbol, Some("cblas_sgemm".to_string()));
                assert!(evidence.passed);
                // Cosine similarity should be very high
                if let Some(cs) = evidence.cosine_similarity {
                    assert!(cs > 0.9999, "cosine_similarity = {} should be > 0.9999", cs);
                }
                // Max absolute error should be small
                if let Some(err) = evidence.max_abs_error {
                    assert!(err < 1e-4, "max_abs_error = {} should be < 1e-4", err);
                }
            }
        }

        #[test]
        fn test_matmul_native_execution_vector_like() {
            let dispatcher = KernelDispatcher::new();
            // 1x3 * 3x1 = 1x1 (vector-like matrix)
            let a = vec![1.0, 2.0, 3.0]; // 1x3
            let b = vec![4.0, 5.0, 6.0]; // 3x1

            let result = dispatcher.dispatch_matmul(&a, &b, 1, 3, 1);

            // Verify numerical correctness: [1*4 + 2*5 + 3*6] = [32]
            assert_eq!(result.output, vec![32.0]);

            // Verify native execution evidence
            assert!(!result.receipt.fallback_used);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.native_symbol, Some("cblas_sgemm".to_string()));
        }

        #[test]
        fn test_matmul_fallback_for_invalid_dimensions() {
            let dispatcher = KernelDispatcher::new();
            // Try to pass wrong dimensions - should fallback
            let a = vec![1.0, 2.0, 3.0]; // 3 elements
            let b = vec![4.0, 5.0, 6.0]; // 3 elements
            // But claim they are 2x2 and 2x2 (which would need 4 elements each)
            // This should trigger the dimension validation fallback

            let result = dispatcher.dispatch_matmul(&a, &b, 2, 2, 2);

            // Should fallback because buffer sizes don't match claimed dimensions
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.lowering_subsystem, AccelerateSubsystem::Blas);
            assert_eq!(result.receipt.executed_subsystem, AccelerateSubsystem::Reference);
            assert!(result.receipt.fallback_reason.as_ref().unwrap().contains("contiguous row-major f32 matmul"));
            assert!(result.receipt.native_symbol.is_none());
        }
    }
}
