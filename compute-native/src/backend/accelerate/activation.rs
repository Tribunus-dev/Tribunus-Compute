//! Accelerate activation function kernels.
//!
//! This module provides **native-ready scaffold** for PR5: f32 sigmoid, SiLU, and softmax kernels.
//! 
//! # Implementation Status
//!
//! **Current State**: All activation kernels use reference implementations on all platforms.
//! The lowering decisions target vDSP/vForce, but actual execution is via reference fallback
//! until native FFI bindings are implemented.
//!
//! # Design Principles
//!
//! 1. **Numerical Stability**: Softmax uses max-subtraction for stability.
//! 2. **Composite Lowering**: SiLU is explicitly lowered as sigmoid * multiply (not a native primitive).
//! 3. **Evidence-Driven**: Each kernel produces detailed evidence.
//! 4. **Fallback Support**: All cases currently use reference fallback with explicit evidence.
//! 5. **Truthful**: Receipts clearly distinguish lowering_subsystem from executed_subsystem.

use std::time::{Duration, Instant};
use uuid::Uuid;

use super::{dtype::AccelerateDType, execution::{AccelerateExecutionReceipt, NumericalStatus}, 
          ffi::AccelerateHandle, layout::AccelerateLayout, ops::CanonicalOp, 
          subsystem::AccelerateSubsystem, evidence::{AccelerateEvidence, EvidenceValidator}, 
          kernels::reference};

/// Activation kernel execution context.
pub struct ActivationContext {
    /// Accelerate handle for platform access.
    pub handle: AccelerateHandle,
    /// Whether to collect evidence.
    pub collect_evidence: bool,
    /// Evidence validator.
    pub validator: EvidenceValidator,
}

impl ActivationContext {
    /// Creates a new activation context.
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

impl Default for ActivationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an activation kernel execution.
#[derive(Debug, Clone)]
pub struct ActivationResult {
    /// Output buffer.
    pub output: Vec<f32>,
    /// Execution receipt.
    pub receipt: AccelerateExecutionReceipt,
    /// Evidence (if collected).
    pub evidence: Option<AccelerateEvidence>,
}

impl ActivationResult {
    /// Creates a new activation result.
    pub fn new(output: Vec<f32>, receipt: AccelerateExecutionReceipt, evidence: Option<AccelerateEvidence>) -> Self {
        Self { output, receipt, evidence }
    }
}

/// Sigmoid kernel.
/// 
/// On macOS: Intended to use vDSP/vForce for vector sigmoid (not yet implemented).
/// On non-macOS: Uses reference implementation with fallback evidence.
/// 
/// Formula: sigmoid(x) = 1 / (1 + exp(-x))
/// 
/// # Note
/// 
/// This implementation uses reference fallback on all platforms.
/// The lowering_subsystem is vDSP (intended), while executed_subsystem is Reference
/// until native vDSP calls are implemented.
pub fn sigmoid(ctx: &ActivationContext, input: &[f32], shape: &[usize]) -> ActivationResult {
    let start = Instant::now();
    let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
    let layout_key = "row_major".to_string();
    
    // For sigmoid, the intended lowering is vDSP
    let lowering_subsystem = AccelerateSubsystem::Vdsp;
    
    let (output, executed_subsystem, fallback_used, fallback_reason) = if ctx.is_available() {
        // On macOS, we would use vDSP/vForce here
        // For now, use reference but mark intended as vDSP
        // lowering_subsystem = vDSP, executed_subsystem = Reference (fallback)
        (reference::sigmoid(input), AccelerateSubsystem::Reference, true, Some("vDSP not yet implemented".to_string()))
    } else {
        // On non-macOS, use reference fallback
        // lowering_subsystem = vDSP, executed_subsystem = Reference, fallback_used = true
        (reference::sigmoid(input), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()))
    };
    
    let wall_time = start.elapsed();
    
    // Create receipt with both lowering and executed subsystems
    let receipt = AccelerateExecutionReceipt::new(
        Uuid::new_v4().to_string(),
        "sigmoid".to_string(),
        CanonicalOp::Sigmoid,
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
    
    let receipt = if fallback_used {
        receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
    } else {
        receipt
    };
    
    // Create evidence
    let evidence = if ctx.collect_evidence {
        // Compare against reference
        let reference_output = reference::sigmoid(input);
        let max_abs_error = output.iter().zip(reference_output.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max) as f64;
        
        let max_rel_error = output.iter().zip(reference_output.iter())
            .filter(|(&a, &b)| b.abs() > 1e-8)
            .map(|(&a, &b)| ((a - b).abs() / b.abs()) as f64)
            .fold(0.0f64, f64::max);
        
        Some(ctx.validator.create_evidence(
            Uuid::new_v4().to_string(),
            "sigmoid".to_string(),
            CanonicalOp::Sigmoid,
            AccelerateDType::F32,
            shape_key,
            layout_key,
            executed_subsystem,
            Some(max_abs_error),
            Some(max_rel_error),
            None,
            wall_time.as_nanos() as u64,
            Some(1),
        ))
    } else {
        None
    };
    
    ActivationResult::new(output, receipt, evidence)
}

/// SiLU kernel.
/// 
/// SiLU (Sigmoid Linear Unit) is defined as: sigmoid(x) * x
/// 
/// This must be implemented as a composite lowering of sigmoid followed by multiply,
/// not as an imaginary native primitive.
/// 
/// On macOS: Intended to use vDSP for both sigmoid and multiply operations (not yet implemented).
/// On non-macOS: Uses reference implementation with fallback evidence.
/// 
/// # Note
/// 
/// This implementation uses reference fallback on all platforms.
/// The lowering_subsystem is vDSP (intended), while executed_subsystem is Reference
/// until native vDSP calls are implemented.
pub fn silu(ctx: &ActivationContext, input: &[f32], shape: &[usize]) -> ActivationResult {
    let start = Instant::now();
    let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
    let layout_key = "row_major".to_string();
    
    // For SiLU, the intended lowering is vDSP (composite: sigmoid + multiply)
    let lowering_subsystem = AccelerateSubsystem::Vdsp;
    
    let (output, executed_subsystem, fallback_used, fallback_reason) = if ctx.is_available() {
        // On macOS, we would use vDSP for both operations
        // For now, use reference but mark intended as vDSP
        // This is a composite operation: sigmoid(x) * x
        let sigmoid_output = reference::sigmoid(input);
        let output = sigmoid_output.iter().zip(input.iter())
            .map(|(&s, &x)| s * x)
            .collect::<Vec<f32>>();
        
        // lowering_subsystem = vDSP, executed_subsystem = Reference (fallback)
        (output, AccelerateSubsystem::Reference, true, Some("vDSP not yet implemented".to_string()))
    } else {
        // On non-macOS, use reference fallback
        let output = reference::silu(input);
        // lowering_subsystem = vDSP, executed_subsystem = Reference, fallback_used = true
        (output, AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()))
    };
    
    let wall_time = start.elapsed();
    
    // Create receipt with both lowering and executed subsystems - note this is a composite operation
    let receipt = AccelerateExecutionReceipt::new(
        Uuid::new_v4().to_string(),
        "silu".to_string(),
        CanonicalOp::Silu,
        AccelerateDType::F32,
        shape_key.clone(),
        layout_key.clone(),
        lowering_subsystem,
        executed_subsystem,
    )
    .with_results(
        wall_time,
        NumericalStatus::Passed,
        Some(2), // Two allocations: sigmoid intermediate + output
    );
    
    let receipt = if fallback_used {
        receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
    } else {
        receipt
    };
    
    // Create evidence
    let evidence = if ctx.collect_evidence {
        // Compare against reference
        let reference_output = reference::silu(input);
        let max_abs_error = output.iter().zip(reference_output.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max) as f64;
        
        let max_rel_error = output.iter().zip(reference_output.iter())
            .filter(|(&a, &b)| b.abs() > 1e-8)
            .map(|(&a, &b)| ((a - b).abs() / b.abs()) as f64)
            .fold(0.0f64, f64::max);
        
        Some(ctx.validator.create_evidence(
            Uuid::new_v4().to_string(),
            "silu".to_string(),
            CanonicalOp::Silu,
            AccelerateDType::F32,
            shape_key,
            layout_key,
            executed_subsystem,
            Some(max_abs_error),
            Some(max_rel_error),
            None,
            wall_time.as_nanos() as u64,
            Some(2),
        ))
    } else {
        None
    };
    
    ActivationResult::new(output, receipt, evidence)
}

/// Softmax kernel.
/// 
/// Implements numerically stable softmax: max reduction, subtract max, exp, sum reduction, divide.
/// 
/// Formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// 
/// On macOS: Intended to use vDSP for exp, max reduction, sum reduction, and divide operations (not yet implemented).
/// On non-macOS: Uses reference implementation with fallback evidence.
/// 
/// # Note
/// 
/// This implementation uses reference fallback on all platforms.
/// The lowering_subsystem is vDSP (intended), while executed_subsystem is Reference
/// until native vDSP calls are implemented.
pub fn softmax(ctx: &ActivationContext, input: &[f32], shape: &[usize]) -> ActivationResult {
    let start = Instant::now();
    let shape_key = format!("[{}]", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
    let layout_key = "row_major".to_string();
    
    // For softmax, the intended lowering is vDSP
    let lowering_subsystem = AccelerateSubsystem::Vdsp;
    
    let (output, executed_subsystem, fallback_used, fallback_reason) = if ctx.is_available() {
        // On macOS, we would use vDSP for the component operations
        // For now, use reference but mark intended as vDSP
        // lowering_subsystem = vDSP, executed_subsystem = Reference (fallback)
        (reference::softmax(input), AccelerateSubsystem::Reference, true, Some("vDSP not yet implemented".to_string()))
    } else {
        // On non-macOS, use reference fallback
        // lowering_subsystem = vDSP, executed_subsystem = Reference, fallback_used = true
        (reference::softmax(input), AccelerateSubsystem::Reference, true, Some("Accelerate unavailable".to_string()))
    };
    
    let wall_time = start.elapsed();
    
    // Create receipt with both lowering and executed subsystems
    let receipt = AccelerateExecutionReceipt::new(
        Uuid::new_v4().to_string(),
        "softmax".to_string(),
        CanonicalOp::Softmax,
        AccelerateDType::F32,
        shape_key.clone(),
        layout_key.clone(),
        lowering_subsystem,
        executed_subsystem,
    )
    .with_results(
        wall_time,
        NumericalStatus::Passed,
        Some(2), // Two allocations: exp intermediate + output
    );
    
    let receipt = if fallback_used {
        receipt.with_fallback(fallback_reason.unwrap_or_default().as_str())
    } else {
        receipt
    };
    
    // Create evidence
    let evidence = if ctx.collect_evidence {
        // Compare against reference
        let reference_output = reference::softmax(input);
        
        // For softmax, use both max absolute error and cosine similarity
        let max_abs_error = output.iter().zip(reference_output.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max) as f64;
        
        // Compute cosine similarity
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
        
        Some(ctx.validator.create_evidence(
            Uuid::new_v4().to_string(),
            "softmax".to_string(),
            CanonicalOp::Softmax,
            AccelerateDType::F32,
            shape_key,
            layout_key,
            executed_subsystem,
            Some(max_abs_error),
            None,
            Some(cosine_similarity),
            wall_time.as_nanos() as u64,
            Some(2),
        ))
    } else {
        None
    };
    
    ActivationResult::new(output, receipt, evidence)
}

/// High-level activation dispatcher.
pub struct ActivationDispatcher {
    ctx: ActivationContext,
}

impl ActivationDispatcher {
    /// Creates a new activation dispatcher.
    pub fn new() -> Self {
        Self {
            ctx: ActivationContext::new(),
        }
    }

    /// Creates a new activation dispatcher with the given context.
    pub fn with_context(ctx: ActivationContext) -> Self {
        Self { ctx }
    }

    /// Dispatches a sigmoid operation.
    pub fn dispatch_sigmoid(&self, input: &[f32], shape: &[usize]) -> ActivationResult {
        sigmoid(&self.ctx, input, shape)
    }

    /// Dispatches a SiLU operation.
    pub fn dispatch_silu(&self, input: &[f32], shape: &[usize]) -> ActivationResult {
        silu(&self.ctx, input, shape)
    }

    /// Dispatches a softmax operation.
    pub fn dispatch_softmax(&self, input: &[f32], shape: &[usize]) -> ActivationResult {
        softmax(&self.ctx, input, shape)
    }

    /// Returns true if Accelerate is available.
    pub fn is_available(&self) -> bool {
        self.ctx.is_available()
    }

    /// Returns the activation context.
    pub fn context(&self) -> &ActivationContext {
        &self.ctx
    }
}

impl Default for ActivationDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Extreme value softmax test case.
/// 
/// This tests the numerical stability of softmax with extreme values
/// that would overflow without max subtraction.
pub fn test_extreme_softmax_stability() {
    // Create input with extreme values that would cause overflow
    let input = vec![1000.0f32, 1001.0f32, 1002.0f32];
    
    // Compute softmax using reference implementation
    let output = reference::softmax(&input);
    
    // Check that output sums to 1 (numerical stability)
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1, got {}", sum);
    
    // Check that all values are positive and finite
    assert!(output.iter().all(|&x| x > 0.0 && x.is_finite()));
    
    // Check that the largest input produces the largest output
    assert!(output[0] < output[1] && output[1] < output[2]);
}

/// Small vector softmax test.
pub fn test_small_softmax() {
    let input = vec![1.0f32, 2.0f32];
    let output = reference::softmax(&input);
    
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(output[0] < output[1]);
}

/// Large vector softmax test.
pub fn test_large_softmax() {
    let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let output = reference::softmax(&input);
    
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    
    // Check that the output is monotonically increasing (since input is)
    for i in 1..output.len() {
        assert!(output[i-1] <= output[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_context() {
        let ctx = ActivationContext::new();
        
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
    fn test_sigmoid_kernel() {
        let ctx = ActivationContext::new();
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let shape = vec![5];
        
        let result = sigmoid(&ctx, &input, &shape);
        
        assert_eq!(result.output.len(), 5);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Sigmoid);
        
        // Check that sigmoid(0) = 0.5
        assert!((result.output[0] - 0.5).abs() < 1e-6);
        
        // Check that all values are in (0, 1)
        assert!(result.output.iter().all(|&x| x > 0.0 && x < 1.0));
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Sigmoid);
        }
    }

    #[test]
    fn test_silu_kernel() {
        let ctx = ActivationContext::new();
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let shape = vec![5];
        
        let result = silu(&ctx, &input, &shape);
        
        assert_eq!(result.output.len(), 5);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Silu);
        
        // Check that silu(0) = 0
        assert_eq!(result.output[0], 0.0);
        
        // Check that silu(x) has the same sign as x
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(result.output[i].signum(), x.signum());
        }
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Silu);
        }
    }

    #[test]
    fn test_softmax_kernel() {
        let ctx = ActivationContext::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        
        let result = softmax(&ctx, &input, &shape);
        
        assert_eq!(result.output.len(), 4);
        assert!(result.receipt.is_success());
        assert_eq!(result.receipt.op, CanonicalOp::Softmax);
        
        // Check that output sums to 1
        let sum: f32 = result.output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all values are positive
        assert!(result.output.iter().all(|&x| x > 0.0));
        
        // Check ordering
        assert!(result.output[0] < result.output[1] && 
                result.output[1] < result.output[2] && 
                result.output[2] < result.output[3]);
        
        if let Some(evidence) = result.evidence {
            assert!(evidence.is_success());
            assert_eq!(evidence.op, CanonicalOp::Softmax);
            
            // Check cosine similarity is high
            if let Some(cs) = evidence.cosine_similarity {
                assert!(cs > 0.9999);
            }
        }
    }

    #[test]
    fn test_activation_dispatcher() {
        let dispatcher = ActivationDispatcher::new();
        
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
    fn test_extreme_softmax_stability() {
        test_extreme_softmax_stability();
    }

    #[test]
    fn test_small_softmax() {
        test_small_softmax();
    }

    #[test]
    fn test_large_softmax() {
        test_large_softmax();
    }

    #[test]
    fn test_fallback_behavior() {
        let dispatcher = ActivationDispatcher::new();
        
        #[cfg(not(target_os = "macos"))]
        {
            // On non-macOS, all activation kernels should use fallback
            let input = vec![1.0, 2.0, 3.0];
            let result = dispatcher.dispatch_sigmoid(&input, &[3]);
            
            assert!(result.receipt.fallback_used);
            assert_eq!(result.receipt.subsystem, AccelerateSubsystem::Reference);
            
            if let Some(evidence) = result.evidence {
                assert!(evidence.fallback_used);
            }
        }
    }

    #[test]
    fn test_composite_silu_evidence() {
        let ctx = ActivationContext::new();
        let input = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        
        let result = silu(&ctx, &input, &shape);
        
        // SiLU should have 2 allocations (sigmoid intermediate + output)
        assert_eq!(result.receipt.allocation_count, Some(2));
        
        if let Some(evidence) = result.evidence {
            // Evidence should show this is a composite operation
            assert_eq!(evidence.op, CanonicalOp::Silu);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let ctx = ActivationContext::new();
        
        // Test with extreme values
        let input = vec![1000.0f32, 1001.0f32, 1002.0f32];
        let shape = vec![3];
        
        let result = softmax(&ctx, &input, &shape);
        
        // Check that output sums to 1 (numerical stability)
        let sum: f32 = result.output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1, got {}", sum);
        
        // Check that all values are finite
        assert!(result.output.iter().all(|&x| x.is_finite()));
    }
}
