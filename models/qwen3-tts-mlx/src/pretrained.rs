//! Extension traits adding `from_pretrained` constructors for MLX `nn` types.
//!
//! These traits provide weight-loaded constructors that create modules with
//! pre-trained weights when provided, or fall back to random initialization.
//! This matches the Python MLX `from_pretrained` pattern.

use std::collections::HashMap;

use mlx_rs::{
    module::Param,
    nn,
    quantization::{MaybeQuantized, Quantizable},
    Array,
};

use crate::config::QuantizationConfig;
use crate::error::Result;

// ── Linear ──────────────────────────────────────────────────────────────────

pub trait LinearFromPretrained {
    fn from_pretrained(
        input_dims: i32,
        output_dims: i32,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::Linear>;
}

impl LinearFromPretrained for nn::Linear {
    fn from_pretrained(
        input_dims: i32,
        output_dims: i32,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::Linear> {
        let mut lin = nn::Linear::new(input_dims, output_dims)?;
        if let Some(w) = weight {
            lin.weight = Param::new(w.clone());
        }
        if let Some(b) = bias {
            lin.bias = Param::new(Some(b.clone()));
        } else {
            lin.bias = Param::new(None);
        }
        Ok(lin)
    }
}

// ── Embedding ───────────────────────────────────────────────────────────────

pub trait EmbeddingFromPretrained {
    fn from_pretrained(
        embedding_count: i32,
        dimensions: i32,
        weight: Option<&Array>,
    ) -> Result<nn::Embedding>;
}

impl EmbeddingFromPretrained for nn::Embedding {
    fn from_pretrained(
        embedding_count: i32,
        dimensions: i32,
        weight: Option<&Array>,
    ) -> Result<nn::Embedding> {
        let mut emb = nn::Embedding::new(embedding_count, dimensions)?;
        if let Some(w) = weight {
            emb.weight = Param::new(w.clone());
        }
        Ok(emb)
    }
}

// ── RmsNorm ─────────────────────────────────────────────────────────────────

pub trait RmsNormFromPretrained {
    fn from_pretrained(
        dimensions: i32,
        weight: Option<&Array>,
        eps: f32,
    ) -> Result<nn::RmsNorm>;
}

impl RmsNormFromPretrained for nn::RmsNorm {
    fn from_pretrained(
        dimensions: i32,
        weight: Option<&Array>,
        eps: f32,
    ) -> Result<nn::RmsNorm> {
        let mut norm = nn::RmsNorm::new(dimensions)?;
        norm.eps = eps;
        if let Some(w) = weight {
            norm.weight = Param::new(w.clone());
        }
        Ok(norm)
    }
}

// ── Conv1d ──────────────────────────────────────────────────────────────────

pub trait Conv1dFromPretrained {
    fn from_pretrained(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        dilation: i32,
        has_bias: bool,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::Conv1d>;
}

impl Conv1dFromPretrained for nn::Conv1d {
    fn from_pretrained(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        dilation: i32,
        has_bias: bool,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::Conv1d> {
        use mlx_rs::builder::Builder;
        let mut b = nn::Conv1dBuilder::new(input_channels, output_channels, kernel_size)
            .dilation(dilation);
        if !has_bias {
            b = b.bias(false);
        }
        let mut conv = b.build()?;
        if let Some(w) = weight {
            conv.weight = Param::new(w.clone());
        }
        let bval: Option<Array> = if has_bias {
            bias.cloned()
        } else {
            None
        };
        conv.bias = Param::new(bval);
        Ok(conv)
    }
}

// ── ConvTranspose1d ─────────────────────────────────────────────────────────

pub trait ConvTranspose1dFromPretrained {
    fn from_pretrained(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        has_bias: bool,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::ConvTranspose1d>;
}

impl ConvTranspose1dFromPretrained for nn::ConvTranspose1d {
    fn from_pretrained(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        has_bias: bool,
        weight: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<nn::ConvTranspose1d> {
        use mlx_rs::builder::Builder;
        let mut b = nn::ConvTranspose1dBuilder::new(input_channels, output_channels, kernel_size)
            .stride(stride)
            .padding(padding);
        if !has_bias {
            b = b.bias(false);
        }
        let mut conv = b.build()?;
        if let Some(w) = weight {
            conv.weight = Param::new(w.clone());
        }
        let bval: Option<Array> = if has_bias {
            bias.cloned()
        } else {
            None
        };
        conv.bias = Param::new(bval);
        Ok(conv)
    }
}

// ── MaybeQuantized ──────────────────────────────────────────────────────────

pub trait MaybeQuantizedFromLinear {
    fn from_linear(
        linear: nn::Linear,
        quant: Option<&QuantizationConfig>,
        prefix: &str,
        weights: &HashMap<String, Array>,
    ) -> MaybeQuantized<nn::Linear>;
}

impl MaybeQuantizedFromLinear for MaybeQuantized<nn::Linear> {
    fn from_linear(
        linear: nn::Linear,
        quant: Option<&QuantizationConfig>,
        _prefix: &str,
        _weights: &HashMap<String, Array>,
    ) -> MaybeQuantized<nn::Linear> {
        if let Some(q) = quant {
            // Try quantization; if it fails, keep the original
            match linear.clone().try_into_quantized(q.group_size, q.bits) {
                Ok(qlinear) => MaybeQuantized::Quantized(qlinear),
                Err(_) => MaybeQuantized::Original(linear),
            }
        } else {
            MaybeQuantized::Original(linear)
        }
    }
}
