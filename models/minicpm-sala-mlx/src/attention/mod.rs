pub mod lightning;
pub mod sparse;

pub use lightning::{LightningAttention, LightningCache};
pub use sparse::{SparseAttention, SparseKVCache};

use mlx_rs::{
    error::Exception,
    module::{
        ModuleParamMut, ModuleParamRef, ModuleParameters as ModuleParametersTrait,
    },
    Array,
};

use crate::config::ModelArgs;

/// Per-layer cache: either a SparseKVCache (sparse layers) or a recurrent
/// state (lightning layers).
#[derive(Debug, Clone)]
pub enum LayerCache {
    Sparse(SparseKVCache),
    Lightning(LightningCache),
}

impl LayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Sparse(cache) => cache.offset(),
            Self::Lightning(cache) => cache.offset(),
        }
    }
}

/// Hybrid attention dispatch: sparse (SDPA) or lightning (GLA).
#[derive(Debug)]
pub enum HybridAttention {
    Sparse(SparseAttention),
    Lightning(LightningAttention),
}

impl HybridAttention {
    pub fn new(args: &ModelArgs, layer_idx: usize) -> Result<Self, Exception> {
        if args.is_sparse_layer(layer_idx) {
            Ok(Self::Sparse(SparseAttention::new(args)?))
        } else {
            Ok(Self::Lightning(LightningAttention::new(args)?))
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Sparse(attn), LayerCache::Sparse(cache)) => attn.forward(x, mask, cache),
            (Self::Lightning(attn), LayerCache::Lightning(cache)) => attn.forward(x, cache),
            _ => Err(Exception::from("Cache type mismatch in HybridAttention")),
        }
    }

    pub fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Sparse(_) => {}
            Self::Lightning(_) => {}
        }
    }
}

// Manual ModuleParameters impl for enum dispatch
impl ModuleParametersTrait for HybridAttention {
    fn parameters(&self) -> Vec<(&'static str, &dyn ModuleParamRef)> {
        match self {
            Self::Sparse(attn) => attn.parameters(),
            Self::Lightning(attn) => attn.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> Vec<(&'static str, &mut dyn ModuleParamMut)> {
        match self {
            Self::Sparse(attn) => attn.parameters_mut(),
            Self::Lightning(attn) => attn.parameters_mut(),
        }
    }

    fn flatten(&self) -> Vec<(&str, &Array)> {
        match self {
            Self::Sparse(attn) => attn.flatten(),
            Self::Lightning(attn) => attn.flatten(),
        }
    }

    fn flatten_mut(&mut self) -> Vec<(&str, &mut Array)> {
        match self {
            Self::Sparse(attn) => attn.flatten_mut(),
            Self::Lightning(attn) => attn.flatten_mut(),
        }
    }
}

/// Create the per-layer cache vector based on mixer_types.
pub fn create_layer_caches(args: &ModelArgs) -> Vec<LayerCache> {
    let num_layers = args.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        if args.is_sparse_layer(i) {
            caches.push(LayerCache::Sparse(SparseKVCache::new()));
        } else {
            caches.push(LayerCache::Lightning(LightningCache::new(args)));
        }
    }

    caches
}
