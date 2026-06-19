//! Namespace resolution for text model tensors.
//!
//! Discovers the text model namespace by probing candidate prefixes
//! against the tensor names in a safetensors file. Candidates are
//! checked in order; the first that matches all anchor tensors wins.

use serde::Serialize;

/// Selected namespace root for text model tensors.
#[derive(Clone, Debug, Serialize)]
pub struct NamespaceBinding {
    pub root: String,
    /// How the root was discovered.
    pub discovery: String,
    /// Where lm_head.weight lives (may alias embed_tokens if tied).
    pub lm_head_key: String,
    pub lm_head_aliased: bool,
}

/// Anchor tensors that must exist under the text model root.
const ANCHORS: &[&str] = &[
    "embed_tokens.weight",
    "norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",
];

/// Discover the text model namespace by probing candidate prefixes.
/// Candidates are checked in order; first to match all anchors wins.
pub fn resolve_namespace(tensor_names: &[String]) -> Option<NamespaceBinding> {
    let candidates = &["language_model.model", "model"];

    for &candidate in candidates {
        let all_found = ANCHORS.iter().all(|anchor| {
            let full = format!("{}.{}", candidate, anchor);
            tensor_names.iter().any(|n| n == &full)
        });
        if all_found {
            let lm_head_key = format!("{}.lm_head.weight", candidate);
            let embed_key = format!("{}.embed_tokens.weight", candidate);
            let lm_head_exists = tensor_names.iter().any(|n| n == &lm_head_key);
            return Some(NamespaceBinding {
                root: candidate.to_string(),
                discovery: format!("matched {} anchors under '{}'", ANCHORS.len(), candidate),
                lm_head_key: if lm_head_exists {
                    lm_head_key
                } else {
                    embed_key
                },
                lm_head_aliased: !lm_head_exists,
            });
        }
    }
    None
}
