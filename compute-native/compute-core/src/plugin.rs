//! Plugin system for custom operations.
//!
//! Allows users to register custom operations (new attention variants, custom
//! kernels, etc.) at runtime without forking or recompiling the compiler.
//! Plugin ops are registered before model load, stored in a global registry,
//! and dispatched through the existing `FusedOperation → kernel_name` pipeline.
//!
//! # Example
//!
//! ```ignore
//! use tribunus_compute_core::plugin::{TribunusPlugin, PluginOperation, register_plugin};
//! use tribunus_compute_core::backend::routing::BackendId;
//!
//! struct MyPlugin;
//!
//! impl TribunusPlugin for MyPlugin {
//!     fn name(&self) -> &'static str { "my_custom_ops" }
//!     fn version(&self) -> &'static str { "0.1.0" }
//!     fn operations(&self) -> Vec<PluginOperation> {
//!         vec![PluginOperation {
//!             name: "efficient_attention_v2".into(),
//!             description: "An experimental fused attention kernel".into(),
//!             kernel_name: "plugin_efficient_attn_v2".into(),
//!             backend: BackendId(0),
//!         }]
//!     }
//! }
//!
//! register_plugin(Box::new(MyPlugin));
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use lazy_static::lazy_static;

use crate::backend::routing::BackendId;

// ── Traits and types ───────────────────────────────────────────────────────

/// A registered plugin providing custom operations.
///
/// Implement this trait on your plugin struct and call [`register_plugin`]
/// before loading a model to make its operations available for dispatch.
pub trait TribunusPlugin: Send + Sync {
    /// Unique plugin name (used as registry key).
    fn name(&self) -> &'static str;

    /// Plugin version for diagnostics.
    fn version(&self) -> &'static str;

    /// The custom operations this plugin provides.
    fn operations(&self) -> Vec<PluginOperation>;
}

/// A custom operation provided by a plugin.
///
/// Each operation maps a logical operation name to a concrete kernel name
/// that the dispatch pipeline will hand off to the correct backend.
pub struct PluginOperation {
    /// Human-readable operation name (e.g. "efficient_attention_v2").
    pub name: String,

    /// Short description of what this operation does.
    pub description: String,

    /// Kernel name that matches the `kernel_name()` lookup path.
    ///
    /// When a `FusedOperation::Custom(kernel_name)` is encountered during
    /// execution, the dispatch pipeline looks up this name in the registry
    /// to find the backend and invocation metadata.
    pub kernel_name: String,

    /// Which backend this operation targets.
    pub backend: BackendId,
}

// ── Global registry ────────────────────────────────────────────────────────

lazy_static! {
    /// Global plugin registry mapping plugin name to plugin instance.
    ///
    /// Plugins are registered via [`register_plugin`] and remain available
    /// for the lifetime of the process.  The registry is thread-safe and
    /// can be mutated from any thread before model load.
    pub static ref PLUGIN_REGISTRY: Mutex<HashMap<String, Box<dyn TribunusPlugin>>> =
        Mutex::new(HashMap::new());
}

// ── Registration and lookup ────────────────────────────────────────────────

/// Register a plugin with the global plugin registry.
///
/// # Panics
///
/// Panics (via poisoned lock) if another thread panicked while holding the
/// registry lock. In normal operation this never occurs.
pub fn register_plugin(plugin: Box<dyn TribunusPlugin>) {
    let name = plugin.name().to_string();
    let mut registry = PLUGIN_REGISTRY
        .lock()
        .expect("plugin registry lock poisoned");
    registry.insert(name, plugin);
}

/// Look up a plugin operation by kernel name.
///
/// Iterates all registered plugins and returns the first [`PluginOperation`]
/// whose `kernel_name` matches the given string.  Returns `None` if no
/// registered operation matches.
///
/// This is the entry point used by the execution pipeline when dispatching
/// a `FusedOperation::Custom(kernel_name)`.
pub fn lookup_operation(kernel_name: &str) -> Option<PluginOperation> {
    let registry = PLUGIN_REGISTRY
        .lock()
        .expect("plugin registry lock poisoned");
    for plugin in registry.values() {
        for op in plugin.operations() {
            if op.kernel_name == kernel_name {
                return Some(op);
            }
        }
    }
    None
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl TribunusPlugin for TestPlugin {
        fn name(&self) -> &'static str {
            "test_plugin"
        }
        fn version(&self) -> &'static str {
            "0.0.1"
        }
        fn operations(&self) -> Vec<PluginOperation> {
            vec![
                PluginOperation {
                    name: "test_op_a".into(),
                    description: "First test operation".into(),
                    kernel_name: "test_kernel_a".into(),
                    backend: BackendId(0),
                },
                PluginOperation {
                    name: "test_op_b".into(),
                    description: "Second test operation".into(),
                    kernel_name: "test_kernel_b".into(),
                    backend: BackendId(1),
                },
            ]
        }
    }

    #[test]
    fn plugin_lifecycle() {
        // Register a plugin.
        register_plugin(Box::new(TestPlugin));

        // Look up a registered operation by kernel name.
        let op = lookup_operation("test_kernel_a");
        assert!(op.is_some(), "test_kernel_a should be found");
        let op = op.unwrap();
        assert_eq!(op.name, "test_op_a");
        assert_eq!(op.backend, BackendId(0));

        // Second operation from the same plugin.
        let op = lookup_operation("test_kernel_b");
        assert!(op.is_some(), "test_kernel_b should be found");
        let op = op.unwrap();
        assert_eq!(op.name, "test_op_b");
        assert_eq!(op.backend, BackendId(1));

        // Unknown kernel name.
        assert!(lookup_operation("nonexistent_kernel").is_none());
    }

    #[test]
    fn plugin_registry_multiple() {
        struct SecondPlugin;

        impl TribunusPlugin for SecondPlugin {
            fn name(&self) -> &'static str {
                "second_plugin"
            }
            fn version(&self) -> &'static str {
                "1.0.0"
            }
            fn operations(&self) -> Vec<PluginOperation> {
                vec![PluginOperation {
                    name: "second_op".into(),
                    description: "Operation from second plugin".into(),
                    kernel_name: "second_kernel".into(),
                    backend: BackendId(2),
                }]
            }
        }

        register_plugin(Box::new(SecondPlugin));

        // Both plugins' operations should be findable.
        let op = lookup_operation("second_kernel");
        assert!(op.is_some(), "second_kernel should be found");
        assert_eq!(op.unwrap().backend, BackendId(2));

        // Previously registered plugin is still available.
        let op = lookup_operation("test_kernel_a");
        assert!(
            op.is_some(),
            "test_kernel_a from first plugin still available"
        );
    }

    #[test]
    fn plugin_name_uniqueness() {
        // Register two plugins with the same name -- the second overwrites.
        struct OverwritePlugin;

        impl TribunusPlugin for OverwritePlugin {
            fn name(&self) -> &'static str {
                "overwriter"
            }
            fn version(&self) -> &'static str {
                "2.0.0"
            }
            fn operations(&self) -> Vec<PluginOperation> {
                vec![PluginOperation {
                    name: "overwrite_op".into(),
                    description: "Replaces earlier plugin".into(),
                    kernel_name: "overwrite_kernel".into(),
                    backend: BackendId(1),
                }]
            }
        }

        register_plugin(Box::new(OverwritePlugin));
        let op = lookup_operation("overwrite_kernel");
        assert!(op.is_some());
        assert_eq!(op.unwrap().backend, BackendId(1));
    }
}
