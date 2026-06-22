//! Model registry: list available models, load from Hub, compile to ComputeImage.

use std::path::PathBuf;
use std::sync::Arc;

/// A model available for inference
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameter_size: String,
    pub quantization: String,
    pub is_loaded: bool,
    pub source: ModelSource,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ModelSource {
    HuggingFace { repo: String, filename: String },
    Local { path: PathBuf },
    URL { url: String },
}

/// Model registry — manages model lifecycle
pub struct ModelRegistry {
    models: Vec<ModelEntry>,
    /// After compilation, stores the ComputeImage path
    compiled_images: std::collections::HashMap<String, PathBuf>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            compiled_images: std::collections::HashMap::new(),
        }
    }

    /// List all available models (from registry + cache)
    pub fn list(&self) -> &[ModelEntry] {
        &self.models
    }

    /// Add a model to the registry
    pub fn register(&mut self, entry: ModelEntry) {
        self.models.push(entry);
    }

    /// Recommend the best model for a given task and hardware
    pub fn recommend(&self, _task: &str) -> Option<&ModelEntry> {
        // Simple heuristic: pick the model that best fits available RAM
        self.models.first()
    }

    /// Download a model from HuggingFace
    pub async fn download_hf(
        repo: &str,
        filename: &str,
        cache_dir: &PathBuf,
    ) -> Result<PathBuf, String> {
        // TODO: use hf-hub crate for proper download
        Err("download not yet implemented — use local path instead".into())
    }

    /// Compile a model into a ComputeImage
    pub fn compile(&mut self, id: &str, source_dir: &PathBuf) -> Result<(), String> {
        // Use the existing ComputeImage compiler
        // compile_unchecked reads config.json + safetensors
        // and produces execution-ordered segments
        let _ = id;
        let _ = source_dir;
        Err(
            "ComputeImage compilation: use the existing compute_image::compile route in server"
                .into(),
        )
    }

    /// Check if a model is loaded and ready
    pub fn is_loaded(&self, id: &str) -> bool {
        self.compiled_images.contains_key(id)
    }
}

/// Get a list of recommended models based on hardware and task
pub fn recommend_models(_chip: &str, ram_gb: u64, _task: &str) -> Vec<ModelEntry> {
    let mut entries = Vec::new();

    // Small models that run on any Apple Silicon
    entries.push(ModelEntry {
        id: "gemma-2-2b-it".into(),
        name: "Gemma 2 2B Instruct".into(),
        description: "Google's lightweight instruction-tuned model".into(),
        parameter_size: "2B".into(),
        quantization: "Q4_K".into(),
        is_loaded: false,
        source: ModelSource::HuggingFace {
            repo: "google/gemma-2-2b-it".into(),
            filename: "model.safetensors".into(),
        },
    });

    entries.push(ModelEntry {
        id: "qwen2.5-1.5b-instruct".into(),
        name: "Qwen 2.5 1.5B Instruct".into(),
        description: "Alibaba's efficient instruction model".into(),
        parameter_size: "1.5B".into(),
        quantization: "Q4_K".into(),
        is_loaded: false,
        source: ModelSource::HuggingFace {
            repo: "Qwen/Qwen2.5-1.5B-Instruct".into(),
            filename: "model.safetensors".into(),
        },
    });

    // Medium models for 8GB+ RAM
    if ram_gb >= 8 {
        entries.push(ModelEntry {
            id: "mistral-7b-instruct".into(),
            name: "Mistral 7B Instruct".into(),
            description: "Mistral AI's 7B instruction model".into(),
            parameter_size: "7B".into(),
            quantization: "Q4_K".into(),
            is_loaded: false,
            source: ModelSource::HuggingFace {
                repo: "mistralai/Mistral-7B-Instruct-v0.3".into(),
                filename: "model.safetensors".into(),
            },
        });
    }

    // Large models for 24GB+ RAM
    if ram_gb >= 24 {
        entries.push(ModelEntry {
            id: "qwen2.5-14b-instruct".into(),
            name: "Qwen 2.5 14B Instruct".into(),
            description: "Alibaba's powerful 14B model".into(),
            parameter_size: "14B".into(),
            quantization: "Q4_K".into(),
            is_loaded: false,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2.5-14B-Instruct".into(),
                filename: "model.safetensors".into(),
            },
        });
    }

    entries
}
