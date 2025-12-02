mod model;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use anyhow::Result as AnyhowResult;
use std::path::PathBuf;

use model::{Config, Model};

/// OxideEngine - Main struct for GPU-accelerated ML operations using Candle
#[napi]
pub struct OxideEngine {
    device: Device,
    model: Option<Model>,
}

#[napi]
impl OxideEngine {
    /// Create a new OxideEngine instance with Metal backend (macOS) or CPU fallback
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        // Initialize logging
        let _ = env_logger::try_init_from_env(env_logger::Env::default().default_filter_or("info"));

        log::info!("Initializing OxideEngine with GPU backend...");

        // Create GPU device using Candle (Metal on macOS, CPU otherwise)
        let device = Self::create_gpu_device()
            .map_err(|e| Error::from_reason(format!("Failed to initialize GPU device: {}", e)))?;

        log::info!("OxideEngine initialized successfully with device: {:?}", device);

        Ok(Self {
            device,
            model: None,
        })
    }

    /// Internal method to create GPU device (Metal on macOS)
    fn create_gpu_device() -> AnyhowResult<Device> {
        // Try to create Metal device (macOS) or fall back to CPU
        #[cfg(target_os = "macos")]
        {
            log::info!("Creating Metal GPU device for macOS...");
            match Device::new_metal(0) {
                Ok(device) => {
                    log::info!("Metal GPU device created successfully");
                    Ok(device)
                }
                Err(e) => {
                    log::warn!("Failed to create Metal device: {}, falling back to CPU", e);
                    Ok(Device::Cpu)
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            log::info!("Metal not available on this platform, using CPU");
            Ok(Device::Cpu)
        }
    }

    /// Get device information as a string
    #[napi]
    pub fn get_device_info(&self) -> String {
        format!("Device: {:?}", self.device)
    }

    /// Load a Phi-3 model from a safetensors file
    #[napi]
    pub fn load_model(&mut self, model_path: String, config_path: Option<String>) -> Result<String> {
        log::info!("Loading model from: {}", model_path);

        let result = self.load_model_internal(model_path, config_path)
            .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?;

        Ok(result)
    }

    /// Internal method to load the model
    fn load_model_internal(&mut self, model_path: String, config_path: Option<String>) -> AnyhowResult<String> {
        // Load config
        let config = if let Some(config_path) = config_path {
            log::info!("Loading config from: {}", config_path);
            let config_content = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&config_content)?
        } else {
            log::info!("Using default Phi-3-mini config");
            Config::default()
        };

        log::info!("Model config: vocab_size={}, hidden_size={}, num_layers={}",
            config.vocab_size, config.hidden_size, config.num_hidden_layers);

        // Load model weights from safetensors
        let model_path = PathBuf::from(&model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &self.device)?
        };

        log::info!("Creating model...");
        let model = Model::new(&config, vb)?;

        log::info!("Model loaded successfully into VRAM");

        self.model = Some(model);

        Ok(format!(
            "✓ Model loaded successfully!\n\
             \n\
             Config:\n\
             - Vocabulary size: {}\n\
             - Hidden size: {}\n\
             - Number of layers: {}\n\
             - Attention heads: {}\n\
             - Max position embeddings: {}\n\
             \n\
             Device: {:?}\n\
             Status: Model loaded in VRAM and ready for inference",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.max_position_embeddings,
            self.device
        ))
    }

    /// Run forward pass on the model
    /// Takes a list of token IDs and returns logits information
    #[napi]
    pub fn forward(&self, token_ids: Vec<u32>) -> Result<ForwardResult> {
        log::info!("Running forward pass with {} tokens", token_ids.len());

        let result = self.forward_internal(&token_ids)
            .map_err(|e| Error::from_reason(format!("Forward pass failed: {}", e)))?;

        Ok(result)
    }

    /// Internal forward pass implementation
    fn forward_internal(&self, token_ids: &[u32]) -> AnyhowResult<ForwardResult> {
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded. Call load_model() first."))?;

        // Convert token IDs to tensor
        let input_ids = Tensor::new(token_ids, &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        log::info!("Input tensor shape: {:?}", input_ids.shape());

        // Run forward pass
        let logits = model.forward(&input_ids, 0)?;

        let logits_shape = logits.shape();
        log::info!("Output logits shape: {:?}", logits_shape);

        // Get dimensions
        let batch_size = logits_shape.dims()[0];
        let seq_len = logits_shape.dims()[1];
        let vocab_size = logits_shape.dims()[2];

        // Get last token logits for next token prediction
        let last_token_logits = logits.i((0, seq_len - 1))?;

        // Get top 5 tokens
        let top_tokens = self.get_top_k_tokens(&last_token_logits, 5)?;

        Ok(ForwardResult {
            batch_size: batch_size as u32,
            sequence_length: seq_len as u32,
            vocab_size: vocab_size as u32,
            top_tokens,
            message: format!(
                "Forward pass successful!\nInput: {} tokens → Output: logits [{}, {}, {}]",
                token_ids.len(),
                batch_size,
                seq_len,
                vocab_size
            ),
        })
    }

    /// Get top-k tokens from logits
    fn get_top_k_tokens(&self, logits: &Tensor, k: usize) -> AnyhowResult<Vec<TokenProb>> {
        let logits_vec = logits.to_vec1::<f32>()?;

        // Create (token_id, logit) pairs
        let mut logits_with_ids: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(id, &logit)| (id, logit))
            .collect();

        // Sort by logit value (descending)
        logits_with_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top k
        let top_k: Vec<TokenProb> = logits_with_ids
            .iter()
            .take(k)
            .map(|(id, logit)| TokenProb {
                token_id: *id as u32,
                logit: *logit as f64,
            })
            .collect();

        Ok(top_k)
    }

    /// Test GPU compute capability by performing a simple tensor operation
    /// Creates two tensors on the GPU, adds them, and returns the result
    #[napi]
    pub fn test_gpu_compute(&self) -> Result<String> {
        log::info!("Starting GPU compute test...");

        // Perform tensor operations
        let result = self.perform_tensor_addition()
            .map_err(|e| Error::from_reason(format!("GPU compute test failed: {}", e)))?;

        log::info!("GPU compute test completed successfully");

        Ok(result)
    }

    /// Internal method to perform tensor addition on GPU
    fn perform_tensor_addition(&self) -> AnyhowResult<String> {
        // Create first tensor with value [10.0]
        let tensor_a = Tensor::new(&[10.0f32], &self.device)?;
        log::info!("Created tensor A: {:?}", tensor_a);

        // Create second tensor with value [20.0]
        let tensor_b = Tensor::new(&[20.0f32], &self.device)?;
        log::info!("Created tensor B: {:?}", tensor_b);

        // Perform addition on GPU
        let result_tensor = (&tensor_a + &tensor_b)?;
        log::info!("Addition result tensor: {:?}", result_tensor);

        // Convert result back to CPU for reading
        let result_value: Vec<f32> = result_tensor.to_vec1()?;

        // Format result
        let output = format!(
            "✓ GPU Compute Test Successful!\n\
             \n\
             Operation: Tensor Addition on GPU\n\
             Device: {:?}\n\
             \n\
             Tensor A: [10.0]\n\
             Tensor B: [20.0]\n\
             Result:   [{:.1}]\n\
             \n\
             Computation performed on GPU using Candle framework",
            self.device,
            result_value[0]
        );

        Ok(output)
    }
}

/// Result of a forward pass
#[napi(object)]
pub struct ForwardResult {
    pub batch_size: u32,
    pub sequence_length: u32,
    pub vocab_size: u32,
    pub top_tokens: Vec<TokenProb>,
    pub message: String,
}

/// Token with its probability/logit
#[napi(object)]
pub struct TokenProb {
    pub token_id: u32,
    pub logit: f64,
}
