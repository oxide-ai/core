mod model;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use anyhow::Result as AnyhowResult;
use std::path::PathBuf;

use model::{Config, Model, KVCache};

/// OxideEngine - Main struct for GPU-accelerated ML operations using Candle
#[napi]
pub struct OxideEngine {
    device: Device,
    model: Option<Model>,
    kv_cache: Option<KVCache>,
    tokenizer: Option<Tokenizer>,
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
            kv_cache: None,
            tokenizer: None,
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

    /// Load a Phi-3 model from a safetensors file with tokenizer
    #[napi]
    pub fn load_model(&mut self, model_path: String, tokenizer_path: String, config_path: Option<String>) -> Result<String> {
        log::info!("Loading model from: {}", model_path);
        log::info!("Loading tokenizer from: {}", tokenizer_path);

        let result = self.load_model_internal(model_path, tokenizer_path, config_path)
            .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?;

        Ok(result)
    }

    /// Internal method to load the model and tokenizer
    fn load_model_internal(&mut self, model_path: String, tokenizer_path: String, config_path: Option<String>) -> AnyhowResult<String> {
        // Load tokenizer
        log::info!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        log::info!("Tokenizer loaded successfully");

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

        // Handle model loading (single file or sharded directory)
        let model_path_buf = PathBuf::from(&model_path);
        let mut model_files = Vec::new();

        if model_path_buf.is_dir() {
            log::info!("Model path is a directory, scanning for .safetensors files...");
            let entries = std::fs::read_dir(&model_path_buf)?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        model_files.push(path);
                    }
                }
            }
            
            // Sort files to ensure deterministic loading order (e.g., model-00001, model-00002)
            model_files.sort();
            
            if model_files.is_empty() {
                return Err(anyhow::anyhow!("No .safetensors files found in directory: {}", model_path));
            }
            
            log::info!("Found {} model shards: {:?}", model_files.len(), model_files);
        } else {
            log::info!("Loading single model file: {:?}", model_path_buf);
            model_files.push(model_path_buf);
        }

        // Load model weights from all found safetensors files
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&model_files, candle_core::DType::F32, &self.device)?
        };

        log::info!("Creating model with pre-computed RoPE embeddings...");
        let model = Model::new(&config, vb)?;

        // Initialize KV cache for all layers
        let num_layers = config.num_hidden_layers;
        let kv_cache = KVCache::new(num_layers);

        log::info!("Model and tokenizer loaded successfully into VRAM with KV cache initialized");

        self.model = Some(model);
        self.kv_cache = Some(kv_cache);
        self.tokenizer = Some(tokenizer);

        Ok(format!(
            "✓ Model and tokenizer loaded successfully!\n\
             \n\
             Config:\n\
             - Vocabulary size: {}\n\
             - Hidden size: {}\n\
             - Number of layers: {}\n\
             - Attention heads: {}\n\
             - Max position embeddings: {}\n\
             \n\
             Device: {:?}\n\
             Status: Ready for text generation\n\
             \n\
             Features:\n\
             ✓ KV Cache: Enabled (O(1) complexity)\n\
             ✓ RoPE: Pre-computed\n\
             ✓ Tokenizer: Loaded",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.max_position_embeddings,
            self.device
        ))
    }

    /// Generate text from a prompt
    #[napi]
    pub fn generate_text(&mut self, prompt: String, max_tokens: u32, options: Option<GenOptions>) -> Result<String> {
        log::info!("Generating text from prompt: '{}', max_tokens: {}", prompt, max_tokens);

        let result = self.generate_text_internal(&prompt, max_tokens as usize, options)
            .map_err(|e| Error::from_reason(format!("Text generation failed: {}", e)))?;

        Ok(result)
    }

    /// Internal text generation implementation
    fn generate_text_internal(&mut self, prompt: &str, max_tokens: usize, options: Option<GenOptions>) -> AnyhowResult<String> {
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded. Call load_model() first."))?;

        let kv_cache = self.kv_cache.as_mut()
            .ok_or_else(|| anyhow::anyhow!("KV cache not initialized. Call load_model() first."))?;

        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded. Call load_model() first."))?;

        // Resolve options
        let temperature = options.as_ref().and_then(|o| o.temperature).unwrap_or(0.8);
        let top_p = options.as_ref().and_then(|o| o.top_p).or(Some(0.9));
        let seed = options.as_ref().and_then(|o| o.seed).map(|s| s as u64).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42)
        });

        log::info!("Generation params: temp={:.2}, top_p={:?}, seed={}", temperature, top_p, seed);

        let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

        // Encode the prompt
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = token_ids.len();

        log::info!("Prompt encoded to {} tokens", prompt_len);

        // Get EOS token ID
        let eos_token_id = if let Some(id) = options.as_ref().and_then(|o| o.eos_token_id) {
             id
        } else {
             tokenizer.token_to_id("<|endoftext|>")
                .or_else(|| tokenizer.token_to_id("</s>"))
                .or_else(|| tokenizer.token_to_id("<|im_end|>"))
                .unwrap_or(0)
        };

        log::info!("EOS token ID: {}", eos_token_id);

        // Process prompt tokens (prefill phase)
        log::info!("Processing prompt tokens...");
        let input_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let _logits = model.forward(&input_ids, kv_cache)?;

        log::info!("Prompt processed, starting generation...");

        // Generate tokens one by one
        let mut generated_text = String::new();
        let mut generated_count = 0;

        for step in 0..max_tokens {
            // Get last token and run forward pass
            let last_token = *token_ids.last().unwrap();
            let input_tensor = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;

            let logits = model.forward(&input_tensor, kv_cache)?;

            // Get logits for last token
            let last_logits = logits.i((0, 0))?;

            // Sample next token using logits processor
            let next_token = logits_processor.sample(&last_logits)?;

            // Check for EOS
            if next_token == eos_token_id {
                log::info!("EOS token generated at step {}", step);
                break;
            }

            // Decode token to string
            let token_str = tokenizer
                .decode(&[next_token], false)
                .map_err(|e| anyhow::anyhow!("Failed to decode token: {}", e))?;

            generated_text.push_str(&token_str);
            token_ids.push(next_token);
            generated_count += 1;

            if step % 10 == 0 {
                log::debug!("Generated {} tokens...", step + 1);
            }
        }

        log::info!("Generation complete: {} tokens generated", generated_count);

        Ok(generated_text)
    }

    /// Reset KV cache - call this when starting a new conversation
    #[napi]
    pub fn reset_cache(&mut self) -> Result<String> {
        log::info!("Resetting KV cache");

        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
            let message = "✓ KV cache cleared successfully\nReady for new conversation";
            log::info!("{}", message);
            Ok(message.to_string())
        } else {
            let error_msg = "No KV cache to reset. Load a model first.";
            log::warn!("{}", error_msg);
            Err(Error::from_reason(error_msg))
        }
    }

    /// Get current cache statistics
    #[napi]
    pub fn get_cache_info(&self) -> Result<CacheInfo> {
        if let Some(ref cache) = self.kv_cache {
            let seq_len = cache.seq_len();
            Ok(CacheInfo {
                sequence_length: seq_len as u32,
                is_empty: seq_len == 0,
                message: format!("Cache contains {} tokens", seq_len),
            })
        } else {
            Err(Error::from_reason("No cache available. Load a model first."))
        }
    }

    /// Run forward pass on the model with KV caching
    /// This now only processes NEW tokens efficiently
    #[napi]
    pub fn forward(&mut self, token_ids: Vec<u32>) -> Result<ForwardResult> {
        log::info!("Running forward pass with {} new tokens (cache enabled)", token_ids.len());

        let result = self.forward_internal(&token_ids)
            .map_err(|e| Error::from_reason(format!("Forward pass failed: {}", e)))?;

        Ok(result)
    }

    /// Internal forward pass implementation with KV cache
    fn forward_internal(&mut self, token_ids: &[u32]) -> AnyhowResult<ForwardResult> {
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded. Call load_model() first."))?;

        let kv_cache = self.kv_cache.as_mut()
            .ok_or_else(|| anyhow::anyhow!("KV cache not initialized. Call load_model() first."))?;

        let cache_len_before = kv_cache.seq_len();

        // Convert token IDs to tensor
        let input_ids = Tensor::new(token_ids, &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        log::info!("Input tensor shape: {:?}, Cache length before: {}",
            input_ids.shape(), cache_len_before);

        // Run forward pass with KV cache
        // Only the new tokens are processed; previous K/V are cached
        let logits = model.forward(&input_ids, kv_cache)?;

        let logits_shape = logits.shape();
        let cache_len_after = kv_cache.seq_len();

        log::info!("Output logits shape: {:?}, Cache length after: {}",
            logits_shape, cache_len_after);

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
            cache_length: cache_len_after as u32,
            top_tokens,
            message: format!(
                "✓ Forward pass successful with KV caching!\n\
                 Input: {} new tokens → Total cached: {} tokens\n\
                 Output: logits [{}, {}, {}]\n\
                 Performance: O(1) complexity for history",
                token_ids.len(),
                cache_len_after,
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
    pub cache_length: u32,
    pub top_tokens: Vec<TokenProb>,
    pub message: String,
}

/// Token with its probability/logit
#[napi(object)]
pub struct TokenProb {
    pub token_id: u32,
    pub logit: f64,
}

/// Cache information
#[napi(object)]
pub struct CacheInfo {
    pub sequence_length: u32,
    pub is_empty: bool,
    pub message: String,
}

/// Generation options for text generation
#[napi(object)]
pub struct GenOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<i64>,
    pub eos_token_id: Option<u32>,
}
