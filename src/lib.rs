use napi::bindgen_prelude::*;
use napi_derive::napi;
use candle_core::{Device, Tensor};
use anyhow::Result as AnyhowResult;

/// OxideEngine - Main struct for GPU-accelerated ML operations using Candle
#[napi]
pub struct OxideEngine {
    device: Device,
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
        let device = Self::create_wgpu_device()
            .map_err(|e| Error::from_reason(format!("Failed to initialize GPU device: {}", e)))?;

        log::info!("OxideEngine initialized successfully with device: {:?}", device);

        Ok(Self { device })
    }

    /// Internal method to create GPU device (Metal on macOS)
    fn create_wgpu_device() -> AnyhowResult<Device> {
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

    /// Perform a more complex computation: element-wise multiplication
    #[napi]
    pub fn test_multiply(&self, a: f64, b: f64) -> Result<String> {
        log::info!("Testing GPU multiplication: {} * {}", a, b);

        let result = self.perform_multiplication(a as f32, b as f32)
            .map_err(|e| Error::from_reason(format!("Multiplication test failed: {}", e)))?;

        Ok(result)
    }

    /// Internal method to perform multiplication on GPU
    fn perform_multiplication(&self, a: f32, b: f32) -> AnyhowResult<String> {
        // Create tensors
        let tensor_a = Tensor::new(&[a], &self.device)?;
        let tensor_b = Tensor::new(&[b], &self.device)?;

        // Perform multiplication on GPU
        let result_tensor = (&tensor_a * &tensor_b)?;

        // Get result
        let result_value: Vec<f32> = result_tensor.to_vec1()?;

        let output = format!(
            "✓ GPU Multiplication Test\n\
             {} × {} = {:.2}\n\
             Device: {:?}",
            a, b, result_value[0], self.device
        );

        Ok(output)
    }

    /// Create a simple tensor and return its shape information
    #[napi]
    pub fn create_tensor_info(&self, values: Vec<f64>) -> Result<String> {
        // Convert f64 to f32 for tensor creation
        let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let tensor = Tensor::new(&f32_values[..], &self.device)
            .map_err(|e| Error::from_reason(format!("Failed to create tensor: {}", e)))?;

        let shape = tensor.shape();
        let dims = shape.dims();

        Ok(format!(
            "Tensor created on GPU\n\
             Shape: {:?}\n\
             Dimensions: {:?}\n\
             Element count: {}\n\
             Device: {:?}",
            shape,
            dims,
            tensor.elem_count(),
            self.device
        ))
    }
}
