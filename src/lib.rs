use napi::bindgen_prelude::*;
use napi_derive::napi;
use wgpu::{Instance, InstanceDescriptor};

/// Initialize WebGPU adapter and return GPU information
#[napi]
pub fn initialize_webgpu() -> Result<String> {
    // Initialize logging for debugging
    let _ = env_logger::try_init_from_env(env_logger::Env::default().default_filter_or("info"));

    // Create a Tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| Error::from_reason(format!("Failed to create runtime: {}", e)))?;

    // Run async code in blocking context
    rt.block_on(async {
        // Create WebGPU instance with default backends
        let instance = Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter (GPU access)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        match adapter {
            Some(adapter) => {
                // Get adapter info
                let info = adapter.get_info();

                // Format success message with GPU details
                let message = format!(
                    "✓ WebGPU initialized successfully!\n\
                     GPU Name: {}\n\
                     Backend: {:?}\n\
                     Driver: {}\n\
                     Vendor: {:#x}\n\
                     Device: {:#x}",
                    info.name,
                    info.backend,
                    info.driver,
                    info.vendor,
                    info.device
                );

                log::info!("WebGPU adapter acquired: {}", info.name);

                Ok(message)
            }
            None => {
                let error_msg = "✗ Failed to find a suitable GPU adapter";
                log::error!("{}", error_msg);
                Err(Error::from_reason(error_msg))
            }
        }
    })
}

/// Get available GPU adapters without initializing
#[napi]
pub fn list_adapters() -> Result<Vec<String>> {
    let instance = Instance::new(InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // Enumerate all available adapters
    let adapters: Vec<String> = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            format!("{} ({:?})", info.name, info.backend)
        })
        .collect();

    Ok(adapters)
}

/// Get detailed GPU capabilities
#[napi]
pub fn get_gpu_capabilities() -> Result<GpuCapabilities> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| Error::from_reason(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        let instance = Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::from_reason("No GPU adapter found"))?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();

        Ok(GpuCapabilities {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_bind_groups: limits.max_bind_groups,
            // Convert u64 to f64 for JavaScript compatibility
            max_buffer_size: limits.max_buffer_size as f64,
            supports_timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        })
    })
}

/// Struct representing GPU capabilities (exported to Node.js)
#[napi(object)]
pub struct GpuCapabilities {
    pub name: String,
    pub backend: String,
    pub max_texture_dimension_2d: u32,
    pub max_bind_groups: u32,
    // Use f64 instead of u64 for JavaScript compatibility (JS doesn't have 64-bit integers)
    pub max_buffer_size: f64,
    pub supports_timestamps: bool,
}
