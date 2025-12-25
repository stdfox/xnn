//! Kernel source generation for GPU compute shaders.

pub(crate) mod activation;
pub(crate) mod ops;

use alloc::sync::Arc;
use core::hash::{Hash, Hasher};

use rustc_hash::FxHasher;
use spin::RwLock;
use wgpu::naga::FastHashMap;

/// Compiler for GPU compute shaders with caching.
pub(crate) struct KernelCompiler {
    cache: RwLock<FastHashMap<u64, Arc<wgpu::ComputePipeline>>>,
    device: wgpu::Device,
}

impl KernelCompiler {
    /// Creates a new kernel compiler.
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            cache: RwLock::new(FastHashMap::default()),
            device: device.clone(),
        }
    }

    /// Compiles WGSL source code, using cache if available.
    pub fn compile(&self, wgsl: &str, label: &str) -> Arc<wgpu::ComputePipeline> {
        let hash = {
            let mut hasher = FxHasher::default();
            wgsl.hash(&mut hasher);
            hasher.finish()
        };

        if let Some(pipeline) = self.cache.read().get(&hash) {
            return Arc::clone(pipeline);
        }

        let mut cache = self.cache.write();

        if let Some(pipeline) = cache.get(&hash) {
            return Arc::clone(pipeline);
        }

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        cache.insert(hash, Arc::clone(&pipeline));

        pipeline
    }
}
