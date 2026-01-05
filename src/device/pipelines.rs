//! Compute pipeline cache.

use core::any::TypeId;

use alloc::string::String;
use alloc::sync::Arc;

use spin::RwLock;
use wgpu::naga::FastHashMap;

/// Thread-safe cache for compute pipelines keyed by type.
#[derive(Default)]
pub(crate) struct PipelineCache {
    cache: RwLock<FastHashMap<TypeId, Arc<wgpu::ComputePipeline>>>,
}

impl PipelineCache {
    /// Gets or creates a cached compute pipeline.
    pub(crate) fn create_compute_pipeline(
        &self,
        device: &wgpu::Device,
        id: TypeId,
        shader: impl FnOnce() -> String,
        label: &'static str,
    ) -> Arc<wgpu::ComputePipeline> {
        if let Some(pipeline) = self.cache.read().get(&id) {
            return Arc::clone(pipeline);
        }

        let mut cache = self.cache.write();

        if let Some(pipeline) = cache.get(&id) {
            return Arc::clone(pipeline);
        }

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });

        let pipeline = Arc::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        );

        cache.insert(id, Arc::clone(&pipeline));

        pipeline
    }

    /// Returns the number of cached compute pipelines.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns `true` if cache is empty.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl core::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct MarkerA;
    struct MarkerB;

    fn create_test_device() -> wgpu::Device {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            let (device, _) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            device
        })
    }

    fn create_test_shader() -> String {
        "@compute @workgroup_size(1) fn main() {}".into()
    }

    #[test]
    fn test_creates_separate_pipelines() {
        let device = create_test_device();
        let cache = PipelineCache::default();

        let first = cache.create_compute_pipeline(
            &device,
            TypeId::of::<MarkerA>(),
            create_test_shader,
            "a",
        );
        let second = cache.create_compute_pipeline(
            &device,
            TypeId::of::<MarkerB>(),
            create_test_shader,
            "b",
        );

        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_returns_same_pipeline() {
        let device = create_test_device();
        let cache = PipelineCache::default();
        let id = TypeId::of::<MarkerA>();

        let first = cache.create_compute_pipeline(&device, id, create_test_shader, "test");
        let second = cache.create_compute_pipeline(&device, id, create_test_shader, "test");

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_shader_closure_called_once() {
        let device = create_test_device();
        let cache = PipelineCache::default();
        let id = TypeId::of::<MarkerA>();
        let call_count = AtomicUsize::new(0);

        let shader = || {
            call_count.fetch_add(1, Ordering::SeqCst);
            create_test_shader()
        };

        cache.create_compute_pipeline(&device, id, shader, "test");
        cache.create_compute_pipeline(&device, id, create_test_shader, "test");

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_len() {
        let device = create_test_device();
        let cache = PipelineCache::default();

        cache.create_compute_pipeline(&device, TypeId::of::<MarkerA>(), create_test_shader, "a");
        assert_eq!(cache.len(), 1);

        cache.create_compute_pipeline(&device, TypeId::of::<MarkerB>(), create_test_shader, "b");
        assert_eq!(cache.len(), 2);

        cache.create_compute_pipeline(&device, TypeId::of::<MarkerA>(), create_test_shader, "a");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let cache = PipelineCache::default();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
