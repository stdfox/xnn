//! GPU context management for buffer and pipeline operations.

use alloc::sync::Arc;
use core::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError, RwLock};

use wgpu::util::DeviceExt as _;

use crate::{Buffer, Element, Error};

/// Cache for compute pipelines keyed by type.
type ComputePipelineCache = RwLock<HashMap<TypeId, Arc<wgpu::ComputePipeline>>>;

/// Global pool of GPU contexts keyed by adapter index.
type ContextPool = OnceLock<Mutex<HashMap<usize, Arc<GpuContextInner>>>>;

/// Global context pool instance.
static POOL: ContextPool = OnceLock::new();

/// Internal GPU context state shared via Arc.
struct GpuContextInner {
    adapter_index: usize,
    adapter_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    cache: ComputePipelineCache,
}

/// Central GPU context for buffer and pipeline management.
///
/// Wraps wgpu device and queue, caches compute pipelines by element type.
///
/// Contexts are pooled by adapter index — creating multiple contexts for
/// the same adapter returns the same underlying resources.
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

impl GpuContext {
    /// Creates a GPU context for the specified adapter index.
    ///
    /// Contexts are pooled — calling with the same index returns a clone
    /// of the existing context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if no adapter exists at the given index
    /// or device creation fails.
    pub fn new(adapter_index: usize) -> Result<Self, Error> {
        let pool = POOL.get_or_init(|| Mutex::new(HashMap::new()));
        let mut pool = pool.lock().map_err(|e| Error::Device(e.to_string()))?;

        if let Some(inner) = pool.get(&adapter_index) {
            return Ok(Self {
                inner: Arc::clone(inner),
            });
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter = adapters
            .into_iter()
            .nth(adapter_index)
            .ok_or_else(|| Error::Device(format!("no adapter at index {adapter_index}")))?;

        let adapter_name = adapter.get_info().name;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| Error::Device(format!("failed to create device: {e}")))?;

        let inner = Arc::new(GpuContextInner {
            adapter_index,
            adapter_name,
            device,
            queue,
            cache: RwLock::new(HashMap::new()),
        });
        pool.insert(adapter_index, Arc::clone(&inner));

        Ok(Self { inner })
    }

    /// Creates an uninitialized GPU buffer with the given number of elements.
    ///
    /// The buffer is padded to a multiple of 4 elements for vec4 optimization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if buffer size overflows.
    pub fn create_buffer<T: Element>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let padded_len = len
            .div_ceil(4)
            .checked_mul(4)
            .ok_or_else(|| Error::Device("buffer length overflow".into()))?;
        let size: u64 = padded_len
            .checked_mul(core::mem::size_of::<T>())
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| Error::Device("buffer size overflow".into()))?;
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Buffer::new(buffer, len, self.inner.adapter_index))
    }

    /// Creates a GPU buffer initialized with data copied from a slice.
    ///
    /// The buffer is padded to a multiple of 4 elements with zeros.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if buffer length overflows.
    pub fn create_buffer_from_slice<T: Element>(&self, data: &[T]) -> Result<Buffer<T>, Error> {
        let padded_len = data
            .len()
            .div_ceil(4)
            .checked_mul(4)
            .ok_or_else(|| Error::Device("buffer length overflow".into()))?;
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_len, T::zeroed());

        let buffer = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&padded_data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Buffer::new(buffer, data.len(), self.inner.adapter_index))
    }

    /// Copies buffer contents from GPU to CPU memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if the read operation fails.
    pub fn read_buffer<T: Element>(&self, buffer: &Buffer<T>) -> Result<Vec<T>, Error> {
        if buffer.is_empty() {
            return Ok(Vec::new());
        }

        let size: u64 = buffer
            .len()
            .checked_mul(core::mem::size_of::<T>())
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| Error::Device("buffer size overflow".into()))?;

        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer.inner(), 0, &staging, 0, size);
        self.inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result).map_err(|e| Error::Device(e.to_string()));
        });

        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Device(format!("device poll failed: {e}")))?;

        rx.recv()
            .map_err(|_| Error::Device("internal channel error".into()))?
            .map_err(|e| Error::Device(format!("buffer mapping failed: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Returns the adapter index.
    pub(crate) fn adapter_index(&self) -> usize {
        self.inner.adapter_index
    }

    /// Returns the wgpu device.
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Returns the wgpu queue.
    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// Gets or creates a cached compute pipeline for type `T` and factory `F`.
    pub(crate) fn get_or_create_pipeline<T: 'static, F>(
        &self,
        create_fn: F,
    ) -> Arc<wgpu::ComputePipeline>
    where
        F: FnOnce(&wgpu::Device) -> wgpu::ComputePipeline + 'static,
    {
        let type_id = TypeId::of::<(T, F)>();

        {
            let cache = self
                .inner
                .cache
                .read()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(pipeline) = cache.get(&type_id) {
                return Arc::clone(pipeline);
            }
        }

        let mut cache = self
            .inner
            .cache
            .write()
            .unwrap_or_else(PoisonError::into_inner);

        if let Some(pipeline) = cache.get(&type_id) {
            return Arc::clone(pipeline);
        }

        let pipeline = Arc::new(create_fn(&self.inner.device));
        cache.insert(type_id, Arc::clone(&pipeline));

        pipeline
    }
}

impl Default for GpuContext {
    /// Creates a GPU context with a high-performance adapter.
    ///
    /// # Panics
    ///
    /// Panics if no suitable GPU adapter is found or device creation fails.
    fn default() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter_info = adapter.get_info();
        let adapter_index = adapters
            .iter()
            .position(|a| a.get_info().name == adapter_info.name)
            .unwrap_or(0);

        Self::new(adapter_index).expect("failed to create GPU context")
    }
}

impl core::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter_index", &self.inner.adapter_index)
            .field("adapter_name", &self.inner.adapter_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let ctx = GpuContext::new(0).unwrap();
        assert_eq!(ctx.inner.adapter_index, 0);
    }

    #[test]
    fn test_create_buffer() {
        let ctx = GpuContext::default();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.adapter_index(), ctx.inner.adapter_index);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.inner().size(), 16);
    }

    #[test]
    fn test_create_buffer_from_slice() {
        let ctx = GpuContext::default();
        let buf = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(buf.adapter_index(), ctx.inner.adapter_index);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_read_buffer() {
        let ctx = GpuContext::default();
        let buf = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let data = ctx.read_buffer(&buf).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_adapter_index() {
        let ctx = GpuContext::default();
        let _ = ctx.adapter_index();
    }

    #[test]
    fn test_device() {
        let ctx = GpuContext::default();
        let _ = ctx.device().limits();
    }

    #[test]
    fn test_queue() {
        let ctx = GpuContext::default();
        ctx.queue().submit(std::iter::empty());
    }

    #[test]
    fn test_get_or_create_pipeline() {
        let ctx = GpuContext::default();
        let pipeline = ctx.get_or_create_pipeline::<f32, _>(|device| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        assert!(Arc::strong_count(&pipeline) >= 1);
    }

    #[test]
    fn test_default() {
        let ctx = GpuContext::default();
        assert!(!ctx.inner.adapter_name.is_empty());
    }

    #[test]
    fn test_debug() {
        let ctx = GpuContext::default();
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("GpuContext"));
        assert!(debug.contains("adapter_index"));
        assert!(debug.contains("adapter_name"));
    }
}
