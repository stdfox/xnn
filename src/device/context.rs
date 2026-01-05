//! GPU context management for buffer and pipeline operations.

use core::any::TypeId;

use alloc::borrow::ToOwned as _;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use bytemuck::Zeroable as _;
use wgpu::util::DeviceExt as _;

use crate::device::{Allocator, PipelineCache};
use crate::{Buffer, Element, Error};

/// Shared inner state for [`Context`].
struct ContextInner {
    allocator: Arc<Allocator>,
    pipelines: PipelineCache,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// GPU device context for buffer and pipeline management.
pub struct Context(Arc<ContextInner>);

impl Context {
    /// Asynchronously creates a GPU context with the system default adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if no suitable adapter is found.
    pub async fn try_default_async() -> Result<Self, Error> {
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::BROWSER_WEBGPU;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|_| Error::Device("no suitable adapter found".to_owned()))?;

        Self::from_adapter_async(&adapter).await
    }

    /// Creates a GPU context with the system default adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if no suitable adapter is found.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_default() -> Result<Self, Error> {
        pollster::block_on(Self::try_default_async())
    }

    /// Asynchronously creates a GPU context from a wgpu adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if device creation fails.
    pub async fn from_adapter_async(adapter: &wgpu::Adapter) -> Result<Self, Error> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| Error::Device(alloc::format!("failed to create device: {e}")))?;

        Ok(Self::from_device_queue(&device, &queue))
    }

    /// Creates a GPU context from a wgpu adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if device creation fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Result<Self, Error> {
        pollster::block_on(Self::from_adapter_async(adapter))
    }

    /// Asynchronously creates a GPU context from adapter index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if adapter index is invalid or device creation fails.
    pub async fn from_adapter_index_async(adapter_index: usize) -> Result<Self, Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let adapter = adapters
            .into_iter()
            .nth(adapter_index)
            .ok_or_else(|| Error::Device(alloc::format!("no adapter at index {adapter_index}")))?;

        Self::from_adapter_async(&adapter).await
    }

    /// Creates a GPU context from adapter index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if adapter index is invalid or device creation fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_adapter_index(adapter_index: usize) -> Result<Self, Error> {
        pollster::block_on(Self::from_adapter_index_async(adapter_index))
    }

    /// Creates a GPU context from existing wgpu device and queue.
    #[must_use]
    pub fn from_device_queue(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let inner = ContextInner {
            allocator: Arc::new(Allocator::new(device.clone())),
            pipelines: PipelineCache::default(),
            device: device.clone(),
            queue: queue.clone(),
        };

        Self(Arc::new(inner))
    }

    /// Blocks until all submitted GPU work completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if device poll fails.
    pub fn poll(&self) -> Result<(), Error> {
        self.0
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Device(alloc::format!("device poll failed: {e}")))?;

        Ok(())
    }

    /// Creates an uninitialized GPU buffer with the given number of elements.
    ///
    /// The buffer is padded to a multiple of 4 elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if buffer size exceeds device limits.
    pub(crate) fn create_buffer<T: Element>(&self, len: usize) -> Result<Buffer<T>, Error> {
        let padded_len = len.div_ceil(4) * 4;
        let padded_size = padded_len * T::NATIVE_SIZE;
        let buffer = self.0.allocator.allocate(padded_size as u64)?;

        Ok(Buffer::new(Arc::clone(&self.0.allocator), buffer, len))
    }

    /// Creates a GPU buffer initialized from a slice.
    ///
    /// The buffer is padded to a multiple of 4 elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if buffer size exceeds device limits.
    pub(crate) fn create_buffer_from_slice<T: Element>(
        &self,
        data: &[T],
    ) -> Result<Buffer<T>, Error> {
        let len = data.len();
        let padded_len = len.div_ceil(4) * 4;
        let padded_size = padded_len * T::NATIVE_SIZE;
        let buffer = self.0.allocator.allocate(padded_size as u64)?;

        let mut native_data: Vec<T::Native> = data.iter().map(|x| x.to_native()).collect();
        native_data.resize(padded_len, T::Native::zeroed());

        self.0
            .queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&native_data));

        Ok(Buffer::new(Arc::clone(&self.0.allocator), buffer, len))
    }

    /// Creates a uniform buffer from a value.
    pub(crate) fn create_uniform_buffer<T: bytemuck::Pod>(&self, value: &T) -> wgpu::Buffer {
        self.0
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Asynchronously copies buffer contents from GPU to CPU memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if the read operation fails.
    pub(crate) async fn read_buffer_async<T: Element>(
        &self,
        buffer: &Buffer<T>,
    ) -> Result<Vec<T>, Error> {
        if buffer.is_empty() {
            return Ok(Vec::new());
        }

        let staging = self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.byte_size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .0
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer.inner(), 0, &staging, 0, buffer.byte_size());
        self.0.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.0
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Device(alloc::format!("device poll failed: {e}")))?;

        rx.await
            .map_err(|_| Error::Device("channel closed".to_owned()))?
            .map_err(|e| Error::Device(alloc::format!("buffer mapping failed: {e}")))?;

        let data = slice.get_mapped_range();
        let native_data: &[T::Native] = bytemuck::cast_slice(&data);
        let result: Vec<T> = native_data.iter().map(|x| T::from_native(*x)).collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Copies buffer contents from GPU to CPU memory.
    ///
    /// Blocks until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Device`] if the read operation fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read_buffer<T: Element>(&self, buffer: &Buffer<T>) -> Result<Vec<T>, Error> {
        pollster::block_on(self.read_buffer_async(buffer))
    }

    /// Gets or creates a cached compute pipeline.
    pub(crate) fn create_compute_pipeline(
        &self,
        id: TypeId,
        shader: impl FnOnce() -> String,
        label: &'static str,
    ) -> Arc<wgpu::ComputePipeline> {
        self.0
            .pipelines
            .create_compute_pipeline(&self.0.device, id, shader, label)
    }

    /// Creates a command encoder for recording device commands.
    pub(crate) fn create_command_encoder(
        &self,
        label: Option<&'static str>,
    ) -> wgpu::CommandEncoder {
        self.0
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Submits command buffers to the device queue.
    pub(crate) fn submit(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.0.queue.submit(commands);
    }

    /// Returns the wgpu device.
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.0.device
    }
}

impl core::fmt::Debug for Context {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Context")
            .field("allocator", &self.0.allocator)
            .field("pipelines", &self.0.pipelines)
            .field("device", &self.0.device)
            .field("queue", &self.0.queue)
            .finish()
    }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
