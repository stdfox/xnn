//! GPU memory allocator with pooling.

use core::sync::atomic::{AtomicU64, Ordering};

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use spin::Mutex;
use wgpu::BufferUsages;

use crate::Error;

/// Minimum buffer allocation size: `0x100` (256 bytes).
const MIN_BUFFER_SIZE: u64 = 0x100;

/// Buffer pool: maps buffer size to cached buffers.
type Pool = BTreeMap<u64, Vec<wgpu::Buffer>>;

/// GPU memory allocator with buffer pooling.
pub(crate) struct Allocator {
    /// GPU device for buffer creation.
    device: wgpu::Device,
    /// Cached buffers grouped by size.
    pool: Mutex<Pool>,
    /// Memory used by active allocations in bytes.
    allocated: AtomicU64,
    /// Memory reserved in pool in bytes.
    reserved: AtomicU64,
    /// Maximum single buffer size from device limits.
    max_buffer_size: u64,
    /// Minimum allocation unit size.
    min_buffer_size: u64,
    /// Maximum total pool size from device limits.
    max_pool_size: u64,
}

impl Allocator {
    /// Creates allocator with device limits.
    pub(crate) fn new(device: wgpu::Device) -> Self {
        let limits = device.limits();

        Self {
            device,
            pool: Mutex::new(BTreeMap::new()),
            allocated: AtomicU64::new(0),
            reserved: AtomicU64::new(0),
            max_buffer_size: u64::from(limits.max_storage_buffer_binding_size),
            min_buffer_size: MIN_BUFFER_SIZE,
            max_pool_size: limits.max_buffer_size,
        }
    }

    /// Allocates buffer: `size = ceil(size / min_buffer_size) * min_buffer_size`.
    ///
    /// Uses best-fit strategy: selects smallest available buffer from pool
    /// that satisfies the requested size. Creates new buffer if none found.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if size exceeds `max_buffer_size` limit.
    pub(crate) fn allocate(&self, size: u64) -> Result<wgpu::Buffer, Error> {
        let buffer_size = size.div_ceil(self.min_buffer_size).max(1) * self.min_buffer_size;

        if buffer_size > self.max_buffer_size {
            return Err(Error::Device(alloc::format!(
                "buffer size {buffer_size} bytes exceeds limit ({} bytes)",
                self.max_buffer_size
            )));
        }

        {
            let mut pool = self.pool.lock();
            if let Some(&key) = pool.range(buffer_size..).next().map(|(k, _)| k)
                && let Some(buffers) = pool.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                let is_empty = buffers.is_empty();
                if is_empty {
                    pool.remove(&key);
                }
                self.allocated.fetch_add(buffer.size(), Ordering::Relaxed);
                self.reserved.fetch_sub(buffer.size(), Ordering::Relaxed);
                return Ok(buffer);
            }
        }

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.allocated.fetch_add(buffer.size(), Ordering::Relaxed);

        Ok(buffer)
    }

    /// Returns buffer to pool for reuse.
    ///
    /// Buffer is dropped if pool size would exceed `max_pool_size`.
    pub(crate) fn release(&self, buffer: wgpu::Buffer) {
        let mut pool = self.pool.lock();
        let size = buffer.size();

        self.allocated.fetch_sub(size, Ordering::Relaxed);

        if self.reserved.load(Ordering::Relaxed) + size > self.max_pool_size {
            return;
        }

        self.reserved.fetch_add(size, Ordering::Relaxed);
        pool.entry(size).or_default().push(buffer);
    }

    /// Returns memory used by active allocations in bytes.
    #[allow(dead_code)]
    pub(crate) fn memory_allocated(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Returns memory reserved in pool in bytes.
    #[allow(dead_code)]
    pub(crate) fn memory_reserved(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }

    /// Returns total memory in bytes.
    #[allow(dead_code)]
    pub(crate) fn memory_total(&self) -> u64 {
        self.memory_allocated() + self.memory_reserved()
    }
}

impl core::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Allocator")
            .field("memory_allocated", &self.memory_allocated())
            .field("memory_reserved", &self.memory_reserved())
            .field("max_buffer_size", &self.max_buffer_size)
            .field("min_buffer_size", &self.min_buffer_size)
            .field("max_pool_size", &self.max_pool_size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_device() -> wgpu::Device {
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

    #[test]
    fn test_allocate_new_buffer() {
        let device = create_device();
        let allocator = Allocator::new(device);

        let buffer = allocator.allocate(1).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_reserved(), 0);
    }

    #[test]
    fn test_release_and_reuse() {
        let device = create_device();
        let allocator = Allocator::new(device);

        let buffer = allocator.allocate(1).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_reserved(), 0);

        allocator.release(buffer);
        assert_eq!(allocator.memory_allocated(), 0);
        assert_eq!(allocator.memory_reserved(), allocator.min_buffer_size);

        let buffer = allocator.allocate(2).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_reserved(), 0);
    }

    #[test]
    fn test_memory_metrics() {
        let device = create_device();
        let allocator = Allocator::new(device);

        assert_eq!(allocator.memory_allocated(), 0);
        assert_eq!(allocator.memory_reserved(), 0);
        assert_eq!(allocator.memory_total(), 0);

        let buffer1 = allocator.allocate(1).unwrap();
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_reserved(), 0);
        assert_eq!(allocator.memory_total(), allocator.min_buffer_size);

        let buffer2 = allocator.allocate(1).unwrap();
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size * 2);
        assert_eq!(allocator.memory_reserved(), 0);
        assert_eq!(allocator.memory_total(), allocator.min_buffer_size * 2);

        allocator.release(buffer1);
        assert_eq!(allocator.memory_allocated(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_reserved(), allocator.min_buffer_size);
        assert_eq!(allocator.memory_total(), allocator.min_buffer_size * 2);

        allocator.release(buffer2);
        assert_eq!(allocator.memory_allocated(), 0);
        assert_eq!(allocator.memory_reserved(), allocator.min_buffer_size * 2);
        assert_eq!(allocator.memory_total(), allocator.min_buffer_size * 2);
    }

    #[test]
    fn test_sizes() {
        let device = create_device();
        let allocator = Allocator::new(device);

        let buffer = allocator.allocate(allocator.min_buffer_size - 1).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size);

        let buffer = allocator.allocate(allocator.min_buffer_size).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size);

        let buffer = allocator.allocate(allocator.min_buffer_size + 1).unwrap();
        assert_eq!(buffer.size(), allocator.min_buffer_size * 2);
    }
}
