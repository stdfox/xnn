//! Typed GPU buffer for element data.

use core::marker::PhantomData;

use crate::Element;

/// Typed wrapper around a GPU buffer.
pub struct Buffer<T: Element> {
    #[allow(clippy::struct_field_names)]
    buffer: wgpu::Buffer,
    len: usize,
    adapter_index: usize,
    _marker: PhantomData<T>,
}

impl<T: Element> Buffer<T> {
    /// Creates a new buffer wrapper.
    pub(crate) fn new(buffer: wgpu::Buffer, len: usize, adapter_index: usize) -> Self {
        Self {
            buffer,
            len,
            adapter_index,
            _marker: PhantomData,
        }
    }

    /// Returns the adapter index this buffer belongs to.
    pub(crate) fn adapter_index(&self) -> usize {
        self.adapter_index
    }

    /// Returns the number of elements.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the underlying wgpu buffer.
    pub(crate) fn inner(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use crate::GpuContext;

    use super::*;

    #[test]
    fn test_new() {
        let ctx = GpuContext::default();
        let wgpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf: Buffer<f32> = Buffer::new(wgpu_buf, 64, ctx.adapter_index());
        assert_eq!(buf.adapter_index(), ctx.adapter_index());
        assert_eq!(buf.inner().size(), 256);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_adapter_index() {
        let ctx = GpuContext::default();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.adapter_index(), ctx.adapter_index());
    }

    #[test]
    fn test_len() {
        let ctx = GpuContext::default();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_is_empty() {
        let ctx = GpuContext::default();

        let empty = ctx.create_buffer::<f32>(0).unwrap();
        assert!(empty.is_empty());

        let non_empty = ctx.create_buffer::<f32>(4).unwrap();
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_inner() {
        let ctx = GpuContext::default();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.inner().size(), 16);
        assert_eq!(buf.len(), 4);
    }
}
