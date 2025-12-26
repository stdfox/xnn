//! Typed GPU buffer for element data.

use core::marker::PhantomData;

use alloc::format;

use crate::Element;

/// Typed GPU buffer for element storage.
#[derive(Clone)]
pub struct Buffer<T: Element> {
    inner: wgpu::Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Element> Buffer<T> {
    /// Creates a new buffer wrapper.
    pub(crate) fn new(buffer: wgpu::Buffer, len: usize) -> Self {
        Self {
            inner: buffer,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the buffer size in bytes.
    pub(crate) fn byte_size(&self) -> u64 {
        self.inner.size()
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
        &self.inner
    }
}

impl<T: Element> core::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("Buffer<{}>", T::wgsl_type()))
            .field("byte_size", &self.inner.size())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::Context;

    use super::*;

    #[test]
    fn test_new() {
        let ctx = Context::try_default().unwrap();
        let wgpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf: Buffer<f32> = Buffer::new(wgpu_buf, 64);
        assert_eq!(buf.byte_size(), 256);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_byte_size() {
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.byte_size(), 16);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_len() {
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.byte_size(), 16);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_is_empty() {
        let ctx = Context::try_default().unwrap();

        let buf = ctx.create_buffer::<f32>(0).unwrap();
        assert_eq!(buf.byte_size(), 0);
        assert!(buf.is_empty());

        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_debug() {
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        let debug = format!("{buf:?}");
        assert!(debug.contains("Buffer<f32>"));
        assert!(debug.contains("byte_size"));
        assert!(debug.contains("len"));
    }
}
