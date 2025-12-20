//! Typed GPU buffer for element data.

use core::marker::PhantomData;

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
        // create buffer wrapper from wgpu buffer
        let ctx = Context::try_default().unwrap();
        let wgpu_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf: Buffer<f32> = Buffer::new(wgpu_buf, 64);
        assert_eq!(buf.inner().size(), 256);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_len() {
        // returns element count
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_is_empty() {
        // empty buffer has zero length
        let ctx = Context::try_default().unwrap();

        let empty = ctx.create_buffer::<f32>(0).unwrap();
        assert!(empty.is_empty());

        let non_empty = ctx.create_buffer::<f32>(4).unwrap();
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_inner() {
        // access underlying wgpu buffer
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.inner().size(), 16);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_debug() {
        // debug output contains type and length
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        let debug = format!("{buf:?}");
        assert!(debug.contains("Buffer<f32>"));
        assert!(debug.contains("len"));
    }
}