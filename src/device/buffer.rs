//! Typed GPU buffer for element data.

use core::marker::PhantomData;

use alloc::sync::Arc;

use crate::Element;
use crate::device::allocator::Allocator;

/// Typed GPU buffer for element storage.
pub struct Buffer<T: Element> {
    allocator: Arc<Allocator>,
    inner: Option<wgpu::Buffer>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Element> Buffer<T> {
    /// Creates a new buffer wrapper.
    ///
    /// # Panics
    ///
    /// Panics if buffer capacity is less than `len * T::NATIVE_SIZE`.
    pub(crate) fn new(allocator: Arc<Allocator>, buffer: wgpu::Buffer, len: usize) -> Self {
        let byte_size = (len * T::NATIVE_SIZE) as u64;
        let capacity = buffer.size();

        assert!(
            capacity >= byte_size,
            "buffer capacity {capacity} < required size {byte_size}"
        );

        let inner = Some(buffer);

        Self {
            allocator,
            inner,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the underlying wgpu buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has been released. This cannot occur in safe code
    /// because `inner` is only `None` after `drop()`.
    pub(crate) fn inner(&self) -> &wgpu::Buffer {
        self.inner.as_ref().expect("buffer already released")
    }

    /// Returns the binding view of the underlying wgpu buffer.
    pub(crate) fn as_entire_binding(&self) -> wgpu::BindingResource<'_> {
        self.inner().as_entire_binding()
    }

    /// Returns the logical data size in bytes.
    #[must_use]
    pub(crate) fn byte_size(&self) -> u64 {
        (self.len * T::NATIVE_SIZE) as u64
    }

    /// Returns the allocated buffer capacity in bytes.
    #[must_use]
    pub(crate) fn capacity(&self) -> u64 {
        self.inner().size()
    }

    /// Returns the number of elements.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if empty.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Element> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            allocator: Arc::clone(&self.allocator),
            inner: self.inner.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<T: Element> Drop for Buffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.inner.take() {
            self.allocator.release(buffer);
        }
    }
}

impl<T: Element> core::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&alloc::format!("Buffer<{}>", T::wgsl_type()))
            .field("byte_size", &self.byte_size())
            .field("capacity", &self.capacity())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::Context;

    #[test]
    fn test_new() {
        let ctx = Context::try_default().unwrap();

        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert_eq!(buf.byte_size(), 16);
        assert_eq!(buf.capacity(), 256);
        assert_eq!(buf.len(), 4);

        let buf = ctx.create_buffer::<f32>(64).unwrap();
        assert_eq!(buf.byte_size(), 256);
        assert_eq!(buf.capacity(), 256);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_is_empty() {
        let ctx = Context::try_default().unwrap();

        let buf = ctx.create_buffer::<f32>(0).unwrap();
        assert!(buf.is_empty());

        let buf = ctx.create_buffer::<f32>(4).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_debug() {
        let ctx = Context::try_default().unwrap();
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        let debug = alloc::format!("{buf:?}");
        assert!(debug.contains("Buffer<f32>"));
        assert!(debug.contains("byte_size"));
        assert!(debug.contains("capacity"));
        assert!(debug.contains("len"));
    }
}
