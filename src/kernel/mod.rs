//! GPU compute kernels.
//!
//! Provides WGSL-based compute shaders for tensor operations.
//! All kernels operate on [`Buffer`](crate::Buffer) and require a [`GpuContext`](crate::GpuContext).
//!
//! # Categories
//!
//! - [`arithmetic`]: Element-wise arithmetic (add, sub, mul, div, rem, pow, scalar variants)
//! - [`initializer`]: Buffer initialization (fill)
//! - [`linalg`]: Linear algebra (gemm)
//! - [`reduction`]: Reduction operations (sum)

pub mod arithmetic;
pub mod initializer;
pub mod linalg;
pub mod reduction;

pub use arithmetic::{
    add, add_scalar, div, div_scalar, mul, mul_scalar, pow, pow_scalar, rem, rem_scalar, sub,
    sub_scalar,
};
pub use initializer::fill;
pub use linalg::gemm;
pub use reduction::sum;

use crate::Element;
use crate::GpuContext;
use crate::device::Buffer;
use crate::error::Error;

/// Synchronizes GPU operations.
///
/// Waits for all pending GPU commands to complete.
#[inline]
pub fn sync(ctx: &GpuContext) -> Result<(), Error> {
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Device(e.to_string()))?;

    Ok(())
}

/// Debug assertion that buffer belongs to the given context.
#[inline]
pub(crate) fn debug_assert_same_device<T: Element>(ctx: &GpuContext, buf: &Buffer<T>, name: &str) {
    debug_assert!(
        ctx.adapter_index() == buf.adapter_index(),
        "buffer `{name}` belongs to a different device"
    );
}

/// Asserts that two buffers have the same length.
#[inline]
pub(crate) fn assert_same_len<T: Element>(
    a: &Buffer<T>,
    b: &Buffer<T>,
    name: &str,
) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::Kernel(format!(
            "buffer length mismatch: a={}, {name}={}",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

/// Asserts that buffer length matches expected size.
#[inline]
pub(crate) fn assert_len<T: Element>(
    buf: &Buffer<T>,
    expected: usize,
    name: &str,
) -> Result<(), Error> {
    if buf.len() != expected {
        return Err(Error::Kernel(format!(
            "buffer {name} length mismatch: expected {expected}, got {}",
            buf.len()
        )));
    }
    Ok(())
}
