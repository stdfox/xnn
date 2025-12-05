//! GPU compute kernels.
//!
//! Provides WGSL-based compute shaders for tensor operations.
//! All kernels operate on [`Buffer`](crate::Buffer) and require a [`GpuContext`](crate::GpuContext).
//!
//! # Categories
//!
//! - [`arithmetic`]: Element-wise arithmetic (add, sub, mul, div, rem, pow)
//! - [`initializer`]: Buffer initialization (fill)
//! - [`linalg`]: Linear algebra (gemm)

pub mod arithmetic;
pub mod initializer;
pub mod linalg;

pub use arithmetic::{add, div, mul, pow, rem, sub};
pub use initializer::fill;
pub use linalg::gemm;

use crate::Element;
use crate::device::Buffer;
use crate::error::Error;

/// Asserts that two buffers have the same length.
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
