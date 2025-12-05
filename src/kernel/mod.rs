//! GPU compute kernels.
//!
//! Provides WGSL-based compute shaders for tensor operations.
//! All kernels operate on [`Buffer`](crate::Buffer) and require a [`GpuContext`](crate::GpuContext).
//!
//! # Categories
//!
//! - [`arithmetic`]: Element-wise arithmetic (add, sub, mul, div, rem, pow)
//! - [`initializer`]: Buffer initialization (fill)

pub mod arithmetic;
pub mod initializer;

pub use arithmetic::{add, div, mul, pow, rem, sub};
pub use initializer::fill;

use crate::Element;
use crate::device::Buffer;
use crate::error::Error;

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
