//! GPU compute kernels.
//!
//! WGSL-based compute shaders for tensor operations. All kernels operate
//! on [`Buffer`] and require a [`GpuContext`].
//!
//! # Categories
//!
//! - **activation** — `relu`, `sigmoid`.
//! - **arithmetic** — `add`, `sub`, `mul`, `div`, `rem`, `pow` (and scalar variants).
//! - **initializer** — `fill`.
//! - **linalg** — `gemm`, `transpose`.
//! - **reduction** — `sum`.
//! - **shape** — `broadcast_rows`.

mod activation;
mod arithmetic;
mod initializer;
mod linalg;
mod reduction;
mod shape;

pub use activation::{relu, sigmoid};
pub use arithmetic::{
    add, add_scalar, div, div_scalar, mul, mul_scalar, pow, pow_scalar, rem, rem_scalar, sub,
    sub_scalar,
};
pub use initializer::fill;
pub use linalg::{gemm, transpose};
pub use reduction::sum;
pub use shape::broadcast_rows;

use crate::Element;
use crate::GpuContext;
use crate::device::Buffer;

/// Synchronizes GPU operations.
///
/// Waits for all pending GPU commands to complete.
///
/// # Panics
///
/// - GPU device poll fails.
#[inline]
pub fn sync(ctx: &GpuContext) {
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
}

/// Debug assertion that buffer belongs to the given context.
#[inline]
pub(crate) fn debug_assert_same_device<T: Element>(ctx: &GpuContext, buf: &Buffer<T>, name: &str) {
    debug_assert!(
        ctx.adapter_index() == buf.adapter_index(),
        "buffer `{name}` belongs to a different device"
    );
}

/// Debug assertion that two buffers have the same length.
#[inline]
pub(crate) fn debug_assert_same_len<T: Element, U: Element>(
    a: &Buffer<T>,
    b: &Buffer<U>,
    name: &str,
) {
    debug_assert!(
        a.len() == b.len(),
        "buffer length mismatch: a={}, {name}={}",
        a.len(),
        b.len()
    );
}

/// Debug assertion that buffer length matches expected size.
#[inline]
pub(crate) fn debug_assert_len<T: Element>(buf: &Buffer<T>, expected: usize, name: &str) {
    debug_assert!(
        buf.len() == expected,
        "buffer `{name}` length mismatch: expected {expected}, got {}",
        buf.len()
    );
}
