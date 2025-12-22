//! GPU compute kernels.
//!
//! WGSL-based compute shaders for tensor operations. All kernels operate
//! on [`Buffer`] and require a [`Context`].
//!
//! # Categories
//!
//! - **activation** — `relu`, `sigmoid`.
//! - **linalg** — `transpose`.
//! - **reduction** — `sum`.
//! - **shape** — `broadcast_rows`.

mod activation;
mod linalg;
mod reduction;
mod shape;

pub use activation::{relu, sigmoid};
pub use linalg::transpose;
pub use reduction::sum;
pub use shape::broadcast_rows;

use crate::{Buffer, Context, Element};

/// Synchronizes GPU operations.
///
/// Waits for all pending GPU commands to complete.
///
/// # Panics
///
/// - GPU device poll fails.
#[inline]
pub fn sync(ctx: &Context) {
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
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
