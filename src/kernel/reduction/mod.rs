//! Reduction kernels.
//!
//! Provides kernels for reducing buffers to scalar values.
//!
//! # Operations
//!
//! - [`sum`]: Sum all elements in a buffer.

mod sum;

pub use sum::sum;
