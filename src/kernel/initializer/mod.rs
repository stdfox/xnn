//! Buffer initialization kernels.
//!
//! Provides kernels for initializing GPU buffers with values.
//!
//! # Operations
//!
//! - [`fill`]: Fill buffer with a constant value.

mod fill;

pub use fill::fill;
