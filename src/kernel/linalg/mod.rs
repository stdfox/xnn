//! Linear algebra kernels.
//!
//! Provides kernels for matrix operations on GPU buffers.
//!
//! # Operations
//!
//! - [`transpose`]: Matrix transpose.

mod transpose;

pub use transpose::transpose;
