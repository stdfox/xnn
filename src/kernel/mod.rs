//! GPU compute kernels.
//!
//! Provides WGSL-based compute shaders for tensor operations.
//! All kernels operate on [`Buffer`](crate::Buffer) and require a [`GpuContext`](crate::GpuContext).
//!
//! # Categories
//!
//! - [`initializer`]: Buffer initialization (fill)

pub mod initializer;

pub use initializer::fill;
