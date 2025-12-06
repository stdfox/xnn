//! Linear algebra kernels.
//!
//! Provides kernels for matrix operations on GPU buffers.
//!
//! # Operations
//!
//! - [`gemm`]: General matrix multiplication (C = A × B), f32 only.
//! - [`transpose`]: Matrix transpose.

mod gemm;
mod transpose;

pub use gemm::gemm;
pub use transpose::transpose;
