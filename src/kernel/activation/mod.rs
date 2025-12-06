//! Activation function kernels.
//!
//! Provides kernels for neural network activation functions on GPU buffers.
//!
//! # Operations
//!
//! - [`relu`]: Rectified Linear Unit (b = max(a, 0)), f32 only.
//! - [`sigmoid`]: Sigmoid activation (b = 1 / (1 + exp(-a))), f32 only.

mod relu;
mod sigmoid;

pub use relu::relu;
pub use sigmoid::sigmoid;
