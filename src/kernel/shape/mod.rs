//! Shape manipulation kernels.
//!
//! Provides kernels for tensor shape operations on GPU buffers.
//!
//! # Operations
//!
//! - [`broadcast_rows`]: Broadcast a vector to all rows of a matrix.

mod broadcast;

pub use broadcast::broadcast_rows;
