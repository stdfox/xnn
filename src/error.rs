//! Error types for GPU operations.
//!
//! - [`Error`] — top-level error type.
//! - [`TensorError`] — tensor-specific errors.

use alloc::string::String;

/// Top-level error type for GPU operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Tensor operation error.
    #[error(transparent)]
    Tensor(#[from] TensorError),

    /// GPU device operation failed.
    #[error("{0}")]
    Device(String),
}

/// Errors from tensor operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TensorError {
    /// Invalid shape for operation.
    #[error("invalid shape: {0}")]
    InvalidShape(String),
}
