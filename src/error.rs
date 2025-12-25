//! Error types for GPU operations.
//!
//! - [`Error`] — top-level error type.
//! - [`TensorError`] — tensor-specific errors.

/// Top-level error type for GPU operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Tensor operation error.
    #[error(transparent)]
    Tensor(#[from] TensorError),

    /// Kernel generation or compilation error.
    #[error("{0}")]
    Kernel(String),

    /// GPU device operation failed.
    #[error("{0}")]
    Device(String),
}

/// Errors from tensor operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TensorError {
    /// Operation use different device contexts.
    #[error("invalid context: {0}")]
    InvalidContext(String),

    /// Invalid shape for operation.
    #[error("invalid shape: {0}")]
    InvalidShape(String),

    /// Invalid axis, rank, or coordinates for operation.
    #[error("invalid index: {0}")]
    InvalidIndex(String),

    /// Operation constraint violated.
    #[error("{0}")]
    Constraint(String),

    /// Device operation failed.
    #[error("{0}")]
    Device(String),
}
