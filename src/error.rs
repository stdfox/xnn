//! Error types for GPU operations.
//!
//! The [`Error`] enum covers device initialization, buffer operations,
//! and kernel execution failures.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// GPU device operation failed.
    #[error("{0}")]
    Device(String),

    /// Kernel execution failed.
    #[error("{0}")]
    Kernel(String),
}
