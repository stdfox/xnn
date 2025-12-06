//! Error types for GPU operations.
//!
//! [`Error`] covers device initialization, and buffer operations
//! execution failures.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// GPU device operation failed.
    #[error("{0}")]
    Device(String),
}
