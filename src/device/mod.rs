//! GPU device and buffer management.
//!
//! Provides [`Context`] for GPU operations and [`Buffer`] for GPU memory.
//! Contexts are pooled by adapter index, automatically selecting
//! high-performance adapters by default.

mod allocator;
mod buffer;
mod context;

pub use buffer::Buffer;
pub use context::Context;
