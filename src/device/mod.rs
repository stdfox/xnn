//! GPU device and buffer management.
//!
//! Provides [`Context`] for GPU operations and [`Buffer`] for typed GPU memory.
//! Context wraps wgpu device/queue with buffer allocation pooling and compute
//! pipeline caching.

mod allocator;
mod buffer;
mod context;
mod pipelines;

pub use buffer::Buffer;
pub use context::Context;

use allocator::Allocator;
use pipelines::PipelineCache;
