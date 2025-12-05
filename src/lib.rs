//! A lightweight ML framework built with Rust.
//!
//! This library provides GPU-accelerated tensor operations via wgpu,
//! with a focus on simplicity, safety, and cross-platform compatibility.
//!
//! # Main types
//!
//! - [`GpuContext`]: Central GPU context for buffer and pipeline management.
//! - [`Buffer`]: GPU buffer for storing element data.
//! - [`Element`]: Trait for GPU-compatible types (f32, i32, u32).
//! - [`Error`]: Error type for GPU operations.

#![warn(missing_docs)]

extern crate alloc;

mod device;
mod element;
mod error;

pub use device::{Buffer, GpuContext};
pub use element::Element;
pub use error::Error;
