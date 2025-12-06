//! A lightweight ML framework built with Rust.
//!
//! This library provides GPU-accelerated tensor operations via wgpu,
//! with a focus on simplicity, safety, and cross-platform compatibility.
//!
//! # Types
//!
//! - [`GpuContext`] — GPU context for buffer and pipeline management.
//! - [`Buffer`] — typed GPU buffer for element data.
//! - [`Element`] — trait for GPU-compatible types (`f32`, `i32`, `u32`).
//! - [`FloatElement`] — marker trait for floating-point types (`f32`).
//! - [`Error`] — error type for GPU operations.
//!
//! # Feature flags
//!
//! - `unstable-kernels` — exposes the `kernel` module with GPU compute kernels.

#![warn(missing_docs)]

extern crate alloc;

#[cfg(feature = "unstable-kernels")]
pub mod kernel;

#[cfg(not(feature = "unstable-kernels"))]
#[allow(unused_imports)]
#[allow(dead_code)]
pub(crate) mod kernel;

mod device;
mod element;
mod error;

pub use device::{Buffer, GpuContext};
pub use element::{Element, FloatElement};
pub use error::Error;
