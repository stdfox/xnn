//! A lightweight ML framework built with Rust.
//!
//! This library provides GPU-accelerated tensor operations via wgpu,
//! with a focus on simplicity, safety, and cross-platform compatibility.
//!
//! # Types
//!
//! - [`Context`] — GPU context for buffer and pipeline management.
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

pub mod kernel;

mod device;
mod element;
mod error;

pub use device::{Buffer, Context};
pub use element::{Element, FloatElement};
pub use error::Error;
