//! A lightweight ML framework built from scratch in Rust with GPU-first architecture.
//!
//! This library provides GPU-accelerated tensor operations via wgpu,
//! with a focus on simplicity, safety, and cross-platform compatibility.
//!
//! # Types
//!
//! - [`Context`] — GPU context for buffer and pipeline management.
//! - [`Buffer`] — Typed GPU buffer for element data.
//! - [`Element`] — Trait for GPU-compatible types (`f32`, `i32`, `u32`, `bool`).
//! - [`Error`] — Error type for GPU operations.
//! - [`Tensor`] — N-dimensional array with GPU-accelerated operations.

#![warn(missing_docs)]

extern crate alloc;

pub mod element;
pub mod error;

mod device;
mod tensor;

pub use device::{Buffer, Context};
pub use element::Element;
pub use error::Error;
pub use tensor::Tensor;
