//! A lightweight ML framework built from scratch in Rust with GPU-first architecture.
//!
//! This library provides GPU-accelerated tensor operations via wgpu,
//! with a focus on simplicity, safety, and cross-platform compatibility.
//!
//! # Types
//!
//! - [`Context`] — GPU context for buffer and pipeline management.
//! - [`Buffer`] — Typed GPU buffer for element data.
//! - [`Element`] — Trait for GPU-compatible types (`f32`, `i32`, `u32`).
//! - [`NumericElement`] — Marker trait for numeric types (`f32`, `i32`, `u32`).
//! - [`SignedElement`] — Marker trait for signed types (`f32`, `i32`).
//! - [`FloatElement`] — Marker trait for floating-point types (`f32`).
//! - [`Error`] — Error type for GPU operations.

#![warn(missing_docs)]

extern crate alloc;

pub mod error;
pub mod kernel;

mod device;
mod element;
mod tensor;

pub use device::{Buffer, Context};
pub use element::{Element, FloatElement, NumericElement, SignedElement};
pub use error::Error;
pub use tensor::Tensor;
