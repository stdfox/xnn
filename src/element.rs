//! Traits for GPU-compatible element types.
//!
//! - [`Element`] — base trait for GPU buffer types (`f32`, `i32`, `u32`).
//! - [`FloatElement`] — marker for floating-point types (`f32`).

use core::fmt::Display;

use bytemuck::{Pod, Zeroable};

/// Trait for GPU-compatible element types.
pub trait Element: Display + Copy + Clone + Pod + Zeroable + 'static {
    /// Returns the WGSL type name.
    fn wgsl_type() -> &'static str;
}

impl Element for f32 {
    #[inline]
    fn wgsl_type() -> &'static str {
        "f32"
    }
}

impl Element for i32 {
    #[inline]
    fn wgsl_type() -> &'static str {
        "i32"
    }
}

impl Element for u32 {
    #[inline]
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

/// Trait for floating-point GPU-compatible types.
///
/// Used for operations that only work with floats, such as `pow`.
pub trait FloatElement: Element {}

impl FloatElement for f32 {}
