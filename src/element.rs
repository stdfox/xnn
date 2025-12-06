//! Traits for GPU-compatible element types.
//!
//! [`Element`] defines types that can be stored in GPU buffers and used
//! in compute shaders. Implemented for `f32`, `i32`, and `u32`.
//!
//! [`FloatElement`] is a marker trait for floating-point types only.
//! Used by operations that require floats (e.g., `pow`, `gemm`).

use bytemuck::{Pod, Zeroable};

/// Trait for GPU-compatible element types.
pub trait Element: Copy + Clone + Pod + Zeroable + 'static {
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
