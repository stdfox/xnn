//! Element trait for GPU-compatible types.
//!
//! The [`Element`] trait defines types that can be stored in GPU buffers
//! and used in compute shaders. Implemented for f32, i32, and u32.

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
