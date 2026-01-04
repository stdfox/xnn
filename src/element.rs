//! Traits for GPU-compatible element types.
//!
//! - [`Element`] — base trait for GPU buffer types (`f32`, `i32`, `u32`, `bool`).
//! - [`NumericElement`] — marker for numeric types (`f32`, `i32`, `u32`).
//! - [`SignedElement`] — marker for signed types (`f32`, `i32`).
//! - [`IntegerElement`] — marker for integer types (`i32`, `u32`).
//! - [`FloatElement`] — marker for floating-point types (`f32`).
//! - [`LogicalElement`] — marker for logical types (`bool`).

use core::fmt::Display;

use bytemuck::{Pod, Zeroable};

/// Trait for GPU-compatible element types.
pub trait Element: Display + Copy + Clone + 'static {
    /// Size of native representation in bytes.
    const NATIVE_SIZE: usize = core::mem::size_of::<Self::Native>();

    /// Native GPU-compatible representation type.
    type Native: Default + Copy + Pod + Zeroable;

    /// Returns the WGSL type name.
    #[must_use]
    fn wgsl_type() -> &'static str;

    /// Returns the WGSL literal for the zero value.
    #[must_use]
    fn wgsl_zero() -> &'static str;

    /// Returns the WGSL literal for the one value.
    #[must_use]
    fn wgsl_one() -> &'static str;

    /// Returns the WGSL literal for the maximum value.
    #[must_use]
    fn wgsl_max() -> &'static str;

    /// Returns the WGSL literal for the minimum value.
    #[must_use]
    fn wgsl_min() -> &'static str;

    /// Convert from native GPU representation.
    #[must_use]
    fn from_native(native: Self::Native) -> Self;

    /// Convert to native GPU representation.
    #[must_use]
    fn to_native(self) -> Self::Native;
}

impl Element for f32 {
    type Native = f32;

    #[inline]
    fn wgsl_type() -> &'static str {
        "f32"
    }

    #[inline]
    fn wgsl_zero() -> &'static str {
        "0.0"
    }

    #[inline]
    fn wgsl_one() -> &'static str {
        "1.0"
    }

    #[inline]
    fn wgsl_max() -> &'static str {
        "3.402823466e+38"
    }

    #[inline]
    fn wgsl_min() -> &'static str {
        "-3.402823466e+38"
    }

    #[inline]
    fn from_native(native: Self) -> Self {
        native
    }

    #[inline]
    fn to_native(self) -> Self {
        self
    }
}

impl Element for i32 {
    type Native = i32;

    #[inline]
    fn wgsl_type() -> &'static str {
        "i32"
    }

    #[inline]
    fn wgsl_zero() -> &'static str {
        "0i"
    }

    #[inline]
    fn wgsl_one() -> &'static str {
        "1i"
    }

    #[inline]
    fn wgsl_max() -> &'static str {
        "0x7fffffffi"
    }

    #[inline]
    fn wgsl_min() -> &'static str {
        "(-0x7fffffffi - 1i)"
    }

    #[inline]
    fn from_native(native: Self) -> Self {
        native
    }

    #[inline]
    fn to_native(self) -> Self {
        self
    }
}

impl Element for u32 {
    type Native = u32;

    #[inline]
    fn wgsl_type() -> &'static str {
        "u32"
    }

    #[inline]
    fn wgsl_zero() -> &'static str {
        "0u"
    }

    #[inline]
    fn wgsl_one() -> &'static str {
        "1u"
    }

    #[inline]
    fn wgsl_max() -> &'static str {
        "0xffffffffu"
    }

    #[inline]
    fn wgsl_min() -> &'static str {
        "0u"
    }

    #[inline]
    fn from_native(native: Self) -> Self {
        native
    }

    #[inline]
    fn to_native(self) -> Self {
        self
    }
}

impl Element for bool {
    type Native = u32;

    #[inline]
    fn wgsl_type() -> &'static str {
        "u32"
    }

    #[inline]
    fn wgsl_zero() -> &'static str {
        "0u"
    }

    #[inline]
    fn wgsl_one() -> &'static str {
        "1u"
    }

    #[inline]
    fn wgsl_max() -> &'static str {
        "0xffffffffu"
    }

    #[inline]
    fn wgsl_min() -> &'static str {
        "0u"
    }

    #[inline]
    fn from_native(native: u32) -> Self {
        native != 0
    }

    #[inline]
    fn to_native(self) -> u32 {
        u32::from(self)
    }
}

/// Trait for numeric GPU-compatible types.
pub trait NumericElement: Element {}

impl NumericElement for f32 {}
impl NumericElement for i32 {}
impl NumericElement for u32 {}

/// Trait for signed GPU-compatible types.
pub trait SignedElement: Element {}

impl SignedElement for f32 {}
impl SignedElement for i32 {}

/// Trait for integer GPU-compatible types.
pub trait IntegerElement: Element {}

impl IntegerElement for i32 {}
impl IntegerElement for u32 {}

/// Trait for floating-point GPU-compatible types.
pub trait FloatElement: Element {}

impl FloatElement for f32 {}

/// Trait for logical GPU-compatible types.
pub trait LogicalElement: Element {}

impl LogicalElement for bool {}
