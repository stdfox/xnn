//! Kernel source generation for GPU compute shaders.

use alloc::string::String;
use alloc::vec::Vec;

use crate::Element;

pub(crate) mod constant;
pub(crate) mod copy;
pub(crate) mod linalg;
pub(crate) mod math;
pub(crate) mod nn;
pub(crate) mod ops;
pub(crate) mod reduction;

/// Maximum workgroups per dimension.
pub(crate) const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// Base trait for GPU compute kernels.
pub(crate) trait Kernel: 'static {
    /// Kernel label for debugging and pipeline caching.
    const LABEL: &'static str;

    /// Output element type for the kernel.
    type Output: Element;

    /// Generates WGSL shader source code.
    fn wgsl() -> String;
}

/// Converts strides from usize to u32.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn convert_strides(strides: &[usize]) -> Vec<u32> {
    if strides.is_empty() {
        alloc::vec![0]
    } else {
        strides.iter().map(|&s| s as u32).collect()
    }
}
