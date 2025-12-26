//! Math kernels.

use bytemuck::{Pod, Zeroable};

pub(crate) mod clamp;
pub(crate) mod select;

mod binary;
mod unary;

pub(crate) use binary::{add, and, div, eq, ge, gt, le, lt, max, min, mul, ne, or, pow, rem, sub};
pub(crate) use unary::{
    abs, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, exp, floor, log, log2, neg, not,
    rcp, round, rsqr, rsqrt, sign, sin, sinh, sqr, sqrt, tan, tanh,
};

use crate::kernel::{MAX_WORKGROUPS, WORKGROUP_SIZE};

/// Kernel parameters passed to shader as uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct Params {
    pub(super) rank: u32,
    pub(super) len: u32,
}

/// Computes workgroup dimensions for dispatch.
pub(super) fn compute_workgroups(len: u32) -> (u32, u32) {
    let workgroups = len.div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);
    (x, y)
}
