//! Reduce operations.

mod max;
mod min;
mod sum;

pub(crate) use max::*;
pub(crate) use min::*;
pub(crate) use sum::*;

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Maximum tensor rank supported.
const MAX_RANK: usize = 8;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;
