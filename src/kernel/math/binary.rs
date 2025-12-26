//! Binary element-wise kernels.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::format;
use alloc::string::String;

use wgpu::util::DeviceExt;

use crate::element::{FloatElement, IntegerElement, LogicalElement, NumericElement};
use crate::kernel::math::Params;
use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context, Element};

/// Defines a binary kernel module.
macro_rules! define_kernel {
    ($in_bound:ident, $out_bound:ident, $kernel:ident, $mod_name:ident, $label:literal, $ty:expr, $out_ty:expr, $op:literal) => {
        pub(crate) mod $mod_name {
            use super::*;

            /// Kernel marker type.
            pub(crate) struct $kernel<T, U>(PhantomData<(T, U)>);

            /// Kernel trait implementation.
            impl<T: $in_bound, U: $out_bound> Kernel for $kernel<T, U> {
                const LABEL: &'static str = $label;
                type Output = U;

                fn wgsl() -> String {
                    let ty = $ty;
                    let out_ty = $out_ty;

                    format!(
                        r"
                            struct Params {{
                                rank: u32,
                                len: u32,
                            }}

                            @group(0) @binding(0) var<storage, read> a: array<{ty}>;
                            @group(0) @binding(1) var<storage, read> b: array<{ty}>;
                            @group(0) @binding(2) var<storage, read_write> c: array<{out_ty}>;
                            @group(0) @binding(3) var<storage, read> a_strides: array<u32>;
                            @group(0) @binding(4) var<storage, read> b_strides: array<u32>;
                            @group(0) @binding(5) var<storage, read> c_strides: array<u32>;
                            @group(0) @binding(6) var<uniform> params: Params;

                            @compute @workgroup_size({WORKGROUP_SIZE})
                            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;

                                if tid >= params.len {{
                                    return;
                                }}

                                var remaining = tid;
                                var a_idx = 0u;
                                var b_idx = 0u;

                                for (var i = 0u; i < params.rank; i++) {{
                                    let coord = remaining / c_strides[i];
                                    remaining = remaining % c_strides[i];
                                    a_idx += coord * a_strides[i];
                                    b_idx += coord * b_strides[i];
                                }}

                                c[tid] = {op};
                            }}
                        ",
                        op = $op
                    )
                }
            }

            /// Executes the kernel.
            pub(crate) fn execute<T: $in_bound, U: $out_bound>(
                ctx: &Context,
                a: &Buffer<T>,
                b: &Buffer<T>,
                c: &Buffer<U>,
                a_strides: &[usize],
                b_strides: &[usize],
                c_strides: &[usize],
            ) {
                super::execute::<$kernel<T, U>, T, U>(
                    ctx, a, b, c, a_strides, b_strides, c_strides,
                );
            }
        }
    };
}

/// Executes a binary kernel.
///
/// # Panics
///
/// - Output length exceeds max size
/// - Output rank exceeds max size
/// - Output buffer too small
#[allow(clippy::too_many_lines)]
fn execute<K: Kernel, T: Element, U: Element>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<U>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    let byte_size = (c.len() * U::NATIVE_SIZE) as u64;
    assert!(c.byte_size() >= byte_size, "output buffer too small");

    let rank = u32::try_from(c_strides.len()).expect("output rank exceeds max size");
    let len = u32::try_from(c.len()).expect("output length exceeds max size");

    let pipeline = ctx.get_or_create_pipeline(TypeId::of::<K>(), K::wgsl, K::LABEL);

    let a_strides = crate::kernel::convert_strides(a_strides);
    let b_strides = crate::kernel::convert_strides(b_strides);
    let c_strides = crate::kernel::convert_strides(c_strides);

    let a_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&a_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let b_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&b_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let c_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&c_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let params = ctx.create_uniform_buffer(&Params { rank, len });

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(K::LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: a_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: c_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let (x, y) = super::compute_workgroups(len);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(K::LABEL),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(K::LABEL),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}

// Arithmetic
define_kernel!(
    NumericElement,
    NumericElement,
    Add,
    add,
    "add",
    T::wgsl_type(),
    U::wgsl_type(),
    "a[a_idx] + b[b_idx]"
);
define_kernel!(
    NumericElement,
    NumericElement,
    Sub,
    sub,
    "sub",
    T::wgsl_type(),
    U::wgsl_type(),
    "a[a_idx] - b[b_idx]"
);
define_kernel!(
    NumericElement,
    NumericElement,
    Mul,
    mul,
    "mul",
    T::wgsl_type(),
    U::wgsl_type(),
    "a[a_idx] * b[b_idx]"
);
define_kernel!(
    NumericElement,
    NumericElement,
    Div,
    div,
    "div",
    T::wgsl_type(),
    U::wgsl_type(),
    "a[a_idx] / b[b_idx]"
);
define_kernel!(
    NumericElement,
    NumericElement,
    Max,
    max,
    "max",
    T::wgsl_type(),
    U::wgsl_type(),
    "max(a[a_idx], b[b_idx])"
);
define_kernel!(
    NumericElement,
    NumericElement,
    Min,
    min,
    "min",
    T::wgsl_type(),
    U::wgsl_type(),
    "min(a[a_idx], b[b_idx])"
);
define_kernel!(
    IntegerElement,
    IntegerElement,
    Rem,
    rem,
    "rem",
    T::wgsl_type(),
    U::wgsl_type(),
    "a[a_idx] % b[b_idx]"
);
define_kernel!(
    FloatElement,
    FloatElement,
    Pow,
    pow,
    "pow",
    T::wgsl_type(),
    U::wgsl_type(),
    "pow(a[a_idx], b[b_idx])"
);

// Comparison
define_kernel!(
    NumericElement,
    LogicalElement,
    Eq,
    eq,
    "eq",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] == b[b_idx])"
);
define_kernel!(
    NumericElement,
    LogicalElement,
    Ne,
    ne,
    "ne",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] != b[b_idx])"
);
define_kernel!(
    NumericElement,
    LogicalElement,
    Gt,
    gt,
    "gt",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] > b[b_idx])"
);
define_kernel!(
    NumericElement,
    LogicalElement,
    Ge,
    ge,
    "ge",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] >= b[b_idx])"
);
define_kernel!(
    NumericElement,
    LogicalElement,
    Lt,
    lt,
    "lt",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] < b[b_idx])"
);
define_kernel!(
    NumericElement,
    LogicalElement,
    Le,
    le,
    "le",
    T::wgsl_type(),
    "u32",
    "u32(a[a_idx] <= b[b_idx])"
);

// Logical
define_kernel!(
    LogicalElement,
    LogicalElement,
    And,
    and,
    "and",
    "u32",
    "u32",
    "u32(a[a_idx] != 0u && b[b_idx] != 0u)"
);
define_kernel!(
    LogicalElement,
    LogicalElement,
    Or,
    or,
    "or",
    "u32",
    "u32",
    "u32(a[a_idx] != 0u || b[b_idx] != 0u)"
);
