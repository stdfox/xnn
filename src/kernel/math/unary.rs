//! Unary element-wise kernels.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::string::String;

use crate::element::{FloatElement, LogicalElement, SignedElement};
use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context, Element};

/// Defines a unary kernel module.
macro_rules! define_kernel {
    ($bound:ident, $kernel:ident, $mod_name:ident, $label:literal, $op:literal) => {
        pub(crate) mod $mod_name {
            use super::*;

            /// Kernel marker type.
            pub(crate) struct $kernel<T>(PhantomData<T>);

            /// Kernel trait implementation.
            impl<T: $bound> Kernel for $kernel<T> {
                const LABEL: &'static str = $label;
                type Output = T;

                fn wgsl() -> String {
                    let ty = T::wgsl_type();
                    let op = $op.replace("{ty}", ty).replace("{one}", T::wgsl_one());

                    alloc::format!(
                        r"
                            @group(0) @binding(0) var<storage, read> x: array<vec4<{ty}>>;
                            @group(0) @binding(1) var<storage, read_write> y: array<vec4<{ty}>>;

                            @compute @workgroup_size({WORKGROUP_SIZE})
                            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                                if tid < arrayLength(&x) {{
                                    y[tid] = {op};
                                }}
                            }}
                        "
                    )
                }
            }

            /// Executes the kernel.
            pub(crate) fn execute<T: $bound>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
                super::execute::<$kernel<T>, T>(ctx, x, y);
            }
        }
    };
}

/// Executes a unary kernel.
///
/// # Panics
///
/// - Buffer lengths do not match
/// - Output length exceeds max size
fn execute<K: Kernel, T: Element>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    assert_eq!(x.len(), y.len(), "buffer length mismatch");

    let len = u32::try_from(x.len().div_ceil(4)).expect("output length exceeds max size");

    if len == 0 {
        return;
    }

    let pipeline = ctx.get_or_create_pipeline(TypeId::of::<K>(), K::wgsl, K::LABEL);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(K::LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.as_entire_binding(),
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
define_kernel!(SignedElement, Abs, abs, "abs", "abs(x[tid])");
define_kernel!(SignedElement, Neg, neg, "neg", "-x[tid]");
define_kernel!(SignedElement, Sign, sign, "sign", "sign(x[tid])");

// Trigonometric
define_kernel!(FloatElement, Sin, sin, "sin", "sin(x[tid])");
define_kernel!(FloatElement, Cos, cos, "cos", "cos(x[tid])");
define_kernel!(FloatElement, Tan, tan, "tan", "tan(x[tid])");

// Inverse trigonometric
define_kernel!(FloatElement, Asin, asin, "asin", "asin(x[tid])");
define_kernel!(FloatElement, Acos, acos, "acos", "acos(x[tid])");
define_kernel!(FloatElement, Atan, atan, "atan", "atan(x[tid])");

// Hyperbolic
define_kernel!(FloatElement, Sinh, sinh, "sinh", "sinh(x[tid])");
define_kernel!(FloatElement, Cosh, cosh, "cosh", "cosh(x[tid])");
define_kernel!(FloatElement, Tanh, tanh, "tanh", "tanh(x[tid])");

// Inverse hyperbolic
define_kernel!(FloatElement, Asinh, asinh, "asinh", "asinh(x[tid])");
define_kernel!(FloatElement, Acosh, acosh, "acosh", "acosh(x[tid])");
define_kernel!(FloatElement, Atanh, atanh, "atanh", "atanh(x[tid])");

// Exponential and logarithmic
define_kernel!(FloatElement, Exp, exp, "exp", "exp(x[tid])");
define_kernel!(FloatElement, Log, log, "log", "log(x[tid])");
define_kernel!(FloatElement, Log2, log2, "log2", "log2(x[tid])");

// Power
define_kernel!(FloatElement, Sqr, sqr, "sqr", "x[tid] * x[tid]");
define_kernel!(FloatElement, Sqrt, sqrt, "sqrt", "sqrt(x[tid])");
define_kernel!(
    FloatElement,
    Rsqr,
    rsqr,
    "rsqr",
    "vec4<{ty}>({one}) / (x[tid] * x[tid])"
);
define_kernel!(FloatElement, Rsqrt, rsqrt, "rsqrt", "inverseSqrt(x[tid])");
define_kernel!(FloatElement, Rcp, rcp, "rcp", "vec4<{ty}>({one}) / x[tid]");

// Rounding
define_kernel!(FloatElement, Ceil, ceil, "ceil", "ceil(x[tid])");
define_kernel!(FloatElement, Floor, floor, "floor", "floor(x[tid])");
define_kernel!(FloatElement, Round, round, "round", "round(x[tid])");

// Logical
define_kernel!(
    LogicalElement,
    Not,
    not,
    "not",
    "vec4<{ty}>({one}) - min(x[tid], vec4<{ty}>({one}))"
);
