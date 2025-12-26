//! Activation function kernels.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::format;
use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::element::FloatElement;
use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context, Element};

/// Uniform parameters for activation kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    alpha: f32,
    lambda: f32,
}

/// Defines an activation kernel module.
macro_rules! define_kernel {
    ($kernel:ident, $mod_name:ident, $label:literal, $op:literal) => {
        pub(crate) mod $mod_name {
            use super::*;

            /// Kernel marker type.
            pub(crate) struct $kernel<T>(PhantomData<T>);

            /// Kernel trait implementation.
            impl<T: FloatElement> Kernel for $kernel<T> {
                const LABEL: &'static str = $label;
                type Output = T;

                fn wgsl() -> String {
                    let ty = T::wgsl_type();
                    let op = $op;

                    format!(
                        r"
                            struct Params {{ alpha: f32, lambda: f32 }}

                            @group(0) @binding(0) var<storage, read> x: array<vec4<{ty}>>;
                            @group(0) @binding(1) var<storage, read_write> y: array<vec4<{ty}>>;
                            @group(0) @binding(2) var<uniform> params: Params;

                            @compute @workgroup_size({WORKGROUP_SIZE})
                            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                                if tid < arrayLength(&x) {{
                                    let alpha = vec4(params.alpha);
                                    let lambda = vec4(params.lambda);
                                    let x = x[tid];
                                    y[tid] = {op};
                                }}
                            }}
                        "
                    )
                }
            }

            /// Executes the kernel.
            pub(crate) fn execute<T: FloatElement>(
                ctx: &Context,
                x: &Buffer<T>,
                y: &Buffer<T>,
                alpha: f32,
                lambda: f32,
            ) {
                super::execute::<$kernel<T>, T>(ctx, x, y, alpha, lambda);
            }
        }
    };
}

/// Executes an activation kernel.
///
/// # Panics
///
/// - Buffer sizes do not match
fn execute<K: Kernel, T: Element>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: f32,
    lambda: f32,
) {
    assert_eq!(x.byte_size(), y.byte_size(), "buffer size mismatch");

    let len = u32::try_from(x.byte_size() / (T::NATIVE_SIZE * 4) as u64)
        .expect("output length exceeds max size");

    if len == 0 {
        return;
    }

    let pipeline = ctx.get_or_create_pipeline(TypeId::of::<K>(), K::wgsl, K::LABEL);

    let params = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(K::LABEL),
            contents: bytemuck::bytes_of(&Params { alpha, lambda }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(K::LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let workgroups = len.div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);

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

define_kernel!(
    Elu,
    elu,
    "elu",
    "select(alpha * (exp(x) - vec4(1.0)), x, x >= vec4(0.0))"
);
define_kernel!(Gelu, gelu, "gelu", "x * (1.0 / (1.0 + exp(-1.702 * x)))");
define_kernel!(
    LeakyRelu,
    leaky_relu,
    "leaky_relu",
    "select(alpha * x, x, x >= vec4(0.0))"
);
define_kernel!(Relu, relu, "relu", "max(x, vec4(0.0))");
define_kernel!(
    Selu,
    selu,
    "selu",
    "lambda * select(alpha * (exp(x) - vec4(1.0)), x, x >= vec4(0.0))"
);
define_kernel!(Sigmoid, sigmoid, "sigmoid", "1.0 / (1.0 + exp(-x))");
define_kernel!(Silu, silu, "silu", "x * (1.0 / (1.0 + exp(-x)))");
define_kernel!(Softplus, softplus, "softplus", "log(exp(x) + vec4(1.0))");

/// `PReLU` activation kernel module.
#[allow(clippy::wildcard_imports)]
pub(crate) mod prelu {
    use super::*;

    /// Kernel marker type.
    pub(crate) struct Prelu<T>(PhantomData<T>);

    /// Kernel trait implementation.
    impl<T: FloatElement> Kernel for Prelu<T> {
        const LABEL: &'static str = "prelu";
        type Output = T;

        fn wgsl() -> String {
            let ty = T::wgsl_type();

            format!(
                r"
                    @group(0) @binding(0) var<storage, read> x: array<vec4<{ty}>>;
                    @group(0) @binding(1) var<storage, read_write> y: array<vec4<{ty}>>;
                    @group(0) @binding(2) var<storage, read> alpha: array<vec4<{ty}>>;

                    @compute @workgroup_size({WORKGROUP_SIZE})
                    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                        let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                        if tid < arrayLength(&x) {{
                            let alpha = alpha[tid];
                            let x = x[tid];
                            y[tid] = select(alpha * x, x, x >= vec4(0.0));
                        }}
                    }}
                "
            )
        }
    }

    /// Executes the kernel.
    pub(crate) fn execute<T: FloatElement>(
        ctx: &Context,
        x: &Buffer<T>,
        y: &Buffer<T>,
        alpha: &Buffer<T>,
    ) {
        assert_eq!(x.byte_size(), y.byte_size(), "buffer size mismatch");
        assert_eq!(
            x.byte_size(),
            alpha.byte_size(),
            "alpha buffer size mismatch"
        );

        let len = u32::try_from(x.byte_size() / (T::NATIVE_SIZE * 4) as u64)
            .expect("output length exceeds max size");

        if len == 0 {
            return;
        }

        let pipeline = ctx.get_or_create_pipeline(
            TypeId::of::<Prelu<T>>(),
            Prelu::<T>::wgsl,
            Prelu::<T>::LABEL,
        );

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(Prelu::<T>::LABEL),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.inner().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y.inner().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: alpha.inner().as_entire_binding(),
                },
            ],
        });

        let workgroups = len.div_ceil(WORKGROUP_SIZE);
        let x = workgroups.min(MAX_WORKGROUPS);
        let y = workgroups.div_ceil(MAX_WORKGROUPS);

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(Prelu::<T>::LABEL),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(Prelu::<T>::LABEL),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }

        ctx.queue().submit(Some(encoder.finish()));
    }
}
