//! Activation kernels.

mod with_alpha;
mod with_params;

pub(crate) use with_alpha::*;
pub(crate) use with_params::*;

use alloc::string::String;

use crate::{Context, Error};

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// `ReLU` activation: `y = max(x, 0)`.
pub(crate) struct Relu;
impl ActivationKernel for Relu {
    const LABEL: &'static str = "relu";
    const OP: &'static str = "max(x, vec4(0.0))";
}

/// `GELU` activation: `y = x * sigmoid(1.702 * x)`.
pub(crate) struct Gelu;
impl ActivationKernel for Gelu {
    const LABEL: &'static str = "gelu";
    const OP: &'static str = "x * (1.0 / (1.0 + exp(-1.702 * x)))";
}

/// `Sigmoid` activation: `y = 1 / (1 + exp(-x))`.
pub(crate) struct Sigmoid;
impl ActivationKernel for Sigmoid {
    const LABEL: &'static str = "sigmoid";
    const OP: &'static str = "1.0 / (1.0 + exp(-x))";
}

/// `SiLU` activation: `y = x * sigmoid(x)`.
pub(crate) struct Silu;
impl ActivationKernel for Silu {
    const LABEL: &'static str = "silu";
    const OP: &'static str = "x * (1.0 / (1.0 + exp(-x)))";
}

/// `Softplus` activation: `y = log(exp(x) + 1)`.
pub(crate) struct Softplus;
impl ActivationKernel for Softplus {
    const LABEL: &'static str = "softplus";
    const OP: &'static str = "log(exp(x) + vec4(1.0))";
}

/// Activation kernel trait.
pub(crate) trait ActivationKernel {
    const LABEL: &'static str;
    const OP: &'static str;

    fn wgsl() -> String {
        format!(
            r"
                @group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read_write> y: array<vec4<f32>>;

                @compute @workgroup_size({WORKGROUP_SIZE})
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                    let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                    if tid < arrayLength(&x) {{
                        let x = x[tid];
                        y[tid] = {op};
                    }}
                }}
            ",
            op = Self::OP
        )
    }
}

/// Executes activation kernel.
pub(crate) fn execute<A: ActivationKernel>(
    ctx: &Context,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Result<(), Error> {
    let len = u32::try_from(x.size() / 4)
        .map_err(|_| Error::Kernel("output length exceeds u32::MAX".into()))?;

    let pipeline = ctx.compiler().compile(&A::wgsl(), A::LABEL);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(A::LABEL),
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

    let workgroups = len.div_ceil(4).div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(A::LABEL),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(A::LABEL),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
