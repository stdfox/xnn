//! Activation kernel with params.

use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::kernel::activation::{MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Context, Error};

/// Uniform parameters for activation kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    alpha: f32,
    lambda: f32,
}

/// `ELU` activation: `y = select(x < 0, alpha * (exp(x) - 1), x)`.
pub(crate) struct Elu;
impl ActivationWithParamsKernel for Elu {
    const LABEL: &'static str = "elu";
    const OP: &'static str = "select(alpha * (exp(x) - vec4(1.0)), x, x >= vec4(0.0))";
}

/// `LeakyReLU` activation: `y = select(x < 0, alpha * x, x)`.
pub(crate) struct LeakyRelu;
impl ActivationWithParamsKernel for LeakyRelu {
    const LABEL: &'static str = "leaky_relu";
    const OP: &'static str = "select(alpha * x, x, x >= vec4(0.0))";
}

/// `SELU` activation: `y = lambda * select(x < 0, alpha * (exp(x) - 1), x)`.
pub(crate) struct Selu;
impl ActivationWithParamsKernel for Selu {
    const LABEL: &'static str = "selu";
    const OP: &'static str = "lambda * select(alpha * (exp(x) - vec4(1.0)), x, x >= vec4(0.0))";
}

/// Activation kernel with params trait.
pub(crate) trait ActivationWithParamsKernel {
    const LABEL: &'static str;
    const OP: &'static str;

    fn wgsl() -> String {
        format!(
            r"
                struct Params {{ alpha: f32, lambda: f32 }}

                @group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read_write> y: array<vec4<f32>>;
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
            ",
            op = Self::OP
        )
    }
}

/// Executes activation kernel with params.
pub(crate) fn execute_with_param<A: ActivationWithParamsKernel>(
    ctx: &Context,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    alpha: f32,
    lambda: f32,
) -> Result<(), Error> {
    let len = u32::try_from(x.size() / 4)
        .map_err(|_| Error::Kernel("output length exceeds u32::MAX".into()))?;

    let pipeline = ctx.compiler().compile(&A::wgsl(), A::LABEL);

    let params = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(A::LABEL),
            contents: bytemuck::bytes_of(&Params { alpha, lambda }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
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
