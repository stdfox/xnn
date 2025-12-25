//! Activation kernel with alpha buffer.

use alloc::string::String;

use crate::kernel::activation::{MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Context, Error};

/// `PReLU` activation: `y = select(x < 0, alpha * x, x)`.
pub(crate) struct Prelu;

impl ActivationWithAlphaKernel for Prelu {
    const LABEL: &'static str = "prelu";
    const OP: &'static str = "select(alpha * x, x, x >= vec4(0.0))";
}

/// Activation kernel with alpha buffer trait.
pub(crate) trait ActivationWithAlphaKernel {
    const LABEL: &'static str;
    const OP: &'static str;

    fn wgsl() -> String {
        format!(
            r"
                @group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read> alpha: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;

                @compute @workgroup_size({WORKGROUP_SIZE})
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                    let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                    if tid < arrayLength(&x) {{
                        let alpha = alpha[tid];
                        let x = x[tid];
                        y[tid] = {op};
                    }}
                }}
            ",
            op = Self::OP
        )
    }
}

/// Executes activation kernel with alpha buffer.
pub(crate) fn execute_with_alpha<A: ActivationWithAlphaKernel>(
    ctx: &Context,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    alpha: &wgpu::Buffer,
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
                resource: alpha.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
