//! Unary operations.

mod arithmetic;
mod logical;
mod rounding;

pub(crate) use arithmetic::*;
pub(crate) use logical::*;
pub(crate) use rounding::*;

use crate::{Context, Error};

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// Generates WGSL shader for unary operations with vec4 optimization.
fn unary_shader(ty: &str, expr: &str) -> String {
    let vec4_type = format!("vec4<{ty}>");

    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read_write> b: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    b[tid] = {expr};
                }}
            }}
        "
    )
}

/// Dispatches a unary operation compute shader.
fn dispatch(
    ctx: &Context,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    len: usize,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
) -> Result<(), Error> {
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
        ],
    });

    let len =
        u32::try_from(len).map_err(|_| Error::Device("buffer length exceeds u32::MAX".into()))?;
    let workgroups = len.div_ceil(4).div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
