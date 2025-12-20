//! Ceiling operation for tensor buffers.

use core::any::TypeId;

use crate::{Buffer, Context, Error, FloatElement};

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// Pipeline label for debugging.
const LABEL: &str = "ceil";

/// Marker type for pipeline caching.
struct Ceil<T>(core::marker::PhantomData<T>);

/// Generates WGSL shader for ceil operation.
fn shader() -> String {
    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    b[tid] = ceil(a[tid]);
                }}
            }}
        "
    )
}

/// Computes ceiling element-wise.
///
/// # Errors
///
/// Returns [`Error::Device`] if pipeline creation fails or buffer length exceeds `u32::MAX`.
pub(crate) fn ceil<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::Device("buffer lengths mismatch".into()));
    }

    if a.is_empty() {
        return Ok(());
    }

    let pipeline = ctx.get_or_create_pipeline(TypeId::of::<Ceil<T>>(), shader, LABEL)?;

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(LABEL),
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
        ],
    });

    let len = u32::try_from(a.len())
        .map_err(|_| Error::Device("buffer length exceeds u32::MAX".into()))?;
    let workgroups = len.div_ceil(4).div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(LABEL) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(LABEL),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
