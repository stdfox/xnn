//! Sign operation for tensor buffers.

use core::any::TypeId;

use crate::element::NumericElement;
use crate::{Buffer, Context, Error};

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// Pipeline label for debugging.
const LABEL: &str = "sign";

/// Marker type for pipeline caching.
struct Sign<T>(core::marker::PhantomData<T>);

/// Marker type for u32 sign pipeline caching.
struct SignU32;

/// Generates WGSL shader for sign operation (f32/i32).
fn shader<T: NumericElement>() -> String {
    let wgsl_type = T::wgsl_type();
    let vec4_type = format!("vec4<{wgsl_type}>");

    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read_write> b: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    b[tid] = sign(a[tid]);
                }}
            }}
        "
    )
}

/// Generates WGSL shader for u32 sign operation (min(x, 1)).
fn shader_u32() -> String {
    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<vec4<u32>>;
            @group(0) @binding(1) var<storage, read_write> b: array<vec4<u32>>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    b[tid] = min(a[tid], vec4<u32>(1u));
                }}
            }}
        "
    )
}

/// Computes sign element-wise.
///
/// # Errors
///
/// Returns [`Error::Device`] if pipeline creation fails or buffer length exceeds `u32::MAX`.
pub(crate) fn sign<T: NumericElement>(
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

    // Use optimized u32 shader
    let pipeline = if TypeId::of::<T>() == TypeId::of::<u32>() {
        ctx.get_or_create_pipeline(TypeId::of::<SignU32>(), shader_u32, LABEL)?
    } else {
        ctx.get_or_create_pipeline(TypeId::of::<Sign<T>>(), shader::<T>, LABEL)?
    };

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
