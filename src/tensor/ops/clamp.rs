//! Clamp operation.

use core::any::TypeId;

use crate::element::NumericElement;
use crate::{Buffer, Context, Error};

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Threads per workgroup.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum tensor rank supported (must be multiple of 4).
const MAX_RANK: usize = 8;

/// Broadcast parameters for ternary operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TernaryBroadcastParams {
    len: u32,
    rank: u32,
    _pad: [u32; 2],
    dimensions: [[u32; 4]; MAX_RANK / 4],
    x_strides: [[u32; 4]; MAX_RANK / 4],
    a_strides: [[u32; 4]; MAX_RANK / 4],
    b_strides: [[u32; 4]; MAX_RANK / 4],
}

struct ClampMarker<T>(core::marker::PhantomData<T>);

impl<T: NumericElement> ClampMarker<T> {
    fn shader() -> String {
        let ty = T::wgsl_type();
        let max_vec4 = MAX_RANK / 4;

        format!(
            r"
                const MAX_VEC4: u32 = {max_vec4}u;

                struct Params {{
                    len: u32,
                    rank: u32,
                    _pad: vec2<u32>,
                    dimensions: array<vec4<u32>, MAX_VEC4>,
                    x_strides: array<vec4<u32>, MAX_VEC4>,
                    a_strides: array<vec4<u32>, MAX_VEC4>,
                    b_strides: array<vec4<u32>, MAX_VEC4>,
                }}

                fn get_dimension(idx: u32) -> u32 {{
                    let vec_idx = idx / 4u;
                    let comp_idx = idx % 4u;
                    return params.dimensions[vec_idx][comp_idx];
                }}

                fn get_x_stride(idx: u32) -> u32 {{
                    let vec_idx = idx / 4u;
                    let comp_idx = idx % 4u;
                    return params.x_strides[vec_idx][comp_idx];
                }}

                fn get_a_stride(idx: u32) -> u32 {{
                    let vec_idx = idx / 4u;
                    let comp_idx = idx % 4u;
                    return params.a_strides[vec_idx][comp_idx];
                }}

                fn get_b_stride(idx: u32) -> u32 {{
                    let vec_idx = idx / 4u;
                    let comp_idx = idx % 4u;
                    return params.b_strides[vec_idx][comp_idx];
                }}

                @group(0) @binding(0) var<storage, read> x: array<{ty}>;
                @group(0) @binding(1) var<storage, read> a: array<{ty}>;
                @group(0) @binding(2) var<storage, read> b: array<{ty}>;
                @group(0) @binding(3) var<storage, read_write> y: array<{ty}>;
                @group(0) @binding(4) var<uniform> params: Params;

                @compute @workgroup_size({WORKGROUP_SIZE})
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                    let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                    if tid >= params.len {{
                        return;
                    }}

                    var remaining = tid;
                    var x_idx = 0u;
                    var a_idx = 0u;
                    var b_idx = 0u;

                    for (var i = 0u; i < params.rank; i++) {{
                        var stride = 1u;
                        for (var j = i + 1u; j < params.rank; j++) {{
                            stride *= get_dimension(j);
                        }}
                        let coord = remaining / stride;
                        remaining = remaining % stride;

                        x_idx += coord * get_x_stride(i);
                        a_idx += coord * get_a_stride(i);
                        b_idx += coord * get_b_stride(i);
                    }}

                    y[tid] = max(min(x[x_idx], b[b_idx]), a[a_idx]);
                }}
            "
        )
    }
}

/// Clamp operation: y = max(min(x, b), a).
#[allow(clippy::too_many_arguments)]
pub(crate) fn clamp<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    a: &Buffer<T>,
    b: &Buffer<T>,
    y: &Buffer<T>,
    dimensions: &[usize],
    x_strides: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
) -> Result<(), Error> {
    const LABEL: &str = "clamp";

    let out_len = dimensions.iter().product::<usize>().max(1);
    let dim_len = dimensions.len();

    if dim_len > MAX_RANK {
        return Err(Error::Device(format!(
            "tensor rank {dim_len} exceeds maximum {MAX_RANK}"
        )));
    }

    if out_len == 0 {
        return Ok(());
    }

    if out_len > u32::MAX as usize {
        return Err(Error::Device("output length exceeds u32::MAX".into()));
    }

    if y.len() != out_len {
        return Err(Error::Device("output buffer length mismatch".into()));
    }

    #[allow(clippy::cast_possible_truncation)]
    let len = out_len as u32;

    #[allow(clippy::cast_possible_truncation)]
    let rank = dim_len as u32;

    let mut params = TernaryBroadcastParams {
        len,
        rank,
        _pad: [0; 2],
        dimensions: [[0; 4]; MAX_RANK / 4],
        x_strides: [[0; 4]; MAX_RANK / 4],
        a_strides: [[0; 4]; MAX_RANK / 4],
        b_strides: [[0; 4]; MAX_RANK / 4],
    };

    #[allow(clippy::cast_possible_truncation)]
    for (i, &dim) in dimensions.iter().enumerate() {
        params.dimensions[i / 4][i % 4] = dim as u32;
    }

    #[allow(clippy::cast_possible_truncation)]
    for (i, &s) in x_strides.iter().enumerate() {
        params.x_strides[i / 4][i % 4] = s as u32;
    }

    #[allow(clippy::cast_possible_truncation)]
    for (i, &s) in a_strides.iter().enumerate() {
        params.a_strides[i / 4][i % 4] = s as u32;
    }

    #[allow(clippy::cast_possible_truncation)]
    for (i, &s) in b_strides.iter().enumerate() {
        params.b_strides[i / 4][i % 4] = s as u32;
    }

    let params = ctx.create_uniform_buffer(&params);

    let pipeline = ctx.get_or_create_pipeline(
        TypeId::of::<ClampMarker<T>>(),
        ClampMarker::<T>::shader,
        LABEL,
    )?;

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let workgroups = len.div_ceil(WORKGROUP_SIZE);
    let wx = workgroups.min(MAX_WORKGROUPS);
    let wy = workgroups.div_ceil(MAX_WORKGROUPS);

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
        pass.dispatch_workgroups(wx, wy, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
