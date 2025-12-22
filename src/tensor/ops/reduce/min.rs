//! Min reduction operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::NumericElement;
use crate::{Buffer, Context, Error};

use super::{MAX_RANK, MAX_WORKGROUPS, WORKGROUP_SIZE};

const LABEL: &str = "min_reduce";

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MinReduceParams {
    output_len: u32,
    reduction_len: u32,
    rank: u32,
    _pad: u32,
    input_dims: [[u32; 4]; MAX_RANK / 4],
    input_strides: [[u32; 4]; MAX_RANK / 4],
    output_strides: [[u32; 4]; MAX_RANK / 4],
    reduce_mask: [[u32; 4]; MAX_RANK / 4],
}

struct MinReduceMarker<T>(PhantomData<T>);

/// Min reduction along specified axes.
pub(crate) fn min_reduce<T: NumericElement>(
    ctx: &Context,
    input: &Buffer<T>,
    output: &Buffer<T>,
    input_dims: &[usize],
    axes: &[usize],
) -> Result<(), Error> {
    let rank = input_dims.len();

    if rank > MAX_RANK {
        return Err(Error::Device(format!(
            "tensor rank {rank} exceeds maximum {MAX_RANK}"
        )));
    }

    let output_len: usize = input_dims
        .iter()
        .enumerate()
        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
        .product();

    let reduction_len: usize = axes.iter().map(|&a| input_dims[a]).product();

    if output_len == 0 || reduction_len == 0 {
        return Ok(());
    }

    let input_strides = compute_strides(input_dims);
    let output_dims: Vec<usize> = input_dims
        .iter()
        .enumerate()
        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
        .collect();
    let output_strides = compute_strides(&output_dims);

    let mut reduce_mask = [[0u32; 4]; MAX_RANK / 4];
    for &axis in axes {
        reduce_mask[axis / 4][axis % 4] = 1;
    }

    #[allow(clippy::cast_possible_truncation)]
    let params = {
        let mut p = MinReduceParams {
            output_len: output_len as u32,
            reduction_len: reduction_len as u32,
            rank: rank as u32,
            _pad: 0,
            input_dims: [[0; 4]; MAX_RANK / 4],
            input_strides: [[0; 4]; MAX_RANK / 4],
            output_strides: [[0; 4]; MAX_RANK / 4],
            reduce_mask,
        };

        for (i, &d) in input_dims.iter().enumerate() {
            p.input_dims[i / 4][i % 4] = d as u32;
        }

        for (i, &s) in input_strides.iter().enumerate() {
            p.input_strides[i / 4][i % 4] = s as u32;
        }

        for (i, &s) in output_strides.iter().enumerate() {
            p.output_strides[i / 4][i % 4] = s as u32;
        }

        p
    };

    let pipeline = ctx.get_or_create_pipeline(
        TypeId::of::<MinReduceMarker<T>>(),
        create_shader::<T>,
        LABEL,
    )?;

    dispatch(ctx, input, output, &params, &pipeline)
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

#[allow(clippy::too_many_lines)]
fn create_shader<T: NumericElement>() -> String {
    let ty = T::wgsl_type();
    let max_val = T::wgsl_max();
    let max_vec4 = MAX_RANK / 4;

    format!(
        r"
            const WG_SIZE: u32 = {WORKGROUP_SIZE}u;
            const MAX_VEC4: u32 = {max_vec4}u;

            struct Params {{
                output_len: u32,
                reduction_len: u32,
                rank: u32,
                _pad: u32,
                input_dims: array<vec4<u32>, MAX_VEC4>,
                input_strides: array<vec4<u32>, MAX_VEC4>,
                output_strides: array<vec4<u32>, MAX_VEC4>,
                reduce_mask: array<vec4<u32>, MAX_VEC4>,
            }}

            @group(0) @binding(0) var<storage, read> input: array<{ty}>;
            @group(0) @binding(1) var<storage, read_write> output: array<{ty}>;
            @group(0) @binding(2) var<uniform> params: Params;

            var<workgroup> sdata: array<{ty}, WG_SIZE>;

            fn get_input_dim(idx: u32) -> u32 {{
                return params.input_dims[idx / 4u][idx % 4u];
            }}

            fn get_input_stride(idx: u32) -> u32 {{
                return params.input_strides[idx / 4u][idx % 4u];
            }}

            fn get_output_stride(idx: u32) -> u32 {{
                return params.output_strides[idx / 4u][idx % 4u];
            }}

            fn is_reduced(idx: u32) -> bool {{
                return params.reduce_mask[idx / 4u][idx % 4u] != 0u;
            }}

            @compute @workgroup_size(WG_SIZE)
            fn main(
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>
            ) {{
                let tid = lid.x;
                let output_idx = wid.x;

                if output_idx >= params.output_len {{
                    return;
                }}

                var base_coords: array<u32, 8>;
                var remaining = output_idx;
                for (var i = 0u; i < params.rank; i++) {{
                    let stride = get_output_stride(i);
                    if stride > 0u {{
                        base_coords[i] = remaining / stride;
                        remaining = remaining % stride;
                    }} else {{
                        base_coords[i] = 0u;
                    }}
                }}

                var acc: {ty} = {max_val};
                var reduction_idx = tid;

                while reduction_idx < params.reduction_len {{
                    var input_idx = 0u;
                    var red_remaining = reduction_idx;

                    for (var i = 0u; i < params.rank; i++) {{
                        var coord: u32;
                        if is_reduced(i) {{
                            var red_stride = 1u;
                            for (var j = i + 1u; j < params.rank; j++) {{
                                if is_reduced(j) {{
                                    red_stride *= get_input_dim(j);
                                }}
                            }}
                            coord = red_remaining / red_stride;
                            red_remaining = red_remaining % red_stride;
                        }} else {{
                            coord = base_coords[i];
                        }}
                        input_idx += coord * get_input_stride(i);
                    }}

                    acc = min(acc, input[input_idx]);
                    reduction_idx += WG_SIZE;
                }}

                sdata[tid] = acc;
                workgroupBarrier();

                if tid < 128u {{ sdata[tid] = min(sdata[tid], sdata[tid + 128u]); }}
                workgroupBarrier();
                if tid < 64u {{ sdata[tid] = min(sdata[tid], sdata[tid + 64u]); }}
                workgroupBarrier();
                if tid < 32u {{ sdata[tid] = min(sdata[tid], sdata[tid + 32u]); }}
                workgroupBarrier();
                if tid < 16u {{ sdata[tid] = min(sdata[tid], sdata[tid + 16u]); }}
                workgroupBarrier();
                if tid < 8u {{ sdata[tid] = min(sdata[tid], sdata[tid + 8u]); }}
                workgroupBarrier();
                if tid < 4u {{ sdata[tid] = min(sdata[tid], sdata[tid + 4u]); }}
                workgroupBarrier();
                if tid < 2u {{ sdata[tid] = min(sdata[tid], sdata[tid + 2u]); }}
                workgroupBarrier();
                if tid < 1u {{ sdata[tid] = min(sdata[tid], sdata[tid + 1u]); }}

                if tid == 0u {{
                    output[output_idx] = sdata[0];
                }}
            }}
        "
    )
}

fn dispatch<T: NumericElement>(
    ctx: &Context,
    input: &Buffer<T>,
    output: &Buffer<T>,
    params: &MinReduceParams,
    pipeline: &wgpu::ComputePipeline,
) -> Result<(), Error> {
    let output_len = params.output_len;

    if output_len > MAX_WORKGROUPS {
        return Err(Error::Device(format!(
            "output length {output_len} exceeds maximum workgroups {MAX_WORKGROUPS}"
        )));
    }

    let params = ctx.create_uniform_buffer(params);

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(LABEL) });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(LABEL),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(output_len, 1, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
