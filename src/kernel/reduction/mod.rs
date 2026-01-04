//! Reduction operation kernels.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};

use crate::element::NumericElement;
use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context};

pub(crate) mod sum;

/// Reduction parameters passed to shader as uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    rank: u32,
    len: u32,
    reduction_len: u32,
    _pad: u32,
}

/// Defines a reduction kernel.
macro_rules! define_kernel {
    ($kernel:ident, $label:literal, $init:ident, $op:literal) => {
        /// Kernel marker type.
        pub(crate) struct $kernel<T>(PhantomData<T>);

        /// Kernel trait implementation.
        impl<T: NumericElement> Kernel for $kernel<T> {
            const LABEL: &'static str = $label;
            type Output = T;

            fn wgsl() -> String {
                let ty = T::wgsl_type();
                let init = T::$init();

                alloc::format!(
                    r"
                        const WG_SIZE: u32 = {WORKGROUP_SIZE}u;

                        struct Params {{
                            rank: u32,
                            len: u32,
                            reduction_len: u32,
                            _pad: u32,
                        }}

                        @group(0) @binding(0) var<storage, read> x: array<{ty}>;
                        @group(0) @binding(1) var<storage, read_write> y: array<{ty}>;
                        @group(0) @binding(2) var<storage, read> x_dims: array<u32>;
                        @group(0) @binding(3) var<storage, read> x_strides: array<u32>;
                        @group(0) @binding(4) var<storage, read> y_strides: array<u32>;
                        @group(0) @binding(5) var<storage, read> reduce_mask: array<u32>;
                        @group(0) @binding(6) var<uniform> params: Params;

                        var<workgroup> sdata: array<{ty}, WG_SIZE>;

                        @compute @workgroup_size(WG_SIZE)
                        fn main(
                            @builtin(local_invocation_id) lid: vec3<u32>,
                            @builtin(workgroup_id) wid: vec3<u32>
                        ) {{
                            let tid = lid.x;
                            let y_idx = wid.x;

                            if y_idx >= params.len {{
                                return;
                            }}

                            var base_coords: array<u32, 32>;
                            var remaining = y_idx;
                            for (var i = 0u; i < params.rank; i++) {{
                                let stride = y_strides[i];
                                if stride > 0u {{
                                    base_coords[i] = remaining / stride;
                                    remaining = remaining % stride;
                                }} else {{
                                    base_coords[i] = 0u;
                                }}
                            }}

                            var acc: {ty} = {init};
                            var reduction_idx = tid;

                            while reduction_idx < params.reduction_len {{
                                var input_idx = 0u;
                                var red_remaining = reduction_idx;

                                for (var i = 0u; i < params.rank; i++) {{
                                    var coord: u32;
                                    if reduce_mask[i] != 0u {{
                                        var red_stride = 1u;
                                        for (var j = i + 1u; j < params.rank; j++) {{
                                            if reduce_mask[j] != 0u {{
                                                red_stride *= x_dims[j];
                                            }}
                                        }}
                                        coord = red_remaining / red_stride;
                                        red_remaining = red_remaining % red_stride;
                                    }} else {{
                                        coord = base_coords[i];
                                    }}
                                    input_idx += coord * x_strides[i];
                                }}

                                acc = {op}(acc, x[input_idx]);
                                reduction_idx += WG_SIZE;
                            }}

                            sdata[tid] = acc;
                            workgroupBarrier();

                            if tid < 128u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 128u]); }}
                            workgroupBarrier();
                            if tid < 64u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 64u]); }}
                            workgroupBarrier();
                            if tid < 32u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 32u]); }}
                            workgroupBarrier();
                            if tid < 16u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 16u]); }}
                            workgroupBarrier();
                            if tid < 8u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 8u]); }}
                            workgroupBarrier();
                            if tid < 4u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 4u]); }}
                            workgroupBarrier();
                            if tid < 2u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 2u]); }}
                            workgroupBarrier();
                            if tid < 1u {{ sdata[tid] = {op}(sdata[tid], sdata[tid + 1u]); }}

                            if tid == 0u {{
                                y[y_idx] = sdata[0];
                            }}
                        }}
                    ",
                    op = $op
                )
            }
        }
    };
}

define_kernel!(MaxReduce, "max_reduce", wgsl_min, "max");
define_kernel!(MinReduce, "min_reduce", wgsl_max, "min");

/// Executes a reduction kernel along specified axes.
///
/// # Panics
///
/// - Output rank exceeds max size
/// - Output length exceeds max size
/// - Output length exceeds maximum workgroups
/// - Reduction length exceeds max size
#[allow(clippy::too_many_lines)]
pub(crate) fn execute<K: Kernel + 'static, T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    x_dimensions: &[usize],
    x_strides: &[usize],
    y_strides: &[usize],
    axes: &[usize],
) {
    let rank = u32::try_from(y_strides.len()).expect("output rank exceeds max size");
    let len = u32::try_from(y.len()).expect("output length exceeds max size");
    let reduction_len = u32::try_from(axes.iter().map(|&a| x_dimensions[a]).product::<usize>())
        .expect("reduction length exceeds max size");

    if len == 0 || reduction_len == 0 {
        return;
    }

    assert!(
        len <= MAX_WORKGROUPS,
        "output length exceeds maximum workgroups"
    );

    let pipeline = ctx.get_or_create_pipeline(TypeId::of::<K>(), K::wgsl, K::LABEL);

    let x_dimensions = crate::kernel::convert_strides(x_dimensions);
    let x_strides = crate::kernel::convert_strides(x_strides);
    let y_strides = crate::kernel::convert_strides(y_strides);

    let reduce_mask: Vec<u32> = (0..rank as usize)
        .map(|i| u32::from(axes.contains(&i)))
        .collect();

    let params = Params {
        rank,
        len,
        reduction_len,
        _pad: 0,
    };

    let x_dimensions = ctx
        .create_buffer_from_slice(&x_dimensions)
        .expect("failed to create x_dimensions buffer");

    let x_strides = ctx
        .create_buffer_from_slice(&x_strides)
        .expect("failed to create x_strides buffer");

    let y_strides = ctx
        .create_buffer_from_slice(&y_strides)
        .expect("failed to create y_strides buffer");

    let reduce_mask = ctx
        .create_buffer_from_slice(&reduce_mask)
        .expect("failed to create reduce_mask buffer");

    let params = ctx.create_uniform_buffer(&params);

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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_dimensions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: y_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: reduce_mask.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
    });

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
        pass.dispatch_workgroups(len, 1, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}
