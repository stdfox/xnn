//! Clamp ternary element-wise kernel.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::format;
use alloc::string::String;

use wgpu::util::DeviceExt;

use crate::element::NumericElement;
use crate::kernel::math::Params;
use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context};

/// Kernel marker type.
struct Clamp<T>(PhantomData<T>);

/// Kernel trait implementation.
impl<T: NumericElement> Kernel for Clamp<T> {
    const LABEL: &'static str = "clamp";
    type Output = T;

    fn wgsl() -> String {
        let ty = T::wgsl_type();

        format!(
            r"
                struct Params {{
                    rank: u32,
                    len: u32,
                }}

                @group(0) @binding(0) var<storage, read> x: array<{ty}>;
                @group(0) @binding(1) var<storage, read> a: array<{ty}>;
                @group(0) @binding(2) var<storage, read> b: array<{ty}>;
                @group(0) @binding(3) var<storage, read_write> y: array<{ty}>;
                @group(0) @binding(4) var<storage, read> x_strides: array<u32>;
                @group(0) @binding(5) var<storage, read> a_strides: array<u32>;
                @group(0) @binding(6) var<storage, read> b_strides: array<u32>;
                @group(0) @binding(7) var<storage, read> y_strides: array<u32>;
                @group(0) @binding(8) var<uniform> params: Params;

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
                        let coord = remaining / y_strides[i];
                        remaining = remaining % y_strides[i];
                        x_idx += coord * x_strides[i];
                        a_idx += coord * a_strides[i];
                        b_idx += coord * b_strides[i];
                    }}

                    y[tid] = max(min(x[x_idx], b[b_idx]), a[a_idx]);
                }}
            "
        )
    }
}

/// Executes the clamp kernel.
///
/// # Panics
///
/// - Output length exceeds max size
/// - Output rank exceeds max size
/// - Output buffer too small
#[allow(clippy::too_many_lines)]
pub(crate) fn execute<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    a: &Buffer<T>,
    b: &Buffer<T>,
    y: &Buffer<T>,
    x_strides: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
    y_strides: &[usize],
) {
    let byte_size = (y.len() * T::NATIVE_SIZE) as u64;
    assert!(y.byte_size() >= byte_size, "output buffer too small");

    let rank = u32::try_from(y_strides.len()).expect("output rank exceeds max size");
    let len = u32::try_from(y.len()).expect("output length exceeds max size");

    let x_strides = crate::kernel::convert_strides(x_strides);
    let a_strides = crate::kernel::convert_strides(a_strides);
    let b_strides = crate::kernel::convert_strides(b_strides);
    let y_strides = crate::kernel::convert_strides(y_strides);

    let pipeline = ctx.get_or_create_pipeline(
        TypeId::of::<Clamp<T>>(),
        Clamp::<T>::wgsl,
        Clamp::<T>::LABEL,
    );

    let x_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&x_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let a_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&a_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let b_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&b_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let y_strides = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&y_strides),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let params = ctx.create_uniform_buffer(&Params { rank, len });

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(Clamp::<T>::LABEL),
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
                resource: x_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: a_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: y_strides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let (x, y) = super::compute_workgroups(len);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(Clamp::<T>::LABEL),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(Clamp::<T>::LABEL),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}
