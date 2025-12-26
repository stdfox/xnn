//! Constant fill kernel.

use core::any::TypeId;
use core::marker::PhantomData;

use alloc::format;
use alloc::string::String;

use crate::kernel::{Kernel, MAX_WORKGROUPS, WORKGROUP_SIZE};
use crate::{Buffer, Context, Element};

/// Constant fill kernel: fills buffer with a uniform value.
pub(crate) struct Constant<T>(PhantomData<T>);

/// Kernel trait implementation.
impl<T: Element> Kernel for Constant<T> {
    const LABEL: &'static str = "constant";
    type Output = T;

    fn wgsl() -> String {
        let ty = T::wgsl_type();

        format!(
            r"
                @group(0) @binding(0) var<storage, read_write> buffer: array<vec4<{ty}>>;
                @group(0) @binding(1) var<uniform> value: {ty};

                @compute @workgroup_size({WORKGROUP_SIZE})
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                    let tid = gid.x + gid.y * {MAX_WORKGROUPS}u * {WORKGROUP_SIZE}u;
                    if tid < arrayLength(&buffer) {{
                        buffer[tid] = vec4<{ty}>(value);
                    }}
                }}
            "
        )
    }
}

/// Fills buffer with constant value.
///
/// # Panics
///
/// - Output length exceeds max size
pub(crate) fn execute<T: Element>(ctx: &Context, buffer: &Buffer<T>, value: &wgpu::Buffer) {
    let len = u32::try_from(buffer.byte_size() / (T::NATIVE_SIZE * 4) as u64)
        .expect("output length exceeds max size");

    if len == 0 {
        return;
    }

    let pipeline = ctx.get_or_create_pipeline(
        TypeId::of::<Constant<T>>(),
        Constant::<T>::wgsl,
        Constant::<T>::LABEL,
    );

    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(Constant::<T>::LABEL),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: value.as_entire_binding(),
            },
        ],
    });

    let workgroups = len.div_ceil(WORKGROUP_SIZE);
    let x = workgroups.min(MAX_WORKGROUPS);
    let y = workgroups.div_ceil(MAX_WORKGROUPS);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(Constant::<T>::LABEL),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(Constant::<T>::LABEL),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}
