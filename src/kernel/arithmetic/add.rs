//! Element-wise addition kernel.
//!
//! Adds two buffers element-wise using a compute shader.

use crate::kernel::debug_assert_same_len;
use crate::{Buffer, Context, Element};

/// Workgroup size for the add kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Adds two buffers element-wise.
///
/// Computes `c[i] = a[i] + b[i]` for all elements.
///
/// # Panics
///
/// - Buffer length exceeds `u32::MAX`.
/// - (debug) Buffer lengths do not match.
pub fn add<T: Element>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>, c: &Buffer<T>) {
    debug_assert_same_len(a, b, "b");
    debug_assert_same_len(a, c, "c");

    if a.is_empty() {
        return;
    }

    let pipeline = ctx.get_or_create_kernel_pipeline::<T, _>(create_pipeline::<T>);

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.inner().as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let len = u32::try_from(a.len()).expect("buffer length exceeds u32::MAX");
        let vec4_count = len.div_ceil(4);

        let total_workgroups = vec4_count.div_ceil(WORKGROUP_SIZE);
        let workgroups_x = total_workgroups.min(MAX_WORKGROUPS_PER_DIM);
        let workgroups_y = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DIM);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}

fn create_shader_source<T: Element>() -> String {
    let wgsl_type = T::wgsl_type();
    let vec4_type = format!("vec4<{wgsl_type}>");

    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read> b: array<{vec4_type}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    c[tid] = a[tid] + b[tid];
                }}
            }}
        "
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("add_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("add_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let ctx = Context::try_default().unwrap();

        // f32
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let b = ctx
            .create_buffer_from_slice(&[5.0f32, 6.0, 7.0, 8.0])
            .unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();
        add(&ctx, &a, &b, &c);
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![6.0, 8.0, 10.0, 12.0]);

        // i32
        let a = ctx.create_buffer_from_slice(&[1i32, 2, 3, 4]).unwrap();
        let b = ctx.create_buffer_from_slice(&[5i32, 6, 7, 8]).unwrap();
        let c = ctx.create_buffer::<i32>(4).unwrap();
        add(&ctx, &a, &b, &c);
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![6, 8, 10, 12]);

        // u32
        let a = ctx.create_buffer_from_slice(&[1u32, 2, 3, 4]).unwrap();
        let b = ctx.create_buffer_from_slice(&[5u32, 6, 7, 8]).unwrap();
        let c = ctx.create_buffer::<u32>(4).unwrap();
        add(&ctx, &a, &b, &c);
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![6, 8, 10, 12]);

        // empty
        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        let c = ctx.create_buffer::<f32>(0).unwrap();
        add(&ctx, &a, &b, &c);
        assert!(ctx.read_buffer(&c).unwrap().is_empty());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "buffer length mismatch"))]
    fn test_add_assert_same_len() {
        let ctx = Context::try_default().unwrap();

        let a = ctx.create_buffer::<f32>(4).unwrap();
        let b = ctx.create_buffer::<f32>(8).unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        add(&ctx, &a, &b, &c);
    }
}
