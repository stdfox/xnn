//! Fill buffer with constant value kernel.
//!
//! Sets all elements in a buffer to the same value using a compute shader.

use wgpu::util::DeviceExt as _;

use crate::kernel::debug_assert_same_device;
use crate::{Buffer, Element, Error, GpuContext};

/// Workgroup size for the fill kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Fills a buffer with a constant value.
///
/// Sets every element in the buffer to the specified value: `buf[i] = value`.
///
/// # Errors
///
/// Returns [`Error::Device`](crate::Error::Device) if the buffer length exceeds u32
/// or the GPU operation fails.
///
/// # Panics
///
/// Debug builds panic if the buffer belongs to a different device than `ctx`.
pub fn fill<T: Element>(ctx: &GpuContext, buf: &Buffer<T>, value: T) -> Result<(), Error> {
    debug_assert_same_device(ctx, buf, "buf");

    if buf.is_empty() {
        return Ok(());
    }

    let pipeline = ctx.get_or_create_pipeline::<T, _>(create_pipeline::<T>);

    let value_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&value),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: value_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf.inner().as_entire_binding(),
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

        let vec4_count = u32::try_from(buf.len())
            .map_err(|_| Error::Device("buffer length exceeds u32".into()))?
            .div_ceil(4);

        let total_workgroups = vec4_count.div_ceil(WORKGROUP_SIZE);
        let workgroups_x = total_workgroups.min(MAX_WORKGROUPS_PER_DIM);
        let workgroups_y = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DIM);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Device(e.to_string()))?;

    Ok(())
}

fn create_shader_source<T: Element>() -> String {
    let wgsl_type = T::wgsl_type();
    let vec4_type = format!("vec4<{wgsl_type}>");

    format!(
        r"
            @group(0) @binding(0) var<uniform> value: {wgsl_type};
            @group(0) @binding(1) var<storage, read_write> buf: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&buf) {{
                    buf[tid] = {vec4_type}(value);
                }}
            }}
        "
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fill_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fill_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_fill() {
        let ctx = GpuContext::default();

        // f32
        let buf = ctx.create_buffer::<f32>(4).unwrap();
        fill(&ctx, &buf, 42.0f32).unwrap();
        assert_eq!(ctx.read_buffer(&buf).unwrap(), vec![42.0, 42.0, 42.0, 42.0]);

        // i32
        let buf = ctx.create_buffer::<i32>(4).unwrap();
        fill(&ctx, &buf, 42i32).unwrap();
        assert_eq!(ctx.read_buffer(&buf).unwrap(), vec![42, 42, 42, 42]);

        // u32
        let buf = ctx.create_buffer::<u32>(4).unwrap();
        fill(&ctx, &buf, 42u32).unwrap();
        assert_eq!(ctx.read_buffer(&buf).unwrap(), vec![42, 42, 42, 42]);

        // non-aligned
        let buf = ctx.create_buffer::<f32>(42).unwrap();
        fill(&ctx, &buf, 42f32).unwrap();
        assert_eq!(ctx.read_buffer(&buf).unwrap(), vec![42.0; 42]);

        // large
        let len = 4096 * 4096;
        let buf = ctx.create_buffer::<f32>(len).unwrap();
        fill(&ctx, &buf, PI).unwrap();
        let result = ctx.read_buffer(&buf).unwrap();
        for val in &result {
            assert_relative_eq!(*val, PI, epsilon = 1e-5);
        }

        // empty
        let buf = ctx.create_buffer::<f32>(0).unwrap();
        fill(&ctx, &buf, PI).unwrap();
        assert!(ctx.read_buffer(&buf).unwrap().is_empty());
    }
}
