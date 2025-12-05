//! Element-wise remainder kernel.
//!
//! Computes the remainder of division using a compute shader.

use crate::kernel::assert_same_len;
use crate::{Buffer, Element, Error, GpuContext};

/// Workgroup size for the rem kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Computes the remainder of division element-wise.
///
/// Computes `c[i] = a[i] % b[i]` for all elements.
///
/// # Errors
///
/// Returns [`Error::Kernel`](crate::Error::Kernel) if buffer lengths do not match.
/// Returns [`Error::Device`](crate::Error::Device) if buffer length exceeds u32
/// or the GPU operation fails.
pub fn rem<T: Element>(
    ctx: &GpuContext,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
) -> Result<(), Error> {
    assert_same_len(a, b, "b")?;
    assert_same_len(a, c, "c")?;

    if a.is_empty() {
        return Ok(());
    }

    let pipeline = ctx.get_or_create_pipeline::<T, _>(create_pipeline::<T>);

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

        let vec4_count = u32::try_from(a.len())
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
        r#"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read> b: array<{vec4_type}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    c[tid] = a[tid] % b[tid];
                }}
            }}
        "#
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rem_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rem_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::kernel::fill;

    use super::*;

    #[test]
    fn test_rem() {
        let ctx = GpuContext::default();

        // f32
        let a = ctx
            .create_buffer_from_slice(&[5.5f32, 7.0, 10.0, 3.0])
            .unwrap();
        let b = ctx
            .create_buffer_from_slice(&[2.0f32, 3.0, 4.0, 5.0])
            .unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        let result = ctx.read_buffer(&c).unwrap();
        assert_relative_eq!(result[0], 1.5, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-5);

        // i32
        let a = ctx.create_buffer_from_slice(&[7i32, 10, 15, 3]).unwrap();
        let b = ctx.create_buffer_from_slice(&[3i32, 4, 6, 5]).unwrap();
        let c = ctx.create_buffer::<i32>(4).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1, 2, 3, 3]);

        // u32
        let a = ctx.create_buffer_from_slice(&[7u32, 10, 15, 3]).unwrap();
        let b = ctx.create_buffer_from_slice(&[3u32, 4, 6, 5]).unwrap();
        let c = ctx.create_buffer::<u32>(4).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1, 2, 3, 3]);

        // non-aligned
        let a = ctx.create_buffer::<i32>(42).unwrap();
        let b = ctx.create_buffer::<i32>(42).unwrap();
        let c = ctx.create_buffer::<i32>(42).unwrap();
        fill(&ctx, &a, 10i32).unwrap();
        fill(&ctx, &b, 3i32).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1; 42]);

        // large
        let len = 4096 * 4096;
        let a = ctx.create_buffer::<i32>(len).unwrap();
        let b = ctx.create_buffer::<i32>(len).unwrap();
        let c = ctx.create_buffer::<i32>(len).unwrap();
        fill(&ctx, &a, 17i32).unwrap();
        fill(&ctx, &b, 5i32).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        let result = ctx.read_buffer(&c).unwrap();
        for val in &result {
            assert_eq!(*val, 2);
        }

        // empty
        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        let c = ctx.create_buffer::<f32>(0).unwrap();
        rem(&ctx, &a, &b, &c).unwrap();
        assert!(ctx.read_buffer(&c).unwrap().is_empty());
    }

    #[test]
    fn test_rem_length_mismatch() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(4).unwrap();
        let b = ctx.create_buffer::<f32>(8).unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        let result = rem(&ctx, &a, &b, &c);
        assert!(result.is_err());
    }
}
