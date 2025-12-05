//! Element-wise scalar division kernel.
//!
//! Divides each element of a buffer by a scalar using a compute shader.

use crate::kernel::{assert_len, assert_same_len, debug_assert_same_device};
use crate::{Buffer, Element, Error, GpuContext};

/// Workgroup size for the `div_scalar` kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Divides each element of a buffer by a scalar.
///
/// Computes `c[i] = a[i] / b[0]` for all elements.
///
/// The `b` buffer must contain exactly one element.
///
/// # Errors
///
/// Returns [`Error::Kernel`](crate::Error::Kernel) if buffer lengths do not match
/// or if `b` does not have exactly one element.
/// Returns [`Error::Device`](crate::Error::Device) if buffer length exceeds u32
/// or the GPU operation fails.
///
/// # Panics
///
/// Debug builds panic if any buffer belongs to a different device than `ctx`.
pub fn div_scalar<T: Element>(
    ctx: &GpuContext,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
) -> Result<(), Error> {
    debug_assert_same_device(ctx, a, "a");
    debug_assert_same_device(ctx, b, "b");
    debug_assert_same_device(ctx, c, "c");
    assert_same_len(a, c, "c")?;
    assert_len(b, 1, "b")?;

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
        r"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read> b: {wgsl_type};
            @group(0) @binding(2) var<storage, read_write> c: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    c[tid] = a[tid] / {vec4_type}(b);
                }}
            }}
        "
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("div_scalar_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("div_scalar_pipeline"),
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

    use crate::kernel::fill;

    use super::*;

    #[test]
    fn test_div_scalar() {
        let ctx = GpuContext::default();

        // f32
        let a = ctx
            .create_buffer_from_slice(&[42.0f32, 84.0, 126.0, 168.0])
            .unwrap();
        let b = ctx.create_buffer_from_slice(&[42.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // i32
        let a = ctx
            .create_buffer_from_slice(&[42i32, 84, 126, 168])
            .unwrap();
        let b = ctx.create_buffer_from_slice(&[42i32]).unwrap();
        let c = ctx.create_buffer::<i32>(4).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1, 2, 3, 4]);

        // u32
        let a = ctx
            .create_buffer_from_slice(&[42u32, 84, 126, 168])
            .unwrap();
        let b = ctx.create_buffer_from_slice(&[42u32]).unwrap();
        let c = ctx.create_buffer::<u32>(4).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        assert_eq!(ctx.read_buffer(&c).unwrap(), vec![1, 2, 3, 4]);

        // non-aligned
        let a = ctx.create_buffer::<f32>(42).unwrap();
        let b = ctx.create_buffer_from_slice(&[PI]).unwrap();
        let c = ctx.create_buffer::<f32>(42).unwrap();
        fill(&ctx, &a, PI).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        let result = ctx.read_buffer(&c).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 1.0, epsilon = 1e-5);
        }

        // large
        let len = 4096 * 4096;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[PI]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();
        fill(&ctx, &a, 42.0f32).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        let result = ctx.read_buffer(&c).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 42.0 / PI, epsilon = 1e-5);
        }

        // empty
        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer_from_slice(&[42.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(0).unwrap();
        div_scalar(&ctx, &a, &b, &c).unwrap();
        assert!(ctx.read_buffer(&c).unwrap().is_empty());
    }

    #[test]
    fn test_div_scalar_length_mismatch() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(4).unwrap();
        let b = ctx.create_buffer_from_slice(&[42.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(8).unwrap();

        let result = div_scalar(&ctx, &a, &b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_scalar_wrong_b_length() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(4).unwrap();
        let b = ctx.create_buffer_from_slice(&[1.0f32, 2.0]).unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        let result = div_scalar(&ctx, &a, &b, &c);
        assert!(result.is_err());
    }
}
