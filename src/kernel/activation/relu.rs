//! `ReLU` activation kernel.
//!
//! Applies the Rectified Linear Unit activation function using a compute shader.

use crate::kernel::{debug_assert_same_device, debug_assert_same_len};
use crate::{Buffer, FloatElement, GpuContext};

/// Workgroup size for the `relu` kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Applies `ReLU` activation element-wise.
///
/// Computes `b[i] = max(a[i], 0)` for all elements.
///
/// # Panics
///
/// - Buffer length exceeds `u32::MAX`.
/// - (debug) Buffer lengths do not match.
/// - (debug) Buffer belongs to a different device than `ctx`.
pub fn relu<T: FloatElement>(ctx: &GpuContext, a: &Buffer<T>, b: &Buffer<T>) {
    debug_assert_same_device(ctx, a, "a");
    debug_assert_same_device(ctx, b, "b");
    debug_assert_same_len(a, b, "b");

    if a.is_empty() {
        return;
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

fn create_shader_source<T: FloatElement>() -> String {
    let wgsl_type = T::wgsl_type();
    let vec4_type = format!("vec4<{wgsl_type}>");

    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read_write> b: array<{vec4_type}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                if tid < arrayLength(&a) {{
                    b[tid] = max(a[tid], {vec4_type}(0.0));
                }}
            }}
        "
    )
}

fn create_pipeline<T: FloatElement>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("relu_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("relu_pipeline"),
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
    fn test_relu() {
        let ctx = GpuContext::default();

        // f32 - basic
        let a = ctx
            .create_buffer_from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5])
            .unwrap();
        let b = ctx.create_buffer::<f32>(8).unwrap();
        relu(&ctx, &a, &b);
        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5]);

        // non-aligned
        let a = ctx.create_buffer::<f32>(42).unwrap();
        let b = ctx.create_buffer::<f32>(42).unwrap();
        fill(&ctx, &a, -1.0f32);
        relu(&ctx, &a, &b);
        let result = ctx.read_buffer(&b).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-5);
        }

        // non-aligned positive
        let a = ctx.create_buffer::<f32>(42).unwrap();
        let b = ctx.create_buffer::<f32>(42).unwrap();
        fill(&ctx, &a, 5.0f32);
        relu(&ctx, &a, &b);
        let result = ctx.read_buffer(&b).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 5.0, epsilon = 1e-5);
        }

        // large
        let len = 4096 * 4096;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        fill(&ctx, &a, -3.0f32);
        relu(&ctx, &a, &b);
        let result = ctx.read_buffer(&b).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 0.0, epsilon = 1e-5);
        }

        // empty
        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        relu(&ctx, &a, &b);
        assert!(ctx.read_buffer(&b).unwrap().is_empty());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "buffer length mismatch"))]
    fn test_relu_length_mismatch() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(4).unwrap();
        let b = ctx.create_buffer::<f32>(8).unwrap();

        relu(&ctx, &a, &b);
    }
}
