//! Broadcast kernel.
//!
//! Broadcasts a vector to all rows of a matrix using a compute shader.

use wgpu::util::DeviceExt as _;

use crate::kernel::{assert_len, debug_assert_same_device};
use crate::{Buffer, Element, Error, GpuContext};

/// Workgroup size for compute shader.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum workgroups per dimension (WebGPU limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Broadcasts a vector to all rows of a matrix.
///
/// Computes `b[i * cols + j] = a[j]` for all `i` in `0..rows`, `j` in `0..cols`.
///
/// # Arguments
/// * `a` - Source vector of length `cols`
/// * `b` - Destination matrix of length `rows * cols`
/// * `rows` - Number of rows in output
/// * `cols` - Number of columns (must equal `a.len()`)
///
/// # Errors
///
/// Returns [`Error::Kernel`](crate::Error::Kernel) if `a.len() != cols` or
/// `b.len() != rows * cols`.
/// Returns [`Error::Device`](crate::Error::Device) if dimensions exceed `u32::MAX`.
///
/// # Panics
///
/// Debug builds panic if any buffer belongs to a different device than `ctx`.
pub fn broadcast_rows<T: Element>(
    ctx: &GpuContext,
    a: &Buffer<T>,
    b: &Buffer<T>,
    rows: usize,
    cols: usize,
) -> Result<(), Error> {
    debug_assert_same_device(ctx, a, "a");
    debug_assert_same_device(ctx, b, "b");
    assert_len(a, cols, "a")?;
    assert_len(b, rows * cols, "b")?;

    if rows == 0 || cols == 0 {
        return Ok(());
    }

    let rows32 = u32::try_from(rows).map_err(|_| Error::Device("rows exceeds u32::MAX".into()))?;
    let cols32 = u32::try_from(cols).map_err(|_| Error::Device("cols exceeds u32::MAX".into()))?;

    let pipeline = ctx.get_or_create_pipeline::<T, _>(create_pipeline::<T>);

    let dims = [rows32, cols32];
    let dims_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

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
                resource: dims_buffer.as_entire_binding(),
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

        let total_elements = rows32 * cols32;
        let total_workgroups = total_elements.div_ceil(WORKGROUP_SIZE);
        let workgroups_x = total_workgroups.min(MAX_WORKGROUPS_PER_DIM);
        let workgroups_y = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DIM);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}

fn create_shader_source<T: Element>() -> String {
    let ty = T::wgsl_type();

    format!(
        r"
            @group(0) @binding(0) var<storage, read> a: array<{ty}>;
            @group(0) @binding(1) var<storage, read_write> b: array<{ty}>;
            @group(0) @binding(2) var<uniform> dims: vec2<u32>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let tid = gid.x + gid.y * {MAX_WORKGROUPS_PER_DIM}u * {WORKGROUP_SIZE}u;
                let total = dims.x * dims.y;
                if tid >= total {{
                    return;
                }}
                let col = tid % dims.y;
                b[tid] = a[col];
            }}
        "
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("broadcast_rows_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("broadcast_rows_pipeline"),
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
    use crate::kernel::fill;
    use approx::assert_relative_eq;

    #[test]
    fn test_broadcast_rows() {
        let ctx = GpuContext::default();

        // [1, 2, 3] -> [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
        let a = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = ctx.create_buffer::<f32>(12).unwrap();
        broadcast_rows(&ctx, &a, &b, 4, 3).unwrap();
        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(
            result,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_broadcast_rows_single_col() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer_from_slice(&[5.0f32]).unwrap();
        let b = ctx.create_buffer::<f32>(4).unwrap();
        broadcast_rows(&ctx, &a, &b, 4, 1).unwrap();
        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_broadcast_rows_non_aligned() {
        let ctx = GpuContext::default();

        // 21 rows × 2 cols = 42 elements
        let a = ctx.create_buffer_from_slice(&[1.0f32, 2.0]).unwrap();
        let b = ctx.create_buffer::<f32>(42).unwrap();
        broadcast_rows(&ctx, &a, &b, 21, 2).unwrap();
        let result = ctx.read_buffer(&b).unwrap();
        for i in 0..21 {
            assert_eq!(result[i * 2], 1.0);
            assert_eq!(result[i * 2 + 1], 2.0);
        }
    }

    #[test]
    fn test_broadcast_rows_large() {
        let ctx = GpuContext::default();

        // 4096 rows × 4096 cols
        let size = 4096;
        let a = ctx.create_buffer::<f32>(size).unwrap();
        let b = ctx.create_buffer::<f32>(size * size).unwrap();
        fill(&ctx, &a, 3.14f32).unwrap();
        broadcast_rows(&ctx, &a, &b, size, size).unwrap();
        let result = ctx.read_buffer(&b).unwrap();
        for val in &result {
            assert_relative_eq!(*val, 3.14, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_broadcast_rows_empty() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        broadcast_rows(&ctx, &a, &b, 0, 0).unwrap();
    }

    #[test]
    fn test_broadcast_rows_wrong_len() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(3).unwrap();
        let b = ctx.create_buffer::<f32>(12).unwrap();
        let result = broadcast_rows(&ctx, &a, &b, 4, 4);
        assert!(result.is_err());

        let a = ctx.create_buffer::<f32>(3).unwrap();
        let b = ctx.create_buffer::<f32>(10).unwrap();
        let result = broadcast_rows(&ctx, &a, &b, 4, 3);
        assert!(result.is_err());
    }
}
