//! Matrix transpose kernel.
//!
//! Transposes a 2D matrix using a tiled compute shader for cache efficiency.

use wgpu::util::DeviceExt as _;

use crate::kernel::debug_assert_len;
use crate::{Buffer, Context, Element};

/// Tile size for workgroup (16×16 threads).
const TILE_SIZE: u32 = 16;

/// Transposes a 2D matrix.
///
/// Computes `b[j * rows + i] = a[i * cols + j]` for all elements.
///
/// # Arguments
/// * `a` - Source matrix of shape (rows, cols)
/// * `b` - Destination matrix of shape (cols, rows)
/// * `rows` - Number of rows in input
/// * `cols` - Number of columns in input
///
/// # Panics
///
/// - Matrix dimensions exceed `u32::MAX`.
/// - (debug) Buffer `a` length does not match `rows * cols`.
/// - (debug) Buffer `b` length does not match `rows * cols`.
pub fn transpose<T: Element>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    rows: usize,
    cols: usize,
) {
    debug_assert_len(a, rows * cols, "a");
    debug_assert_len(b, rows * cols, "b");

    if rows == 0 || cols == 0 {
        return;
    }

    let rows32 = u32::try_from(rows).expect("rows exceeds u32::MAX");
    let cols32 = u32::try_from(cols).expect("cols exceeds u32::MAX");

    let pipeline = ctx.get_or_create_kernel_pipeline::<T, _>(create_pipeline::<T>);

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

        // Dispatch enough workgroups to cover the input matrix
        let workgroups_x = cols32.div_ceil(TILE_SIZE);
        let workgroups_y = rows32.div_ceil(TILE_SIZE);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
}

fn create_shader_source<T: Element>() -> String {
    let ty = T::wgsl_type();

    format!(
        r"
            const TILE_SIZE: u32 = {TILE_SIZE}u;

            @group(0) @binding(0) var<storage, read> a: array<{ty}>;
            @group(0) @binding(1) var<storage, read_write> b: array<{ty}>;
            @group(0) @binding(2) var<uniform> dims: vec2<u32>;

            var<workgroup> tile: array<array<{ty}, {tile_pad}>, {TILE_SIZE}>;

            @compute @workgroup_size({TILE_SIZE}, {TILE_SIZE})
            fn main(
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>
            ) {{
                let rows = dims.x;
                let cols = dims.y;

                // Input coordinates
                let in_row = wid.y * TILE_SIZE + lid.y;
                let in_col = wid.x * TILE_SIZE + lid.x;

                // Load tile (coalesced read)
                if in_row < rows && in_col < cols {{
                    tile[lid.y][lid.x] = a[in_row * cols + in_col];
                }}

                workgroupBarrier();

                // Output coordinates (transposed tile position)
                let out_row = wid.x * TILE_SIZE + lid.y;
                let out_col = wid.y * TILE_SIZE + lid.x;

                // Write transposed (coalesced write)
                if out_row < cols && out_col < rows {{
                    b[out_row * rows + out_col] = tile[lid.x][lid.y];
                }}
            }}
        ",
        tile_pad = TILE_SIZE + 1
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("transpose_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("transpose_pipeline"),
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
    fn test_transpose() {
        let ctx = Context::try_default().unwrap();

        // 2x3 -> 3x2
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let b = ctx.create_buffer::<f32>(6).unwrap();

        transpose(&ctx, &a, &b, 2, 3);

        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_square() {
        let ctx = Context::try_default().unwrap();

        // 3x3 transpose
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let b = ctx.create_buffer::<f32>(9).unwrap();

        transpose(&ctx, &a, &b, 3, 3);

        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_transpose_single_row() {
        let ctx = Context::try_default().unwrap();

        // 1x4 -> 4x1
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let b = ctx.create_buffer::<f32>(4).unwrap();

        transpose(&ctx, &a, &b, 1, 4);

        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_single_col() {
        let ctx = Context::try_default().unwrap();

        // 4x1 -> 1x4
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let b = ctx.create_buffer::<f32>(4).unwrap();

        transpose(&ctx, &a, &b, 4, 1);

        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_non_aligned() {
        let ctx = Context::try_default().unwrap();

        // 7x6 -> 6x7 (42 elements, non-tile-aligned)
        let rows = 7;
        let cols = 6;
        let a_data: Vec<f32> = (0..42).map(|i| i as f32).collect();
        let a = ctx.create_buffer_from_slice(&a_data).unwrap();
        let b = ctx.create_buffer::<f32>(42).unwrap();

        transpose(&ctx, &a, &b, rows, cols);

        let result = ctx.read_buffer(&b).unwrap();
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(result[j * rows + i], a_data[i * cols + j]);
            }
        }
    }

    #[test]
    fn test_transpose_large() {
        let ctx = Context::try_default().unwrap();

        // 4096x4096 square matrix
        let size = 4096;
        let len = size * size;
        let a_data: Vec<f32> = (0..len as u32).map(|i| i as f32).collect();
        let a = ctx.create_buffer_from_slice(&a_data).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        transpose(&ctx, &a, &b, size, size);

        let result = ctx.read_buffer(&b).unwrap();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[size], 1.0);
        assert_eq!(result[1], a_data[size]);
        assert_eq!(result[size + 1], a_data[size + 1]);
    }

    #[test]
    fn test_transpose_empty() {
        let ctx = Context::try_default().unwrap();
        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        transpose(&ctx, &a, &b, 0, 0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "length mismatch"))]
    fn test_transpose_assert_len() {
        let ctx = Context::try_default().unwrap();
        let a = ctx.create_buffer::<f32>(6).unwrap();
        let b = ctx.create_buffer::<f32>(8).unwrap();
        transpose(&ctx, &a, &b, 2, 3);
    }
}
