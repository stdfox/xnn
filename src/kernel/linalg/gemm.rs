//! General matrix multiply kernel.
//!
//! Performs matrix multiplication using a tiled compute shader.
//! Only works with floating-point types.

use wgpu::util::DeviceExt as _;

use crate::kernel::assert_len;
use crate::{Buffer, Error, FloatElement, GpuContext};

/// Block size for register tiling (each thread computes BM×BN elements).
const BLOCK_SIZE: u32 = 4;

/// Workgroup dimensions.
const WG_SIZE: u32 = 16;

/// Output tile size (WG_SIZE * BLOCK_SIZE).
const TILE_SIZE: u32 = WG_SIZE * BLOCK_SIZE;

/// K-dimension tile size.
const TILE_K: u32 = 16;

/// Padded tile sizes to avoid shared memory bank conflicts.
const TILE_SIZE_PAD: u32 = TILE_SIZE + 1;
const TILE_K_PAD: u32 = TILE_K + 1;

/// Maximum workgroups per dispatch to avoid GPU watchdog timeout.
const MAX_WORKGROUPS_PER_DISPATCH: u32 = 65536;

/// Performs general matrix multiplication.
///
/// Computes `C = A × B` where matrices are stored in row-major order:
/// - A is `m × k`
/// - B is `k × n`
/// - C is `m × n`
///
/// Only works with floating-point types.
///
/// # Errors
///
/// Returns [`Error::Kernel`](crate::Error::Kernel) if buffer sizes do not match
/// matrix dimensions.
pub fn gemm<T: FloatElement>(
    ctx: &GpuContext,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), Error> {
    assert_len(a, m * k, "a")?;
    assert_len(b, k * n, "b")?;
    assert_len(c, m * n, "c")?;

    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let pipeline = ctx.get_or_create_pipeline::<T, _>(create_pipeline::<T>);

    let m_tiles = (m as u32).div_ceil(TILE_SIZE);
    let n_tiles = (n as u32).div_ceil(TILE_SIZE);
    let total_workgroups = m_tiles * n_tiles;

    let num_dispatches = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DISPATCH);
    let n_tiles_per_dispatch = n_tiles.div_ceil(num_dispatches);

    // Pre-create all uniform buffers and bind groups
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut bind_groups = Vec::with_capacity(num_dispatches as usize);

    for dispatch in 0..num_dispatches {
        let n_tile_start = dispatch * n_tiles_per_dispatch;
        if n_tile_start >= n_tiles {
            break;
        }

        let col_start = n_tile_start * TILE_SIZE;
        let dims = [m as u32, k as u32, n as u32, col_start];
        let dims_buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        bind_groups.push((
            bind_group,
            n_tile_start,
            ((dispatch + 1) * n_tiles_per_dispatch).min(n_tiles),
        ));
    }

    // Single encoder for all passes
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for (bind_group, n_tile_start, n_tile_end) in &bind_groups {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(m_tiles, n_tile_end - n_tile_start, 1);
    }

    ctx.queue().submit(Some(encoder.finish()));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Device(e.to_string()))?;

    Ok(())
}

fn create_shader_source<T: FloatElement>() -> String {
    let ty = T::wgsl_type();
    let vec4_ty = format!("vec4<{ty}>");

    // Shared memory: As[TILE_SIZE × TILE_K_PAD], Bs[TILE_K × TILE_SIZE_PAD]
    let as_size = TILE_SIZE * TILE_K_PAD;
    let bs_size = TILE_K * TILE_SIZE_PAD;

    format!(
        r"
            const TILE: u32 = {TILE_SIZE}u;
            const TILE_K: u32 = {TILE_K}u;
            const TILE_PAD: u32 = {TILE_SIZE_PAD}u;
            const TILE_K_PAD: u32 = {TILE_K_PAD}u;
            const WG: u32 = {WG_SIZE}u;
            const BLK: u32 = {BLOCK_SIZE}u;

            @group(0) @binding(0) var<storage, read> a: array<{vec4_ty}>;
            @group(0) @binding(1) var<storage, read> b: array<{vec4_ty}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{ty}>;
            @group(0) @binding(3) var<uniform> dims: vec4<u32>;

            var<workgroup> As: array<{ty}, {as_size}>;
            var<workgroup> Bs: array<{ty}, {bs_size}>;

            // Load scalar from vec4
            fn load_a(idx: u32) -> {ty} {{
                let v = a[idx >> 2u];
                let c = idx & 3u;
                return select(select(select(v.w, v.z, c == 2u), v.y, c == 1u), v.x, c == 0u);
            }}

            fn load_b(idx: u32) -> {ty} {{
                let v = b[idx >> 2u];
                let c = idx & 3u;
                return select(select(select(v.w, v.z, c == 2u), v.y, c == 1u), v.x, c == 0u);
            }}

            @compute @workgroup_size({WG_SIZE}, {WG_SIZE})
            fn main(
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>
            ) {{
                let M = dims.x;
                let K = dims.y;
                let N = dims.z;
                let col_offset = dims.w;

                let lr = lid.x;
                let lc = lid.y;
                let thread_id = lr * WG + lc;

                // Global output position for this thread's 4x4 block
                let tile_row = wid.x * TILE;
                let tile_col = col_offset + wid.y * TILE;
                let c_row = tile_row + lr * BLK;
                let c_col = tile_col + lc * BLK;

                // 4x4 accumulators (16 registers)
                var acc00: {ty} = 0.0; var acc01: {ty} = 0.0; var acc02: {ty} = 0.0; var acc03: {ty} = 0.0;
                var acc10: {ty} = 0.0; var acc11: {ty} = 0.0; var acc12: {ty} = 0.0; var acc13: {ty} = 0.0;
                var acc20: {ty} = 0.0; var acc21: {ty} = 0.0; var acc22: {ty} = 0.0; var acc23: {ty} = 0.0;
                var acc30: {ty} = 0.0; var acc31: {ty} = 0.0; var acc32: {ty} = 0.0; var acc33: {ty} = 0.0;

                let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

                for (var kt: u32 = 0u; kt < num_k_tiles; kt++) {{
                    let k_start = kt * TILE_K;

                    // Cooperative load As[64×16]: 256 threads load 1024 elements (4 per thread)
                    for (var i: u32 = 0u; i < 4u; i++) {{
                        let a_idx = thread_id * 4u + i;
                        let a_row = a_idx / TILE_K;
                        let a_kcol = a_idx % TILE_K;
                        let g_row = tile_row + a_row;
                        let g_kcol = k_start + a_kcol;
                        if g_row < M && g_kcol < K {{
                            As[a_row * TILE_K_PAD + a_kcol] = load_a(g_row * K + g_kcol);
                        }} else {{
                            As[a_row * TILE_K_PAD + a_kcol] = 0.0;
                        }}
                    }}

                    // Cooperative load Bs[16×64]: 256 threads load 1024 elements (4 per thread)
                    for (var i: u32 = 0u; i < 4u; i++) {{
                        let b_idx = thread_id * 4u + i;
                        let b_krow = b_idx / TILE;
                        let b_col = b_idx % TILE;
                        let g_krow = k_start + b_krow;
                        let g_col = tile_col + b_col;
                        if g_krow < K && g_col < N {{
                            Bs[b_krow * TILE_PAD + b_col] = load_b(g_krow * N + g_col);
                        }} else {{
                            Bs[b_krow * TILE_PAD + b_col] = 0.0;
                        }}
                    }}

                    workgroupBarrier();

                    // Compute 4x4 block
                    for (var k: u32 = 0u; k < TILE_K; k++) {{
                        let a0 = As[(lr * BLK) * TILE_K_PAD + k];
                        let a1 = As[(lr * BLK + 1u) * TILE_K_PAD + k];
                        let a2 = As[(lr * BLK + 2u) * TILE_K_PAD + k];
                        let a3 = As[(lr * BLK + 3u) * TILE_K_PAD + k];
                        let b0 = Bs[k * TILE_PAD + lc * BLK];
                        let b1 = Bs[k * TILE_PAD + lc * BLK + 1u];
                        let b2 = Bs[k * TILE_PAD + lc * BLK + 2u];
                        let b3 = Bs[k * TILE_PAD + lc * BLK + 3u];

                        acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
                        acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
                        acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
                        acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
                    }}

                    workgroupBarrier();
                }}

                // Write 4x4 block
                if c_row < M && c_col < N {{ c[c_row * N + c_col] = acc00; }}
                if c_row < M && c_col + 1u < N {{ c[c_row * N + c_col + 1u] = acc01; }}
                if c_row < M && c_col + 2u < N {{ c[c_row * N + c_col + 2u] = acc02; }}
                if c_row < M && c_col + 3u < N {{ c[c_row * N + c_col + 3u] = acc03; }}

                if c_row + 1u < M && c_col < N {{ c[(c_row + 1u) * N + c_col] = acc10; }}
                if c_row + 1u < M && c_col + 1u < N {{ c[(c_row + 1u) * N + c_col + 1u] = acc11; }}
                if c_row + 1u < M && c_col + 2u < N {{ c[(c_row + 1u) * N + c_col + 2u] = acc12; }}
                if c_row + 1u < M && c_col + 3u < N {{ c[(c_row + 1u) * N + c_col + 3u] = acc13; }}

                if c_row + 2u < M && c_col < N {{ c[(c_row + 2u) * N + c_col] = acc20; }}
                if c_row + 2u < M && c_col + 1u < N {{ c[(c_row + 2u) * N + c_col + 1u] = acc21; }}
                if c_row + 2u < M && c_col + 2u < N {{ c[(c_row + 2u) * N + c_col + 2u] = acc22; }}
                if c_row + 2u < M && c_col + 3u < N {{ c[(c_row + 2u) * N + c_col + 3u] = acc23; }}

                if c_row + 3u < M && c_col < N {{ c[(c_row + 3u) * N + c_col] = acc30; }}
                if c_row + 3u < M && c_col + 1u < N {{ c[(c_row + 3u) * N + c_col + 1u] = acc31; }}
                if c_row + 3u < M && c_col + 2u < N {{ c[(c_row + 3u) * N + c_col + 2u] = acc32; }}
                if c_row + 3u < M && c_col + 3u < N {{ c[(c_row + 3u) * N + c_col + 3u] = acc33; }}
            }}
        "
    )
}

fn create_pipeline<T: FloatElement>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_pipeline"),
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
    fn test_gemm_square() {
        let ctx = GpuContext::default();

        // 2x2 matrix multiplication
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let b = ctx
            .create_buffer_from_slice(&[5.0f32, 6.0, 7.0, 8.0])
            .unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        gemm(&ctx, &a, &b, &c, 2, 2, 2).unwrap();

        let result = ctx.read_buffer(&c).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_rectangular() {
        let ctx = GpuContext::default();

        // A: 2x3, B: 3x2, C: 2x2
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let b = ctx
            .create_buffer_from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        gemm(&ctx, &a, &b, &c, 2, 3, 2).unwrap();

        let result = ctx.read_buffer(&c).unwrap();
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_identity() {
        let ctx = GpuContext::default();

        // Multiply by identity matrix
        let a = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let identity = ctx
            .create_buffer_from_slice(&[1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
        let c = ctx.create_buffer::<f32>(4).unwrap();

        gemm(&ctx, &a, &identity, &c, 2, 2, 2).unwrap();

        let result = ctx.read_buffer(&c).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gemm_large() {
        let ctx = GpuContext::default();

        let m = 64;
        let k = 32;
        let n = 48;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i + 1) % 10) as f32).collect();

        let a = ctx.create_buffer_from_slice(&a_data).unwrap();
        let b = ctx.create_buffer_from_slice(&b_data).unwrap();
        let c = ctx.create_buffer::<f32>(m * n).unwrap();

        gemm(&ctx, &a, &b, &c, m, k, n).unwrap();

        let result = ctx.read_buffer(&c).unwrap();

        // Verify first element manually
        let mut expected_c00: f32 = 0.0;
        for i in 0..k {
            expected_c00 += a_data[i] * b_data[i * n];
        }
        assert!((result[0] - expected_c00).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_empty() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(0).unwrap();
        let b = ctx.create_buffer::<f32>(0).unwrap();
        let c = ctx.create_buffer::<f32>(0).unwrap();

        gemm(&ctx, &a, &b, &c, 0, 0, 0).unwrap();
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let ctx = GpuContext::default();

        let a = ctx.create_buffer::<f32>(6).unwrap(); // 2x3
        let b = ctx.create_buffer::<f32>(6).unwrap(); // 3x2
        let c = ctx.create_buffer::<f32>(4).unwrap(); // 2x2

        // Wrong A dimensions
        let err = gemm(&ctx, &a, &b, &c, 3, 2, 2).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));

        // Wrong B dimensions
        let err = gemm(&ctx, &a, &b, &c, 2, 3, 3).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));

        // Wrong C dimensions
        let c_wrong = ctx.create_buffer::<f32>(6).unwrap();
        let err = gemm(&ctx, &a, &b, &c_wrong, 2, 3, 2).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }
}
