//! Matrix operations.

use core::any::TypeId;

use wgpu::util::DeviceExt as _;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

/// Block size for register tiling (each thread computes BM×BN elements).
const BLOCK_SIZE: u32 = 4;

/// Workgroup dimensions.
const WG_SIZE: u32 = 16;

/// Output tile size (`WG_SIZE * BLOCK_SIZE`).
const TILE_SIZE: u32 = WG_SIZE * BLOCK_SIZE;

/// K-dimension tile size.
const TILE_K: u32 = 16;

/// Padded tile sizes to avoid shared memory bank conflicts.
const TILE_SIZE_PAD: u32 = TILE_SIZE + 1;
const TILE_K_PAD: u32 = TILE_K + 1;

/// Maximum batch dimensions supported.
const MAX_BATCH_RANK: usize = 6;

/// Maximum workgroups per dimension.
const MAX_WORKGROUPS: u32 = 65535;

/// Pipeline label for debugging.
const LABEL: &str = "matmul";

/// Matmul parameters passed to shader as uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
    batch_rank: u32,
    transpose_a: u32,
    transpose_b: u32,
    _pad: u32,
    batch_dims: [[u32; 4]; 2],
    a_batch_strides: [[u32; 4]; 2],
    b_batch_strides: [[u32; 4]; 2],
    a_matrix_stride: u32,
    b_matrix_stride: u32,
    c_matrix_stride: u32,
    _pad2: u32,
}

/// Marker type for pipeline caching.
struct MatmulMarker<T>(core::marker::PhantomData<T>);

/// Batched matrix multiplication with optional transposes.
///
/// `C = A × B`
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_dims: &[usize],
    b_dims: &[usize],
    c_dims: &[usize],
    transpose_a: bool,
    transpose_b: bool,
) -> Result<(), Error> {
    let rank = a_dims.len();
    let batch_rank = rank.saturating_sub(2);

    let (a_rows, a_cols) = match rank {
        0 => (1, 1),
        1 => (1, a_dims[0]),
        _ => (a_dims[rank - 2], a_dims[rank - 1]),
    };

    let (b_rows, b_cols) = match rank {
        0 => (1, 1),
        1 => (1, b_dims[0]),
        _ => (b_dims[rank - 2], b_dims[rank - 1]),
    };

    let (m, k) = if transpose_a {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };
    let n = if transpose_b { b_rows } else { b_cols };

    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let batch_size: usize = c_dims[..batch_rank].iter().product::<usize>().max(1);

    let (a_batch_strides, b_batch_strides) = if batch_rank > 0 {
        compute_batch_strides(
            &a_dims[..batch_rank],
            &b_dims[..batch_rank],
            &c_dims[..batch_rank],
        )
    } else {
        (vec![0; MAX_BATCH_RANK], vec![0; MAX_BATCH_RANK])
    };

    #[allow(clippy::cast_possible_truncation)]
    let params = {
        let mut batch_dims_arr = [[0u32; 4]; 2];
        let mut a_strides_arr = [[0u32; 4]; 2];
        let mut b_strides_arr = [[0u32; 4]; 2];

        for (i, &dim) in c_dims[..batch_rank].iter().enumerate() {
            batch_dims_arr[i / 4][i % 4] = dim as u32;
        }

        for (i, &s) in a_batch_strides.iter().take(batch_rank).enumerate() {
            a_strides_arr[i / 4][i % 4] = s as u32;
        }

        for (i, &s) in b_batch_strides.iter().take(batch_rank).enumerate() {
            b_strides_arr[i / 4][i % 4] = s as u32;
        }

        MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
            batch_rank: batch_rank as u32,
            transpose_a: u32::from(transpose_a),
            transpose_b: u32::from(transpose_b),
            _pad: 0,
            batch_dims: batch_dims_arr,
            a_batch_strides: a_strides_arr,
            b_batch_strides: b_strides_arr,
            a_matrix_stride: (a_rows * a_cols) as u32,
            b_matrix_stride: (b_rows * b_cols) as u32,
            c_matrix_stride: (m * n) as u32,
            _pad2: 0,
        }
    };

    let pipeline =
        ctx.get_or_create_pipeline(TypeId::of::<MatmulMarker<T>>(), create_shader::<T>, LABEL)?;

    dispatch(ctx, a, b, c, &params, &pipeline)
}

fn compute_batch_strides(
    a_batch: &[usize],
    b_batch: &[usize],
    out_batch: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let batch_rank = out_batch.len();
    let a_strides = compute_contiguous_strides(a_batch);
    let b_strides = compute_contiguous_strides(b_batch);

    let mut a_broadcast = vec![0; batch_rank];
    let mut b_broadcast = vec![0; batch_rank];

    let a_offset = batch_rank.saturating_sub(a_batch.len());
    let b_offset = batch_rank.saturating_sub(b_batch.len());

    for (i, &out_dim) in out_batch.iter().enumerate() {
        if i >= a_offset && a_batch[i - a_offset] == out_dim {
            a_broadcast[i] = a_strides[i - a_offset];
        }
        if i >= b_offset && b_batch[i - b_offset] == out_dim {
            b_broadcast[i] = b_strides[i - b_offset];
        }
    }

    (a_broadcast, b_broadcast)
}

fn compute_contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn create_shader<T: FloatElement>() -> String {
    let ty = T::wgsl_type();
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
            const MAX_BATCH: u32 = {MAX_BATCH_RANK}u;

            struct Params {{
                m: u32,
                k: u32,
                n: u32,
                batch_size: u32,
                batch_rank: u32,
                transpose_a: u32,
                transpose_b: u32,
                _pad: u32,
                batch_dims: array<vec4<u32>, 2>,
                a_batch_strides: array<vec4<u32>, 2>,
                b_batch_strides: array<vec4<u32>, 2>,
                a_matrix_stride: u32,
                b_matrix_stride: u32,
                c_matrix_stride: u32,
                _pad2: u32,
            }}

            @group(0) @binding(0) var<storage, read> a: array<{ty}>;
            @group(0) @binding(1) var<storage, read> b: array<{ty}>;
            @group(0) @binding(2) var<storage, read_write> c: array<{ty}>;
            @group(0) @binding(3) var<uniform> params: Params;

            var<workgroup> As: array<{ty}, {as_size}>;
            var<workgroup> Bs: array<{ty}, {bs_size}>;

            fn get_batch_dim(idx: u32) -> u32 {{
                return params.batch_dims[idx / 4u][idx % 4u];
            }}

            fn get_a_batch_stride(idx: u32) -> u32 {{
                return params.a_batch_strides[idx / 4u][idx % 4u];
            }}

            fn get_b_batch_stride(idx: u32) -> u32 {{
                return params.b_batch_strides[idx / 4u][idx % 4u];
            }}

            fn compute_batch_offset(batch_idx: u32, is_a: bool) -> u32 {{
                var offset = 0u;
                var remaining = batch_idx;

                for (var i = 0u; i < params.batch_rank; i++) {{
                    var prod = 1u;
                    for (var j = i + 1u; j < params.batch_rank; j++) {{
                        prod *= get_batch_dim(j);
                    }}
                    let coord = remaining / prod;
                    remaining = remaining % prod;

                    if is_a {{
                        offset += coord * get_a_batch_stride(i);
                    }} else {{
                        offset += coord * get_b_batch_stride(i);
                    }}
                }}

                return offset;
            }}

            @compute @workgroup_size({WG_SIZE}, {WG_SIZE})
            fn main(
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>
            ) {{
                let M = params.m;
                let K = params.k;
                let N = params.n;

                let batch_idx = wid.z;
                if batch_idx >= params.batch_size {{
                    return;
                }}

                let a_batch_offset = compute_batch_offset(batch_idx, true) * params.a_matrix_stride;
                let b_batch_offset = compute_batch_offset(batch_idx, false) * params.b_matrix_stride;
                let c_batch_offset = batch_idx * params.c_matrix_stride;

                let lr = lid.x;
                let lc = lid.y;
                let thread_id = lr * WG + lc;

                let tile_row = wid.x * TILE;
                let tile_col = wid.y * TILE;
                let c_row = tile_row + lr * BLK;
                let c_col = tile_col + lc * BLK;

                var acc00: {ty} = 0.0; var acc01: {ty} = 0.0; var acc02: {ty} = 0.0; var acc03: {ty} = 0.0;
                var acc10: {ty} = 0.0; var acc11: {ty} = 0.0; var acc12: {ty} = 0.0; var acc13: {ty} = 0.0;
                var acc20: {ty} = 0.0; var acc21: {ty} = 0.0; var acc22: {ty} = 0.0; var acc23: {ty} = 0.0;
                var acc30: {ty} = 0.0; var acc31: {ty} = 0.0; var acc32: {ty} = 0.0; var acc33: {ty} = 0.0;

                let num_k_tiles = (K + TILE_K - 1u) / TILE_K;

                let a_rows = select(M, K, params.transpose_a != 0u);
                let a_cols = select(K, M, params.transpose_a != 0u);
                let b_rows = select(K, N, params.transpose_b != 0u);
                let b_cols = select(N, K, params.transpose_b != 0u);

                for (var kt: u32 = 0u; kt < num_k_tiles; kt++) {{
                    let k_start = kt * TILE_K;

                    for (var i: u32 = 0u; i < 4u; i++) {{
                        let a_idx = thread_id * 4u + i;
                        let local_row = a_idx / TILE_K;
                        let local_kcol = a_idx % TILE_K;

                        let g_row = tile_row + local_row;
                        let g_kcol = k_start + local_kcol;

                        var val: {ty} = 0.0;
                        if g_row < M && g_kcol < K {{
                            let orig_row = select(g_row, g_kcol, params.transpose_a != 0u);
                            let orig_col = select(g_kcol, g_row, params.transpose_a != 0u);
                            val = a[a_batch_offset + orig_row * a_cols + orig_col];
                        }}
                        As[local_row * TILE_K_PAD + local_kcol] = val;
                    }}

                    for (var i: u32 = 0u; i < 4u; i++) {{
                        let b_idx = thread_id * 4u + i;
                        let local_krow = b_idx / TILE;
                        let local_col = b_idx % TILE;

                        let g_krow = k_start + local_krow;
                        let g_col = tile_col + local_col;

                        var val: {ty} = 0.0;
                        if g_krow < K && g_col < N {{
                            let orig_row = select(g_krow, g_col, params.transpose_b != 0u);
                            let orig_col = select(g_col, g_krow, params.transpose_b != 0u);
                            val = b[b_batch_offset + orig_row * b_cols + orig_col];
                        }}
                        Bs[local_krow * TILE_PAD + local_col] = val;
                    }}

                    workgroupBarrier();

                    for (var kk: u32 = 0u; kk < TILE_K; kk++) {{
                        let a0 = As[(lr * BLK) * TILE_K_PAD + kk];
                        let a1 = As[(lr * BLK + 1u) * TILE_K_PAD + kk];
                        let a2 = As[(lr * BLK + 2u) * TILE_K_PAD + kk];
                        let a3 = As[(lr * BLK + 3u) * TILE_K_PAD + kk];
                        let b0 = Bs[kk * TILE_PAD + lc * BLK];
                        let b1 = Bs[kk * TILE_PAD + lc * BLK + 1u];
                        let b2 = Bs[kk * TILE_PAD + lc * BLK + 2u];
                        let b3 = Bs[kk * TILE_PAD + lc * BLK + 3u];

                        acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
                        acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
                        acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
                        acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
                    }}

                    workgroupBarrier();
                }}

                if c_row < M && c_col < N {{ c[c_batch_offset + c_row * N + c_col] = acc00; }}
                if c_row < M && c_col + 1u < N {{ c[c_batch_offset + c_row * N + c_col + 1u] = acc01; }}
                if c_row < M && c_col + 2u < N {{ c[c_batch_offset + c_row * N + c_col + 2u] = acc02; }}
                if c_row < M && c_col + 3u < N {{ c[c_batch_offset + c_row * N + c_col + 3u] = acc03; }}

                if c_row + 1u < M && c_col < N {{ c[c_batch_offset + (c_row + 1u) * N + c_col] = acc10; }}
                if c_row + 1u < M && c_col + 1u < N {{ c[c_batch_offset + (c_row + 1u) * N + c_col + 1u] = acc11; }}
                if c_row + 1u < M && c_col + 2u < N {{ c[c_batch_offset + (c_row + 1u) * N + c_col + 2u] = acc12; }}
                if c_row + 1u < M && c_col + 3u < N {{ c[c_batch_offset + (c_row + 1u) * N + c_col + 3u] = acc13; }}

                if c_row + 2u < M && c_col < N {{ c[c_batch_offset + (c_row + 2u) * N + c_col] = acc20; }}
                if c_row + 2u < M && c_col + 1u < N {{ c[c_batch_offset + (c_row + 2u) * N + c_col + 1u] = acc21; }}
                if c_row + 2u < M && c_col + 2u < N {{ c[c_batch_offset + (c_row + 2u) * N + c_col + 2u] = acc22; }}
                if c_row + 2u < M && c_col + 3u < N {{ c[c_batch_offset + (c_row + 2u) * N + c_col + 3u] = acc23; }}

                if c_row + 3u < M && c_col < N {{ c[c_batch_offset + (c_row + 3u) * N + c_col] = acc30; }}
                if c_row + 3u < M && c_col + 1u < N {{ c[c_batch_offset + (c_row + 3u) * N + c_col + 1u] = acc31; }}
                if c_row + 3u < M && c_col + 2u < N {{ c[c_batch_offset + (c_row + 3u) * N + c_col + 2u] = acc32; }}
                if c_row + 3u < M && c_col + 3u < N {{ c[c_batch_offset + (c_row + 3u) * N + c_col + 3u] = acc33; }}
            }}
        "
    )
}

fn dispatch<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    params: &MatmulParams,
    pipeline: &wgpu::ComputePipeline,
) -> Result<(), Error> {
    let m_tiles = params.m.div_ceil(TILE_SIZE);
    let n_tiles = params.n.div_ceil(TILE_SIZE);
    let batch_size = params.batch_size;

    if batch_size == 0 {
        return Ok(());
    }

    if m_tiles > MAX_WORKGROUPS || n_tiles > MAX_WORKGROUPS {
        return Err(Error::Device(format!(
            "matrix dimensions too large: {}x{} tiles exceed limit {}",
            m_tiles, n_tiles, MAX_WORKGROUPS
        )));
    }

    let params_buffer = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(LABEL),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let max_batch_per_dispatch = MAX_WORKGROUPS;
    let num_dispatches = batch_size.div_ceil(max_batch_per_dispatch);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(LABEL) });

    for dispatch_idx in 0..num_dispatches {
        let batch_start = dispatch_idx * max_batch_per_dispatch;
        let batch_count = (batch_size - batch_start).min(max_batch_per_dispatch);

        let mut dispatch_params = *params;
        dispatch_params.batch_size = batch_count;

        // TODO: Add batch_offset to params for multi-dispatch support
        let dispatch_params_buffer = if num_dispatches > 1 {
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(LABEL),
                    contents: bytemuck::bytes_of(&dispatch_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        } else {
            params_buffer.clone()
        };

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(LABEL),
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
                    resource: dispatch_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(LABEL),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(m_tiles, n_tiles, batch_count);
    }

    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
