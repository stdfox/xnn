//! Sum reduction kernel.
//!
//! Reduces a buffer to a single sum value using parallel reduction.

use crate::kernel::debug_assert_len;
use crate::{Buffer, Context, Element};

/// Workgroup size for the sum kernel.
const WORKGROUP_SIZE: u32 = 256;

/// Computes the sum of all elements in a buffer.
///
/// Reduces `input` to a single value stored in `output[0]`.
///
/// # Panics
///
/// - Input buffer length exceeds `u32::MAX`.
/// - (debug) Output buffer length is not 1.
pub fn sum<T: Element>(ctx: &Context, input: &Buffer<T>, output: &Buffer<T>) {
    debug_assert_len(output, 1, "output");

    if input.is_empty() {
        // Write zero to output for empty input
        let zero_buf = ctx
            .create_buffer_from_slice(&[T::zeroed()])
            .expect("failed to create zero buffer");
        copy_buffer(ctx, &zero_buf, output);
        return;
    }

    let len = u32::try_from(input.len()).expect("input length exceeds u32::MAX");
    let vec4_count = len.div_ceil(4);

    // Calculate reduction chain: how many passes and buffer sizes needed
    let mut sizes = Vec::new();
    let mut current_count = vec4_count;
    while current_count > WORKGROUP_SIZE {
        let num_workgroups = current_count.div_ceil(WORKGROUP_SIZE);
        sizes.push(num_workgroups);
        current_count = num_workgroups.div_ceil(4);
    }

    // Create intermediate buffers
    let mut intermediates: Vec<Buffer<T>> = Vec::with_capacity(sizes.len());
    for &num_workgroups in &sizes {
        intermediates.push(
            ctx.create_buffer::<T>(num_workgroups as usize)
                .expect("failed to create intermediate buffer"),
        );
    }

    let pipeline = ctx.get_or_create_kernel_pipeline::<T, _>(create_pipeline::<T>);

    // Batched encoder for all passes
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Chain reduction passes
    let mut current_input = input;
    for (i, &num_workgroups) in sizes.iter().enumerate() {
        let current_output = &intermediates[i];
        encode_reduce_pass(
            ctx,
            &pipeline,
            &mut encoder,
            current_input,
            current_output,
            num_workgroups,
        );
        current_input = current_output;
    }

    // Final pass: reduce to output
    let final_count = if intermediates.is_empty() {
        vec4_count
    } else {
        let len = u32::try_from(current_input.len()).expect("length exceeds u32::MAX");
        len.div_ceil(4)
    };
    let final_workgroups = final_count.div_ceil(WORKGROUP_SIZE).max(1);
    encode_reduce_pass(
        ctx,
        &pipeline,
        &mut encoder,
        current_input,
        output,
        final_workgroups,
    );

    ctx.queue().submit(Some(encoder.finish()));
}

/// Encodes a single reduction pass into the command encoder.
fn encode_reduce_pass<T: Element>(
    ctx: &Context,
    pipeline: &wgpu::ComputePipeline,
    encoder: &mut wgpu::CommandEncoder,
    input: &Buffer<T>,
    output: &Buffer<T>,
    num_workgroups: u32,
) {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.inner().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.inner().as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(num_workgroups, 1, 1);
}

/// Copies one element from src to dst.
fn copy_buffer<T: Element>(ctx: &Context, src: &Buffer<T>, dst: &Buffer<T>) {
    let size = core::mem::size_of::<T>() as u64;
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(src.inner(), 0, dst.inner(), 0, size);
    ctx.queue().submit(Some(encoder.finish()));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
}

fn create_shader_source<T: Element>() -> String {
    let wgsl_type = T::wgsl_type();
    let vec4_type = format!("vec4<{wgsl_type}>");

    format!(
        r"
            @group(0) @binding(0) var<storage, read> input: array<{vec4_type}>;
            @group(0) @binding(1) var<storage, read_write> output: array<{wgsl_type}>;

            var<workgroup> sdata: array<{wgsl_type}, {WORKGROUP_SIZE}>;

            @compute @workgroup_size({WORKGROUP_SIZE})
            fn main(
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>
            ) {{
                let tid = lid.x;
                let gid = wid.x * {WORKGROUP_SIZE}u + tid;
                let len = arrayLength(&input);

                // Each thread loads one vec4 and sums its components
                var acc: {wgsl_type} = {wgsl_type}(0);
                if gid < len {{
                    let v = input[gid];
                    acc = v.x + v.y + v.z + v.w;
                }}

                sdata[tid] = acc;
                workgroupBarrier();

                // Tree reduction with sequential addressing (unrolled)
                if tid < 128u {{ sdata[tid] += sdata[tid + 128u]; }}
                workgroupBarrier();
                if tid < 64u {{ sdata[tid] += sdata[tid + 64u]; }}
                workgroupBarrier();
                if tid < 32u {{ sdata[tid] += sdata[tid + 32u]; }}
                workgroupBarrier();
                if tid < 16u {{ sdata[tid] += sdata[tid + 16u]; }}
                workgroupBarrier();
                if tid < 8u {{ sdata[tid] += sdata[tid + 8u]; }}
                workgroupBarrier();
                if tid < 4u {{ sdata[tid] += sdata[tid + 4u]; }}
                workgroupBarrier();
                if tid < 2u {{ sdata[tid] += sdata[tid + 2u]; }}
                workgroupBarrier();
                if tid < 1u {{ sdata[tid] += sdata[tid + 1u]; }}

                // Thread 0 writes the result
                if tid == 0u {{
                    output[wid.x] = sdata[0];
                }}
            }}
        "
    )
}

fn create_pipeline<T: Element>(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader_source = create_shader_source::<T>();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sum_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sum_pipeline"),
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

    use super::*;

    #[test]
    fn test_sum_single() {
        let ctx = Context::try_default().unwrap();

        let input = ctx.create_buffer_from_slice(&[42.0f32]).unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_relative_eq!(ctx.read_buffer(&output).unwrap()[0], 42.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sum_small() {
        let ctx = Context::try_default().unwrap();

        // f32
        let input = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_relative_eq!(ctx.read_buffer(&output).unwrap()[0], 10.0, epsilon = 1e-5);

        // i32
        let input = ctx.create_buffer_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
        let output = ctx.create_buffer::<i32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_eq!(ctx.read_buffer(&output).unwrap(), vec![15]);

        // u32
        let input = ctx.create_buffer_from_slice(&[10u32, 20, 30, 40]).unwrap();
        let output = ctx.create_buffer::<u32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_eq!(ctx.read_buffer(&output).unwrap(), vec![100]);
    }

    #[test]
    fn test_sum_large() {
        let ctx = Context::try_default().unwrap();

        let len = 4096 * 4096;
        let data: Vec<f32> = vec![1.0; len];
        let expected = len as f32;

        let input = ctx.create_buffer_from_slice(&data).unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();

        sum(&ctx, &input, &output);

        let result = ctx.read_buffer(&output).unwrap()[0];
        assert_relative_eq!(result, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_sum_empty() {
        let ctx = Context::try_default().unwrap();

        let input = ctx.create_buffer::<f32>(0).unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_relative_eq!(ctx.read_buffer(&output).unwrap()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sum_non_aligned() {
        let ctx = Context::try_default().unwrap();

        let input = ctx
            .create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();
        sum(&ctx, &input, &output);
        assert_relative_eq!(ctx.read_buffer(&output).unwrap()[0], 28.0, epsilon = 1e-5);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "length mismatch"))]
    fn test_sum_assert_len() {
        let ctx = Context::try_default().unwrap();

        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0]).unwrap();
        let output = ctx.create_buffer::<f32>(2).unwrap();

        sum(&ctx, &input, &output);
    }
}
