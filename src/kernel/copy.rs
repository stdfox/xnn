//! Copy kernel.

use crate::Context;

/// Pipeline label for debugging.
const LABEL: &str = "copy";

/// Copies buffer contents from source to destination.
///
/// # Panics
///
/// - Source buffer size mismatch
/// - Destination buffer size mismatch
pub(crate) fn execute(ctx: &Context, src: &wgpu::Buffer, dst: &wgpu::Buffer, size_bytes: u64) {
    if size_bytes == 0 {
        return;
    }

    assert!(src.size() >= size_bytes, "source buffer size mismatch");
    assert!(dst.size() >= size_bytes, "destination buffer size mismatch");

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(LABEL) });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, size_bytes);

    ctx.queue().submit(Some(encoder.finish()));
}
