//! Copy operation for tensor buffers.

use crate::{Buffer, Context, Element, Error};

/// Copies buffer contents from source to destination.
///
/// # Errors
///
/// Returns [`Error::Device`] if buffer lengths don't match or size overflows.
pub(crate) fn copy<T: Element>(
    ctx: &Context,
    src: &Buffer<T>,
    dst: &Buffer<T>,
) -> Result<(), Error> {
    if src.len() != dst.len() {
        return Err(Error::Device("buffer lengths mismatch".into()));
    }

    if src.is_empty() {
        return Ok(());
    }

    let size = src
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .and_then(|s| u64::try_from(s).ok())
        .ok_or_else(|| Error::Device("buffer size overflow".into()))?;

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy"),
        });
    encoder.copy_buffer_to_buffer(src.inner(), 0, dst.inner(), 0, size);
    ctx.queue().submit(Some(encoder.finish()));

    Ok(())
}
