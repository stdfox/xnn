//! N-dimensional tensor with GPU-backed storage.

mod layout;
mod ops;

use crate::error::{Error, TensorError};
use crate::{Buffer, Context, Element};
use layout::Layout;

/// N-dimensional tensor with GPU-backed storage.
pub struct Tensor<T: Element> {
    /// GPU buffer storing tensor elements.
    buffer: Buffer<T>,
    /// Shape and stride information.
    layout: Layout,
    /// GPU context for operations.
    ctx: Context,
}

impl<T: Element> Tensor<T> {
    /// Creates a tensor with constant values.
    ///
    /// If `value` has length 1, that single value is broadcast to fill the entire tensor.
    /// Otherwise, `value` length must equal the shape volume.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if `value` is empty, any dimension is zero,
    ///   or value length is neither 1 nor equal to shape volume.
    /// - [`Error::Device`] if operation fails.
    pub fn constant(ctx: &Context, shape: &[usize], value: &[T]) -> Result<Self, Error> {
        if value.is_empty() {
            return Err(TensorError::InvalidShape("value must not be empty".into()).into());
        }

        let layout = Layout::from_shape(shape)?;
        let volume = layout.size();

        let buffer = match value.len() {
            1 => {
                let buffer = ctx.create_buffer(volume)?;
                let uniform = ctx.create_uniform_buffer(value[0]);
                ops::constant(ctx, &buffer, &uniform)?;
                buffer
            }
            n if n == volume => ctx.create_buffer_from_slice(value)?,
            n => {
                return Err(TensorError::InvalidShape(format!(
                    "value length {n} must be 1 or equal to shape volume {volume}"
                ))
                .into());
            }
        };

        Ok(Self {
            buffer,
            layout,
            ctx: ctx.clone(),
        })
    }

    /// Creates a tensor from shape and data slice.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if any dimension is zero or size doesn't match data length.
    /// - [`Error::Device`] if operation fails.
    pub fn from_shape_slice(ctx: &Context, shape: &[usize], data: &[T]) -> Result<Self, Error> {
        Self::constant(ctx, shape, data)
    }

    /// Creates a 1D tensor from a data slice.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if data is empty.
    /// - [`Error::Device`] if operation fails.
    pub fn from_slice(ctx: &Context, data: &[T]) -> Result<Self, Error> {
        Self::constant(ctx, &[data.len()], data)
    }

    /// Creates a copy of this tensor.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn copy(&self) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        ops::copy(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.layout.dimensions()
    }

    /// Copies tensor data from GPU to CPU.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        self.ctx.read_buffer(&self.buffer)
    }
}
