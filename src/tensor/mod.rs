//! N-dimensional tensor with GPU-backed storage.

mod layout;
mod ops;

use core::any::TypeId;

use crate::element::{FloatElement, IntegerElement, LogicalElement, NumericElement, SignedElement};
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

        let layout = Layout::from_dimensions(shape)?;
        let volume = layout.size();

        let buffer = match value.len() {
            1 => {
                let buffer = ctx.create_buffer(volume)?;
                let uniform = ctx.create_uniform_buffer(&value[0].to_native());
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

    /// Returns the tensor dimensions.
    #[must_use]
    pub fn dimensions(&self) -> &[usize] {
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

    /// Computes broadcast parameters for two tensors.
    #[allow(clippy::type_complexity)]
    fn broadcast_with(&self, other: &Self) -> Result<(Layout, Box<[usize]>, Box<[usize]>), Error> {
        let out_dimensions = self
            .layout
            .broadcast_dimensions(&other.layout)
            .ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "dimensions {:?} and {:?} are not broadcast-compatible",
                    self.dimensions(),
                    other.dimensions()
                ))
            })?;

        let a_strides = self.layout.broadcast_strides(&out_dimensions);
        let b_strides = other.layout.broadcast_strides(&out_dimensions);
        let layout = Layout::from_dimensions(&out_dimensions)?;

        Ok((layout, a_strides, b_strides))
    }

    /// Applies a binary operation with broadcasting and returns a new tensor.
    #[allow(clippy::type_complexity)]
    fn binary_op(
        &self,
        other: &Self,
        op: fn(
            &Context,
            &Buffer<T>,
            &Buffer<T>,
            &Buffer<T>,
            &[usize],
            &[usize],
            &[usize],
        ) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let (layout, a_strides, b_strides) = self.broadcast_with(other)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        op(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            &buffer,
            layout.dimensions(),
            &a_strides,
            &b_strides,
        )?;

        Ok(Self {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: NumericElement> Tensor<T> {
    /// Element-wise addition with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::add)
    }

    /// Element-wise subtraction with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn sub(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::sub)
    }

    /// Element-wise multiplication with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn mul(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::mul)
    }

    /// Element-wise division with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn div(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::div)
    }

    /// Element-wise less-than comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::lt)
    }

    /// Element-wise greater-than comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::gt)
    }

    /// Element-wise less-than-or-equal comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::le)
    }

    /// Element-wise greater-than-or-equal comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::ge)
    }

    /// Element-wise equality comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::eq)
    }

    /// Element-wise inequality comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.comparison_op(other, ops::ne)
    }

    /// Computes absolute value element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn abs(&self) -> Result<Self, Error> {
        if TypeId::of::<T>() == TypeId::of::<u32>() {
            return self.copy();
        }

        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        ops::abs(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }

    /// Computes sign element-wise.
    ///
    /// Returns -1, 0, or 1 for signed types; 0 or 1 for unsigned types.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sign(&self) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        ops::sign(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }

    /// Helper for comparison operations.
    fn comparison_op(
        &self,
        other: &Self,
        op: fn(
            &Context,
            &Buffer<T>,
            &Buffer<T>,
            &Buffer<bool>,
            &[usize],
            &[usize],
            &[usize],
        ) -> Result<(), Error>,
    ) -> Result<Tensor<bool>, Error> {
        let (layout, a_strides, b_strides) = self.broadcast_with(other)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        op(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            &buffer,
            layout.dimensions(),
            &a_strides,
            &b_strides,
        )?;

        Ok(Tensor {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: SignedElement> Tensor<T> {
    /// Computes negation element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn neg(&self) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        ops::neg(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: IntegerElement> Tensor<T> {
    /// Element-wise remainder with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn rem(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::rem)
    }
}

impl<T: FloatElement> Tensor<T> {
    /// Element-wise power with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn pow(&self, other: &Self) -> Result<Self, Error> {
        self.binary_op(other, ops::pow)
    }

    /// Computes arc cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn acos(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::acos)
    }

    /// Computes inverse hyperbolic cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn acosh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::acosh)
    }

    /// Computes arc sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn asin(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::asin)
    }

    /// Computes inverse hyperbolic sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn asinh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::asinh)
    }

    /// Computes arc tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn atan(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::atan)
    }

    /// Computes inverse hyperbolic tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn atanh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::atanh)
    }

    /// Computes ceiling element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn ceil(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::ceil)
    }

    /// Computes cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn cos(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::cos)
    }

    /// Computes hyperbolic cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn cosh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::cosh)
    }

    /// Computes exponential (e^x) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn exp(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::exp)
    }

    /// Computes floor element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn floor(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::floor)
    }

    /// Computes natural logarithm element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn log(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::log)
    }

    /// Computes reciprocal (1/x) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn rcp(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::rcp)
    }

    /// Rounds to nearest integer element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn round(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::round)
    }

    /// Computes sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sin(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::sin)
    }

    /// Computes hyperbolic sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sinh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::sinh)
    }

    /// Computes tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn tan(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::tan)
    }

    /// Computes hyperbolic tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn tanh(&self) -> Result<Self, Error> {
        self.unary_float_op(ops::tanh)
    }

    /// Applies a unary float operation and returns a new tensor.
    #[allow(clippy::type_complexity)]
    fn unary_float_op(
        &self,
        op: fn(&Context, &Buffer<T>, &Buffer<T>) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        op(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: LogicalElement> Tensor<T> {
    /// Computes logical NOT element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn not(&self) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        ops::not(&self.ctx, &self.buffer, &buffer)?;

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }
}
