//! N-dimensional tensor with GPU-backed storage.

mod layout;

use alloc::vec::Vec;
use alloc::{format, vec};

use crate::element::{FloatElement, IntegerElement, LogicalElement, NumericElement, SignedElement};
use crate::error::{Error, TensorError};
use crate::kernel::ops;
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
                ops::constant(ctx, &buffer, &uniform);
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
        ops::copy(&self.ctx, &self.buffer, &buffer);

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

    /// Asynchronously copies tensor data from GPU to CPU.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub async fn to_vec_async(&self) -> Result<Vec<T>, Error> {
        self.ctx.read_buffer_async(&self.buffer).await
    }

    /// Copies tensor data from GPU to CPU.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        self.ctx.read_buffer(&self.buffer)
    }

    /// Applies a math binary operation with broadcasting.
    fn math_binary<U: Element>(
        &self,
        other: &Self,
        op: impl FnOnce(&Context, &Buffer<T>, &Buffer<T>, &Buffer<U>, &[usize], &[usize], &[usize]),
    ) -> Result<Tensor<U>, Error> {
        let (dimensions, strides) =
            Layout::broadcast(&[&self.layout, &other.layout]).ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "dimensions {:?} and {:?} are not broadcast-compatible",
                    self.dimensions(),
                    other.dimensions()
                ))
            })?;

        let layout = Layout::from_dimensions(&dimensions)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        op(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            &buffer,
            &strides[0],
            &strides[1],
            layout.strides(),
        );

        Ok(Tensor {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }

    /// Applies a math unary operation and returns a new tensor.
    fn math_unary(&self, op: impl FnOnce(&Context, &Buffer<T>, &Buffer<T>)) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        op(&self.ctx, &self.buffer, &buffer);

        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: NumericElement> Tensor<T> {
    /// Clamps tensor values: `y = max(min(x, b), a)`.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn clamp(&self, a: &Self, b: &Self) -> Result<Self, Error> {
        let (dimensions, strides) = Layout::broadcast(&[&self.layout, &a.layout, &b.layout])
            .ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "dimensions {:?}, {:?}, and {:?} are not broadcast-compatible",
                    self.dimensions(),
                    a.dimensions(),
                    b.dimensions()
                ))
            })?;

        let layout = Layout::from_dimensions(&dimensions)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        ops::clamp(
            &self.ctx,
            &self.buffer,
            &a.buffer,
            &b.buffer,
            &buffer,
            &strides[0],
            &strides[1],
            &strides[2],
            layout.strides(),
        );

        Ok(Self {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }

    /// Element-wise addition with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::add(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise subtraction with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn sub(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::sub(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise multiplication with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn mul(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::mul(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise division with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn div(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::div(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise maximum with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn max(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::max(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise minimum with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn min(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::min(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise equality comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::eq(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise inequality comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::ne(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise greater-than-or-equal comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::ge(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise greater-than comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::gt(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise less-than-or-equal comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::le(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise less-than comparison with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::lt(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Max reduction along specified axes.
    ///
    /// Output shape equals input shape with reduced axes set to 1.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if axes are invalid or duplicate.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn max_reduce(&self, axes: &[usize]) -> Result<Self, Error> {
        self.reduction(axes, ops::max_reduce)
    }

    /// Min reduction along specified axes.
    ///
    /// Output shape equals input shape with reduced axes set to 1.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if axes are invalid or duplicate.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn min_reduce(&self, axes: &[usize]) -> Result<Self, Error> {
        self.reduction(axes, ops::min_reduce)
    }

    /// Sum reduction along specified axes.
    ///
    /// Output shape equals input shape with reduced axes set to 1.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if axes are invalid or duplicate.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn sum_reduce(&self, axes: &[usize], normalize: bool) -> Result<Self, Error> {
        self.reduction(
            axes,
            |ctx, input, output, dims, x_strides, y_strides, axes| {
                ops::sum_reduce(
                    ctx, input, output, dims, x_strides, y_strides, axes, normalize,
                );
            },
        )
    }

    /// Mean reduction along specified axes.
    ///
    /// Output shape equals input shape with reduced axes set to 1.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if axes are invalid or duplicate.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn mean_reduce(&self, axes: &[usize]) -> Result<Self, Error> {
        self.sum_reduce(axes, true)
    }

    /// Applies a reduce operation with strides and returns a new tensor.
    fn reduction<F>(&self, axes: &[usize], op: F) -> Result<Self, Error>
    where
        F: FnOnce(&Context, &Buffer<T>, &Buffer<T>, &[usize], &[usize], &[usize], &[usize]),
    {
        let dimensions = self.layout.dimensions();
        let rank = dimensions.len();

        let mut seen = vec![false; rank];
        for &axis in axes {
            if axis >= rank {
                return Err(TensorError::InvalidShape(format!(
                    "axis {axis} out of bounds for tensor with rank {rank}"
                ))
                .into());
            }
            if seen[axis] {
                return Err(TensorError::InvalidShape(format!("duplicate axis {axis}")).into());
            }
            seen[axis] = true;
        }

        let out_dimensions: Vec<usize> = dimensions
            .iter()
            .enumerate()
            .map(|(i, &d)| if seen[i] { 1 } else { d })
            .collect();

        let layout = Layout::from_dimensions(&out_dimensions)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        op(
            &self.ctx,
            &self.buffer,
            &buffer,
            dimensions,
            self.layout.strides(),
            layout.strides(),
            axes,
        );

        Ok(Self {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: SignedElement> Tensor<T> {
    /// Computes absolute value element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn abs(&self) -> Result<Self, Error> {
        self.math_unary(ops::abs)
    }

    /// Computes negation element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn neg(&self) -> Result<Self, Error> {
        self.math_unary(ops::neg)
    }

    /// Computes sign element-wise.
    ///
    /// Returns -1, 0, or 1.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sign(&self) -> Result<Self, Error> {
        self.math_unary(ops::sign)
    }
}

impl<T: IntegerElement> Tensor<T> {
    /// Element-wise remainder with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn rem(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::rem(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }
}

impl<T: FloatElement> Tensor<T> {
    /// Batched matrix multiplication with optional transposes.
    ///
    /// `A[..., m, k] × B[..., k, n] → C[..., m, n]`
    ///
    /// Batch dimensions are broadcast-compatible.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if ranks differ or are less than 2.
    /// - [`TensorError::InvalidShape`] if inner dimensions don't match.
    /// - [`Error::Device`] if GPU operation fails.
    pub fn matmul(
        &self,
        other: &Self,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Self, Error> {
        let a_dims = self.layout.dimensions();
        let b_dims = other.layout.dimensions();
        let rank = a_dims.len();

        if rank < 2 || b_dims.len() < 2 {
            return Err(
                TensorError::InvalidShape("matmul requires tensors with rank >= 2".into()).into(),
            );
        }

        if rank != b_dims.len() {
            return Err(TensorError::InvalidShape(format!(
                "matmul requires equal ranks, got {} and {}",
                rank,
                b_dims.len()
            ))
            .into());
        }

        let (a_rows, a_cols) = (a_dims[rank - 2], a_dims[rank - 1]);
        let (b_rows, b_cols) = (b_dims[rank - 2], b_dims[rank - 1]);

        let (m, a_k) = if transpose_a {
            (a_cols, a_rows)
        } else {
            (a_rows, a_cols)
        };
        let (b_k, n) = if transpose_b {
            (b_cols, b_rows)
        } else {
            (b_rows, b_cols)
        };

        if a_k != b_k {
            return Err(TensorError::InvalidShape(format!(
                "matmul inner dimensions don't match: {a_k} vs {b_k}"
            ))
            .into());
        }

        let mut out_dims: Vec<usize> = a_dims[..rank - 2]
            .iter()
            .zip(&b_dims[..rank - 2])
            .map(|(&da, &db)| match (da, db) {
                (a, b) if a == b => Ok(a),
                (1, b) => Ok(b),
                (a, 1) => Ok(a),
                _ => Err(TensorError::InvalidShape(format!(
                    "batch dimensions not broadcast-compatible: {da} vs {db}"
                ))),
            })
            .collect::<Result<_, _>>()?;
        out_dims.extend([m, n]);

        let layout = Layout::from_dimensions(&out_dims)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        ops::matmul(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            &buffer,
            a_dims,
            b_dims,
            &out_dims,
            transpose_a,
            transpose_b,
        );

        Ok(Self {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }

    /// Element-wise power with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn pow(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::pow(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Computes sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sin(&self) -> Result<Self, Error> {
        self.math_unary(ops::sin)
    }

    /// Computes cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn cos(&self) -> Result<Self, Error> {
        self.math_unary(ops::cos)
    }

    /// Computes tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn tan(&self) -> Result<Self, Error> {
        self.math_unary(ops::tan)
    }

    /// Computes arc sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn asin(&self) -> Result<Self, Error> {
        self.math_unary(ops::asin)
    }

    /// Computes arc cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn acos(&self) -> Result<Self, Error> {
        self.math_unary(ops::acos)
    }

    /// Computes arc tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn atan(&self) -> Result<Self, Error> {
        self.math_unary(ops::atan)
    }

    /// Computes hyperbolic sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sinh(&self) -> Result<Self, Error> {
        self.math_unary(ops::sinh)
    }

    /// Computes hyperbolic cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn cosh(&self) -> Result<Self, Error> {
        self.math_unary(ops::cosh)
    }

    /// Computes hyperbolic tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn tanh(&self) -> Result<Self, Error> {
        self.math_unary(ops::tanh)
    }

    /// Computes inverse hyperbolic sine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn asinh(&self) -> Result<Self, Error> {
        self.math_unary(ops::asinh)
    }

    /// Computes inverse hyperbolic cosine element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn acosh(&self) -> Result<Self, Error> {
        self.math_unary(ops::acosh)
    }

    /// Computes inverse hyperbolic tangent element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn atanh(&self) -> Result<Self, Error> {
        self.math_unary(ops::atanh)
    }

    /// Computes exponential (e^x) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn exp(&self) -> Result<Self, Error> {
        self.math_unary(ops::exp)
    }

    /// Computes natural logarithm element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn log(&self) -> Result<Self, Error> {
        self.math_unary(ops::log)
    }

    /// Computes base-2 logarithm element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn log2(&self) -> Result<Self, Error> {
        self.math_unary(ops::log2)
    }

    /// Computes square (x²) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sqr(&self) -> Result<Self, Error> {
        self.math_unary(ops::sqr)
    }

    /// Computes square root element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn sqrt(&self) -> Result<Self, Error> {
        self.math_unary(ops::sqrt)
    }

    /// Computes reciprocal of square (1/x²) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn rsqr(&self) -> Result<Self, Error> {
        self.math_unary(ops::rsqr)
    }

    /// Computes reciprocal of square root (1/√x) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn rsqrt(&self) -> Result<Self, Error> {
        self.math_unary(ops::rsqrt)
    }

    /// Computes reciprocal (1/x) element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn rcp(&self) -> Result<Self, Error> {
        self.math_unary(ops::rcp)
    }

    /// Computes ceiling element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn ceil(&self) -> Result<Self, Error> {
        self.math_unary(ops::ceil)
    }

    /// Computes floor element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn floor(&self) -> Result<Self, Error> {
        self.math_unary(ops::floor)
    }

    /// Rounds to nearest integer element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn round(&self) -> Result<Self, Error> {
        self.math_unary(ops::round)
    }

    /// `ELU` activation: `y = x < 0 ? α(eˣ - 1) : x`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Slope for negative values. Default: `1.0`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn elu(&self, alpha: Option<f32>) -> Result<Self, Error> {
        let alpha = alpha.unwrap_or(1.0);
        self.nn_activation(|ctx, x, y| ops::elu(ctx, x, y, alpha))
    }

    /// `GELU` activation: `y = x · σ(1.702x)`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn gelu(&self) -> Result<Self, Error> {
        self.nn_activation(ops::gelu)
    }

    /// `Leaky ReLU` activation: `y = x < 0 ? αx : x`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Slope for negative values. Default: `0.01`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn leaky_relu(&self, alpha: Option<f32>) -> Result<Self, Error> {
        let alpha = alpha.unwrap_or(0.01);
        self.nn_activation(|ctx, x, y| ops::leaky_relu(ctx, x, y, alpha))
    }

    /// `PReLU` activation: `y = x < 0 ? αx : x`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Learnable parameter tensor with the same shape as `self`.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes mismatch.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn prelu(&self, alpha: &Self) -> Result<Self, Error> {
        if self.dimensions() != alpha.dimensions() {
            return Err(TensorError::InvalidShape(format!(
                "prelu shape mismatch: {:?} vs {:?}",
                self.dimensions(),
                alpha.dimensions()
            ))
            .into());
        }
        self.nn_activation(|ctx, x, y| ops::prelu(ctx, x, y, &alpha.buffer))
    }

    /// `ReLU` activation: `y = max(x, 0)`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn relu(&self) -> Result<Self, Error> {
        self.nn_activation(ops::relu)
    }

    /// `SELU` activation: `y = λ(x < 0 ? α(eˣ - 1) : x)`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Scale for negative values. Default: `1.673_263_2`.
    /// * `lambda` - Output scale. Default: `1.050_701`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn selu(&self, alpha: Option<f32>, lambda: Option<f32>) -> Result<Self, Error> {
        let alpha = alpha.unwrap_or(1.673_263_2);
        let lambda = lambda.unwrap_or(1.050_701);
        self.nn_activation(|ctx, x, y| ops::selu(ctx, x, y, alpha, lambda))
    }

    /// `Sigmoid` activation: `y = 1/(1 + e⁻ˣ)`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn sigmoid(&self) -> Result<Self, Error> {
        self.nn_activation(ops::sigmoid)
    }

    /// `SiLU` activation: `y = x · σ(x)`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn silu(&self) -> Result<Self, Error> {
        self.nn_activation(ops::silu)
    }

    /// `Softplus` activation: `y = ln(eˣ + 1)`.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn softplus(&self) -> Result<Self, Error> {
        self.nn_activation(ops::softplus)
    }

    /// Applies an activation operation.
    fn nn_activation(
        &self,
        op: impl FnOnce(&Context, &Buffer<T>, &Buffer<T>),
    ) -> Result<Self, Error> {
        let buffer = self.ctx.create_buffer(self.buffer.len())?;
        op(&self.ctx, &self.buffer, &buffer);
        Ok(Self {
            buffer,
            layout: self.layout.clone(),
            ctx: self.ctx.clone(),
        })
    }
}

impl<T: LogicalElement> Tensor<T> {
    /// Selects elements from `a` or `b` based on condition.
    ///
    /// For each element, returns `a` where condition is true, otherwise `b`.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn select<U: NumericElement>(
        &self,
        a: &Tensor<U>,
        b: &Tensor<U>,
    ) -> Result<Tensor<U>, Error> {
        let (dimensions, strides) = Layout::broadcast(&[&self.layout, &a.layout, &b.layout])
            .ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "dimensions {:?}, {:?}, and {:?} are not broadcast-compatible",
                    self.dimensions(),
                    a.dimensions(),
                    b.dimensions()
                ))
            })?;

        let layout = Layout::from_dimensions(&dimensions)?;
        let buffer = self.ctx.create_buffer(layout.size())?;

        ops::select(
            &self.ctx,
            &self.buffer,
            &a.buffer,
            &b.buffer,
            &buffer,
            &strides[0],
            &strides[1],
            &strides[2],
            layout.strides(),
        );

        Ok(Tensor {
            buffer,
            layout,
            ctx: self.ctx.clone(),
        })
    }

    /// Element-wise logical AND with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn and(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::and(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Element-wise logical OR with broadcasting.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if shapes are not broadcast-compatible.
    /// - [`Error::Device`] if buffer allocation fails.
    pub fn or(&self, other: &Self) -> Result<Self, Error> {
        self.math_binary(other, |ctx, a, b, c, dimensions, a_strides, b_strides| {
            ops::or(ctx, a, b, c, dimensions, a_strides, b_strides);
        })
    }

    /// Computes logical NOT element-wise.
    ///
    /// # Errors
    ///
    /// - [`Error::Device`] if operation fails.
    pub fn not(&self) -> Result<Self, Error> {
        self.math_unary(ops::not)
    }
}
