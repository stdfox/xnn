//! Kernel operations.

use crate::element::FloatElement;
use crate::kernel::activation::{
    self, Elu, Gelu, LeakyRelu, Prelu, Relu, Selu, Sigmoid, Silu, Softplus,
};
use crate::{Buffer, Context, Error};

/// `ELU` activation: `y = select(x < 0, alpha * (exp(x) - 1), x)`.
pub(crate) fn elu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: f32,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute_with_param::<Elu>(ctx, x.inner(), y.inner(), alpha, 0.0)
}

/// `GELU` activation: `y = x * sigmoid(1.702 * x)`.
pub(crate) fn gelu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute::<Gelu>(ctx, x.inner(), y.inner())
}

/// `LeakyReLU` activation: `y = select(x < 0, alpha * x, x)`.
pub(crate) fn leaky_relu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: f32,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute_with_param::<LeakyRelu>(ctx, x.inner(), y.inner(), alpha, 0.0)
}

/// `PReLU` activation: `y = select(x < 0, alpha * x, x)`.
pub(crate) fn prelu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.len() != alpha.len() {
        return Err(Error::Kernel("alpha buffer length mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute_with_alpha::<Prelu>(ctx, x.inner(), y.inner(), alpha.inner())
}

/// `ReLU` activation: `y = max(x, 0)`.
pub(crate) fn relu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute::<Relu>(ctx, x.inner(), y.inner())
}

/// `SELU` activation: `y = lambda * select(x < 0, alpha * (exp(x) - 1), x)`.
pub(crate) fn selu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: f32,
    lambda: f32,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute_with_param::<Selu>(ctx, x.inner(), y.inner(), alpha, lambda)
}

/// `Sigmoid` activation: `y = 1 / (1 + exp(-x))`.
pub(crate) fn sigmoid<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute::<Sigmoid>(ctx, x.inner(), y.inner())
}

/// `SiLU` activation: `y = x * sigmoid(x)`.
pub(crate) fn silu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute::<Silu>(ctx, x.inner(), y.inner())
}

/// `Softplus` activation: `y = log(exp(x) + 1)`.
pub(crate) fn softplus<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
) -> Result<(), Error> {
    if x.len() != y.len() {
        return Err(Error::Kernel("buffer lengths mismatch".into()));
    }
    if x.is_empty() {
        return Ok(());
    }
    activation::execute::<Softplus>(ctx, x.inner(), y.inner())
}
