//! Sigmoid activation operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

const LABEL: &str = "sigmoid";

struct SigmoidMarker<T>(PhantomData<T>);

impl<T: FloatElement> SigmoidMarker<T> {
    fn shader() -> String {
        super::activation_shader(T::wgsl_type(), "1.0 / (1.0 + exp(-a[tid]))")
    }
}

/// Sigmoid activation: `y = 1 / (1 + exp(-x))`.
pub(crate) fn sigmoid<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::Device("buffer lengths mismatch".into()));
    }

    if a.is_empty() {
        return Ok(());
    }

    let pipeline = ctx.get_or_create_pipeline(
        TypeId::of::<SigmoidMarker<T>>(),
        SigmoidMarker::<T>::shader,
        LABEL,
    )?;

    super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
}
