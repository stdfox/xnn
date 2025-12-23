//! `ReLU` activation operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

const LABEL: &str = "relu";

struct ReluMarker<T>(PhantomData<T>);

impl<T: FloatElement> ReluMarker<T> {
    fn shader() -> String {
        super::activation_shader(T::wgsl_type(), "max(a[tid], vec4(0.0))")
    }
}

/// `ReLU` activation: `y = max(x, 0)`.
pub(crate) fn relu<T: FloatElement>(
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
        TypeId::of::<ReluMarker<T>>(),
        ReluMarker::<T>::shader,
        LABEL,
    )?;

    super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
}
