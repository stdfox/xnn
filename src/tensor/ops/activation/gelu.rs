//! `GELU` activation operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

const LABEL: &str = "gelu";

struct GeluMarker<T>(PhantomData<T>);

impl<T: FloatElement> GeluMarker<T> {
    fn shader() -> String {
        super::activation_shader(
            T::wgsl_type(),
            "a[tid] * (1.0 / (1.0 + exp(-1.702 * a[tid])))",
        )
    }
}

/// `GELU` activation: `y = x * sigmoid(1.702 * x)`.
pub(crate) fn gelu<T: FloatElement>(
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
        TypeId::of::<GeluMarker<T>>(),
        GeluMarker::<T>::shader,
        LABEL,
    )?;

    super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
}
