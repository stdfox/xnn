//! Softplus activation operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

const LABEL: &str = "softplus";

struct SoftplusMarker<T>(PhantomData<T>);

impl<T: FloatElement> SoftplusMarker<T> {
    fn shader() -> String {
        super::activation_shader(T::wgsl_type(), "log(exp(a[tid]) + vec4(1.0))")
    }
}

/// Softplus activation: `y = log(exp(x) + 1)`.
pub(crate) fn softplus<T: FloatElement>(
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
        TypeId::of::<SoftplusMarker<T>>(),
        SoftplusMarker::<T>::shader,
        LABEL,
    )?;

    super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
}
