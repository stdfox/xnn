//! `SiLU` activation operation.

use core::any::TypeId;
use core::marker::PhantomData;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

const LABEL: &str = "silu";

struct SiluMarker<T>(PhantomData<T>);

impl<T: FloatElement> SiluMarker<T> {
    fn shader() -> String {
        super::activation_shader(T::wgsl_type(), "a[tid] * (1.0 / (1.0 + exp(-a[tid])))")
    }
}

/// `SiLU` activation: `y = x * sigmoid(x)`.
pub(crate) fn silu<T: FloatElement>(
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
        TypeId::of::<SiluMarker<T>>(),
        SiluMarker::<T>::shader,
        LABEL,
    )?;

    super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
}
