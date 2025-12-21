//! Unary rounding operations.

use core::any::TypeId;

use crate::element::FloatElement;
use crate::{Buffer, Context, Error};

macro_rules! impl_rounding_op {
    ($name:ident, $expr:literal, $marker:ident) => {
        struct $marker<T>(core::marker::PhantomData<T>);

        impl<T: FloatElement> $marker<T> {
            fn shader() -> String {
                super::unary_shader("f32", $expr)
            }
        }

        pub(crate) fn $name<T: FloatElement>(
            ctx: &Context,
            a: &Buffer<T>,
            b: &Buffer<T>,
        ) -> Result<(), Error> {
            const LABEL: &str = stringify!($name);

            if a.len() != b.len() {
                return Err(Error::Device("buffer lengths mismatch".into()));
            }

            if a.is_empty() {
                return Ok(());
            }

            let pipeline = ctx.get_or_create_pipeline(
                TypeId::of::<$marker<T>>(),
                $marker::<T>::shader,
                LABEL,
            )?;

            super::dispatch(ctx, a.inner(), b.inner(), a.len(), &pipeline, LABEL)
        }
    };
}

impl_rounding_op!(ceil, "ceil(a[tid])", Ceil);
impl_rounding_op!(floor, "floor(a[tid])", Floor);
impl_rounding_op!(round, "round(a[tid])", Round);
