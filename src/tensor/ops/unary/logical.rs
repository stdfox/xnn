//! Unary logical operations.

use core::any::TypeId;

use crate::element::LogicalElement;
use crate::{Buffer, Context, Error};

macro_rules! impl_logical_op {
    ($name:ident, $expr:literal, $marker:ident) => {
        struct $marker<T>(core::marker::PhantomData<T>);

        impl<T: LogicalElement> $marker<T> {
            fn shader() -> String {
                super::unary_shader("u32", $expr)
            }
        }

        pub(crate) fn $name<T: LogicalElement>(
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

impl_logical_op!(not, "vec4<u32>(1u) - min(a[tid], vec4<u32>(1u))", Not);
