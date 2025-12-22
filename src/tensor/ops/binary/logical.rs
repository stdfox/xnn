//! Binary logical operations.

use core::any::TypeId;

use crate::element::LogicalElement;
use crate::{Buffer, Context, Error};

macro_rules! impl_logical_op {
    ($name:ident, $expr:literal, $marker:ident) => {
        struct $marker;

        impl $marker {
            fn shader() -> String {
                super::binary_shader("u32", "u32", "u32", $expr)
            }
        }

        pub(crate) fn $name<T: LogicalElement>(
            ctx: &Context,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &Buffer<T>,
            dimensions: &[usize],
            a_strides: &[usize],
            b_strides: &[usize],
        ) -> Result<(), Error> {
            const LABEL: &str = stringify!($name);

            let out_len = dimensions.iter().product::<usize>().max(1);
            if c.len() != out_len {
                return Err(Error::Device("output buffer length mismatch".into()));
            }

            let pipeline =
                ctx.get_or_create_pipeline(TypeId::of::<$marker>(), $marker::shader, LABEL)?;

            super::dispatch(
                ctx,
                a.inner(),
                b.inner(),
                c.inner(),
                dimensions,
                a_strides,
                b_strides,
                &pipeline,
                LABEL,
            )
        }
    };
}

impl_logical_op!(and, "u32(a[a_idx] != 0u && b[b_idx] != 0u)", And);
impl_logical_op!(or, "u32(a[a_idx] != 0u || b[b_idx] != 0u)", Or);
