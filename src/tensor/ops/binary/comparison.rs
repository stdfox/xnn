//! Binary comparison operations.

use core::any::TypeId;

use crate::element::NumericElement;
use crate::{Buffer, Context, Error};

macro_rules! impl_comparison_op {
    ($name:ident, $expr:literal, $marker:ident) => {
        struct $marker<T>(core::marker::PhantomData<T>);

        impl<T: NumericElement> $marker<T> {
            fn shader() -> String {
                let t = T::wgsl_type();
                super::binary_shader(t, t, "u32", &format!("u32({expr})", expr = $expr))
            }
        }

        #[allow(clippy::many_single_char_names)]
        pub(crate) fn $name<T: NumericElement>(
            ctx: &Context,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &Buffer<bool>,
            dimensions: &[usize],
            a_strides: &[usize],
            b_strides: &[usize],
        ) -> Result<(), Error> {
            const LABEL: &str = stringify!($name);

            let out_len = dimensions.iter().product::<usize>().max(1);
            if c.len() != out_len {
                return Err(Error::Device("output buffer length mismatch".into()));
            }

            let pipeline = ctx.get_or_create_pipeline(
                TypeId::of::<$marker<T>>(),
                $marker::<T>::shader,
                LABEL,
            )?;

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

impl_comparison_op!(lt, "a[a_idx] < b[b_idx]", Lt);
impl_comparison_op!(gt, "a[a_idx] > b[b_idx]", Gt);
impl_comparison_op!(le, "a[a_idx] <= b[b_idx]", Le);
impl_comparison_op!(ge, "a[a_idx] >= b[b_idx]", Ge);
impl_comparison_op!(eq, "a[a_idx] == b[b_idx]", Eq);
impl_comparison_op!(ne, "a[a_idx] != b[b_idx]", Ne);
