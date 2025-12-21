//! Binary arithmetic operations.

use core::any::TypeId;

use crate::element::{FloatElement, IntegerElement, NumericElement};
use crate::{Buffer, Context, Error};

macro_rules! impl_arithmetic_op {
    ($name:ident, $expr:literal, $marker:ident, $trait:ident) => {
        struct $marker<T>(core::marker::PhantomData<T>);

        impl<T: $trait> $marker<T> {
            fn shader() -> String {
                let t = T::wgsl_type();
                super::binary_shader(t, t, t, $expr)
            }
        }

        #[allow(clippy::many_single_char_names)]
        pub(crate) fn $name<T: $trait>(
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

impl_arithmetic_op!(add, "a[a_idx] + b[b_idx]", Add, NumericElement);
impl_arithmetic_op!(sub, "a[a_idx] - b[b_idx]", Sub, NumericElement);
impl_arithmetic_op!(mul, "a[a_idx] * b[b_idx]", Mul, NumericElement);
impl_arithmetic_op!(div, "a[a_idx] / b[b_idx]", Div, NumericElement);
impl_arithmetic_op!(rem, "a[a_idx] % b[b_idx]", Rem, IntegerElement);
impl_arithmetic_op!(pow, "pow(a[a_idx], b[b_idx])", Pow, FloatElement);
