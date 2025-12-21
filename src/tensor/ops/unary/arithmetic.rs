//! Unary arithmetic operations.

use core::any::TypeId;

use crate::element::{FloatElement, SignedElement};
use crate::{Buffer, Context, Error};

macro_rules! impl_arithmetic_op {
    ($name:ident, $expr:literal, $marker:ident, $trait:ident) => {
        struct $marker<T>(core::marker::PhantomData<T>);

        impl<T: $trait> $marker<T> {
            fn shader() -> String {
                super::unary_shader(T::wgsl_type(), $expr)
            }
        }

        pub(crate) fn $name<T: $trait>(
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

// Signed operations
impl_arithmetic_op!(abs, "abs(a[tid])", Abs, SignedElement);
impl_arithmetic_op!(neg, "-a[tid]", Neg, SignedElement);
impl_arithmetic_op!(sign, "sign(a[tid])", Sign, SignedElement);

// Float operations
impl_arithmetic_op!(acos, "acos(a[tid])", Acos, FloatElement);
impl_arithmetic_op!(acosh, "acosh(a[tid])", Acosh, FloatElement);
impl_arithmetic_op!(asin, "asin(a[tid])", Asin, FloatElement);
impl_arithmetic_op!(asinh, "asinh(a[tid])", Asinh, FloatElement);
impl_arithmetic_op!(atan, "atan(a[tid])", Atan, FloatElement);
impl_arithmetic_op!(atanh, "atanh(a[tid])", Atanh, FloatElement);
impl_arithmetic_op!(cos, "cos(a[tid])", Cos, FloatElement);
impl_arithmetic_op!(cosh, "cosh(a[tid])", Cosh, FloatElement);
impl_arithmetic_op!(exp, "exp(a[tid])", Exp, FloatElement);
impl_arithmetic_op!(log, "log(a[tid])", Log, FloatElement);
impl_arithmetic_op!(rcp, "1.0 / a[tid]", Rcp, FloatElement);
impl_arithmetic_op!(sin, "sin(a[tid])", Sin, FloatElement);
impl_arithmetic_op!(sinh, "sinh(a[tid])", Sinh, FloatElement);
impl_arithmetic_op!(tan, "tan(a[tid])", Tan, FloatElement);
impl_arithmetic_op!(tanh, "tanh(a[tid])", Tanh, FloatElement);
