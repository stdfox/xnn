//! Arithmetic unary operation tests.

mod abs;
mod acos;
mod acosh;
mod asin;
mod asinh;
mod atan;
mod atanh;
mod cos;
mod cosh;
mod exp;
mod log;
mod log2;
mod neg;
mod rcp;
mod rsqr;
mod rsqrt;
mod sign;
mod sin;
mod sinh;
mod sqr;
mod sqrt;
mod tan;
mod tanh;

/// Generates a unary float op test.
macro_rules! test_unary_op_float {
    ($name:ident, $method:ident, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let a = Tensor::<f32>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<f32>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let result = a.$method().unwrap();
            crate::assert_tensor_relative_eq(&result, &b);
        }
    };
}

/// Generates a unary integer op test.
macro_rules! test_unary_op_integer {
    ($name:ident, $method:ident, $T:ty, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let result = a.$method().unwrap();
            crate::assert_tensor_eq(&result, &b);
        }
    };
}

pub(crate) use test_unary_op_float;
pub(crate) use test_unary_op_integer;
