//! Math operation tests.

mod abs;
mod acos;
mod acosh;
mod add;
mod and;
mod asin;
mod asinh;
mod atan;
mod atanh;
mod ceil;
mod clamp;
mod cos;
mod cosh;
mod div;
mod eq;
mod exp;
mod floor;
mod ge;
mod gt;
mod le;
mod log;
mod log2;
mod lt;
mod max;
mod min;
mod mul;
mod ne;
mod neg;
mod not;
mod or;
mod pow;
mod rcp;
mod rem;
mod round;
mod rsqr;
mod rsqrt;
mod select;
mod sign;
mod sin;
mod sinh;
mod sqr;
mod sqrt;
mod sub;
mod tan;
mod tanh;

/// Generates a binary arithmetic op test for float types.
macro_rules! test_arithmetic_op_float {
    ($name:ident, $method:ident, $T:ty, $a:expr, $b:expr, $c:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (c_shape, c_data) = $c;
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let c = Tensor::<$T>::from_shape_slice(&ctx, c_shape, c_data).unwrap();
            let result = a.$method(&b).unwrap();
            crate::assert_tensor_relative_eq(&result, &c);
        }
    };
}

/// Generates a binary arithmetic op test for integer types.
macro_rules! test_arithmetic_op_integer {
    ($name:ident, $method:ident, $T:ty, $a:expr, $b:expr, $c:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (c_shape, c_data) = $c;
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let c = Tensor::<$T>::from_shape_slice(&ctx, c_shape, c_data).unwrap();
            let result = a.$method(&b).unwrap();
            crate::assert_tensor_eq(&result, &c);
        }
    };
}

/// Generates a binary comparison op test.
macro_rules! test_comparison_op {
    ($name:ident, $method:ident, $T:ty, $a:expr, $b:expr, $c:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (c_shape, c_data) = $c;
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let c = Tensor::<bool>::from_shape_slice(&ctx, c_shape, c_data).unwrap();
            let result = a.$method(&b).unwrap();
            crate::assert_tensor_eq(&result, &c);
        }
    };
}

/// Generates a binary logical op test.
macro_rules! test_logical_op {
    ($name:ident, $method:ident, $a:expr, $b:expr, $c:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (c_shape, c_data) = $c;
            let a = Tensor::<bool>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<bool>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let c = Tensor::<bool>::from_shape_slice(&ctx, c_shape, c_data).unwrap();
            let result = a.$method(&b).unwrap();
            crate::assert_tensor_eq(&result, &c);
        }
    };
}

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

/// Generates a unary rounding op test.
macro_rules! test_unary_rounding_op {
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

/// Generates a unary logical op test.
macro_rules! test_unary_logical_op {
    ($name:ident, $method:ident, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            use xnn::{Context, Tensor};
            let ctx = Context::try_default().unwrap();
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let a = Tensor::<bool>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<bool>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let result = a.$method().unwrap();
            crate::assert_tensor_eq(&result, &b);
        }
    };
}

pub(crate) use test_arithmetic_op_float;
pub(crate) use test_arithmetic_op_integer;
pub(crate) use test_comparison_op;
pub(crate) use test_logical_op;
pub(crate) use test_unary_logical_op;
pub(crate) use test_unary_op_float;
pub(crate) use test_unary_op_integer;
pub(crate) use test_unary_rounding_op;
