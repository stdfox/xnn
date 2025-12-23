//! Binary arithmetic operation tests.

mod add;
mod div;
mod max;
mod min;
mod mul;
mod pow;
mod rem;
mod sub;

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

pub(crate) use test_arithmetic_op_float;
pub(crate) use test_arithmetic_op_integer;
