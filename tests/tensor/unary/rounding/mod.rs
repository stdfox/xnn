//! Rounding unary operation tests.

mod ceil;
mod floor;
mod round;

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

pub(crate) use test_unary_rounding_op;
