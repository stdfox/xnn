//! Logical unary operation tests.

mod not;

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

pub(crate) use test_unary_logical_op;
