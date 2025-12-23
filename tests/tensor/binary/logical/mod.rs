//! Binary logical operation tests.

mod and;
mod or;

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

pub(crate) use test_logical_op;
