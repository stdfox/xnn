//! Binary comparison operation tests.

mod eq;
mod ge;
mod gt;
mod le;
mod lt;
mod ne;

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

pub(crate) use test_comparison_op;
