//! Tests for `Tensor::pow` operation.

use std::f32::consts::SQRT_2;

use xnn::{Context, Tensor};

use super::test_arithmetic_op_float;

// vector

test_arithmetic_op_float!(
    test_pow_f32_vector,
    pow,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[2.0, 2.0, 2.0, 2.0]),
    (&[4], &[1.0, 4.0, 9.0, 16.0])
);

// matrix

test_arithmetic_op_float!(
    test_pow_f32_matrix,
    pow,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    (&[2, 3], &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
);

// scalar

test_arithmetic_op_float!(
    test_pow_f32_scalar,
    pow,
    f32,
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[4.0]),
    (&[] as &[usize], &[81.0])
);

// broadcast

test_arithmetic_op_float!(
    test_pow_f32_broadcast_multi_expand,
    pow,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[0.0, 1.0, 2.0]),
    (
        &[2, 3, 4],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 4.0, 9.0, 16.0, 1.0, 1.0, 1.0, 1.0, 5.0,
            6.0, 7.0, 8.0, 25.0, 36.0, 49.0, 64.0
        ]
    )
);

test_arithmetic_op_float!(
    test_pow_f32_broadcast_expand,
    pow,
    f32,
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1, 4], &[0.0, 1.0, 2.0, 3.0]),
    (
        &[3, 4],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 8.0, 1.0, 3.0, 9.0, 27.0]
    )
);

test_arithmetic_op_float!(
    test_pow_f32_broadcast_trailing,
    pow,
    f32,
    (&[2, 3], &[1.0, 2.0, 4.0, 4.0, 9.0, 16.0]),
    (&[3], &[2.0, 0.5, 0.5]),
    (&[2, 3], &[1.0, SQRT_2, 2.0, 16.0, 3.0, 4.0])
);

test_arithmetic_op_float!(
    test_pow_f32_broadcast_scalar,
    pow,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[] as &[usize], &[2.0]),
    (&[4], &[1.0, 4.0, 9.0, 16.0])
);

test_arithmetic_op_float!(
    test_pow_f32_broadcast_scalar_reverse,
    pow,
    f32,
    (&[] as &[usize], &[2.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[2.0, 4.0, 8.0, 16.0])
);

// error

#[test]
fn test_pow_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.pow(&b).is_err());
}
