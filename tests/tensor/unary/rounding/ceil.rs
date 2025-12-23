//! Tests for `Tensor::ceil` operation.

use super::test_unary_rounding_op;

test_unary_rounding_op!(
    test_ceil_f32_vector,
    ceil,
    (&[4], &[1.2, 2.7, -1.2, -2.7]),
    (&[4], &[2.0, 3.0, -1.0, -2.0])
);

test_unary_rounding_op!(
    test_ceil_f32_matrix,
    ceil,
    (&[2, 3], &[0.1, 0.9, -0.1, -0.9, 1.5, -1.5]),
    (&[2, 3], &[1.0, 1.0, 0.0, 0.0, 2.0, -1.0])
);

test_unary_rounding_op!(
    test_ceil_f32_scalar,
    ceil,
    (&[] as &[usize], &[1.5]),
    (&[] as &[usize], &[2.0])
);
