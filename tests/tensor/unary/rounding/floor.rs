//! Tests for `Tensor::floor` operation.

use super::test_unary_rounding_op;

test_unary_rounding_op!(
    test_floor_f32_vector,
    floor,
    (&[4], &[1.2, 2.7, -1.2, -2.7]),
    (&[4], &[1.0, 2.0, -2.0, -3.0])
);

test_unary_rounding_op!(
    test_floor_f32_matrix,
    floor,
    (&[2, 3], &[0.1, 0.9, -0.1, -0.9, 1.5, -1.5]),
    (&[2, 3], &[0.0, 0.0, -1.0, -1.0, 1.0, -2.0])
);

test_unary_rounding_op!(
    test_floor_f32_scalar,
    floor,
    (&[] as &[usize], &[1.5]),
    (&[] as &[usize], &[1.0])
);
