//! Tests for `Tensor::round` operation.

use super::test_unary_rounding_op;

test_unary_rounding_op!(
    test_round_f32_vector,
    round,
    (&[4], &[1.2, 2.7, -1.2, -2.7]),
    (&[4], &[1.0, 3.0, -1.0, -3.0])
);

test_unary_rounding_op!(
    test_round_f32_matrix,
    round,
    (&[2, 3], &[0.1, 0.9, -0.1, -0.9, 1.7, -1.7]),
    (&[2, 3], &[0.0, 1.0, 0.0, -1.0, 2.0, -2.0])
);

test_unary_rounding_op!(
    test_round_f32_scalar,
    round,
    (&[] as &[usize], &[1.7]),
    (&[] as &[usize], &[2.0])
);
