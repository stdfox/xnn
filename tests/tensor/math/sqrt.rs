//! Tests for `Tensor::sqrt` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_sqrt_f32_vector,
    sqrt,
    (&[4], &[0.0, 1.0, 4.0, 9.0]),
    (&[4], &[0.0, 1.0, 2.0, 3.0])
);

test_unary_op_float!(
    test_sqrt_f32_matrix,
    sqrt,
    (&[2, 3], &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
);

test_unary_op_float!(
    test_sqrt_f32_scalar,
    sqrt,
    (&[] as &[usize], &[9.0]),
    (&[] as &[usize], &[3.0])
);
