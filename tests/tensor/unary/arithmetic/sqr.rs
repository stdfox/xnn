//! Tests for `Tensor::sqr` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_sqr_f32_vector,
    sqr,
    (&[4], &[0.0, 1.0, 2.0, -3.0]),
    (&[4], &[0.0, 1.0, 4.0, 9.0])
);

test_unary_op_float!(
    test_sqr_f32_matrix,
    sqr,
    (&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    (&[2, 3], &[1.0, 4.0, 9.0, 1.0, 4.0, 9.0])
);

test_unary_op_float!(
    test_sqr_f32_scalar,
    sqr,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[25.0])
);
