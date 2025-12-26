//! Tests for `Tensor::rsqrt` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_rsqrt_f32_vector,
    rsqrt,
    (&[4], &[1.0, 4.0, 9.0, 16.0]),
    (&[4], &[1.0, 0.5, 1.0 / 3.0, 0.25])
);

test_unary_op_float!(
    test_rsqrt_f32_matrix,
    rsqrt,
    (&[2, 3], &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
    (&[2, 3], &[1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0])
);

test_unary_op_float!(
    test_rsqrt_f32_scalar,
    rsqrt,
    (&[] as &[usize], &[16.0]),
    (&[] as &[usize], &[0.25])
);
