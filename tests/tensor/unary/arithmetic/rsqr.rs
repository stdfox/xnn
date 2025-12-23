//! Tests for `Tensor::rsqr` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_rsqr_f32_vector,
    rsqr,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[1.0, 0.25, 1.0 / 9.0, 0.0625])
);

test_unary_op_float!(
    test_rsqr_f32_matrix,
    rsqr,
    (&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    (&[2, 3], &[1.0, 0.25, 1.0 / 9.0, 1.0, 0.25, 1.0 / 9.0])
);

test_unary_op_float!(
    test_rsqr_f32_scalar,
    rsqr,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[0.04])
);
