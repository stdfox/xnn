//! Tests for `Tensor::rcp` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_rcp_f32_vector,
    rcp,
    (&[4], &[0.5, 1.0, 2.0, 4.0]),
    (&[4], &[2.0, 1.0, 0.5, 0.25])
);

test_unary_op_float!(
    test_rcp_f32_matrix,
    rcp,
    (&[2, 3], &[1.0, 2.0, 4.0, 5.0, 10.0, 20.0]),
    (&[2, 3], &[1.0, 0.5, 0.25, 0.2, 0.1, 0.05])
);

test_unary_op_float!(
    test_rcp_f32_scalar,
    rcp,
    (&[] as &[usize], &[8.0]),
    (&[] as &[usize], &[0.125])
);
