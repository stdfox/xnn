//! Tests for `Tensor::log2` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_log2_f32_vector,
    log2,
    (&[4], &[1.0, 2.0, 4.0, 8.0]),
    (&[4], &[0.0, 1.0, 2.0, 3.0])
);

test_unary_op_float!(
    test_log2_f32_matrix,
    log2,
    (&[2, 3], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    (&[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
);

test_unary_op_float!(
    test_log2_f32_scalar,
    log2,
    (&[] as &[usize], &[16.0]),
    (&[] as &[usize], &[4.0])
);
