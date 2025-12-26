//! Tests for `Tensor::tan` operation.

use std::f32::consts::FRAC_PI_4;

use super::test_unary_op_float;

test_unary_op_float!(
    test_tan_f32_vector,
    tan,
    (&[4], &[0.0, FRAC_PI_4, -FRAC_PI_4, 0.0]),
    (&[4], &[0.0, 1.0, -1.0, 0.0])
);

test_unary_op_float!(
    test_tan_f32_matrix,
    tan,
    (&[2, 3], &[0.0, FRAC_PI_4, -FRAC_PI_4, 0.0, FRAC_PI_4, 0.0]),
    (&[2, 3], &[0.0, 1.0, -1.0, 0.0, 1.0, 0.0])
);

test_unary_op_float!(
    test_tan_f32_scalar,
    tan,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
