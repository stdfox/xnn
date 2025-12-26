//! Tests for `Tensor::exp` operation.

use std::f32::consts::E;

use super::test_unary_op_float;

test_unary_op_float!(
    test_exp_f32_vector,
    exp,
    (&[4], &[0.0, 1.0, 2.0, -1.0]),
    (&[4], &[1.0, E, E * E, 1.0 / E])
);

test_unary_op_float!(
    test_exp_f32_matrix,
    exp,
    (&[2, 3], &[0.0, 1.0, -1.0, 2.0, -2.0, 0.0]),
    (&[2, 3], &[1.0, E, 1.0 / E, E * E, 1.0 / (E * E), 1.0])
);

test_unary_op_float!(
    test_exp_f32_scalar,
    exp,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[1.0])
);
