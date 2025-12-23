//! Tests for `Tensor::cos` operation.

use std::f32::consts::{FRAC_PI_2, PI};

use super::test_unary_op_float;

test_unary_op_float!(
    test_cos_f32_vector,
    cos,
    (&[4], &[0.0, FRAC_PI_2, PI, 2.0 * PI]),
    (&[4], &[1.0, 0.0, -1.0, 1.0])
);

test_unary_op_float!(
    test_cos_f32_matrix,
    cos,
    (&[2, 3], &[0.0, FRAC_PI_2, PI, -FRAC_PI_2, 0.0, -PI]),
    (&[2, 3], &[1.0, 0.0, -1.0, 0.0, 1.0, -1.0])
);

test_unary_op_float!(
    test_cos_f32_scalar,
    cos,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[1.0])
);
