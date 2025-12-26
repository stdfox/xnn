//! Tests for `Tensor::acos` operation.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, PI};

use super::test_unary_op_float;

test_unary_op_float!(
    test_acos_f32_vector,
    acos,
    (&[4], &[1.0, 0.0, -1.0, 0.5]),
    (&[4], &[0.0, FRAC_PI_2, PI, FRAC_PI_3])
);

test_unary_op_float!(
    test_acos_f32_matrix,
    acos,
    (&[2, 3], &[1.0, 0.5, 0.0, -0.5, -1.0, 0.25]),
    (
        &[2, 3],
        &[0.0, FRAC_PI_3, FRAC_PI_2, 2.094_395_2, PI, 1.318_116_1]
    )
);

test_unary_op_float!(
    test_acos_f32_scalar,
    acos,
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[0.0])
);
