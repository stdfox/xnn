//! Tests for `Tensor::sin` operation.

use std::f32::consts::FRAC_PI_2;

use super::test_unary_op_float;

test_unary_op_float!(
    test_sin_f32_vector,
    sin,
    (&[4], &[0.0, 0.5, FRAC_PI_2, -FRAC_PI_2]),
    (&[4], &[0.0, 0.479_425_55, 1.0, -1.0])
);

test_unary_op_float!(
    test_sin_f32_matrix,
    sin,
    (&[2, 3], &[0.0, 0.5, FRAC_PI_2, -0.5, -FRAC_PI_2, 1.0]),
    (
        &[2, 3],
        &[0.0, 0.479_425_55, 1.0, -0.479_425_55, -1.0, 0.841_470_96]
    )
);

test_unary_op_float!(
    test_sin_f32_scalar,
    sin,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
