//! Tests for `Tensor::atan` operation.

use std::f32::consts::FRAC_PI_4;

use super::test_unary_op_float;

test_unary_op_float!(
    test_atan_f32_vector,
    atan,
    (&[4], &[0.0, 1.0, -1.0, 2.0]),
    (&[4], &[0.0, FRAC_PI_4, -FRAC_PI_4, 1.107_148_8])
);

test_unary_op_float!(
    test_atan_f32_matrix,
    atan,
    (&[2, 3], &[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.463_647_6,
            FRAC_PI_4,
            -0.463_647_6,
            -FRAC_PI_4,
            1.107_148_8
        ]
    )
);

test_unary_op_float!(
    test_atan_f32_scalar,
    atan,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
