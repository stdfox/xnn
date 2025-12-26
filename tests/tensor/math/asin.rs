//! Tests for `Tensor::asin` operation.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

use super::test_unary_op_float;

test_unary_op_float!(
    test_asin_f32_vector,
    asin,
    (&[4], &[0.0, 0.5, 1.0, -1.0]),
    (&[4], &[0.0, FRAC_PI_6, FRAC_PI_2, -FRAC_PI_2])
);

test_unary_op_float!(
    test_asin_f32_matrix,
    asin,
    (&[2, 3], &[0.0, 0.25, 0.5, -0.25, -0.5, 1.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.252_680_24,
            FRAC_PI_6,
            -0.252_680_24,
            -FRAC_PI_6,
            FRAC_PI_2
        ]
    )
);

test_unary_op_float!(
    test_asin_f32_scalar,
    asin,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
