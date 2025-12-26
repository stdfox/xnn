//! Tests for `Tensor::asinh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_asinh_f32_vector,
    asinh,
    (&[4], &[0.0, 1.0, -1.0, 2.0]),
    (&[4], &[0.0, 0.881_373_64, -0.881_373_64, 1.443_635_5])
);

test_unary_op_float!(
    test_asinh_f32_matrix,
    asinh,
    (&[2, 3], &[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.481_211_8,
            0.881_373_64,
            -0.481_211_8,
            -0.881_373_64,
            1.443_635_5
        ]
    )
);

test_unary_op_float!(
    test_asinh_f32_scalar,
    asinh,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
