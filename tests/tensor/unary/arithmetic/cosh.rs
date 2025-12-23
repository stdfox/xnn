//! Tests for `Tensor::cosh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_cosh_f32_vector,
    cosh,
    (&[4], &[0.0, 1.0, -1.0, 2.0]),
    (&[4], &[1.0, 1.543_080_6, 1.543_080_6, 3.762_195_6])
);

test_unary_op_float!(
    test_cosh_f32_matrix,
    cosh,
    (&[2, 3], &[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]),
    (
        &[2, 3],
        &[
            1.0,
            1.127_626,
            1.543_080_6,
            1.127_626,
            1.543_080_6,
            3.762_195_6
        ]
    )
);

test_unary_op_float!(
    test_cosh_f32_scalar,
    cosh,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[1.0])
);
