//! Tests for `Tensor::sinh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_sinh_f32_vector,
    sinh,
    (&[4], &[0.0, 1.0, -1.0, 2.0]),
    (&[4], &[0.0, 1.175_201_2, -1.175_201_2, 3.626_860_4])
);

test_unary_op_float!(
    test_sinh_f32_matrix,
    sinh,
    (&[2, 3], &[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.521_095_3,
            1.175_201_2,
            -0.521_095_3,
            -1.175_201_2,
            3.626_860_4
        ]
    )
);

test_unary_op_float!(
    test_sinh_f32_scalar,
    sinh,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
