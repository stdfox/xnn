//! Tests for `Tensor::atanh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_atanh_f32_vector,
    atanh,
    (&[4], &[0.0, 0.5, -0.5, 0.9]),
    (&[4], &[0.0, 0.549_306_2, -0.549_306_2, 1.472_219_3])
);

test_unary_op_float!(
    test_atanh_f32_matrix,
    atanh,
    (&[2, 3], &[0.0, 0.25, 0.5, -0.25, -0.5, 0.75]),
    (
        &[2, 3],
        &[
            0.0,
            0.255_412_82,
            0.549_306_2,
            -0.255_412_82,
            -0.549_306_2,
            0.972_955_05
        ]
    )
);

test_unary_op_float!(
    test_atanh_f32_scalar,
    atanh,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
