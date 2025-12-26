//! Tests for `Tensor::tanh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_tanh_f32_vector,
    tanh,
    (&[4], &[0.0, 1.0, -1.0, 10.0]),
    (&[4], &[0.0, 0.761_594_2, -0.761_594_2, 1.0])
);

test_unary_op_float!(
    test_tanh_f32_matrix,
    tanh,
    (&[2, 3], &[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.462_117_17,
            0.761_594_2,
            -0.462_117_17,
            -0.761_594_2,
            0.964_027_6
        ]
    )
);

test_unary_op_float!(
    test_tanh_f32_scalar,
    tanh,
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[0.0])
);
