//! Tests for `Tensor::sign` operation.

use super::{test_unary_op_float, test_unary_op_integer};

// f32

test_unary_op_float!(
    test_sign_f32_vector,
    sign,
    (&[4], &[-1.0, 2.0, -3.0, 4.0]),
    (&[4], &[-1.0, 1.0, -1.0, 1.0])
);

test_unary_op_float!(
    test_sign_f32_matrix,
    sign,
    (&[2, 3], &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
    (&[2, 3], &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
);

test_unary_op_float!(
    test_sign_f32_scalar,
    sign,
    (&[] as &[usize], &[-42.0]),
    (&[] as &[usize], &[-1.0])
);

test_unary_op_float!(
    test_sign_f32_zero,
    sign,
    (&[2], &[0.0, -0.0]),
    (&[2], &[0.0, 0.0])
);

// i32

test_unary_op_integer!(
    test_sign_i32_vector,
    sign,
    i32,
    (&[4], &[-1, 2, -3, 4]),
    (&[4], &[-1, 1, -1, 1])
);

test_unary_op_integer!(
    test_sign_i32_matrix,
    sign,
    i32,
    (&[2, 3], &[-1, 2, -3, 4, -5, 6]),
    (&[2, 3], &[-1, 1, -1, 1, -1, 1])
);

test_unary_op_integer!(
    test_sign_i32_scalar,
    sign,
    i32,
    (&[] as &[usize], &[-42]),
    (&[] as &[usize], &[-1])
);

test_unary_op_integer!(
    test_sign_i32_zero,
    sign,
    i32,
    (&[4], &[0, 0, 0, 0]),
    (&[4], &[0, 0, 0, 0])
);
