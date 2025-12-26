//! Tests for `Tensor::abs` operation.

use super::{test_unary_op_float, test_unary_op_integer};

// f32

test_unary_op_float!(
    test_abs_f32_vector,
    abs,
    (&[4], &[-1.0, 2.0, -3.0, 4.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0])
);

test_unary_op_float!(
    test_abs_f32_matrix,
    abs,
    (&[2, 3], &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
);

test_unary_op_float!(
    test_abs_f32_scalar,
    abs,
    (&[] as &[usize], &[-42.0]),
    (&[] as &[usize], &[42.0])
);

// i32

test_unary_op_integer!(
    test_abs_i32_vector,
    abs,
    i32,
    (&[4], &[-1, 2, -3, 4]),
    (&[4], &[1, 2, 3, 4])
);

test_unary_op_integer!(
    test_abs_i32_matrix,
    abs,
    i32,
    (&[2, 3], &[-1, 2, -3, 4, -5, 6]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6])
);

test_unary_op_integer!(
    test_abs_i32_scalar,
    abs,
    i32,
    (&[] as &[usize], &[-42]),
    (&[] as &[usize], &[42])
);
