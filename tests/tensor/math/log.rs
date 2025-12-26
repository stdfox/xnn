//! Tests for `Tensor::log` operation.

use std::f32::consts::{E, LN_2, LN_10};

use super::test_unary_op_float;

test_unary_op_float!(
    test_log_f32_vector,
    log,
    (&[4], &[1.0, E, E * E, 10.0]),
    (&[4], &[0.0, 1.0, 2.0, LN_10])
);

test_unary_op_float!(
    test_log_f32_matrix,
    log,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (
        &[2, 3],
        &[0.0, LN_2, 1.098_612_3, 1.386_294_4, 1.609_438, 1.791_759_5]
    )
);

test_unary_op_float!(
    test_log_f32_scalar,
    log,
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[0.0])
);
