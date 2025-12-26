//! Tests for `Tensor::acosh` operation.

use super::test_unary_op_float;

test_unary_op_float!(
    test_acosh_f32_vector,
    acosh,
    (&[4], &[1.0, 2.0, 3.0, 10.0]),
    (&[4], &[0.0, 1.316_958, 1.762_747_2, 2.993_222_7])
);

test_unary_op_float!(
    test_acosh_f32_matrix,
    acosh,
    (&[2, 3], &[1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
    (
        &[2, 3],
        &[
            0.0,
            0.962_423_6,
            1.316_958,
            1.566_799_3,
            1.762_747_2,
            2.063_437_2
        ]
    )
);

test_unary_op_float!(
    test_acosh_f32_scalar,
    acosh,
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[0.0])
);
