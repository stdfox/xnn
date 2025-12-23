//! Tests for `Tensor::not` operation.

use super::test_unary_logical_op;

test_unary_logical_op!(
    test_not_vector,
    not,
    (&[4], &[true, false, true, false]),
    (&[4], &[false, true, false, true])
);

test_unary_logical_op!(
    test_not_matrix,
    not,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_unary_logical_op!(
    test_not_scalar_true,
    not,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[false])
);

test_unary_logical_op!(
    test_not_scalar_false,
    not,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[true])
);
