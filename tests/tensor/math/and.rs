//! Tests for `Tensor::and` operation.

use super::test_logical_op;

// vector

test_logical_op!(
    test_and_vector,
    and,
    (&[4], &[true, true, false, false]),
    (&[4], &[true, false, true, false]),
    (&[4], &[true, false, false, false])
);

// matrix

test_logical_op!(
    test_and_matrix,
    and,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[false, false, true, true, true, false]),
    (&[2, 3], &[false, false, true, false, true, false])
);

// scalar

test_logical_op!(
    test_and_scalar_true_true,
    and,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[true])
);

test_logical_op!(
    test_and_scalar_true_false,
    and,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[false])
);

test_logical_op!(
    test_and_scalar_false_true,
    and,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[false])
);

test_logical_op!(
    test_and_scalar_false_false,
    and,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[false])
);

// broadcast

test_logical_op!(
    test_and_broadcast_multi_expand,
    and,
    (&[3, 1], &[true, false, true]),
    (&[1, 4], &[true, false, true, false]),
    (
        &[3, 4],
        &[
            true, false, true, false, false, false, false, false, true, false, true, false
        ]
    )
);

test_logical_op!(
    test_and_broadcast_expand,
    and,
    (&[2, 1, 3], &[true, false, true, false, true, false]),
    (
        &[2, 2, 3],
        &[
            true, true, false, false, true, true, true, false, true, false, false, true
        ]
    ),
    (
        &[2, 2, 3],
        &[
            true, false, false, false, false, true, false, false, false, false, false, false
        ]
    )
);

test_logical_op!(
    test_and_broadcast_trailing,
    and,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[3], &[true, false, true]),
    (&[2, 3], &[true, false, true, false, false, false])
);

test_logical_op!(
    test_and_broadcast_scalar_true,
    and,
    (&[4], &[true, true, false, false]),
    (&[] as &[usize], &[true]),
    (&[4], &[true, true, false, false])
);

test_logical_op!(
    test_and_broadcast_scalar_false,
    and,
    (&[4], &[true, true, false, false]),
    (&[] as &[usize], &[false]),
    (&[4], &[false, false, false, false])
);

test_logical_op!(
    test_and_broadcast_scalar_reverse_true,
    and,
    (&[] as &[usize], &[true]),
    (&[4], &[true, false, true, false]),
    (&[4], &[true, false, true, false])
);

test_logical_op!(
    test_and_broadcast_scalar_reverse_false,
    and,
    (&[] as &[usize], &[false]),
    (&[4], &[true, false, true, false]),
    (&[4], &[false, false, false, false])
);

// error

#[test]
fn test_and_error_incompatible_shapes() {
    use xnn::{Context, Tensor};
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_shape_slice(&ctx, &[3], &[true, false, true]).unwrap();
    let b = Tensor::<bool>::from_shape_slice(&ctx, &[4], &[true, false, true, false]).unwrap();
    assert!(a.and(&b).is_err());
}
