//! Tests for `Tensor::or` operation.

use super::test_logical_op;

// vector

test_logical_op!(
    test_or_vector,
    or,
    (&[4], &[true, true, false, false]),
    (&[4], &[true, false, true, false]),
    (&[4], &[true, true, true, false])
);

// matrix

test_logical_op!(
    test_or_matrix,
    or,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[false, false, true, true, true, false]),
    (&[2, 3], &[true, false, true, true, true, false])
);

// scalar

test_logical_op!(
    test_or_scalar_true_true,
    or,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[true])
);

test_logical_op!(
    test_or_scalar_true_false,
    or,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[true])
);

test_logical_op!(
    test_or_scalar_false_true,
    or,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[true])
);

test_logical_op!(
    test_or_scalar_false_false,
    or,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[false])
);

// broadcast

test_logical_op!(
    test_or_broadcast_multi_expand,
    or,
    (&[3, 1], &[true, false, true]),
    (&[1, 4], &[true, false, true, false]),
    (
        &[3, 4],
        &[
            true, true, true, true, true, false, true, false, true, true, true, true
        ]
    )
);

test_logical_op!(
    test_or_broadcast_expand,
    or,
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
            true, true, true, true, true, true, true, true, true, false, true, true
        ]
    )
);

test_logical_op!(
    test_or_broadcast_trailing,
    or,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[3], &[true, false, true]),
    (&[2, 3], &[true, false, true, true, true, true])
);

test_logical_op!(
    test_or_broadcast_scalar_false,
    or,
    (&[4], &[true, true, false, false]),
    (&[] as &[usize], &[false]),
    (&[4], &[true, true, false, false])
);

test_logical_op!(
    test_or_broadcast_scalar_true,
    or,
    (&[4], &[true, true, false, false]),
    (&[] as &[usize], &[true]),
    (&[4], &[true, true, true, true])
);

test_logical_op!(
    test_or_broadcast_scalar_reverse_false,
    or,
    (&[] as &[usize], &[false]),
    (&[4], &[true, false, true, false]),
    (&[4], &[true, false, true, false])
);

test_logical_op!(
    test_or_broadcast_scalar_reverse_true,
    or,
    (&[] as &[usize], &[true]),
    (&[4], &[true, false, true, false]),
    (&[4], &[true, true, true, true])
);

// error

#[test]
fn test_or_error_incompatible_shapes() {
    use xnn::{Context, Tensor};
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_shape_slice(&ctx, &[3], &[true, false, true]).unwrap();
    let b = Tensor::<bool>::from_shape_slice(&ctx, &[4], &[true, false, true, false]).unwrap();
    assert!(a.or(&b).is_err());
}
