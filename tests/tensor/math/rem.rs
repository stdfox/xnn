//! Tests for `Tensor::rem` operation.

use xnn::{Context, Tensor};

use super::test_arithmetic_op_integer;

// vector

test_arithmetic_op_integer!(
    test_rem_i32_vector,
    rem,
    i32,
    (&[4], &[10, 23, 35, 47]),
    (&[4], &[3, 4, 5, 6]),
    (&[4], &[1, 3, 0, 5])
);

test_arithmetic_op_integer!(
    test_rem_u32_vector,
    rem,
    u32,
    (&[4], &[10, 23, 35, 47]),
    (&[4], &[3, 4, 5, 6]),
    (&[4], &[1, 3, 0, 5])
);

// matrix

test_arithmetic_op_integer!(
    test_rem_i32_matrix,
    rem,
    i32,
    (&[2, 3], &[10, 23, 35, 47, 58, 69]),
    (&[2, 3], &[3, 4, 5, 6, 7, 8]),
    (&[2, 3], &[1, 3, 0, 5, 2, 5])
);

test_arithmetic_op_integer!(
    test_rem_u32_matrix,
    rem,
    u32,
    (&[2, 3], &[10, 23, 35, 47, 58, 69]),
    (&[2, 3], &[3, 4, 5, 6, 7, 8]),
    (&[2, 3], &[1, 3, 0, 5, 2, 5])
);

// scalar

test_arithmetic_op_integer!(
    test_rem_i32_scalar,
    rem,
    i32,
    (&[] as &[usize], &[17]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[2])
);

test_arithmetic_op_integer!(
    test_rem_u32_scalar,
    rem,
    u32,
    (&[] as &[usize], &[17]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[2])
);

// broadcast

test_arithmetic_op_integer!(
    test_rem_i32_broadcast_multi_expand,
    rem,
    i32,
    (&[2, 1, 4], &[10, 23, 35, 47, 58, 69, 80, 91]),
    (&[3, 1], &[3, 7, 11]),
    (
        &[2, 3, 4],
        &[
            1, 2, 2, 2, 3, 2, 0, 5, 10, 1, 2, 3, 1, 0, 2, 1, 2, 6, 3, 0, 3, 3, 3, 3
        ]
    )
);

test_arithmetic_op_integer!(
    test_rem_u32_broadcast_multi_expand,
    rem,
    u32,
    (&[2, 1, 4], &[10, 23, 35, 47, 58, 69, 80, 91]),
    (&[3, 1], &[3, 7, 11]),
    (
        &[2, 3, 4],
        &[
            1, 2, 2, 2, 3, 2, 0, 5, 10, 1, 2, 3, 1, 0, 2, 1, 2, 6, 3, 0, 3, 3, 3, 3
        ]
    )
);

test_arithmetic_op_integer!(
    test_rem_i32_broadcast_expand,
    rem,
    i32,
    (&[3, 1], &[10, 23, 35]),
    (&[1, 4], &[3, 7, 11, 13]),
    (&[3, 4], &[1, 3, 10, 10, 2, 2, 1, 10, 2, 0, 2, 9])
);

test_arithmetic_op_integer!(
    test_rem_u32_broadcast_expand,
    rem,
    u32,
    (&[3, 1], &[10, 23, 35]),
    (&[1, 4], &[3, 7, 11, 13]),
    (&[3, 4], &[1, 3, 10, 10, 2, 2, 1, 10, 2, 0, 2, 9])
);

test_arithmetic_op_integer!(
    test_rem_i32_broadcast_trailing,
    rem,
    i32,
    (&[2, 3], &[10, 23, 35, 47, 58, 69]),
    (&[3], &[3, 7, 11]),
    (&[2, 3], &[1, 2, 2, 2, 2, 3])
);

test_arithmetic_op_integer!(
    test_rem_u32_broadcast_trailing,
    rem,
    u32,
    (&[2, 3], &[10, 23, 35, 47, 58, 69]),
    (&[3], &[3, 7, 11]),
    (&[2, 3], &[1, 2, 2, 2, 2, 3])
);

test_arithmetic_op_integer!(
    test_rem_i32_broadcast_scalar,
    rem,
    i32,
    (&[4], &[10, 23, 35, 47]),
    (&[] as &[usize], &[7]),
    (&[4], &[3, 2, 0, 5])
);

test_arithmetic_op_integer!(
    test_rem_u32_broadcast_scalar,
    rem,
    u32,
    (&[4], &[10, 23, 35, 47]),
    (&[] as &[usize], &[7]),
    (&[4], &[3, 2, 0, 5])
);

test_arithmetic_op_integer!(
    test_rem_i32_broadcast_scalar_reverse,
    rem,
    i32,
    (&[] as &[usize], &[100]),
    (&[4], &[3, 7, 11, 13]),
    (&[4], &[1, 2, 1, 9])
);

test_arithmetic_op_integer!(
    test_rem_u32_broadcast_scalar_reverse,
    rem,
    u32,
    (&[] as &[usize], &[100]),
    (&[4], &[3, 7, 11, 13]),
    (&[4], &[1, 2, 1, 9])
);

// error

#[test]
fn test_rem_error_incompatible_shapes() {
    let ctx = Context::new().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, &[1, 2, 3]).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, &[1, 2, 3, 4]).unwrap();
    assert!(a.rem(&b).is_err());
}
