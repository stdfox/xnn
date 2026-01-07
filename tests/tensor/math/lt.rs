//! Tests for `Tensor::lt` operation.

use xnn::{Context, Tensor};

use super::test_comparison_op;

// vector

test_comparison_op!(
    test_lt_f32_vector,
    lt,
    f32,
    (&[4], &[1.0, 6.0, 3.0, 8.0]),
    (&[4], &[2.0, 5.0, 4.0, 4.0]),
    (&[4], &[true, false, true, false])
);

test_comparison_op!(
    test_lt_i32_vector,
    lt,
    i32,
    (&[4], &[1, 6, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[true, false, true, false])
);

test_comparison_op!(
    test_lt_u32_vector,
    lt,
    u32,
    (&[4], &[1, 6, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[true, false, true, false])
);

// matrix

test_comparison_op!(
    test_lt_f32_matrix,
    lt,
    f32,
    (&[2, 3], &[1.0, 6.0, 3.0, 8.0, 5.0, 10.0]),
    (&[2, 3], &[2.0, 5.0, 4.0, 4.0, 6.0, 5.0]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_lt_i32_matrix,
    lt,
    i32,
    (&[2, 3], &[1, 6, 3, 8, 5, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_lt_u32_matrix,
    lt,
    u32,
    (&[2, 3], &[1, 6, 3, 8, 5, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

// scalar

test_comparison_op!(
    test_lt_f32_scalar,
    lt,
    f32,
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[true])
);

test_comparison_op!(
    test_lt_i32_scalar,
    lt,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[false])
);

test_comparison_op!(
    test_lt_u32_scalar,
    lt,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[false])
);

// broadcast

test_comparison_op!(
    test_lt_f32_broadcast_multi_expand,
    lt,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[2.0, 4.0, 6.0]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, true, true, true, false, true, true, true, true, false,
            false, false, false, false, false, false, false, true, false, false, false
        ]
    )
);

test_comparison_op!(
    test_lt_i32_broadcast_multi_expand,
    lt,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, true, true, true, false, true, true, true, true, false,
            false, false, false, false, false, false, false, true, false, false, false
        ]
    )
);

test_comparison_op!(
    test_lt_u32_broadcast_multi_expand,
    lt,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, true, true, true, false, true, true, true, true, false,
            false, false, false, false, false, false, false, true, false, false, false
        ]
    )
);

test_comparison_op!(
    test_lt_f32_broadcast_expand,
    lt,
    f32,
    (&[3, 1], &[1.0, 3.0, 5.0]),
    (&[1, 4], &[0.0, 2.0, 4.0, 6.0]),
    (
        &[3, 4],
        &[
            false, true, true, true, false, false, true, true, false, false, false, true
        ]
    )
);

test_comparison_op!(
    test_lt_i32_broadcast_expand,
    lt,
    i32,
    (&[3, 1], &[1, 3, 5]),
    (&[1, 4], &[0, 2, 4, 6]),
    (
        &[3, 4],
        &[
            false, true, true, true, false, false, true, true, false, false, false, true
        ]
    )
);

test_comparison_op!(
    test_lt_u32_broadcast_expand,
    lt,
    u32,
    (&[3, 1], &[1, 3, 5]),
    (&[1, 4], &[0, 2, 4, 6]),
    (
        &[3, 4],
        &[
            false, true, true, true, false, false, true, true, false, false, false, true
        ]
    )
);

test_comparison_op!(
    test_lt_f32_broadcast_trailing,
    lt,
    f32,
    (&[2, 3], &[1.0, 5.0, 3.0, 6.0, 2.0, 8.0]),
    (&[3], &[3.0, 4.0, 5.0]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_lt_i32_broadcast_trailing,
    lt,
    i32,
    (&[2, 3], &[1, 5, 3, 6, 2, 8]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_lt_u32_broadcast_trailing,
    lt,
    u32,
    (&[2, 3], &[1, 5, 3, 6, 2, 8]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_lt_f32_broadcast_scalar,
    lt,
    f32,
    (&[4], &[1.0, 3.0, 5.0, 7.0]),
    (&[] as &[usize], &[4.0]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_lt_i32_broadcast_scalar,
    lt,
    i32,
    (&[4], &[1, 3, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_lt_u32_broadcast_scalar,
    lt,
    u32,
    (&[4], &[1, 3, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_lt_f32_broadcast_scalar_reverse,
    lt,
    f32,
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 3.0, 5.0, 7.0]),
    (&[4], &[false, false, true, true])
);

test_comparison_op!(
    test_lt_i32_broadcast_scalar_reverse,
    lt,
    i32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 3, 5, 7]),
    (&[4], &[false, false, true, true])
);

test_comparison_op!(
    test_lt_u32_broadcast_scalar_reverse,
    lt,
    u32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 3, 5, 7]),
    (&[4], &[false, false, true, true])
);

// error

#[test]
fn test_lt_error_incompatible_shapes() {
    let ctx = Context::new().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.lt(&b).is_err());
}
