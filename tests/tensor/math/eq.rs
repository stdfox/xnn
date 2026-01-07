//! Tests for `Tensor::eq` operation.

use xnn::{Context, Tensor};

use super::test_comparison_op;

// vector

test_comparison_op!(
    test_eq_f32_vector,
    eq,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[1.0, 5.0, 3.0, 8.0]),
    (&[4], &[true, false, true, false])
);

test_comparison_op!(
    test_eq_i32_vector,
    eq,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[true, false, true, false])
);

test_comparison_op!(
    test_eq_u32_vector,
    eq,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[true, false, true, false])
);

// matrix

test_comparison_op!(
    test_eq_f32_matrix,
    eq,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[1.0, 5.0, 3.0, 8.0, 5.0, 9.0]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_eq_i32_matrix,
    eq,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[1, 5, 3, 8, 5, 9]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_eq_u32_matrix,
    eq,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[1, 5, 3, 8, 5, 9]),
    (&[2, 3], &[true, false, true, false, true, false])
);

// scalar

test_comparison_op!(
    test_eq_f32_scalar,
    eq,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[true])
);

test_comparison_op!(
    test_eq_i32_scalar,
    eq,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[false])
);

test_comparison_op!(
    test_eq_u32_scalar,
    eq,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[true])
);

// broadcast

test_comparison_op!(
    test_eq_f32_broadcast_multi_expand,
    eq,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[1.0, 5.0, 9.0]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, false, false, false, false, false, false, false
        ]
    )
);

test_comparison_op!(
    test_eq_i32_broadcast_multi_expand,
    eq,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[1, 5, 9]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, false, false, false, false, false, false, false
        ]
    )
);

test_comparison_op!(
    test_eq_u32_broadcast_multi_expand,
    eq,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[1, 5, 9]),
    (
        &[2, 3, 4],
        &[
            true, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, false, false, false, false, false, false, false
        ]
    )
);

test_comparison_op!(
    test_eq_f32_broadcast_expand,
    eq,
    f32,
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1, 4], &[1.0, 2.0, 3.0, 4.0]),
    (
        &[3, 4],
        &[
            true, false, false, false, false, true, false, false, false, false, true, false
        ]
    )
);

test_comparison_op!(
    test_eq_i32_broadcast_expand,
    eq,
    i32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[1, 2, 3, 4]),
    (
        &[3, 4],
        &[
            true, false, false, false, false, true, false, false, false, false, true, false
        ]
    )
);

test_comparison_op!(
    test_eq_u32_broadcast_expand,
    eq,
    u32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[1, 2, 3, 4]),
    (
        &[3, 4],
        &[
            true, false, false, false, false, true, false, false, false, false, true, false
        ]
    )
);

test_comparison_op!(
    test_eq_f32_broadcast_trailing,
    eq,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[3], &[1.0, 5.0, 3.0]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_eq_i32_broadcast_trailing,
    eq,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[1, 5, 3]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_eq_u32_broadcast_trailing,
    eq,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[1, 5, 3]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_eq_f32_broadcast_scalar,
    eq,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[] as &[usize], &[2.0]),
    (&[4], &[false, true, false, false])
);

test_comparison_op!(
    test_eq_i32_broadcast_scalar,
    eq,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[2]),
    (&[4], &[false, true, false, false])
);

test_comparison_op!(
    test_eq_u32_broadcast_scalar,
    eq,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[2]),
    (&[4], &[false, true, false, false])
);

test_comparison_op!(
    test_eq_f32_broadcast_scalar_reverse,
    eq,
    f32,
    (&[] as &[usize], &[3.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[false, false, true, false])
);

test_comparison_op!(
    test_eq_i32_broadcast_scalar_reverse,
    eq,
    i32,
    (&[] as &[usize], &[3]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[false, false, true, false])
);

test_comparison_op!(
    test_eq_u32_broadcast_scalar_reverse,
    eq,
    u32,
    (&[] as &[usize], &[3]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[false, false, true, false])
);

// error

#[test]
fn test_eq_error_incompatible_shapes() {
    let ctx = Context::new().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.eq(&b).is_err());
}
