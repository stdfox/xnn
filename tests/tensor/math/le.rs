//! Tests for `Tensor::le` operation.

use xnn::{Context, Tensor};

use super::test_comparison_op;

// vector

test_comparison_op!(
    test_le_f32_vector,
    le,
    f32,
    (&[4], &[1.0, 5.0, 3.0, 8.0]),
    (&[4], &[2.0, 5.0, 4.0, 4.0]),
    (&[4], &[true, true, true, false])
);

test_comparison_op!(
    test_le_i32_vector,
    le,
    i32,
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[true, true, true, false])
);

test_comparison_op!(
    test_le_u32_vector,
    le,
    u32,
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[true, true, true, false])
);

// matrix

test_comparison_op!(
    test_le_f32_matrix,
    le,
    f32,
    (&[2, 3], &[1.0, 5.0, 3.0, 8.0, 6.0, 10.0]),
    (&[2, 3], &[2.0, 5.0, 4.0, 4.0, 6.0, 5.0]),
    (&[2, 3], &[true, true, true, false, true, false])
);

test_comparison_op!(
    test_le_i32_matrix,
    le,
    i32,
    (&[2, 3], &[1, 5, 3, 8, 6, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[true, true, true, false, true, false])
);

test_comparison_op!(
    test_le_u32_matrix,
    le,
    u32,
    (&[2, 3], &[1, 5, 3, 8, 6, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[true, true, true, false, true, false])
);

// scalar

test_comparison_op!(
    test_le_f32_scalar,
    le,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[true])
);

test_comparison_op!(
    test_le_i32_scalar,
    le,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[false])
);

test_comparison_op!(
    test_le_u32_scalar,
    le,
    u32,
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[true])
);

// broadcast

test_comparison_op!(
    test_le_f32_broadcast_multi_expand,
    le,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[2.0, 4.0, 6.0]),
    (
        &[2, 3, 4],
        &[
            true, true, false, false, true, true, true, true, true, true, true, true, false, false,
            false, false, false, false, false, false, true, true, false, false
        ]
    )
);

test_comparison_op!(
    test_le_i32_broadcast_multi_expand,
    le,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            true, true, false, false, true, true, true, true, true, true, true, true, false, false,
            false, false, false, false, false, false, true, true, false, false
        ]
    )
);

test_comparison_op!(
    test_le_u32_broadcast_multi_expand,
    le,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            true, true, false, false, true, true, true, true, true, true, true, true, false, false,
            false, false, false, false, false, false, true, true, false, false
        ]
    )
);

test_comparison_op!(
    test_le_f32_broadcast_expand,
    le,
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
    test_le_i32_broadcast_expand,
    le,
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
    test_le_u32_broadcast_expand,
    le,
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
    test_le_f32_broadcast_trailing,
    le,
    f32,
    (&[2, 3], &[1.0, 4.0, 3.0, 3.0, 2.0, 5.0]),
    (&[3], &[3.0, 4.0, 5.0]),
    (&[2, 3], &[true, true, true, true, true, true])
);

test_comparison_op!(
    test_le_i32_broadcast_trailing,
    le,
    i32,
    (&[2, 3], &[1, 4, 3, 3, 2, 5]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, true, true, true, true, true])
);

test_comparison_op!(
    test_le_u32_broadcast_trailing,
    le,
    u32,
    (&[2, 3], &[1, 4, 3, 3, 2, 5]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, true, true, true, true, true])
);

test_comparison_op!(
    test_le_f32_broadcast_scalar,
    le,
    f32,
    (&[4], &[1.0, 4.0, 5.0, 7.0]),
    (&[] as &[usize], &[4.0]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_le_i32_broadcast_scalar,
    le,
    i32,
    (&[4], &[1, 4, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_le_u32_broadcast_scalar,
    le,
    u32,
    (&[4], &[1, 4, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_le_f32_broadcast_scalar_reverse,
    le,
    f32,
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 4.0, 5.0, 7.0]),
    (&[4], &[false, true, true, true])
);

test_comparison_op!(
    test_le_i32_broadcast_scalar_reverse,
    le,
    i32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 4, 5, 7]),
    (&[4], &[false, true, true, true])
);

test_comparison_op!(
    test_le_u32_broadcast_scalar_reverse,
    le,
    u32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 4, 5, 7]),
    (&[4], &[false, true, true, true])
);

// error

#[test]
fn test_le_error_incompatible_shapes() {
    let ctx = Context::new().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.le(&b).is_err());
}
