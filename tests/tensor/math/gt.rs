//! Tests for `Tensor::gt` operation.

use xnn::{Context, Tensor};

use super::test_comparison_op;

// vector

test_comparison_op!(
    test_gt_f32_vector,
    gt,
    f32,
    (&[4], &[1.0, 6.0, 3.0, 8.0]),
    (&[4], &[2.0, 5.0, 4.0, 4.0]),
    (&[4], &[false, true, false, true])
);

test_comparison_op!(
    test_gt_i32_vector,
    gt,
    i32,
    (&[4], &[1, 6, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[false, true, false, true])
);

test_comparison_op!(
    test_gt_u32_vector,
    gt,
    u32,
    (&[4], &[1, 6, 3, 8]),
    (&[4], &[2, 5, 4, 4]),
    (&[4], &[false, true, false, true])
);

// matrix

test_comparison_op!(
    test_gt_f32_matrix,
    gt,
    f32,
    (&[2, 3], &[1.0, 6.0, 3.0, 8.0, 5.0, 10.0]),
    (&[2, 3], &[2.0, 5.0, 4.0, 4.0, 6.0, 5.0]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_gt_i32_matrix,
    gt,
    i32,
    (&[2, 3], &[1, 6, 3, 8, 5, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_gt_u32_matrix,
    gt,
    u32,
    (&[2, 3], &[1, 6, 3, 8, 5, 10]),
    (&[2, 3], &[2, 5, 4, 4, 6, 5]),
    (&[2, 3], &[false, true, false, true, false, true])
);

// scalar

test_comparison_op!(
    test_gt_f32_scalar,
    gt,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[true])
);

test_comparison_op!(
    test_gt_i32_scalar,
    gt,
    i32,
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[false])
);

test_comparison_op!(
    test_gt_u32_scalar,
    gt,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[false])
);

// broadcast

test_comparison_op!(
    test_gt_f32_broadcast_multi_expand,
    gt,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[2.0, 4.0, 6.0]),
    (
        &[2, 3, 4],
        &[
            false, false, true, true, false, false, false, false, false, false, false, false, true,
            true, true, true, true, true, true, true, false, false, true, true
        ]
    )
);

test_comparison_op!(
    test_gt_i32_broadcast_multi_expand,
    gt,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            false, false, true, true, false, false, false, false, false, false, false, false, true,
            true, true, true, true, true, true, true, false, false, true, true
        ]
    )
);

test_comparison_op!(
    test_gt_u32_broadcast_multi_expand,
    gt,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[2, 4, 6]),
    (
        &[2, 3, 4],
        &[
            false, false, true, true, false, false, false, false, false, false, false, false, true,
            true, true, true, true, true, true, true, false, false, true, true
        ]
    )
);

test_comparison_op!(
    test_gt_f32_broadcast_expand,
    gt,
    f32,
    (&[3, 1], &[1.0, 3.0, 5.0]),
    (&[1, 4], &[0.0, 2.0, 4.0, 6.0]),
    (
        &[3, 4],
        &[
            true, false, false, false, true, true, false, false, true, true, true, false
        ]
    )
);

test_comparison_op!(
    test_gt_i32_broadcast_expand,
    gt,
    i32,
    (&[3, 1], &[1, 3, 5]),
    (&[1, 4], &[0, 2, 4, 6]),
    (
        &[3, 4],
        &[
            true, false, false, false, true, true, false, false, true, true, true, false
        ]
    )
);

test_comparison_op!(
    test_gt_u32_broadcast_expand,
    gt,
    u32,
    (&[3, 1], &[1, 3, 5]),
    (&[1, 4], &[0, 2, 4, 6]),
    (
        &[3, 4],
        &[
            true, false, false, false, true, true, false, false, true, true, true, false
        ]
    )
);

test_comparison_op!(
    test_gt_f32_broadcast_trailing,
    gt,
    f32,
    (&[2, 3], &[5.0, 2.0, 7.0, 1.0, 8.0, 3.0]),
    (&[3], &[3.0, 4.0, 5.0]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_gt_i32_broadcast_trailing,
    gt,
    i32,
    (&[2, 3], &[5, 2, 7, 1, 8, 3]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_gt_u32_broadcast_trailing,
    gt,
    u32,
    (&[2, 3], &[5, 2, 7, 1, 8, 3]),
    (&[3], &[3, 4, 5]),
    (&[2, 3], &[true, false, true, false, true, false])
);

test_comparison_op!(
    test_gt_f32_broadcast_scalar,
    gt,
    f32,
    (&[4], &[1.0, 3.0, 5.0, 7.0]),
    (&[] as &[usize], &[4.0]),
    (&[4], &[false, false, true, true])
);

test_comparison_op!(
    test_gt_i32_broadcast_scalar,
    gt,
    i32,
    (&[4], &[1, 3, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[false, false, true, true])
);

test_comparison_op!(
    test_gt_u32_broadcast_scalar,
    gt,
    u32,
    (&[4], &[1, 3, 5, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[false, false, true, true])
);

test_comparison_op!(
    test_gt_f32_broadcast_scalar_reverse,
    gt,
    f32,
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 3.0, 5.0, 7.0]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_gt_i32_broadcast_scalar_reverse,
    gt,
    i32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 3, 5, 7]),
    (&[4], &[true, true, false, false])
);

test_comparison_op!(
    test_gt_u32_broadcast_scalar_reverse,
    gt,
    u32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 3, 5, 7]),
    (&[4], &[true, true, false, false])
);

// error

#[test]
fn test_gt_error_incompatible_shapes() {
    let ctx = Context::new().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.gt(&b).is_err());
}
