//! Tests for `Tensor::ne` operation.

use xnn::{Context, Tensor};

use super::test_comparison_op;

// vector

test_comparison_op!(
    test_ne_f32_vector,
    ne,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[1.0, 5.0, 3.0, 8.0]),
    (&[4], &[false, true, false, true])
);

test_comparison_op!(
    test_ne_i32_vector,
    ne,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[false, true, false, true])
);

test_comparison_op!(
    test_ne_u32_vector,
    ne,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[1, 5, 3, 8]),
    (&[4], &[false, true, false, true])
);

// matrix

test_comparison_op!(
    test_ne_f32_matrix,
    ne,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[1.0, 5.0, 3.0, 8.0, 5.0, 9.0]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_ne_i32_matrix,
    ne,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[1, 5, 3, 8, 5, 9]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_ne_u32_matrix,
    ne,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[1, 5, 3, 8, 5, 9]),
    (&[2, 3], &[false, true, false, true, false, true])
);

// scalar

test_comparison_op!(
    test_ne_f32_scalar,
    ne,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[false])
);

test_comparison_op!(
    test_ne_i32_scalar,
    ne,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[true])
);

test_comparison_op!(
    test_ne_u32_scalar,
    ne,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[false])
);

// broadcast

test_comparison_op!(
    test_ne_f32_broadcast_multi_expand,
    ne,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[1.0, 5.0, 9.0]),
    (
        &[2, 3, 4],
        &[
            false, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, false, true, true, true, true, true, true, true
        ]
    )
);

test_comparison_op!(
    test_ne_i32_broadcast_multi_expand,
    ne,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[1, 5, 9]),
    (
        &[2, 3, 4],
        &[
            false, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, false, true, true, true, true, true, true, true
        ]
    )
);

test_comparison_op!(
    test_ne_u32_broadcast_multi_expand,
    ne,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[1, 5, 9]),
    (
        &[2, 3, 4],
        &[
            false, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, false, true, true, true, true, true, true, true
        ]
    )
);

test_comparison_op!(
    test_ne_f32_broadcast_expand,
    ne,
    f32,
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1, 4], &[1.0, 2.0, 3.0, 4.0]),
    (
        &[3, 4],
        &[
            false, true, true, true, true, false, true, true, true, true, false, true
        ]
    )
);

test_comparison_op!(
    test_ne_i32_broadcast_expand,
    ne,
    i32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[1, 2, 3, 4]),
    (
        &[3, 4],
        &[
            false, true, true, true, true, false, true, true, true, true, false, true
        ]
    )
);

test_comparison_op!(
    test_ne_u32_broadcast_expand,
    ne,
    u32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[1, 2, 3, 4]),
    (
        &[3, 4],
        &[
            false, true, true, true, true, false, true, true, true, true, false, true
        ]
    )
);

test_comparison_op!(
    test_ne_f32_broadcast_trailing,
    ne,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[3], &[1.0, 5.0, 3.0]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_ne_i32_broadcast_trailing,
    ne,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[1, 5, 3]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_ne_u32_broadcast_trailing,
    ne,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[1, 5, 3]),
    (&[2, 3], &[false, true, false, true, false, true])
);

test_comparison_op!(
    test_ne_f32_broadcast_scalar,
    ne,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[] as &[usize], &[2.0]),
    (&[4], &[true, false, true, true])
);

test_comparison_op!(
    test_ne_i32_broadcast_scalar,
    ne,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[2]),
    (&[4], &[true, false, true, true])
);

test_comparison_op!(
    test_ne_u32_broadcast_scalar,
    ne,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[2]),
    (&[4], &[true, false, true, true])
);

test_comparison_op!(
    test_ne_f32_broadcast_scalar_reverse,
    ne,
    f32,
    (&[] as &[usize], &[3.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[true, true, false, true])
);

test_comparison_op!(
    test_ne_i32_broadcast_scalar_reverse,
    ne,
    i32,
    (&[] as &[usize], &[3]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[true, true, false, true])
);

test_comparison_op!(
    test_ne_u32_broadcast_scalar_reverse,
    ne,
    u32,
    (&[] as &[usize], &[3]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[true, true, false, true])
);

// error

#[test]
fn test_ne_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.ne(&b).is_err());
}
