//! Tests for `Tensor::sub` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_sub_f32_vector,
    sub,
    f32,
    (&[4], &[5.0, 6.0, 7.0, 8.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[4.0, 4.0, 4.0, 4.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_vector,
    sub,
    i32,
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[9, 18, 27, 36])
);

test_arithmetic_op_integer!(
    test_sub_u32_vector,
    sub,
    u32,
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[9, 18, 27, 36])
);

// matrix

test_arithmetic_op_float!(
    test_sub_f32_matrix,
    sub,
    f32,
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[9.0, 18.0, 27.0, 36.0, 45.0, 54.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_matrix,
    sub,
    i32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[9, 18, 27, 36, 45, 54])
);

test_arithmetic_op_integer!(
    test_sub_u32_matrix,
    sub,
    u32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[9, 18, 27, 36, 45, 54])
);

// scalar

test_arithmetic_op_float!(
    test_sub_f32_scalar,
    sub,
    f32,
    (&[] as &[usize], &[10.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[7.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_scalar,
    sub,
    i32,
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[7])
);

test_arithmetic_op_integer!(
    test_sub_u32_scalar,
    sub,
    u32,
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[7])
);

// broadcast

test_arithmetic_op_float!(
    test_sub_f32_broadcast_multi_expand,
    sub,
    f32,
    (
        &[2, 1, 4],
        &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    ),
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (
        &[2, 3, 4],
        &[
            10.0, 11.0, 12.0, 13.0, 9.0, 10.0, 11.0, 12.0, 8.0, 9.0, 10.0, 11.0, 14.0, 15.0, 16.0,
            17.0, 13.0, 14.0, 15.0, 16.0, 12.0, 13.0, 14.0, 15.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_sub_i32_broadcast_multi_expand,
    sub,
    i32,
    (&[2, 1, 4], &[11, 12, 13, 14, 15, 16, 17, 18]),
    (&[3, 1], &[1, 2, 3]),
    (
        &[2, 3, 4],
        &[
            10, 11, 12, 13, 9, 10, 11, 12, 8, 9, 10, 11, 14, 15, 16, 17, 13, 14, 15, 16, 12, 13,
            14, 15
        ]
    )
);

test_arithmetic_op_integer!(
    test_sub_u32_broadcast_multi_expand,
    sub,
    u32,
    (&[2, 1, 4], &[11, 12, 13, 14, 15, 16, 17, 18]),
    (&[3, 1], &[1, 2, 3]),
    (
        &[2, 3, 4],
        &[
            10, 11, 12, 13, 9, 10, 11, 12, 8, 9, 10, 11, 14, 15, 16, 17, 13, 14, 15, 16, 12, 13,
            14, 15
        ]
    )
);

test_arithmetic_op_float!(
    test_sub_f32_broadcast_expand,
    sub,
    f32,
    (&[3, 1], &[11.0, 12.0, 13.0]),
    (&[1, 4], &[1.0, 2.0, 3.0, 4.0]),
    (
        &[3, 4],
        &[
            10.0, 9.0, 8.0, 7.0, 11.0, 10.0, 9.0, 8.0, 12.0, 11.0, 10.0, 9.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_sub_i32_broadcast_expand,
    sub,
    i32,
    (&[3, 1], &[11, 12, 13]),
    (&[1, 4], &[1, 2, 3, 4]),
    (&[3, 4], &[10, 9, 8, 7, 11, 10, 9, 8, 12, 11, 10, 9])
);

test_arithmetic_op_integer!(
    test_sub_u32_broadcast_expand,
    sub,
    u32,
    (&[3, 1], &[11, 12, 13]),
    (&[1, 4], &[1, 2, 3, 4]),
    (&[3, 4], &[10, 9, 8, 7, 11, 10, 9, 8, 12, 11, 10, 9])
);

test_arithmetic_op_float!(
    test_sub_f32_broadcast_trailing,
    sub,
    f32,
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[3], &[10.0, 20.0, 30.0]),
    (&[2, 3], &[0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_broadcast_trailing,
    sub,
    i32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[0, 0, 0, 30, 30, 30])
);

test_arithmetic_op_integer!(
    test_sub_u32_broadcast_trailing,
    sub,
    u32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[0, 0, 0, 30, 30, 30])
);

test_arithmetic_op_float!(
    test_sub_f32_broadcast_scalar,
    sub,
    f32,
    (&[4], &[5.0, 6.0, 7.0, 8.0]),
    (&[] as &[usize], &[10.0]),
    (&[4], &[-5.0, -4.0, -3.0, -2.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_broadcast_scalar,
    sub,
    i32,
    (&[4], &[15, 16, 17, 18]),
    (&[] as &[usize], &[10]),
    (&[4], &[5, 6, 7, 8])
);

test_arithmetic_op_integer!(
    test_sub_u32_broadcast_scalar,
    sub,
    u32,
    (&[4], &[15, 16, 17, 18]),
    (&[] as &[usize], &[10]),
    (&[4], &[5, 6, 7, 8])
);

test_arithmetic_op_float!(
    test_sub_f32_broadcast_scalar_reverse,
    sub,
    f32,
    (&[] as &[usize], &[10.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[9.0, 8.0, 7.0, 6.0])
);

test_arithmetic_op_integer!(
    test_sub_i32_broadcast_scalar_reverse,
    sub,
    i32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[9, 8, 7, 6])
);

test_arithmetic_op_integer!(
    test_sub_u32_broadcast_scalar_reverse,
    sub,
    u32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[9, 8, 7, 6])
);

// error

#[test]
fn test_sub_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.sub(&b).is_err());
}
