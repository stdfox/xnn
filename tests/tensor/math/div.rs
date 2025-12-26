//! Tests for `Tensor::div` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_div_f32_vector,
    div,
    f32,
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[4], &[2.0, 4.0, 5.0, 8.0]),
    (&[4], &[5.0, 5.0, 6.0, 5.0])
);

test_arithmetic_op_integer!(
    test_div_i32_vector,
    div,
    i32,
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[2, 4, 5, 8]),
    (&[4], &[5, 5, 6, 5])
);

test_arithmetic_op_integer!(
    test_div_u32_vector,
    div,
    u32,
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[2, 4, 5, 8]),
    (&[4], &[5, 5, 6, 5])
);

// matrix

test_arithmetic_op_float!(
    test_div_f32_matrix,
    div,
    f32,
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[2, 3], &[2.0, 4.0, 5.0, 8.0, 10.0, 12.0]),
    (&[2, 3], &[5.0, 5.0, 6.0, 5.0, 5.0, 5.0])
);

test_arithmetic_op_integer!(
    test_div_i32_matrix,
    div,
    i32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[2, 4, 5, 8, 10, 12]),
    (&[2, 3], &[5, 5, 6, 5, 5, 5])
);

test_arithmetic_op_integer!(
    test_div_u32_matrix,
    div,
    u32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[2, 4, 5, 8, 10, 12]),
    (&[2, 3], &[5, 5, 6, 5, 5, 5])
);

// scalar

test_arithmetic_op_float!(
    test_div_f32_scalar,
    div,
    f32,
    (&[] as &[usize], &[10.0]),
    (&[] as &[usize], &[2.0]),
    (&[] as &[usize], &[5.0])
);

test_arithmetic_op_integer!(
    test_div_i32_scalar,
    div,
    i32,
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[2]),
    (&[] as &[usize], &[5])
);

test_arithmetic_op_integer!(
    test_div_u32_scalar,
    div,
    u32,
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[2]),
    (&[] as &[usize], &[5])
);

// broadcast

test_arithmetic_op_float!(
    test_div_f32_broadcast_multi_expand,
    div,
    f32,
    (
        &[2, 1, 4],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    ),
    (&[3, 1], &[2.0, 4.0, 5.0]),
    (
        &[2, 3, 4],
        &[
            5.0, 10.0, 15.0, 20.0, 2.5, 5.0, 7.5, 10.0, 2.0, 4.0, 6.0, 8.0, 25.0, 30.0, 35.0, 40.0,
            12.5, 15.0, 17.5, 20.0, 10.0, 12.0, 14.0, 16.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_div_i32_broadcast_multi_expand,
    div,
    i32,
    (&[2, 1, 4], &[10, 20, 30, 40, 50, 60, 70, 80]),
    (&[3, 1], &[2, 4, 5]),
    (
        &[2, 3, 4],
        &[
            5, 10, 15, 20, 2, 5, 7, 10, 2, 4, 6, 8, 25, 30, 35, 40, 12, 15, 17, 20, 10, 12, 14, 16
        ]
    )
);

test_arithmetic_op_integer!(
    test_div_u32_broadcast_multi_expand,
    div,
    u32,
    (&[2, 1, 4], &[10, 20, 30, 40, 50, 60, 70, 80]),
    (&[3, 1], &[2, 4, 5]),
    (
        &[2, 3, 4],
        &[
            5, 10, 15, 20, 2, 5, 7, 10, 2, 4, 6, 8, 25, 30, 35, 40, 12, 15, 17, 20, 10, 12, 14, 16
        ]
    )
);

test_arithmetic_op_float!(
    test_div_f32_broadcast_expand,
    div,
    f32,
    (&[3, 1], &[10.0, 20.0, 30.0]),
    (&[1, 4], &[1.0, 2.0, 5.0, 10.0]),
    (
        &[3, 4],
        &[
            10.0, 5.0, 2.0, 1.0, 20.0, 10.0, 4.0, 2.0, 30.0, 15.0, 6.0, 3.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_div_i32_broadcast_expand,
    div,
    i32,
    (&[3, 1], &[10, 20, 30]),
    (&[1, 4], &[1, 2, 5, 10]),
    (&[3, 4], &[10, 5, 2, 1, 20, 10, 4, 2, 30, 15, 6, 3])
);

test_arithmetic_op_integer!(
    test_div_u32_broadcast_expand,
    div,
    u32,
    (&[3, 1], &[10, 20, 30]),
    (&[1, 4], &[1, 2, 5, 10]),
    (&[3, 4], &[10, 5, 2, 1, 20, 10, 4, 2, 30, 15, 6, 3])
);

test_arithmetic_op_float!(
    test_div_f32_broadcast_trailing,
    div,
    f32,
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[3], &[2.0, 4.0, 5.0]),
    (&[2, 3], &[5.0, 5.0, 6.0, 20.0, 12.5, 12.0])
);

test_arithmetic_op_integer!(
    test_div_i32_broadcast_trailing,
    div,
    i32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[5, 5, 6, 20, 12, 12])
);

test_arithmetic_op_integer!(
    test_div_u32_broadcast_trailing,
    div,
    u32,
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[5, 5, 6, 20, 12, 12])
);

test_arithmetic_op_float!(
    test_div_f32_broadcast_scalar,
    div,
    f32,
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[] as &[usize], &[2.0]),
    (&[4], &[5.0, 10.0, 15.0, 20.0])
);

test_arithmetic_op_integer!(
    test_div_i32_broadcast_scalar,
    div,
    i32,
    (&[4], &[10, 20, 30, 40]),
    (&[] as &[usize], &[2]),
    (&[4], &[5, 10, 15, 20])
);

test_arithmetic_op_integer!(
    test_div_u32_broadcast_scalar,
    div,
    u32,
    (&[4], &[10, 20, 30, 40]),
    (&[] as &[usize], &[2]),
    (&[4], &[5, 10, 15, 20])
);

test_arithmetic_op_float!(
    test_div_f32_broadcast_scalar_reverse,
    div,
    f32,
    (&[] as &[usize], &[100.0]),
    (&[4], &[1.0, 2.0, 4.0, 5.0]),
    (&[4], &[100.0, 50.0, 25.0, 20.0])
);

test_arithmetic_op_integer!(
    test_div_i32_broadcast_scalar_reverse,
    div,
    i32,
    (&[] as &[usize], &[100]),
    (&[4], &[1, 2, 4, 5]),
    (&[4], &[100, 50, 25, 20])
);

test_arithmetic_op_integer!(
    test_div_u32_broadcast_scalar_reverse,
    div,
    u32,
    (&[] as &[usize], &[100]),
    (&[4], &[1, 2, 4, 5]),
    (&[4], &[100, 50, 25, 20])
);

// error

#[test]
fn test_div_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.div(&b).is_err());
}
