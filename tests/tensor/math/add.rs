//! Tests for `Tensor::add` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_add_f32_vector,
    add,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[5.0, 6.0, 7.0, 8.0]),
    (&[4], &[6.0, 8.0, 10.0, 12.0])
);

test_arithmetic_op_integer!(
    test_add_i32_vector,
    add,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[11, 22, 33, 44])
);

test_arithmetic_op_integer!(
    test_add_u32_vector,
    add,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[11, 22, 33, 44])
);

// matrix

test_arithmetic_op_float!(
    test_add_f32_matrix,
    add,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[2, 3], &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0])
);

test_arithmetic_op_integer!(
    test_add_i32_matrix,
    add,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[11, 22, 33, 44, 55, 66])
);

test_arithmetic_op_integer!(
    test_add_u32_matrix,
    add,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[11, 22, 33, 44, 55, 66])
);

// scalar

test_arithmetic_op_float!(
    test_add_f32_scalar,
    add,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[8.0])
);

test_arithmetic_op_integer!(
    test_add_i32_scalar,
    add,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[8])
);

test_arithmetic_op_integer!(
    test_add_u32_scalar,
    add,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[8])
);

// broadcast

test_arithmetic_op_float!(
    test_add_f32_broadcast_multi_expand,
    add,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[10.0, 20.0, 30.0]),
    (
        &[2, 3, 4],
        &[
            11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0, 32.0, 33.0, 34.0, 15.0, 16.0,
            17.0, 18.0, 25.0, 26.0, 27.0, 28.0, 35.0, 36.0, 37.0, 38.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_add_i32_broadcast_multi_expand,
    add,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[10, 20, 30]),
    (
        &[2, 3, 4],
        &[
            11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 15, 16, 17, 18, 25, 26, 27, 28, 35, 36,
            37, 38
        ]
    )
);

test_arithmetic_op_integer!(
    test_add_u32_broadcast_multi_expand,
    add,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[10, 20, 30]),
    (
        &[2, 3, 4],
        &[
            11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 15, 16, 17, 18, 25, 26, 27, 28, 35, 36,
            37, 38
        ]
    )
);

test_arithmetic_op_float!(
    test_add_f32_broadcast_expand,
    add,
    f32,
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
    (
        &[3, 4],
        &[
            11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_add_i32_broadcast_expand,
    add,
    i32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[10, 20, 30, 40]),
    (&[3, 4], &[11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43])
);

test_arithmetic_op_integer!(
    test_add_u32_broadcast_expand,
    add,
    u32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[10, 20, 30, 40]),
    (&[3, 4], &[11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43])
);

test_arithmetic_op_float!(
    test_add_f32_broadcast_trailing,
    add,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[3], &[10.0, 20.0, 30.0]),
    (&[2, 3], &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
);

test_arithmetic_op_integer!(
    test_add_i32_broadcast_trailing,
    add,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[11, 22, 33, 14, 25, 36])
);

test_arithmetic_op_integer!(
    test_add_u32_broadcast_trailing,
    add,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[11, 22, 33, 14, 25, 36])
);

test_arithmetic_op_float!(
    test_add_f32_broadcast_scalar,
    add,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[] as &[usize], &[10.0]),
    (&[4], &[11.0, 12.0, 13.0, 14.0])
);

test_arithmetic_op_integer!(
    test_add_i32_broadcast_scalar,
    add,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[10]),
    (&[4], &[11, 12, 13, 14])
);

test_arithmetic_op_integer!(
    test_add_u32_broadcast_scalar,
    add,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[10]),
    (&[4], &[11, 12, 13, 14])
);

test_arithmetic_op_float!(
    test_add_f32_broadcast_scalar_reverse,
    add,
    f32,
    (&[] as &[usize], &[10.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[11.0, 12.0, 13.0, 14.0])
);

test_arithmetic_op_integer!(
    test_add_i32_broadcast_scalar_reverse,
    add,
    i32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[11, 12, 13, 14])
);

test_arithmetic_op_integer!(
    test_add_u32_broadcast_scalar_reverse,
    add,
    u32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[11, 12, 13, 14])
);

// error

#[test]
fn test_add_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.add(&b).is_err());
}
