//! Tests for `Tensor::min` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_min_f32_vector,
    min,
    f32,
    (&[4], &[1.0, 6.0, 3.0, 8.0]),
    (&[4], &[5.0, 2.0, 7.0, 4.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0])
);

test_arithmetic_op_integer!(
    test_min_i32_vector,
    min,
    i32,
    (&[4], &[1, 20, 3, 40]),
    (&[4], &[10, 2, 30, 4]),
    (&[4], &[1, 2, 3, 4])
);

test_arithmetic_op_integer!(
    test_min_u32_vector,
    min,
    u32,
    (&[4], &[1, 20, 3, 40]),
    (&[4], &[10, 2, 30, 4]),
    (&[4], &[1, 2, 3, 4])
);

// matrix

test_arithmetic_op_float!(
    test_min_f32_matrix,
    min,
    f32,
    (&[2, 3], &[1.0, 20.0, 3.0, 40.0, 5.0, 60.0]),
    (&[2, 3], &[10.0, 2.0, 30.0, 4.0, 50.0, 6.0]),
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
);

test_arithmetic_op_integer!(
    test_min_i32_matrix,
    min,
    i32,
    (&[2, 3], &[1, 20, 3, 40, 5, 60]),
    (&[2, 3], &[10, 2, 30, 4, 50, 6]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6])
);

test_arithmetic_op_integer!(
    test_min_u32_matrix,
    min,
    u32,
    (&[2, 3], &[1, 20, 3, 40, 5, 60]),
    (&[2, 3], &[10, 2, 30, 4, 50, 6]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6])
);

// scalar

test_arithmetic_op_float!(
    test_min_f32_scalar,
    min,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[3.0])
);

test_arithmetic_op_integer!(
    test_min_i32_scalar,
    min,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[8]),
    (&[] as &[usize], &[5])
);

test_arithmetic_op_integer!(
    test_min_u32_scalar,
    min,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[3])
);

// broadcast

test_arithmetic_op_float!(
    test_min_f32_broadcast_multi_expand,
    min,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[3.0, 5.0, 7.0]),
    (
        &[2, 3, 4],
        &[
            1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 7.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_min_i32_broadcast_multi_expand,
    min,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[3, 5, 7]),
    (
        &[2, 3, 4],
        &[
            1, 2, 3, 3, 1, 2, 3, 4, 1, 2, 3, 4, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 7, 7
        ]
    )
);

test_arithmetic_op_integer!(
    test_min_u32_broadcast_multi_expand,
    min,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[3, 5, 7]),
    (
        &[2, 3, 4],
        &[
            1, 2, 3, 3, 1, 2, 3, 4, 1, 2, 3, 4, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 7, 7
        ]
    )
);

test_arithmetic_op_float!(
    test_min_f32_broadcast_expand,
    min,
    f32,
    (&[3, 1], &[1.0, 5.0, 3.0]),
    (&[1, 4], &[2.0, 4.0, 6.0, 0.0]),
    (
        &[3, 4],
        &[1.0, 1.0, 1.0, 0.0, 2.0, 4.0, 5.0, 0.0, 2.0, 3.0, 3.0, 0.0]
    )
);

test_arithmetic_op_integer!(
    test_min_i32_broadcast_expand,
    min,
    i32,
    (&[3, 1], &[1, 5, 3]),
    (&[1, 4], &[2, 4, 6, 0]),
    (&[3, 4], &[1, 1, 1, 0, 2, 4, 5, 0, 2, 3, 3, 0])
);

test_arithmetic_op_integer!(
    test_min_u32_broadcast_expand,
    min,
    u32,
    (&[3, 1], &[1, 5, 3]),
    (&[1, 4], &[2, 4, 6, 0]),
    (&[3, 4], &[1, 1, 1, 0, 2, 4, 5, 0, 2, 3, 3, 0])
);

test_arithmetic_op_float!(
    test_min_f32_broadcast_trailing,
    min,
    f32,
    (&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
    (&[3], &[2.0, 4.0, 5.0]),
    (&[2, 3], &[1.0, 4.0, 3.0, 2.0, 2.0, 5.0])
);

test_arithmetic_op_integer!(
    test_min_i32_broadcast_trailing,
    min,
    i32,
    (&[2, 3], &[1, 5, 3, 4, 2, 6]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[1, 4, 3, 2, 2, 5])
);

test_arithmetic_op_integer!(
    test_min_u32_broadcast_trailing,
    min,
    u32,
    (&[2, 3], &[1, 5, 3, 4, 2, 6]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[1, 4, 3, 2, 2, 5])
);

test_arithmetic_op_float!(
    test_min_f32_broadcast_scalar,
    min,
    f32,
    (&[4], &[1.0, 5.0, 3.0, 7.0]),
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 4.0, 3.0, 4.0])
);

test_arithmetic_op_integer!(
    test_min_i32_broadcast_scalar,
    min,
    i32,
    (&[4], &[1, 5, 3, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[1, 4, 3, 4])
);

test_arithmetic_op_integer!(
    test_min_u32_broadcast_scalar,
    min,
    u32,
    (&[4], &[1, 5, 3, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[1, 4, 3, 4])
);

test_arithmetic_op_float!(
    test_min_f32_broadcast_scalar_reverse,
    min,
    f32,
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 5.0, 3.0, 7.0]),
    (&[4], &[1.0, 4.0, 3.0, 4.0])
);

test_arithmetic_op_integer!(
    test_min_i32_broadcast_scalar_reverse,
    min,
    i32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 5, 3, 7]),
    (&[4], &[1, 4, 3, 4])
);

test_arithmetic_op_integer!(
    test_min_u32_broadcast_scalar_reverse,
    min,
    u32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 5, 3, 7]),
    (&[4], &[1, 4, 3, 4])
);

// error

#[test]
fn test_min_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.min(&b).is_err());
}
