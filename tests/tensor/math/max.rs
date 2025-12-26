//! Tests for `Tensor::max` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_max_f32_vector,
    max,
    f32,
    (&[4], &[1.0, 6.0, 3.0, 8.0]),
    (&[4], &[5.0, 2.0, 7.0, 4.0]),
    (&[4], &[5.0, 6.0, 7.0, 8.0])
);

test_arithmetic_op_integer!(
    test_max_i32_vector,
    max,
    i32,
    (&[4], &[1, 20, 3, 40]),
    (&[4], &[10, 2, 30, 4]),
    (&[4], &[10, 20, 30, 40])
);

test_arithmetic_op_integer!(
    test_max_u32_vector,
    max,
    u32,
    (&[4], &[1, 20, 3, 40]),
    (&[4], &[10, 2, 30, 4]),
    (&[4], &[10, 20, 30, 40])
);

// matrix

test_arithmetic_op_float!(
    test_max_f32_matrix,
    max,
    f32,
    (&[2, 3], &[1.0, 20.0, 3.0, 40.0, 5.0, 60.0]),
    (&[2, 3], &[10.0, 2.0, 30.0, 4.0, 50.0, 6.0]),
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
);

test_arithmetic_op_integer!(
    test_max_i32_matrix,
    max,
    i32,
    (&[2, 3], &[1, 20, 3, 40, 5, 60]),
    (&[2, 3], &[10, 2, 30, 4, 50, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60])
);

test_arithmetic_op_integer!(
    test_max_u32_matrix,
    max,
    u32,
    (&[2, 3], &[1, 20, 3, 40, 5, 60]),
    (&[2, 3], &[10, 2, 30, 4, 50, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60])
);

// scalar

test_arithmetic_op_float!(
    test_max_f32_scalar,
    max,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[5.0])
);

test_arithmetic_op_integer!(
    test_max_i32_scalar,
    max,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[8]),
    (&[] as &[usize], &[8])
);

test_arithmetic_op_integer!(
    test_max_u32_scalar,
    max,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[5])
);

// broadcast

test_arithmetic_op_float!(
    test_max_f32_broadcast_multi_expand,
    max,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[3.0, 5.0, 7.0]),
    (
        &[2, 3, 4],
        &[
            3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 7.0, 8.0, 5.0,
            6.0, 7.0, 8.0, 7.0, 7.0, 7.0, 8.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_max_i32_broadcast_multi_expand,
    max,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[3, 5, 7]),
    (
        &[2, 3, 4],
        &[
            3, 3, 3, 4, 5, 5, 5, 5, 7, 7, 7, 7, 5, 6, 7, 8, 5, 6, 7, 8, 7, 7, 7, 8
        ]
    )
);

test_arithmetic_op_integer!(
    test_max_u32_broadcast_multi_expand,
    max,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[3, 5, 7]),
    (
        &[2, 3, 4],
        &[
            3, 3, 3, 4, 5, 5, 5, 5, 7, 7, 7, 7, 5, 6, 7, 8, 5, 6, 7, 8, 7, 7, 7, 8
        ]
    )
);

test_arithmetic_op_float!(
    test_max_f32_broadcast_expand,
    max,
    f32,
    (&[3, 1], &[1.0, 5.0, 3.0]),
    (&[1, 4], &[2.0, 4.0, 6.0, 0.0]),
    (
        &[3, 4],
        &[2.0, 4.0, 6.0, 1.0, 5.0, 5.0, 6.0, 5.0, 3.0, 4.0, 6.0, 3.0]
    )
);

test_arithmetic_op_integer!(
    test_max_i32_broadcast_expand,
    max,
    i32,
    (&[3, 1], &[1, 5, 3]),
    (&[1, 4], &[2, 4, 6, 0]),
    (&[3, 4], &[2, 4, 6, 1, 5, 5, 6, 5, 3, 4, 6, 3])
);

test_arithmetic_op_integer!(
    test_max_u32_broadcast_expand,
    max,
    u32,
    (&[3, 1], &[1, 5, 3]),
    (&[1, 4], &[2, 4, 6, 0]),
    (&[3, 4], &[2, 4, 6, 1, 5, 5, 6, 5, 3, 4, 6, 3])
);

test_arithmetic_op_float!(
    test_max_f32_broadcast_trailing,
    max,
    f32,
    (&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
    (&[3], &[2.0, 4.0, 5.0]),
    (&[2, 3], &[2.0, 5.0, 5.0, 4.0, 4.0, 6.0])
);

test_arithmetic_op_integer!(
    test_max_i32_broadcast_trailing,
    max,
    i32,
    (&[2, 3], &[1, 5, 3, 4, 2, 6]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[2, 5, 5, 4, 4, 6])
);

test_arithmetic_op_integer!(
    test_max_u32_broadcast_trailing,
    max,
    u32,
    (&[2, 3], &[1, 5, 3, 4, 2, 6]),
    (&[3], &[2, 4, 5]),
    (&[2, 3], &[2, 5, 5, 4, 4, 6])
);

test_arithmetic_op_float!(
    test_max_f32_broadcast_scalar,
    max,
    f32,
    (&[4], &[1.0, 5.0, 3.0, 7.0]),
    (&[] as &[usize], &[4.0]),
    (&[4], &[4.0, 5.0, 4.0, 7.0])
);

test_arithmetic_op_integer!(
    test_max_i32_broadcast_scalar,
    max,
    i32,
    (&[4], &[1, 5, 3, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[4, 5, 4, 7])
);

test_arithmetic_op_integer!(
    test_max_u32_broadcast_scalar,
    max,
    u32,
    (&[4], &[1, 5, 3, 7]),
    (&[] as &[usize], &[4]),
    (&[4], &[4, 5, 4, 7])
);

test_arithmetic_op_float!(
    test_max_f32_broadcast_scalar_reverse,
    max,
    f32,
    (&[] as &[usize], &[4.0]),
    (&[4], &[1.0, 5.0, 3.0, 7.0]),
    (&[4], &[4.0, 5.0, 4.0, 7.0])
);

test_arithmetic_op_integer!(
    test_max_i32_broadcast_scalar_reverse,
    max,
    i32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 5, 3, 7]),
    (&[4], &[4, 5, 4, 7])
);

test_arithmetic_op_integer!(
    test_max_u32_broadcast_scalar_reverse,
    max,
    u32,
    (&[] as &[usize], &[4]),
    (&[4], &[1, 5, 3, 7]),
    (&[4], &[4, 5, 4, 7])
);

// error

#[test]
fn test_max_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.max(&b).is_err());
}
