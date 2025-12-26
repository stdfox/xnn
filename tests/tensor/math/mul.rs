//! Tests for `Tensor::mul` operation.

use xnn::{Context, Tensor};

use super::{test_arithmetic_op_float, test_arithmetic_op_integer};

// vector

test_arithmetic_op_float!(
    test_mul_f32_vector,
    mul,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[5.0, 6.0, 7.0, 8.0]),
    (&[4], &[5.0, 12.0, 21.0, 32.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_vector,
    mul,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[10, 40, 90, 160])
);

test_arithmetic_op_integer!(
    test_mul_u32_vector,
    mul,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40]),
    (&[4], &[10, 40, 90, 160])
);

// matrix

test_arithmetic_op_float!(
    test_mul_f32_matrix,
    mul,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[2, 3], &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_matrix,
    mul,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[10, 40, 90, 160, 250, 360])
);

test_arithmetic_op_integer!(
    test_mul_u32_matrix,
    mul,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[10, 40, 90, 160, 250, 360])
);

// scalar

test_arithmetic_op_float!(
    test_mul_f32_scalar,
    mul,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[15.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_scalar,
    mul,
    i32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[15])
);

test_arithmetic_op_integer!(
    test_mul_u32_scalar,
    mul,
    u32,
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[3]),
    (&[] as &[usize], &[15])
);

// broadcast

test_arithmetic_op_float!(
    test_mul_f32_broadcast_multi_expand,
    mul,
    f32,
    (&[2, 1, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    (&[3, 1], &[10.0, 20.0, 30.0]),
    (
        &[2, 3, 4],
        &[
            10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0, 30.0, 60.0, 90.0, 120.0, 50.0, 60.0,
            70.0, 80.0, 100.0, 120.0, 140.0, 160.0, 150.0, 180.0, 210.0, 240.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_mul_i32_broadcast_multi_expand,
    mul,
    i32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[10, 20, 30]),
    (
        &[2, 3, 4],
        &[
            10, 20, 30, 40, 20, 40, 60, 80, 30, 60, 90, 120, 50, 60, 70, 80, 100, 120, 140, 160,
            150, 180, 210, 240
        ]
    )
);

test_arithmetic_op_integer!(
    test_mul_u32_broadcast_multi_expand,
    mul,
    u32,
    (&[2, 1, 4], &[1, 2, 3, 4, 5, 6, 7, 8]),
    (&[3, 1], &[10, 20, 30]),
    (
        &[2, 3, 4],
        &[
            10, 20, 30, 40, 20, 40, 60, 80, 30, 60, 90, 120, 50, 60, 70, 80, 100, 120, 140, 160,
            150, 180, 210, 240
        ]
    )
);

test_arithmetic_op_float!(
    test_mul_f32_broadcast_expand,
    mul,
    f32,
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
    (
        &[3, 4],
        &[
            10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0, 30.0, 60.0, 90.0, 120.0
        ]
    )
);

test_arithmetic_op_integer!(
    test_mul_i32_broadcast_expand,
    mul,
    i32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[10, 20, 30, 40]),
    (&[3, 4], &[10, 20, 30, 40, 20, 40, 60, 80, 30, 60, 90, 120])
);

test_arithmetic_op_integer!(
    test_mul_u32_broadcast_expand,
    mul,
    u32,
    (&[3, 1], &[1, 2, 3]),
    (&[1, 4], &[10, 20, 30, 40]),
    (&[3, 4], &[10, 20, 30, 40, 20, 40, 60, 80, 30, 60, 90, 120])
);

test_arithmetic_op_float!(
    test_mul_f32_broadcast_trailing,
    mul,
    f32,
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[3], &[10.0, 20.0, 30.0]),
    (&[2, 3], &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_broadcast_trailing,
    mul,
    i32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[10, 40, 90, 40, 100, 180])
);

test_arithmetic_op_integer!(
    test_mul_u32_broadcast_trailing,
    mul,
    u32,
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[3], &[10, 20, 30]),
    (&[2, 3], &[10, 40, 90, 40, 100, 180])
);

test_arithmetic_op_float!(
    test_mul_f32_broadcast_scalar,
    mul,
    f32,
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[] as &[usize], &[10.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_broadcast_scalar,
    mul,
    i32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[10]),
    (&[4], &[10, 20, 30, 40])
);

test_arithmetic_op_integer!(
    test_mul_u32_broadcast_scalar,
    mul,
    u32,
    (&[4], &[1, 2, 3, 4]),
    (&[] as &[usize], &[10]),
    (&[4], &[10, 20, 30, 40])
);

test_arithmetic_op_float!(
    test_mul_f32_broadcast_scalar_reverse,
    mul,
    f32,
    (&[] as &[usize], &[10.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0])
);

test_arithmetic_op_integer!(
    test_mul_i32_broadcast_scalar_reverse,
    mul,
    i32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40])
);

test_arithmetic_op_integer!(
    test_mul_u32_broadcast_scalar_reverse,
    mul,
    u32,
    (&[] as &[usize], &[10]),
    (&[4], &[1, 2, 3, 4]),
    (&[4], &[10, 20, 30, 40])
);

// error

#[test]
fn test_mul_error_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.mul(&b).is_err());
}
