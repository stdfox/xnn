//! Tests for `Tensor::clamp` operation.

use xnn::{Context, Tensor};

macro_rules! test_clamp_op_float {
    ($name:ident, $T:ty, $x:expr, $a:expr, $b:expr, $y:expr) => {
        #[test]
        fn $name() {
            let ctx = Context::new().unwrap();
            let (x_shape, x_data) = $x;
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (y_shape, y_data) = $y;
            let x = Tensor::<$T>::from_shape_slice(&ctx, x_shape, x_data).unwrap();
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let y = Tensor::<$T>::from_shape_slice(&ctx, y_shape, y_data).unwrap();
            let result = x.clamp(&a, &b).unwrap();
            crate::assert_tensor_relative_eq(&result, &y);
        }
    };
}

macro_rules! test_clamp_op_integer {
    ($name:ident, $T:ty, $x:expr, $a:expr, $b:expr, $y:expr) => {
        #[test]
        fn $name() {
            let ctx = Context::new().unwrap();
            let (x_shape, x_data) = $x;
            let (a_shape, a_data) = $a;
            let (b_shape, b_data) = $b;
            let (y_shape, y_data) = $y;
            let x = Tensor::<$T>::from_shape_slice(&ctx, x_shape, x_data).unwrap();
            let a = Tensor::<$T>::from_shape_slice(&ctx, a_shape, a_data).unwrap();
            let b = Tensor::<$T>::from_shape_slice(&ctx, b_shape, b_data).unwrap();
            let y = Tensor::<$T>::from_shape_slice(&ctx, y_shape, y_data).unwrap();
            let result = x.clamp(&a, &b).unwrap();
            crate::assert_tensor_eq(&result, &y);
        }
    };
}

// vector

test_clamp_op_float!(
    test_clamp_f32_vector,
    f32,
    (&[6], &[-1.0, 0.5, 1.5, 2.5, 3.5, 5.0]),
    (&[6], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (&[6], &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    (&[6], &[0.0, 0.5, 1.5, 2.5, 3.0, 3.0])
);

test_clamp_op_integer!(
    test_clamp_i32_vector,
    i32,
    (&[6], &[-10, 0, 5, 10, 15, 20]),
    (&[6], &[0, 0, 0, 0, 0, 0]),
    (&[6], &[10, 10, 10, 10, 10, 10]),
    (&[6], &[0, 0, 5, 10, 10, 10])
);

test_clamp_op_integer!(
    test_clamp_u32_vector,
    u32,
    (&[6], &[0, 1, 5, 10, 15, 20]),
    (&[6], &[2, 2, 2, 2, 2, 2]),
    (&[6], &[12, 12, 12, 12, 12, 12]),
    (&[6], &[2, 2, 5, 10, 12, 12])
);

// matrix

test_clamp_op_float!(
    test_clamp_f32_matrix,
    f32,
    (&[2, 3], &[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]),
    (&[2, 3], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (&[2, 3], &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
    (&[2, 3], &[0.0, 0.0, 2.0, 4.0, 5.0, 5.0])
);

test_clamp_op_integer!(
    test_clamp_i32_matrix,
    i32,
    (&[2, 3], &[-10, 0, 5, 10, 15, 20]),
    (&[2, 3], &[0, 0, 0, 0, 0, 0]),
    (&[2, 3], &[10, 10, 10, 10, 10, 10]),
    (&[2, 3], &[0, 0, 5, 10, 10, 10])
);

test_clamp_op_integer!(
    test_clamp_u32_matrix,
    u32,
    (&[2, 3], &[0, 1, 5, 10, 15, 20]),
    (&[2, 3], &[2, 2, 2, 2, 2, 2]),
    (&[2, 3], &[12, 12, 12, 12, 12, 12]),
    (&[2, 3], &[2, 2, 5, 10, 12, 12])
);

// scalar

test_clamp_op_float!(
    test_clamp_f32_scalar,
    f32,
    (&[] as &[usize], &[5.0]),
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[3.0])
);

test_clamp_op_float!(
    test_clamp_f32_scalar_below,
    f32,
    (&[] as &[usize], &[-5.0]),
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[0.0])
);

test_clamp_op_float!(
    test_clamp_f32_scalar_in_range,
    f32,
    (&[] as &[usize], &[1.5]),
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[3.0]),
    (&[] as &[usize], &[1.5])
);

test_clamp_op_integer!(
    test_clamp_i32_scalar,
    i32,
    (&[] as &[usize], &[15]),
    (&[] as &[usize], &[0]),
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[10])
);

test_clamp_op_integer!(
    test_clamp_u32_scalar,
    u32,
    (&[] as &[usize], &[1]),
    (&[] as &[usize], &[5]),
    (&[] as &[usize], &[10]),
    (&[] as &[usize], &[5])
);

// broadcast scalar min/max

test_clamp_op_float!(
    test_clamp_f32_broadcast_scalar_bounds,
    f32,
    (&[4], &[-1.0, 0.5, 1.5, 5.0]),
    (&[] as &[usize], &[0.0]),
    (&[] as &[usize], &[3.0]),
    (&[4], &[0.0, 0.5, 1.5, 3.0])
);

test_clamp_op_integer!(
    test_clamp_i32_broadcast_scalar_bounds,
    i32,
    (&[4], &[-10, 0, 5, 15]),
    (&[] as &[usize], &[0]),
    (&[] as &[usize], &[10]),
    (&[4], &[0, 0, 5, 10])
);

test_clamp_op_integer!(
    test_clamp_u32_broadcast_scalar_bounds,
    u32,
    (&[4], &[0, 5, 10, 20]),
    (&[] as &[usize], &[2]),
    (&[] as &[usize], &[15]),
    (&[4], &[2, 5, 10, 15])
);

// broadcast vector bounds

test_clamp_op_float!(
    test_clamp_f32_broadcast_vector_bounds,
    f32,
    (&[4], &[-1.0, 0.5, 1.5, 5.0]),
    (&[4], &[0.0, 0.0, 1.0, 2.0]),
    (&[4], &[3.0, 1.0, 2.0, 4.0]),
    (&[4], &[0.0, 0.5, 1.5, 4.0])
);

test_clamp_op_integer!(
    test_clamp_i32_broadcast_vector_bounds,
    i32,
    (&[4], &[-5, 0, 5, 15]),
    (&[4], &[0, 1, 2, 3]),
    (&[4], &[10, 5, 8, 12]),
    (&[4], &[0, 1, 5, 12])
);

// broadcast mixed dimensions

test_clamp_op_float!(
    test_clamp_f32_broadcast_mixed,
    f32,
    (&[2, 3], &[-1.0, 0.5, 5.0, -2.0, 3.0, 10.0]),
    (&[1, 3], &[0.0, 0.0, 0.0]),
    (&[2, 1], &[4.0, 8.0]),
    (&[2, 3], &[0.0, 0.5, 4.0, 0.0, 3.0, 8.0])
);

test_clamp_op_integer!(
    test_clamp_i32_broadcast_mixed,
    i32,
    (&[2, 3], &[-1, 0, 5, -2, 3, 10]),
    (&[1, 3], &[0, 0, 0]),
    (&[2, 1], &[4, 8]),
    (&[2, 3], &[0, 0, 4, 0, 3, 8])
);

// broadcast multi-expand

test_clamp_op_float!(
    test_clamp_f32_broadcast_multi_expand,
    f32,
    (&[2, 1, 4], &[-1.0, 0.5, 1.5, 5.0, -2.0, 1.0, 3.0, 8.0]),
    (&[3, 1], &[0.0, 0.5, 1.0]),
    (&[1], &[4.0]),
    (
        &[2, 3, 4],
        &[
            0.0, 0.5, 1.5, 4.0, 0.5, 0.5, 1.5, 4.0, 1.0, 1.0, 1.5, 4.0, 0.0, 1.0, 3.0, 4.0, 0.5,
            1.0, 3.0, 4.0, 1.0, 1.0, 3.0, 4.0
        ]
    )
);

test_clamp_op_integer!(
    test_clamp_i32_broadcast_multi_expand,
    i32,
    (&[2, 1, 4], &[-1, 0, 2, 5, -2, 1, 3, 8]),
    (&[3, 1], &[0, 1, 2]),
    (&[1], &[4]),
    (
        &[2, 3, 4],
        &[
            0, 0, 2, 4, 1, 1, 2, 4, 2, 2, 2, 4, 0, 1, 3, 4, 1, 1, 3, 4, 2, 2, 3, 4
        ]
    )
);

// error

#[test]
fn test_clamp_error_incompatible_shapes_x_a() {
    let ctx = Context::new().unwrap();
    let x = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[0.0, 0.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0]).unwrap();
    assert!(x.clamp(&a, &b).is_err());
}

#[test]
fn test_clamp_error_incompatible_shapes_x_b() {
    let ctx = Context::new().unwrap();
    let x = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[0.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0]).unwrap();
    assert!(x.clamp(&a, &b).is_err());
}

#[test]
fn test_clamp_error_incompatible_shapes_a_b() {
    let ctx = Context::new().unwrap();
    let x = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, &[3, 1], &[0.0, 0.0, 0.0]).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[1, 2], &[1.0, 2.0]).unwrap();
    assert!(x.clamp(&a, &b).is_err());
}
