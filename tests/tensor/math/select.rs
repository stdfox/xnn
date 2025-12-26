//! Tests for `Tensor::select` operation.

use xnn::{Context, Tensor};

macro_rules! test_select_op_float {
    ($name:ident, $T:ty, $cond:expr, $true_val:expr, $false_val:expr, $y:expr) => {
        #[test]
        fn $name() {
            let ctx = Context::try_default().unwrap();
            let (cond_shape, cond_data) = $cond;
            let (true_shape, true_data) = $true_val;
            let (false_shape, false_data) = $false_val;
            let (y_shape, y_data) = $y;
            let cond = Tensor::<bool>::from_shape_slice(&ctx, cond_shape, cond_data).unwrap();
            let true_val = Tensor::<$T>::from_shape_slice(&ctx, true_shape, true_data).unwrap();
            let false_val = Tensor::<$T>::from_shape_slice(&ctx, false_shape, false_data).unwrap();
            let y = Tensor::<$T>::from_shape_slice(&ctx, y_shape, y_data).unwrap();
            let result = cond.select(&true_val, &false_val).unwrap();
            crate::assert_tensor_relative_eq(&result, &y);
        }
    };
}

macro_rules! test_select_op_integer {
    ($name:ident, $T:ty, $cond:expr, $true_val:expr, $false_val:expr, $y:expr) => {
        #[test]
        fn $name() {
            let ctx = Context::try_default().unwrap();
            let (cond_shape, cond_data) = $cond;
            let (true_shape, true_data) = $true_val;
            let (false_shape, false_data) = $false_val;
            let (y_shape, y_data) = $y;
            let cond = Tensor::<bool>::from_shape_slice(&ctx, cond_shape, cond_data).unwrap();
            let true_val = Tensor::<$T>::from_shape_slice(&ctx, true_shape, true_data).unwrap();
            let false_val = Tensor::<$T>::from_shape_slice(&ctx, false_shape, false_data).unwrap();
            let y = Tensor::<$T>::from_shape_slice(&ctx, y_shape, y_data).unwrap();
            let result = cond.select(&true_val, &false_val).unwrap();
            crate::assert_tensor_eq(&result, &y);
        }
    };
}

// vector

test_select_op_float!(
    test_select_f32_vector,
    f32,
    (&[6], &[true, false, true, false, true, false]),
    (&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[6], &[1.0, 20.0, 3.0, 40.0, 5.0, 60.0])
);

test_select_op_integer!(
    test_select_i32_vector,
    i32,
    (&[6], &[true, false, true, false, true, false]),
    (&[6], &[1, 2, 3, 4, 5, 6]),
    (&[6], &[10, 20, 30, 40, 50, 60]),
    (&[6], &[1, 20, 3, 40, 5, 60])
);

test_select_op_integer!(
    test_select_u32_vector,
    u32,
    (&[6], &[true, false, true, false, true, false]),
    (&[6], &[1, 2, 3, 4, 5, 6]),
    (&[6], &[10, 20, 30, 40, 50, 60]),
    (&[6], &[1, 20, 3, 40, 5, 60])
);

// matrix

test_select_op_float!(
    test_select_f32_matrix,
    f32,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    (&[2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    (&[2, 3], &[1.0, 20.0, 3.0, 40.0, 5.0, 60.0])
);

test_select_op_integer!(
    test_select_i32_matrix,
    i32,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[1, 20, 3, 40, 5, 60])
);

test_select_op_integer!(
    test_select_u32_matrix,
    u32,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[2, 3], &[1, 2, 3, 4, 5, 6]),
    (&[2, 3], &[10, 20, 30, 40, 50, 60]),
    (&[2, 3], &[1, 20, 3, 40, 5, 60])
);

// scalar

test_select_op_float!(
    test_select_f32_scalar_true,
    f32,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[2.0]),
    (&[] as &[usize], &[1.0])
);

test_select_op_float!(
    test_select_f32_scalar_false,
    f32,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[2.0]),
    (&[] as &[usize], &[2.0])
);

test_select_op_integer!(
    test_select_i32_scalar_true,
    i32,
    (&[] as &[usize], &[true]),
    (&[] as &[usize], &[1]),
    (&[] as &[usize], &[2]),
    (&[] as &[usize], &[1])
);

test_select_op_integer!(
    test_select_u32_scalar_false,
    u32,
    (&[] as &[usize], &[false]),
    (&[] as &[usize], &[1]),
    (&[] as &[usize], &[2]),
    (&[] as &[usize], &[2])
);

// all true / all false

test_select_op_float!(
    test_select_f32_all_true,
    f32,
    (&[4], &[true, true, true, true]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0])
);

test_select_op_float!(
    test_select_f32_all_false,
    f32,
    (&[4], &[false, false, false, false]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0])
);

// broadcast scalar condition

test_select_op_float!(
    test_select_f32_broadcast_scalar_cond_true,
    f32,
    (&[] as &[usize], &[true]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[4], &[1.0, 2.0, 3.0, 4.0])
);

test_select_op_float!(
    test_select_f32_broadcast_scalar_cond_false,
    f32,
    (&[] as &[usize], &[false]),
    (&[4], &[1.0, 2.0, 3.0, 4.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0]),
    (&[4], &[10.0, 20.0, 30.0, 40.0])
);

// broadcast scalar values

test_select_op_float!(
    test_select_f32_broadcast_scalar_values,
    f32,
    (&[4], &[true, false, true, false]),
    (&[] as &[usize], &[1.0]),
    (&[] as &[usize], &[0.0]),
    (&[4], &[1.0, 0.0, 1.0, 0.0])
);

test_select_op_integer!(
    test_select_i32_broadcast_scalar_values,
    i32,
    (&[4], &[true, false, true, false]),
    (&[] as &[usize], &[1]),
    (&[] as &[usize], &[0]),
    (&[4], &[1, 0, 1, 0])
);

// broadcast mixed dimensions

test_select_op_float!(
    test_select_f32_broadcast_mixed,
    f32,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[1, 3], &[1.0, 2.0, 3.0]),
    (&[2, 1], &[10.0, 20.0]),
    (&[2, 3], &[1.0, 10.0, 3.0, 20.0, 2.0, 20.0])
);

test_select_op_integer!(
    test_select_i32_broadcast_mixed,
    i32,
    (&[2, 3], &[true, false, true, false, true, false]),
    (&[1, 3], &[1, 2, 3]),
    (&[2, 1], &[10, 20]),
    (&[2, 3], &[1, 10, 3, 20, 2, 20])
);

// broadcast multi-expand

test_select_op_float!(
    test_select_f32_broadcast_multi_expand,
    f32,
    (
        &[2, 1, 4],
        &[true, false, true, false, false, true, false, true]
    ),
    (&[3, 1], &[1.0, 2.0, 3.0]),
    (&[1], &[0.0]),
    (
        &[2, 3, 4],
        &[
            1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0
        ]
    )
);

// error

#[test]
fn test_select_error_incompatible_shapes_cond_true() {
    let ctx = Context::try_default().unwrap();
    let cond = Tensor::<bool>::from_slice(&ctx, &[true, false, true]).unwrap();
    let true_val = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0]).unwrap();
    let false_val = Tensor::<f32>::from_slice(&ctx, &[0.0]).unwrap();
    assert!(cond.select(&true_val, &false_val).is_err());
}

#[test]
fn test_select_error_incompatible_shapes_cond_false() {
    let ctx = Context::try_default().unwrap();
    let cond = Tensor::<bool>::from_slice(&ctx, &[true, false, true]).unwrap();
    let true_val = Tensor::<f32>::from_slice(&ctx, &[1.0]).unwrap();
    let false_val = Tensor::<f32>::from_slice(&ctx, &[0.0, 0.0]).unwrap();
    assert!(cond.select(&true_val, &false_val).is_err());
}

#[test]
fn test_select_error_incompatible_shapes_true_false() {
    let ctx = Context::try_default().unwrap();
    let cond = Tensor::<bool>::from_slice(&ctx, &[true, false, true]).unwrap();
    let true_val = Tensor::<f32>::from_shape_slice(&ctx, &[3, 1], &[1.0, 2.0, 3.0]).unwrap();
    let false_val = Tensor::<f32>::from_shape_slice(&ctx, &[1, 2], &[0.0, 0.0]).unwrap();
    assert!(cond.select(&true_val, &false_val).is_err());
}
