//! Tests for `Tensor::div` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

use super::assert_vec_relative_eq;
use super::{
    COLUMN, COLUMN_SHAPE, MATRIX_A, MATRIX_B, MATRIX_SHAPE, ROW, ROW_SHAPE, SCALAR, TRAILING,
    VECTOR_A, VECTOR_B, VECTOR_I32_A, VECTOR_I32_B, VECTOR_U32_A, VECTOR_U32_B,
};

#[test]
fn test_div_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_B).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[5.0, 3.0, 7.0 / 3.0, 2.0]);
}

#[test]
fn test_div_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_B).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
}

#[test]
fn test_div_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_B).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[0.5, 0.6, 0.7, 0.8]);
}

#[test]
fn test_div_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 5.0, 10.0 / 3.0, 2.5]);
}

#[test]
fn test_div_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    // [2, 3] / [3] -> [2, 3]
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_B).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, TRAILING).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 1.0, 1.0, 4.0, 2.5, 2.0]);
}

#[test]
fn test_div_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, ROW_SHAPE, ROW).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(
        &out,
        &[
            0.1,
            0.05,
            1.0 / 30.0,
            0.025,
            0.2,
            0.1,
            2.0 / 30.0,
            0.05,
            0.3,
            0.15,
            0.1,
            0.075,
        ],
    );
}

#[test]
fn test_div_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, COLUMN).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    assert!(a.div(&b).is_err());
}

#[test]
fn test_div_i32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![10, 10, 10, 10]);
}

#[test]
fn test_div_u32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_B).unwrap();
    let b = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_A).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![10, 10, 10, 10]);
}

#[test]
fn test_div_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data_a: Vec<f32> = (1_u8..43).map(|i| f32::from(i) * 10.0).collect();
    let data_b: Vec<f32> = (1_u8..43).map(f32::from).collect();
    let a = Tensor::<f32>::from_slice(&ctx, &data_a).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &data_b).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[42]);
    let out = c.to_vec().unwrap();
    let expected: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x / y)
        .collect();
    assert_vec_relative_eq(&out, &expected);
}

#[test]
fn test_div_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let a = Tensor::<f32>::constant(&ctx, &[len], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[len], &[2.0]).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[len]);
    let out = c.to_vec().unwrap();
    for val in &out[..100] {
        assert_relative_eq!(*val, 5.0, epsilon = 1e-4);
    }
}

#[test]
fn test_div_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[2.0]).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_relative_eq!(out[0], 5.0, epsilon = 1e-4);
}
