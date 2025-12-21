//! Tests for `Tensor::mul` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

use super::assert_vec_relative_eq;
use super::{
    COLUMN, COLUMN_SHAPE, MATRIX_A, MATRIX_B, MATRIX_SHAPE, ROW, ROW_SHAPE, SCALAR, TRAILING,
    VECTOR_A, VECTOR_B, VECTOR_I32_A, VECTOR_I32_B, VECTOR_U32_A, VECTOR_U32_B,
};

#[test]
fn test_mul_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_B).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_mul_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_B).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0]);
}

#[test]
fn test_mul_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_mul_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_mul_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, TRAILING).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn test_mul_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, ROW_SHAPE, ROW).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(
        &out,
        &[
            10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0, 30.0, 60.0, 90.0, 120.0,
        ],
    );
}

#[test]
fn test_mul_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, COLUMN).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    assert!(a.mul(&b).is_err());
}

#[test]
fn test_mul_i32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![10, 40, 90, 160]);
}

#[test]
fn test_mul_u32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_A).unwrap();
    let b = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_B).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![10, 40, 90, 160]);
}

#[test]
fn test_mul_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data_a: Vec<f32> = (1_u8..43).map(f32::from).collect();
    let data_b: Vec<f32> = (1_u8..43).map(|i| f32::from(i) * 0.1).collect();
    let a = Tensor::<f32>::from_slice(&ctx, &data_a).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &data_b).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[42]);
    let out = c.to_vec().unwrap();
    let expected: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x * y)
        .collect();
    assert_vec_relative_eq(&out, &expected);
}

#[test]
fn test_mul_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let a = Tensor::<f32>::constant(&ctx, &[len], &[2.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[len], &[3.0]).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[len]);
    let out = c.to_vec().unwrap();
    for val in &out[..100] {
        assert_relative_eq!(*val, 6.0, epsilon = 1e-4);
    }
}

#[test]
fn test_mul_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[3.0]).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_relative_eq!(out[0], 30.0, epsilon = 1e-4);
}
