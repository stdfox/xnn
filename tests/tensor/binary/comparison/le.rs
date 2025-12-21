//! Tests for `Tensor::le` operation.

use xnn::{Context, Tensor};

use super::{
    COLUMN, COLUMN_SHAPE, MATRIX_A, MATRIX_B, MATRIX_SHAPE, ROW, ROW_SHAPE, SCALAR, TRAILING,
    VECTOR_A, VECTOR_B, VECTOR_I32_A, VECTOR_I32_B, VECTOR_U32_A, VECTOR_U32_B,
};

#[test]
fn test_le_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_B).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true]);
}

#[test]
fn test_le_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_B).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true, true, true]);
}

#[test]
fn test_le_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true]);
}

#[test]
fn test_le_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![false, false, false, false]);
}

#[test]
fn test_le_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, TRAILING).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true, true, true]);
}

#[test]
fn test_le_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, ROW_SHAPE, ROW).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_eq!(
        out,
        vec![
            true, true, true, true, true, true, true, true, true, true, true, true
        ]
    );
}

#[test]
fn test_le_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, COLUMN).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    assert!(a.le(&b).is_err());
}

#[test]
fn test_le_i32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true]);
}

#[test]
fn test_le_u32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_A).unwrap();
    let b = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_B).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, true, true]);
}

#[test]
fn test_le_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let c = a.le(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert!(out[0]);
}
