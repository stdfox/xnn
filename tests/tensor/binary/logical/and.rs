//! Tests for `Tensor::and` operation.

use xnn::{Context, Tensor};

use super::{
    COLUMN_BOOL, COLUMN_SHAPE, MATRIX_BOOL_A, MATRIX_BOOL_B, MATRIX_SHAPE, ROW_BOOL, ROW_SHAPE,
    TRAILING_BOOL, VECTOR_BOOL_A, VECTOR_BOOL_B,
};

#[test]
fn test_and_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_A).unwrap();
    let b = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_B).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, false, false, false]);
}

#[test]
fn test_and_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_BOOL_A).unwrap();
    let b = Tensor::<bool>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_BOOL_B).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![false, false, true, false, true, false]);
}

#[test]
fn test_and_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_A).unwrap();
    let b = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, false, false]);
}

#[test]
fn test_and_scalar_broadcast_false() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_A).unwrap();
    let b = Tensor::<bool>::constant(&ctx, &[], &[false]).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![false, false, false, false]);
}

#[test]
fn test_and_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let b = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_A).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, true, false, false]);
}

#[test]
fn test_and_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_BOOL_A).unwrap();
    let b = Tensor::<bool>::from_slice(&ctx, TRAILING_BOOL).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![true, false, true, false, false, false]);
}

#[test]
fn test_and_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN_BOOL).unwrap();
    let b = Tensor::<bool>::from_shape_slice(&ctx, ROW_SHAPE, ROW_BOOL).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_eq!(
        out,
        vec![
            true, false, true, false, false, false, false, false, true, false, true, false
        ]
    );
}

#[test]
fn test_and_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::from_slice(&ctx, COLUMN_BOOL).unwrap();
    let b = Tensor::<bool>::from_slice(&ctx, VECTOR_BOOL_A).unwrap();
    assert!(a.and(&b).is_err());
}

#[test]
fn test_and_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let b = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_eq!(out[0], true);

    let a = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let b = Tensor::<bool>::constant(&ctx, &[], &[false]).unwrap();
    let c = a.and(&b).unwrap();
    assert_eq!(out[0], true);
    let out = c.to_vec().unwrap();
    assert_eq!(out[0], false);
}
