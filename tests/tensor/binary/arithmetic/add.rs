//! Tests for `Tensor::add` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

use super::assert_vec_relative_eq;
use super::{
    COLUMN, COLUMN_SHAPE, MATRIX_A, MATRIX_B, MATRIX_SHAPE, ROW, ROW_SHAPE, SCALAR, TRAILING,
    VECTOR_A, VECTOR_B, VECTOR_I32_A, VECTOR_I32_B, VECTOR_U32_A, VECTOR_U32_B,
};

#[test]
fn test_add_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_B).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_add_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_B).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}

#[test]
fn test_add_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_add_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[SCALAR]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_add_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, TRAILING).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_add_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, ROW_SHAPE, ROW).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(
        &out,
        &[
            11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
        ],
    );
}

#[test]
fn test_add_multi_expand() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(
        &ctx,
        &[2, 1, 4],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[3, 1], &[10.0, 20.0, 30.0]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[2, 3, 4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(
        &out,
        &[
            11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0, 32.0, 33.0, 34.0, 15.0, 16.0,
            17.0, 18.0, 25.0, 26.0, 27.0, 28.0, 35.0, 36.0, 37.0, 38.0,
        ],
    );
}

#[test]
fn test_add_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(a.add(&b).is_err());
}

#[test]
fn test_add_i32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![11, 22, 33, 44]);
}

#[test]
fn test_add_u32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_A).unwrap();
    let b = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_B).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![11, 22, 33, 44]);
}

#[test]
fn test_add_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data_a: Vec<f32> = (0_u8..42).map(f32::from).collect();
    let data_b: Vec<f32> = (0_u8..42).map(|i| f32::from(i) * 10.0).collect();
    let a = Tensor::<f32>::from_slice(&ctx, &data_a).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &data_b).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[42]);
    let out = c.to_vec().unwrap();
    let expected: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x + y)
        .collect();
    assert_vec_relative_eq(&out, &expected);
}

#[test]
fn test_add_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let a = Tensor::<f32>::constant(&ctx, &[len], &[1.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[len], &[2.0]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[len]);
    let out = c.to_vec().unwrap();
    for val in &out[..100] {
        assert_relative_eq!(*val, 3.0, epsilon = 1e-4);
    }
}

#[test]
fn test_add_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[5.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[3.0]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_relative_eq!(out[0], 8.0, epsilon = 1e-4);
}
