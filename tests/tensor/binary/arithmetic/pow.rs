//! Tests for `Tensor::pow` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

use super::assert_vec_relative_eq;
use super::{COLUMN, COLUMN_SHAPE, MATRIX_A, MATRIX_SHAPE, ROW_SHAPE, SCALAR, VECTOR_A};

#[test]
fn test_pow_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[2.0, 2.0, 2.0, 2.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_pow_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        .unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
}

#[test]
fn test_pow_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[2.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_pow_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[2.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[2.0, 4.0, 8.0, 16.0]);
}

#[test]
fn test_pow_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, MATRIX_SHAPE, MATRIX_A).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 0.5]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), MATRIX_SHAPE);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 4.0, 3.0_f32.sqrt(), 4.0, 25.0, 6.0_f32.sqrt()]);
}

#[test]
fn test_pow_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_shape_slice(&ctx, COLUMN_SHAPE, COLUMN).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, ROW_SHAPE, &[0.0, 1.0, 2.0, 3.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(
        &out,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 8.0, 1.0, 3.0, 9.0, 27.0],
    );
}

#[test]
fn test_pow_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, COLUMN).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    assert!(a.pow(&b).is_err());
}

#[test]
fn test_pow_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data_a: Vec<f32> = (1_u8..43).map(f32::from).collect();
    let data_b: Vec<f32> = (1_u8..43).map(|i| f32::from(i) * 0.1).collect();
    let a = Tensor::<f32>::from_slice(&ctx, &data_a).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &data_b).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[42]);
    let out = c.to_vec().unwrap();
    let expected: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x.powf(*y))
        .collect();
    for (a, e) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, e, max_relative = 1e-4);
    }
}

#[test]
fn test_pow_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let a = Tensor::<f32>::constant(&ctx, &[len], &[2.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[len], &[SCALAR]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[len]);
    let out = c.to_vec().unwrap();
    for val in &out[..100] {
        assert_relative_eq!(*val, 1024.0, epsilon = 1e-4);
    }
}

#[test]
fn test_pow_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::constant(&ctx, &[], &[3.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[4.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_relative_eq!(out[0], 81.0, epsilon = 1e-4);
}

#[test]
fn test_pow_fractional_exponent() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, &[4.0, 9.0, 16.0, 25.0]).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[0.5]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_pow_zero_exponent() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, VECTOR_A).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_pow_negative_exponent() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<f32>::from_slice(&ctx, COLUMN).unwrap();
    let b = Tensor::<f32>::constant(&ctx, &[], &[-1.0]).unwrap();
    let c = a.pow(&b).unwrap();
    assert_eq!(c.dimensions(), &[3]);
    let out = c.to_vec().unwrap();
    assert_vec_relative_eq(&out, &[1.0, 0.5, 1.0 / 3.0]);
}
