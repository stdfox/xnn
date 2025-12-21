//! Tests for `Tensor::ceil` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_ceil_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.2, 2.7, -1.2, -2.7];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.ceil().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [2.0, 3.0, -1.0, -2.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, -0.0, 0.0, -0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.ceil().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_integers() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, -1.0, -2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.ceil().unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.1, 0.9, -0.1, -0.9, 1.5, -1.5];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.ceil().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.ceil()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i16..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.ceil().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.ceil()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[1.5]).unwrap();
    let result = t.ceil().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 2.0, epsilon = 1e-4);
    }
}

#[test]
fn test_ceil_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[1.5]).unwrap();
    let result = t.ceil().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 2.0, epsilon = 1e-4);
}
