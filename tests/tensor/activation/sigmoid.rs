//! Tests for `Tensor::sigmoid` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_sigmoid_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0f32];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sigmoid().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    assert_relative_eq!(out[0], 0.5, epsilon = 1e-4);
}

#[test]
fn test_sigmoid_large_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![10.0f32, 20.0, 50.0, 100.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sigmoid().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_large_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-10.0f32, -20.0, -50.0, -100.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sigmoid().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_values() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sigmoid().unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.sigmoid().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sigmoid().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.sigmoid().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.5, epsilon = 1e-4);
    }
}

#[test]
fn test_sigmoid_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.sigmoid().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.5, epsilon = 1e-4);
}
