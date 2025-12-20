//! Tests for `Tensor::asin` operation.

use std::f32::consts::FRAC_PI_2;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_asin_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.5, 1.0, -1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asin().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asin()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asin().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_one() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asin().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, FRAC_PI_2, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.25, 0.5, -0.25, -0.5, 0.75];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.asin().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asin()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21..21).map(|i| i as f32 / 21.0).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asin().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asin()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.asin().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_asin_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.5]).unwrap();
    let result = t.asin().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.5_f32.asin(), epsilon = 1e-4);
}

#[test]
fn test_asin_sin_inverse() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.25, 0.5, -0.25];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asin().unwrap().sin().unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
