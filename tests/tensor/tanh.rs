//! Tests for `Tensor::tanh` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_tanh_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 1.0, -1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tanh().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tanh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tanh().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_bounds() {
    let ctx = Context::try_default().unwrap();
    let data = vec![10.0, -10.0, 100.0, -100.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tanh().unwrap();
    let out = result.to_vec().unwrap();
    assert_relative_eq!(out[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(out[1], -1.0, epsilon = 1e-4);
    assert_relative_eq!(out[2], 1.0, epsilon = 1e-4);
    assert_relative_eq!(out[3], -1.0, epsilon = 1e-4);
}

#[test]
fn test_tanh_odd() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.5, 1.0, 1.5, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let pos = t.tanh().unwrap().to_vec().unwrap();
    let neg_data: Vec<f32> = data.iter().map(|x| -x).collect();
    let t_neg = Tensor::<f32>::from_slice(&ctx, &neg_data).unwrap();
    let neg = t_neg.tanh().unwrap().to_vec().unwrap();
    for (p, n) in pos.iter().zip(neg.iter()) {
        assert_relative_eq!(*p, -*n, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.5, 1.0, -0.5, -1.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.tanh().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tanh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21..21).map(|i| i as f32 * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tanh().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tanh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.tanh().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_tanh_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.tanh().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}
