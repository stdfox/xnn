//! Tests for `Tensor::log` operation.

use std::f32::consts::E;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_log_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, E, E * E, 10.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.log().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.ln()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_log_one() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.log().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_log_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.log().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.ln()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_log_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (1..43).map(|i| i as f32).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.log().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.ln()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_log_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[E]).unwrap();
    let result = t.log().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_log_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[1.0]).unwrap();
    let result = t.log().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}

#[test]
fn test_log_exp_inverse() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.log().unwrap().exp().unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
