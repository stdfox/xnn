//! Tests for `Tensor::cos` operation.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_cos_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, FRAC_PI_2, PI, 2.0 * PI];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.cos().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [1.0, 0.0, -1.0, 1.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_cos_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.cos().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_cos_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, FRAC_PI_4, FRAC_PI_2, PI, -FRAC_PI_2, -PI];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.cos().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.cos()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_cos_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.cos().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.cos()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_cos_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.cos().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_cos_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.cos().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 1.0, epsilon = 1e-4);
}

#[test]
fn test_sin_cos_identity() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.5, 1.0, 1.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let sin_sq = t.sin().unwrap();
    let cos_sq = t.cos().unwrap();
    let sin_out = sin_sq.to_vec().unwrap();
    let cos_out = cos_sq.to_vec().unwrap();
    for (s, c) in sin_out.iter().zip(cos_out.iter()) {
        assert_relative_eq!(s * s + c * c, 1.0, epsilon = 1e-4);
    }
}
