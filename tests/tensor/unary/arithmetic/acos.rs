//! Tests for `Tensor::acos` operation.

use std::f32::consts::{FRAC_PI_2, PI};

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_acos_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 0.0, -1.0, 0.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acos().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acos()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_one() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acos().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acos().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, FRAC_PI_2, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_neg_one() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, -1.0, -1.0, -1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acos().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, PI, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.25, 0.5, -0.25, -0.5, 0.75];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.acos().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acos()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) / 21.0).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acos().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acos()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[1.0]).unwrap();
    let result = t.acos().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_acos_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.5]).unwrap();
    let result = t.acos().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.5_f32.acos(), epsilon = 1e-4);
}
