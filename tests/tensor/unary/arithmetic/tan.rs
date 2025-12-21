//! Tests for `Tensor::tan` operation.

use std::f32::consts::FRAC_PI_4;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_tan_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, FRAC_PI_4, -FRAC_PI_4, 0.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tan().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tan()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tan().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_pi_4() {
    let ctx = Context::try_default().unwrap();
    let data = vec![FRAC_PI_4, FRAC_PI_4, FRAC_PI_4, FRAC_PI_4];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tan().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.tan().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tan()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.05).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.tan().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.tan()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[FRAC_PI_4]).unwrap();
    let result = t.tan().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_tan_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.tan().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}
