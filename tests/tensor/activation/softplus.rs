//! Tests for `Tensor::softplus` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

fn softplus_ref(x: f32) -> f32 {
    (x.exp() + 1.0).ln()
}

#[test]
fn test_softplus_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.softplus().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| softplus_ref(x)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_softplus_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0f32];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.softplus().unwrap();
    // softplus(0) = log(2) ≈ 0.693
    assert_relative_eq!(result.to_vec().unwrap()[0], 2.0_f32.ln(), epsilon = 1e-4);
}

#[test]
fn test_softplus_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.softplus().unwrap();
    let out = result.to_vec().unwrap();
    // softplus is always positive and approaches x for large x
    for (i, &val) in out.iter().enumerate() {
        assert!(val > 0.0);
        assert!(val >= data[i]); // softplus(x) >= x for x >= 0
    }
}

#[test]
fn test_softplus_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0f32, -2.0, -3.0, -4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.softplus().unwrap();
    let out = result.to_vec().unwrap();
    // softplus approaches 0 for large negative x
    for &val in &out {
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}

#[test]
fn test_softplus_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.softplus().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| softplus_ref(x)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_softplus_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.softplus().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| softplus_ref(x)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_softplus_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.softplus().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 2.0_f32.ln(), epsilon = 1e-4);
    }
}

#[test]
fn test_softplus_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.softplus().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 2.0_f32.ln(), epsilon = 1e-4);
}
