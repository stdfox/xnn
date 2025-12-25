//! Tests for `Tensor::selu` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

const SELU_ALPHA: f32 = 1.673_263_2;
const SELU_LAMBDA: f32 = 1.050_701;

fn selu_ref(x: f32, alpha: f32, lambda: f32) -> f32 {
    if x < 0.0 {
        lambda * alpha * (x.exp() - 1.0)
    } else {
        lambda * x
    }
}

#[test]
fn test_selu_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.selu(None, None).unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| selu_ref(x, SELU_ALPHA, SELU_LAMBDA))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_selu_custom_params() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = 1.5;
    let lambda = 1.2;
    let result = t.selu(Some(alpha), Some(lambda)).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| selu_ref(x, alpha, lambda)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_selu_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0f32];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.selu(None, None).unwrap();
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}

#[test]
fn test_selu_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.selu(None, None).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| selu_ref(x, SELU_ALPHA, SELU_LAMBDA))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_selu_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0f32, -2.0, -3.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.selu(None, None).unwrap();
    let out = result.to_vec().unwrap();
    for val in &out {
        assert!(*val < 0.0);
    }
}

#[test]
fn test_selu_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.selu(None, None).unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| selu_ref(x, SELU_ALPHA, SELU_LAMBDA))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_selu_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.selu(None, None).unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| selu_ref(x, SELU_ALPHA, SELU_LAMBDA))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_selu_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.selu(None, None).unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}
