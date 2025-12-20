//! Tests for `Tensor::neg` operation.

use std::f32::consts::PI;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_neg_f32_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [-1.0, -2.0, -3.0, -4.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_neg_f32_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, -2.0, -3.0, -4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_neg_f32_mixed() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, 2.0, -3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [1.0, -2.0, 3.0, -4.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_neg_f32_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, -0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.to_vec().unwrap()[1], 0.0, epsilon = 1e-4);
}

#[test]
fn test_neg_i32_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1, 2, 3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    assert_eq!(result.to_vec().unwrap(), vec![-1, -2, -3, -4]);
}

#[test]
fn test_neg_i32_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1, -2, -3, -4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    assert_eq!(result.to_vec().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn test_neg_i32_mixed() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1, 2, -3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    assert_eq!(result.to_vec().unwrap(), vec![1, -2, 3, -4]);
}

#[test]
fn test_neg_i32_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0, 0, 0, 0];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    assert_eq!(result.to_vec().unwrap(), vec![0, 0, 0, 0]);
}

#[test]
fn test_neg_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_neg_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21..21).map(|i| i as f32).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| -x).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_neg_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[PI]).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, -PI, epsilon = 1e-4);
    }
}

#[test]
fn test_neg_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[42.0]).unwrap();
    let result = t.neg().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], -42.0, epsilon = 1e-4);
}

#[test]
fn test_neg_double() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, -2.0, 3.0, -4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.neg().unwrap().neg().unwrap();
    assert_eq!(result.shape(), t.shape());
    for (a, b) in result.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
