//! Tests for `Tensor::leaky_relu` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

fn leaky_relu_ref(x: f32, alpha: f32) -> f32 {
    if x < 0.0 { alpha * x } else { x }
}

#[test]
fn test_leaky_relu_basic() {
    let ctx = Context::new().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| leaky_relu_ref(x, 0.01)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_custom_alpha() {
    let ctx = Context::new().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = 0.2;
    let result = t.leaky_relu(Some(alpha)).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| leaky_relu_ref(x, alpha)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_zero() {
    let ctx = Context::new().unwrap();
    let data = vec![0.0f32];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}

#[test]
fn test_leaky_relu_positive() {
    let ctx = Context::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_negative() {
    let ctx = Context::new().unwrap();
    let data = vec![-1.0f32, -2.0, -3.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| leaky_relu_ref(x, 0.01)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_2d() {
    let ctx = Context::new().unwrap();
    let data = vec![-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| leaky_relu_ref(x, 0.01)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_non_aligned() {
    let ctx = Context::new().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.leaky_relu(None).unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| leaky_relu_ref(x, 0.01)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_leaky_relu_scalar() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let result = t.leaky_relu(None).unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}
