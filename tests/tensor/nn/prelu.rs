//! Tests for `Tensor::prelu` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

fn prelu_ref(x: f32, alpha: f32) -> f32 {
    if x < 0.0 { alpha * x } else { x }
}

#[test]
fn test_prelu_basic() {
    let ctx = Context::new().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .zip(alpha_data.iter())
        .map(|(&x, &a)| prelu_ref(x, a))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_prelu_uniform_alpha() {
    let ctx = Context::new().unwrap();
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha_data = vec![0.25f32; 5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| prelu_ref(x, 0.25)).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_prelu_zero() {
    let ctx = Context::new().unwrap();
    let data = vec![0.0f32];
    let alpha_data = vec![0.5f32];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}

#[test]
fn test_prelu_positive() {
    let ctx = Context::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let alpha_data = vec![0.1f32, 0.2, 0.3, 0.4];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_prelu_negative() {
    let ctx = Context::new().unwrap();
    let data = vec![-1.0f32, -2.0, -3.0];
    let alpha_data = vec![0.1f32, 0.2, 0.3];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .zip(alpha_data.iter())
        .map(|(&x, &a)| prelu_ref(x, a))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_prelu_2d() {
    let ctx = Context::new().unwrap();
    let data = vec![-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0];
    let alpha_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let alpha = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .zip(alpha_data.iter())
        .map(|(&x, &a)| prelu_ref(x, a))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_prelu_non_aligned() {
    let ctx = Context::new().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let alpha_data: Vec<f32> = (0..42).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &alpha_data).unwrap();
    let result = t.prelu(&alpha).unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .zip(alpha_data.iter())
        .map(|(&x, &a)| prelu_ref(x, a))
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_prelu_scalar() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[0.0]).unwrap();
    let alpha = Tensor::<f32>::constant(&ctx, &[], &[0.5]).unwrap();
    let result = t.prelu(&alpha).unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
}

#[test]
fn test_prelu_shape_mismatch() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let alpha = Tensor::<f32>::from_slice(&ctx, &[0.1, 0.2]).unwrap();
    let result = t.prelu(&alpha);
    assert!(result.is_err());
}
