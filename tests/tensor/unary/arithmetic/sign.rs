//! Tests for `Tensor::sign` operation.

use std::f32::consts::PI;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_sign_f32_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_f32_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, -2.0, -3.0, -4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, -1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_f32_mixed() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, 2.0, -3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected = [-1.0, 1.0, -1.0, 1.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_f32_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, -0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.to_vec().unwrap()[1], 0.0, epsilon = 1e-4);
}

#[test]
fn test_sign_i32_positive() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1, 2, 3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_eq!(result.to_vec().unwrap(), vec![1, 1, 1, 1]);
}

#[test]
fn test_sign_i32_negative() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1, -2, -3, -4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_eq!(result.to_vec().unwrap(), vec![-1, -1, -1, -1]);
}

#[test]
fn test_sign_i32_mixed() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1, 2, -3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_eq!(result.to_vec().unwrap(), vec![-1, 1, -1, 1]);
}

#[test]
fn test_sign_i32_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0, 0, 0, 0];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_eq!(result.to_vec().unwrap(), vec![0, 0, 0, 0]);
}

#[test]
fn test_sign_u32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0u32, 1, 2, 100];
    let t = Tensor::<u32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    assert_eq!(result.to_vec().unwrap(), vec![0, 1, 1, 1]);
}

#[test]
fn test_sign_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(f32::from).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x < 0.0 {
                -1.0
            } else if x > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[-PI]).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, -1.0, epsilon = 1e-4);
    }
}

#[test]
fn test_sign_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[-42.0]).unwrap();
    let result = t.sign().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], -1.0, epsilon = 1e-4);
}
