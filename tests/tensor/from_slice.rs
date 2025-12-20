//! Tests for `Tensor::from_slice` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_from_slice_f32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    assert_eq!(t.shape(), &[4]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_from_slice_i32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1, 2, 3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    assert_eq!(t.shape(), &[4]);
    assert_eq!(t.to_vec().unwrap(), data);
}

#[test]
fn test_from_slice_u32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1u32, 2, 3, 4];
    let t = Tensor::<u32>::from_slice(&ctx, &data).unwrap();
    assert_eq!(t.shape(), &[4]);
    assert_eq!(t.to_vec().unwrap(), data);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_from_slice_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (0..42).map(|i| i as f32).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    assert_eq!(t.shape(), &[42]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_from_slice_single() {
    let ctx = Context::try_default().unwrap();
    let data = vec![42.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    assert_eq!(t.shape(), &[1]);
    let out = t.to_vec().unwrap();
    assert_relative_eq!(out[0], 42.0, epsilon = 1e-4);
}

#[test]
fn test_from_slice_empty_error() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = vec![];
    let result = Tensor::<f32>::from_slice(&ctx, &data);
    assert!(result.is_err());
}
