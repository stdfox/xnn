//! Tests for `Tensor::from_shape_slice` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_from_shape_slice_f32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_from_shape_slice_i32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1, 2, 3, 4, 5, 6];
    let t = Tensor::<i32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec().unwrap(), data);
}

#[test]
fn test_from_shape_slice_u32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1u32, 2, 3, 4, 5, 6];
    let t = Tensor::<u32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec().unwrap(), data);
}

#[test]
fn test_from_shape_slice_3d() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (0_u8..24).map(f32::from).collect();
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3, 4], &data).unwrap();
    assert_eq!(t.shape(), &[2, 3, 4]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_from_shape_slice_1d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[4], &data).unwrap();
    assert_eq!(t.shape(), &[4]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_from_shape_slice_scalar() {
    let ctx = Context::try_default().unwrap();
    let data = vec![42.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[], &data).unwrap();
    assert_eq!(t.shape(), &[] as &[usize]);
    let out = t.to_vec().unwrap();
    assert_relative_eq!(out[0], 42.0, epsilon = 1e-4);
}

#[test]
fn test_from_shape_slice_length_mismatch() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0];
    let result = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data);
    assert!(result.is_err());
}
