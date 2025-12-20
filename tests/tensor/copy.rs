//! Tests for `Tensor::copy` operation.

use std::f32::consts::PI;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_copy_f32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), t.shape());
    for (a, b) in copy.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_copy_i32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1, 2, 3, 4];
    let t = Tensor::<i32>::from_slice(&ctx, &data).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), t.shape());
    assert_eq!(copy.to_vec().unwrap(), data);
}

#[test]
fn test_copy_u32() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1u32, 2, 3, 4];
    let t = Tensor::<u32>::from_slice(&ctx, &data).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), t.shape());
    assert_eq!(copy.to_vec().unwrap(), data);
}

#[test]
fn test_copy_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), &[2, 3]);
    for (a, b) in copy.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_copy_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (0..42).map(|i| i as f32).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), &[42]);
    for (a, b) in copy.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_copy_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[PI]).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), &[len]);
    for val in &copy.to_vec().unwrap() {
        assert_relative_eq!(*val, PI, epsilon = 1e-4);
    }
}

#[test]
fn test_copy_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[42.0]).unwrap();
    let copy = t.copy().unwrap();
    assert_eq!(copy.shape(), &[] as &[usize]);
    assert_relative_eq!(copy.to_vec().unwrap()[0], 42.0, epsilon = 1e-4);
}

#[test]
fn test_copy_independence() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let original = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let copy = original.copy().unwrap();
    let out_orig = original.to_vec().unwrap();
    let out_copy = copy.to_vec().unwrap();
    for (a, b) in out_orig.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
    for (a, b) in out_copy.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
