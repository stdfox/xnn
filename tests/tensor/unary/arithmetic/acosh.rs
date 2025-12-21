//! Tests for `Tensor::acosh` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_acosh_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 10.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acosh().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acosh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acosh_one() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acosh().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_acosh_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.acosh().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acosh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acosh_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (1_u8..43).map(f32::from).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acosh().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.acosh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_acosh_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[1.0]).unwrap();
    let result = t.acosh().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_acosh_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[2.0]).unwrap();
    let result = t.acosh().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 2.0_f32.acosh(), epsilon = 1e-4);
}

#[test]
fn test_acosh_cosh_inverse() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 1.5, 2.0, 3.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.acosh().unwrap().cosh().unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
