//! Tests for `Tensor::rcp` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_rcp_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.5, 1.0, 2.0, 4.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.rcp().unwrap();
    assert_eq!(result.shape(), t.shape());
    let out = result.to_vec().unwrap();
    let expected = [2.0, 1.0, 0.5, 0.25];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_rcp_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 4.0, 5.0, 10.0, 20.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.rcp().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_rcp_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (1_u8..43).map(f32::from).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.rcp().unwrap();
    assert_eq!(result.shape(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| 1.0 / x).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_rcp_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[4.0]).unwrap();
    let result = t.rcp().unwrap();
    assert_eq!(result.shape(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.25, epsilon = 1e-4);
    }
}

#[test]
fn test_rcp_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[8.0]).unwrap();
    let result = t.rcp().unwrap();
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 0.125, epsilon = 1e-4);
}
