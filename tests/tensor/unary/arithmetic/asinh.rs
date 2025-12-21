//! Tests for `Tensor::asinh` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_asinh_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 1.0, -1.0, 2.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asinh().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asinh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asinh().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_odd() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.0, 2.0, 3.0, 0.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let pos = t.asinh().unwrap().to_vec().unwrap();
    let neg_data: Vec<f32> = data.iter().map(|x| -x).collect();
    let t_neg = Tensor::<f32>::from_slice(&ctx, &neg_data).unwrap();
    let neg = t_neg.asinh().unwrap().to_vec().unwrap();
    for (p, n) in pos.iter().zip(neg.iter()) {
        assert_relative_eq!(*p, -*n, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.5, 1.0, -0.5, -1.0, 2.0];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.asinh().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asinh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i8..21).map(|i| f32::from(i) * 0.1).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asinh().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<f32> = data.iter().map(|x| x.asinh()).collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[0.0]).unwrap();
    let result = t.asinh().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_asinh_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[1.0]).unwrap();
    let result = t.asinh().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 1.0_f32.asinh(), epsilon = 1e-4);
}

#[test]
fn test_asinh_sinh_inverse() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, 0.5, 1.0, -0.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.asinh().unwrap().sinh().unwrap();
    let out = result.to_vec().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}
