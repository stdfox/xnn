//! Tests for `Tensor::round` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_round_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![1.2, 2.7, -1.2, -2.7];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.round().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected = [1.0, 3.0, -1.0, -3.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_round_zero() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.0, -0.0, 0.0, -0.0];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.round().unwrap();
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-4);
    }
}

#[test]
fn test_round_half() {
    let ctx = Context::try_default().unwrap();
    // WGSL round uses round-half-to-even (banker's rounding)
    let data = vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5];
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.round().unwrap();
    let out = result.to_vec().unwrap();
    // Round-half-to-even: 0.5->0, 1.5->2, 2.5->2, 3.5->4
    let expected = [0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0, -4.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_round_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![0.1, 0.9, -0.1, -0.9, 1.5, -1.5];
    let t = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.round().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected = [0.0, 1.0, 0.0, -1.0, 2.0, -2.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_round_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<f32> = (-21_i16..21).map(|i| f32::from(i) * 0.3).collect();
    let t = Tensor::<f32>::from_slice(&ctx, &data).unwrap();
    let result = t.round().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    // Use round-half-to-even for comparison
    let expected: Vec<f32> = data
        .iter()
        .map(|x| {
            let r = x.round();
            let diff = (x - r).abs();
            if (diff - 0.5).abs() < 1e-6 {
                if r.rem_euclid(2.0) < 0.5 {
                    r
                } else {
                    r - x.signum()
                }
            } else {
                r
            }
        })
        .collect();
    for (a, b) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_round_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[1.7]).unwrap();
    let result = t.round().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert_relative_eq!(*val, 2.0, epsilon = 1e-4);
    }
}

#[test]
fn test_round_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[1.7]).unwrap();
    let result = t.round().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert_relative_eq!(result.to_vec().unwrap()[0], 2.0, epsilon = 1e-4);
}
