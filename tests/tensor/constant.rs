//! Tests for `Tensor::constant` operation.

use std::f32::consts::PI;

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

#[test]
fn test_constant_broadcast_f32() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[4], &[42.0]).unwrap();
    assert_eq!(t.dimensions(), &[4]);
    for val in &t.to_vec().unwrap() {
        assert_relative_eq!(*val, 42.0, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_broadcast_i32() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<i32>::constant(&ctx, &[4], &[42]).unwrap();
    assert_eq!(t.dimensions(), &[4]);
    assert_eq!(t.to_vec().unwrap(), vec![42, 42, 42, 42]);
}

#[test]
fn test_constant_broadcast_u32() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<u32>::constant(&ctx, &[4], &[42]).unwrap();
    assert_eq!(t.dimensions(), &[4]);
    assert_eq!(t.to_vec().unwrap(), vec![42, 42, 42, 42]);
}

#[test]
fn test_constant_broadcast_2d() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[2, 3], &[1.5]).unwrap();
    assert_eq!(t.dimensions(), &[2, 3]);
    for val in &t.to_vec().unwrap() {
        assert_relative_eq!(*val, 1.5, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_exact_fill() {
    let ctx = Context::new().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32>::constant(&ctx, &[4], &data).unwrap();
    assert_eq!(t.dimensions(), &[4]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_exact_fill_2d() {
    let ctx = Context::new().unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32>::constant(&ctx, &[2, 3], &data).unwrap();
    assert_eq!(t.dimensions(), &[2, 3]);
    for (a, b) in t.to_vec().unwrap().iter().zip(data.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_non_aligned() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[5], &[PI]).unwrap();
    assert_eq!(t.dimensions(), &[5]);
    for val in &t.to_vec().unwrap() {
        assert_relative_eq!(*val, PI, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_large() {
    let ctx = Context::new().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<f32>::constant(&ctx, &[len], &[PI]).unwrap();
    assert_eq!(t.dimensions(), &[len]);
    for val in &t.to_vec().unwrap() {
        assert_relative_eq!(*val, PI, epsilon = 1e-4);
    }
}

#[test]
fn test_constant_scalar() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::constant(&ctx, &[], &[42.0]).unwrap();
    assert_eq!(t.dimensions(), &[] as &[usize]);
    assert_relative_eq!(t.to_vec().unwrap()[0], 42.0, epsilon = 1e-4);
}

#[test]
fn test_constant_empty_value_error() {
    let ctx = Context::new().unwrap();
    let result = Tensor::<f32>::constant(&ctx, &[4], &[]);
    assert!(result.is_err());
}

#[test]
fn test_constant_zero_dimension_error() {
    let ctx = Context::new().unwrap();
    let result = Tensor::<f32>::constant(&ctx, &[0], &[1.0]);
    assert!(result.is_err());
}

#[test]
fn test_constant_length_mismatch_error() {
    let ctx = Context::new().unwrap();
    let result = Tensor::<f32>::constant(&ctx, &[4], &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}
