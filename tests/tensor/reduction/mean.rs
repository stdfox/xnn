//! Mean reduction tests.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

fn assert_approx(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(a, e, epsilon = epsilon);
    }
}

#[test]
fn test_mean_reduce_2d_axis0() {
    let ctx = Context::new().unwrap();

    let a =
        Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = a.mean_reduce(&[0]).unwrap();

    assert_eq!(result.dimensions(), &[1, 3]);
    assert_approx(&result.to_vec().unwrap(), &[2.5, 3.5, 4.5], 1e-4);
}

#[test]
fn test_mean_reduce_2d_axis1() {
    let ctx = Context::new().unwrap();

    let a =
        Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = a.mean_reduce(&[1]).unwrap();

    assert_eq!(result.dimensions(), &[2, 1]);
    assert_approx(&result.to_vec().unwrap(), &[2.0, 5.0], 1e-4);
}

#[test]
fn test_mean_reduce_2d_all_axes() {
    let ctx = Context::new().unwrap();

    let a =
        Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = a.mean_reduce(&[0, 1]).unwrap();

    assert_eq!(result.dimensions(), &[1, 1]);
    assert_approx(&result.to_vec().unwrap(), &[3.5], 1e-4);
}

#[test]
fn test_mean_reduce_3d_axis0() {
    let ctx = Context::new().unwrap();

    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2, 3], &data).unwrap();
    let result = a.mean_reduce(&[0]).unwrap();

    assert_eq!(result.dimensions(), &[1, 2, 3]);
    assert_approx(
        &result.to_vec().unwrap(),
        &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-4,
    );
}

#[test]
fn test_mean_reduce_3d_axis1() {
    let ctx = Context::new().unwrap();

    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2, 3], &data).unwrap();
    let result = a.mean_reduce(&[1]).unwrap();

    assert_eq!(result.dimensions(), &[2, 1, 3]);
    assert_approx(
        &result.to_vec().unwrap(),
        &[2.5, 3.5, 4.5, 8.5, 9.5, 10.5],
        1e-4,
    );
}

#[test]
fn test_mean_reduce_3d_axis2() {
    let ctx = Context::new().unwrap();

    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2, 3], &data).unwrap();
    let result = a.mean_reduce(&[2]).unwrap();

    assert_eq!(result.dimensions(), &[2, 2, 1]);
    assert_approx(&result.to_vec().unwrap(), &[2.0, 5.0, 8.0, 11.0], 1e-4);
}

#[test]
fn test_mean_reduce_3d_multiple_axes() {
    let ctx = Context::new().unwrap();

    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2, 3], &data).unwrap();
    let result = a.mean_reduce(&[1, 2]).unwrap();

    assert_eq!(result.dimensions(), &[2, 1, 1]);
    assert_approx(&result.to_vec().unwrap(), &[3.5, 9.5], 1e-4);
}

#[test]
fn test_mean_reduce_non_aligned() {
    let ctx = Context::new().unwrap();

    let a = Tensor::<f32>::from_shape_slice(
        &ctx,
        &[2, 5],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let result = a.mean_reduce(&[1]).unwrap();

    assert_eq!(result.dimensions(), &[2, 1]);
    assert_approx(&result.to_vec().unwrap(), &[3.0, 8.0], 1e-4);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_mean_reduce_large() {
    let ctx = Context::new().unwrap();

    let size = 1024;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[size, size], &data).unwrap();
    let result = a.mean_reduce(&[1]).unwrap();

    assert_eq!(result.dimensions(), &[size, 1]);

    let out = result.to_vec().unwrap();
    for (row, &val) in out.iter().enumerate().take(size) {
        let expected = (row * size) as f32 + (size - 1) as f32 / 2.0;
        assert_relative_eq!(val, expected, epsilon = 1e-2);
    }
}

#[test]
fn test_mean_reduce_invalid_axis() {
    let ctx = Context::new().unwrap();

    let a =
        Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = a.mean_reduce(&[5]);

    assert!(result.is_err());
}

#[test]
fn test_mean_reduce_duplicate_axis() {
    let ctx = Context::new().unwrap();

    let a =
        Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = a.mean_reduce(&[1, 1]);

    assert!(result.is_err());
}
