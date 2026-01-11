//! Tests for `Tensor::random_normal` operation.

use approx::assert_relative_eq;
use xnn::{Context, Tensor};

const SAMPLE_SIZE: usize = 10000;

#[test]
fn test_random_normal_shape() {
    let ctx = Context::new().unwrap();

    let t = Tensor::<f32>::random_normal(&ctx, &[], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[] as &[usize]);

    let t = Tensor::<f32>::random_normal(&ctx, &[10], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[10]);

    let t = Tensor::<f32>::random_normal(&ctx, &[2, 3], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[2, 3]);

    let t = Tensor::<f32>::random_normal(&ctx, &[2, 3, 4], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[2, 3, 4]);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_random_normal_default_mean() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::random_normal(&ctx, &[SAMPLE_SIZE], None, None, Some(42.0)).unwrap();
    let data = t.to_vec().unwrap();
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert_relative_eq!(mean, 0.0, epsilon = 0.1);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_random_normal_default_variance() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::random_normal(&ctx, &[SAMPLE_SIZE], None, None, Some(42.0)).unwrap();
    let data = t.to_vec().unwrap();
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    assert_relative_eq!(variance, 1.0, epsilon = 0.1);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_random_normal_custom_mean() {
    let ctx = Context::new().unwrap();
    let t =
        Tensor::<f32>::random_normal(&ctx, &[SAMPLE_SIZE], Some(5.0), None, Some(42.0)).unwrap();
    let data = t.to_vec().unwrap();
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert_relative_eq!(mean, 5.0, epsilon = 0.1);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_random_normal_custom_scale() {
    let ctx = Context::new().unwrap();
    let t =
        Tensor::<f32>::random_normal(&ctx, &[SAMPLE_SIZE], None, Some(2.0), Some(42.0)).unwrap();
    let data = t.to_vec().unwrap();
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    assert_relative_eq!(variance, 4.0, epsilon = 0.1);
}

#[test]
fn test_random_normal_seed_reproducibility() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, Some(42.0)).unwrap();
    let t2 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, Some(42.0)).unwrap();
    assert_eq!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_normal_different_seeds() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, Some(42.0)).unwrap();
    let t2 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, Some(43.0)).unwrap();
    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_normal_auto_seed() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, None).unwrap();
    let t2 = Tensor::<f32>::random_normal(&ctx, &[100], None, None, None).unwrap();
    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_normal_zero_dimension_error() {
    let ctx = Context::new().unwrap();
    assert!(Tensor::<f32>::random_normal(&ctx, &[0], None, None, Some(42.0)).is_err());
}

#[test]
fn test_random_normal_zero_dimension_middle_error() {
    let ctx = Context::new().unwrap();
    assert!(Tensor::<f32>::random_normal(&ctx, &[2, 0, 3], None, None, Some(42.0)).is_err());
}
