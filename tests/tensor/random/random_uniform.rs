//! Tests for `Tensor::random_uniform` operation.

use xnn::{Context, Tensor};

const SAMPLE_SIZE: usize = 10000;

#[test]
fn test_random_uniform_shape() {
    let ctx = Context::new().unwrap();

    let t = Tensor::<f32>::random_uniform(&ctx, &[], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[] as &[usize]);

    let t = Tensor::<f32>::random_uniform(&ctx, &[10], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[10]);

    let t = Tensor::<f32>::random_uniform(&ctx, &[2, 3], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[2, 3]);

    let t = Tensor::<f32>::random_uniform(&ctx, &[2, 3, 4], None, None, Some(42.0)).unwrap();
    assert_eq!(t.dimensions(), &[2, 3, 4]);
}

#[test]
fn test_random_uniform_default_range() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::random_uniform(&ctx, &[SAMPLE_SIZE], None, None, Some(42.0)).unwrap();
    for &v in &t.to_vec().unwrap() {
        assert!((0.0..1.0).contains(&v));
    }
}

#[test]
fn test_random_uniform_custom_range() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::random_uniform(&ctx, &[SAMPLE_SIZE], Some(5.0), Some(10.0), Some(42.0))
        .unwrap();
    for &v in &t.to_vec().unwrap() {
        assert!((5.0..10.0).contains(&v));
    }
}

#[test]
fn test_random_uniform_negative_range() {
    let ctx = Context::new().unwrap();
    let t = Tensor::<f32>::random_uniform(&ctx, &[SAMPLE_SIZE], Some(-5.0), Some(-1.0), Some(42.0))
        .unwrap();
    for &v in &t.to_vec().unwrap() {
        assert!((-5.0..-1.0).contains(&v));
    }
}

#[test]
fn test_random_uniform_seed_reproducibility() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, Some(42.0)).unwrap();
    let t2 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, Some(42.0)).unwrap();
    assert_eq!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_uniform_different_seeds() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, Some(42.0)).unwrap();
    let t2 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, Some(43.0)).unwrap();
    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_uniform_auto_seed() {
    let ctx = Context::new().unwrap();
    let t1 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, None).unwrap();
    let t2 = Tensor::<f32>::random_uniform(&ctx, &[100], None, None, None).unwrap();
    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn test_random_uniform_zero_dimension_error() {
    let ctx = Context::new().unwrap();
    assert!(Tensor::<f32>::random_uniform(&ctx, &[0], None, None, Some(42.0)).is_err());
}

#[test]
fn test_random_uniform_zero_dimension_middle_error() {
    let ctx = Context::new().unwrap();
    assert!(Tensor::<f32>::random_uniform(&ctx, &[2, 0, 3], None, None, Some(42.0)).is_err());
}
