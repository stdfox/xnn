//! Tests for `Tensor::not` operation.

use xnn::{Context, Tensor};

#[test]
fn test_not_basic() {
    let ctx = Context::try_default().unwrap();
    let data = vec![true, false, true, false];
    let t = Tensor::<bool>::from_slice(&ctx, &data).unwrap();
    let result = t.not().unwrap();
    assert_eq!(result.dimensions(), t.dimensions());
    let out = result.to_vec().unwrap();
    let expected = [false, true, false, true];
    assert_eq!(out, expected);
}

#[test]
fn test_not_all_true() {
    let ctx = Context::try_default().unwrap();
    let data = vec![true; 8];
    let t = Tensor::<bool>::from_slice(&ctx, &data).unwrap();
    let result = t.not().unwrap();
    for val in &result.to_vec().unwrap() {
        assert!(!*val);
    }
}

#[test]
fn test_not_all_false() {
    let ctx = Context::try_default().unwrap();
    let data = vec![false; 8];
    let t = Tensor::<bool>::from_slice(&ctx, &data).unwrap();
    let result = t.not().unwrap();
    for val in &result.to_vec().unwrap() {
        assert!(*val);
    }
}

#[test]
fn test_not_2d() {
    let ctx = Context::try_default().unwrap();
    let data = vec![true, false, true, false, true, false];
    let t = Tensor::<bool>::from_shape_slice(&ctx, &[2, 3], &data).unwrap();
    let result = t.not().unwrap();
    assert_eq!(result.dimensions(), &[2, 3]);
    let out = result.to_vec().unwrap();
    let expected = vec![false, true, false, true, false, true];
    assert_eq!(out, expected);
}

#[test]
fn test_not_non_aligned() {
    let ctx = Context::try_default().unwrap();
    let data: Vec<bool> = (0..42).map(|i| i % 2 == 0).collect();
    let t = Tensor::<bool>::from_slice(&ctx, &data).unwrap();
    let result = t.not().unwrap();
    assert_eq!(result.dimensions(), &[42]);
    let out = result.to_vec().unwrap();
    let expected: Vec<bool> = data.iter().map(|x| !*x).collect();
    assert_eq!(out, expected);
}

#[test]
fn test_not_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let t = Tensor::<bool>::constant(&ctx, &[len], &[true]).unwrap();
    let result = t.not().unwrap();
    assert_eq!(result.dimensions(), &[len]);
    for val in &result.to_vec().unwrap() {
        assert!(!*val);
    }
}

#[test]
fn test_not_scalar() {
    let ctx = Context::try_default().unwrap();
    let t = Tensor::<bool>::constant(&ctx, &[], &[true]).unwrap();
    let result = t.not().unwrap();
    assert_eq!(result.dimensions(), &[] as &[usize]);
    assert!(!result.to_vec().unwrap()[0]);
}

#[test]
fn test_not_double() {
    // not(not(x)) == x
    let ctx = Context::try_default().unwrap();
    let data = vec![true, false, true, false];
    let t = Tensor::<bool>::from_slice(&ctx, &data).unwrap();
    let result = t.not().unwrap().not().unwrap();
    assert_eq!(result.to_vec().unwrap(), data);
}
