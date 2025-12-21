//! Tests for `Tensor::rem` operation.

use xnn::{Context, Tensor};

use super::{VECTOR_I32_A, VECTOR_I32_B, VECTOR_U32_A, VECTOR_U32_B};

#[test]
fn test_rem_same_shape() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 0, 0]);
}

#[test]
fn test_rem_same_shape_2d() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_shape_slice(&ctx, &[2, 3], &[10, 20, 30, 40, 50, 60]).unwrap();
    let b = Tensor::<i32>::from_shape_slice(&ctx, &[2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[2, 3]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_rem_scalar_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, &[5, 6, 7, 8]).unwrap();
    let b = Tensor::<i32>::constant(&ctx, &[], &[3]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![2, 0, 1, 2]);
}

#[test]
fn test_rem_scalar_broadcast_reverse() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::constant(&ctx, &[], &[10]).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 1, 2]);
}

#[test]
fn test_rem_trailing_broadcast() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_shape_slice(&ctx, &[2, 3], &[10, 20, 30, 40, 50, 60]).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, &[10, 20, 30]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[2, 3]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 0, 0, 10, 0]);
}

#[test]
fn test_rem_expand_both() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_shape_slice(&ctx, &[3, 1], &[1, 2, 3]).unwrap();
    let b = Tensor::<i32>::from_shape_slice(&ctx, &[1, 4], &[10, 20, 30, 40]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[3, 4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
}

#[test]
fn test_rem_incompatible_shapes() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, &[1, 2, 3]).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    assert!(a.rem(&b).is_err());
}

#[test]
fn test_rem_i32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_B).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, VECTOR_I32_A).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 0, 0]);
}

#[test]
fn test_rem_u32() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_B).unwrap();
    let b = Tensor::<u32>::from_slice(&ctx, VECTOR_U32_A).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[4]);
    let out = c.to_vec().unwrap();
    assert_eq!(out, vec![0, 0, 0, 0]);
}

#[test]
fn test_rem_non_aligned_i32() {
    let ctx = Context::try_default().unwrap();
    let data_a: Vec<i32> = (1_i32..43).map(|i| i * 10 + 3).collect();
    let data_b: Vec<i32> = (1_i32..43).map(|i| i + 1).collect();
    let a = Tensor::<i32>::from_slice(&ctx, &data_a).unwrap();
    let b = Tensor::<i32>::from_slice(&ctx, &data_b).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[42]);
    let out = c.to_vec().unwrap();
    let expected: Vec<i32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x % y)
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn test_rem_large() {
    let ctx = Context::try_default().unwrap();
    let len = 4096 * 4096;
    let a = Tensor::<i32>::constant(&ctx, &[len], &[17]).unwrap();
    let b = Tensor::<i32>::constant(&ctx, &[len], &[5]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[len]);
    let out = c.to_vec().unwrap();
    for val in &out[..100] {
        assert_eq!(*val, 2);
    }
}

#[test]
fn test_rem_scalar_to_scalar() {
    let ctx = Context::try_default().unwrap();
    let a = Tensor::<i32>::constant(&ctx, &[], &[10]).unwrap();
    let b = Tensor::<i32>::constant(&ctx, &[], &[3]).unwrap();
    let c = a.rem(&b).unwrap();
    assert_eq!(c.dimensions(), &[] as &[usize]);
    let out = c.to_vec().unwrap();
    assert_eq!(out[0], 1);
}
