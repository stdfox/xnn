//! Tests for `Tensor::matmul` operation.

#![allow(clippy::cast_precision_loss)]

use xnn::{Context, Tensor};

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_transpose_a(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[l * m + i] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_transpose_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_transpose_both(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[l * m + i] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[test]
fn test_matmul_2d_basic() {
    let ctx = Context::new().unwrap();

    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[3, 4], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[2, 4]);
    crate::assert_vec_relative_eq(
        &c.to_vec().unwrap(),
        &cpu_matmul(&a_data, &b_data, 2, 3, 4),
        1e-4,
    );
}

#[test]
fn test_matmul_2d_square() {
    let ctx = Context::new().unwrap();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[2, 2], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[2, 2]);
    crate::assert_vec_relative_eq(&c.to_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-4);
}

#[test]
fn test_matmul_2d_transpose_a() {
    let ctx = Context::new().unwrap();

    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[3, 2], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[3, 4], &b_data).unwrap();
    let c = a.matmul(&b, true, false).unwrap();

    assert_eq!(c.dimensions(), &[2, 4]);
    crate::assert_vec_relative_eq(
        &c.to_vec().unwrap(),
        &cpu_matmul_transpose_a(&a_data, &b_data, 2, 3, 4),
        1e-4,
    );
}

#[test]
fn test_matmul_2d_transpose_b() {
    let ctx = Context::new().unwrap();

    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[4, 3], &b_data).unwrap();
    let c = a.matmul(&b, false, true).unwrap();

    assert_eq!(c.dimensions(), &[2, 4]);
    crate::assert_vec_relative_eq(
        &c.to_vec().unwrap(),
        &cpu_matmul_transpose_b(&a_data, &b_data, 2, 3, 4),
        1e-4,
    );
}

#[test]
fn test_matmul_2d_transpose_both() {
    let ctx = Context::new().unwrap();

    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[3, 2], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[4, 3], &b_data).unwrap();
    let c = a.matmul(&b, true, true).unwrap();

    assert_eq!(c.dimensions(), &[2, 4]);
    crate::assert_vec_relative_eq(
        &c.to_vec().unwrap(),
        &cpu_matmul_transpose_both(&a_data, &b_data, 2, 3, 4),
        1e-4,
    );
}

#[test]
fn test_matmul_2d_large() {
    let ctx = Context::new().unwrap();

    let m = 128;
    let k = 64;
    let n = 96;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i % 10) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i + 1) % 10) as f32 * 0.1).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[m, k], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[k, n], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[m, n]);
    crate::assert_vec_relative_eq(
        &c.to_vec().unwrap(),
        &cpu_matmul(&a_data, &b_data, m, k, n),
        1e-4,
    );
}

#[test]
fn test_matmul_3d() {
    let ctx = Context::new().unwrap();

    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 5;

    let a_data: Vec<f32> = (0..(batch * m * k)).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..(batch * k * n))
        .map(|i| ((i + 1) % 10) as f32)
        .collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[batch, m, k], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[batch, k, n], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[batch, m, n]);
    let result = c.to_vec().unwrap();

    for b_idx in 0..batch {
        let a_slice = &a_data[b_idx * m * k..(b_idx + 1) * m * k];
        let b_slice = &b_data[b_idx * k * n..(b_idx + 1) * k * n];
        let result = &result[b_idx * m * n..(b_idx + 1) * m * n];
        crate::assert_vec_relative_eq(result, &cpu_matmul(a_slice, b_slice, m, k, n), 1e-4);
    }
}

#[test]
fn test_matmul_3d_broadcast() {
    let ctx = Context::new().unwrap();

    let m = 3;
    let k = 4;
    let n = 5;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..(2 * k * n)).map(|i| ((i + 1) % 10) as f32).collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[1, m, k], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[2, k, n], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[2, m, n]);
    let result = c.to_vec().unwrap();

    for b_idx in 0..2 {
        let a_slice = &a_data[0..m * k];
        let b_slice = &b_data[b_idx * k * n..(b_idx + 1) * k * n];
        let result = &result[b_idx * m * n..(b_idx + 1) * m * n];
        crate::assert_vec_relative_eq(result, &cpu_matmul(a_slice, b_slice, m, k, n), 1e-4);
    }
}

#[test]
fn test_matmul_4d() {
    let ctx = Context::new().unwrap();

    let b0 = 2;
    let b1 = 3;
    let m = 4;
    let k = 5;
    let n = 6;

    let a_data: Vec<f32> = (0..(b0 * b1 * m * k)).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..(b0 * b1 * k * n))
        .map(|i| ((i + 1) % 10) as f32)
        .collect();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[b0, b1, m, k], &a_data).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[b0, b1, k, n], &b_data).unwrap();
    let c = a.matmul(&b, false, false).unwrap();

    assert_eq!(c.dimensions(), &[b0, b1, m, n]);
    let result = c.to_vec().unwrap();

    let a_slice = &a_data[0..m * k];
    let b_slice = &b_data[0..k * n];
    let result = &result[0..m * n];
    crate::assert_vec_relative_eq(result, &cpu_matmul(a_slice, b_slice, m, k, n), 1e-4);
}

#[test]
fn test_matmul_error_rank_too_low() {
    let ctx = Context::new().unwrap();

    let a = Tensor::<f32>::from_slice(&ctx, &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::<f32>::from_slice(&ctx, &[4.0, 5.0, 6.0]).unwrap();

    assert!(a.matmul(&b, false, false).is_err());
}

#[test]
fn test_matmul_error_rank_mismatch() {
    let ctx = Context::new().unwrap();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[0.0; 6]).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3, 4], &[0.0; 24]).unwrap();

    assert!(a.matmul(&b, false, false).is_err());
}

#[test]
fn test_matmul_error_dim_mismatch() {
    let ctx = Context::new().unwrap();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3], &[0.0; 6]).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[4, 5], &[0.0; 20]).unwrap();

    assert!(a.matmul(&b, false, false).is_err());
}

#[test]
fn test_matmul_error_batch_incompatible() {
    let ctx = Context::new().unwrap();

    let a = Tensor::<f32>::from_shape_slice(&ctx, &[2, 3, 4], &[0.0; 24]).unwrap();
    let b = Tensor::<f32>::from_shape_slice(&ctx, &[3, 4, 5], &[0.0; 60]).unwrap();

    assert!(a.matmul(&b, false, false).is_err());
}
