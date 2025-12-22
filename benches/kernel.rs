//! Kernel benchmarks.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xnn::{Context, kernel};

const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096];

fn configure(group: &mut BenchmarkGroup<WallTime>) {
    group.warm_up_time(Duration::from_millis(3000));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);
}

/// Generates a vector of random f32 values in [0, 1).
fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| rng.random()).collect()
}

fn bench_transpose(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/transpose");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::transpose(&ctx, &a, &b, size, size);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/relu");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::relu(&ctx, &a, &b);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/sigmoid");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sigmoid(&ctx, &a, &b);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_broadcast_rows(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/broadcast_rows");
    configure(&mut group);

    for &size in SIZES {
        let a = ctx.create_buffer_from_slice(&random_vec(size)).unwrap();
        let b = ctx.create_buffer::<f32>(size * size).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::broadcast_rows(&ctx, &a, &b, size, size);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

criterion::criterion_group!(
    benches,
    bench_transpose,
    bench_relu,
    bench_sigmoid,
    bench_broadcast_rows
);
criterion::criterion_main!(benches);
