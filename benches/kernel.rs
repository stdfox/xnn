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

fn bench_fill(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/fill");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let buf = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::fill(&ctx, &buf, 42.0f32);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/add");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::add(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/sub");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sub(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/mul");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::mul(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_div(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/div");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::div(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_rem(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/rem");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::rem(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_pow(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/pow");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::pow(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_add_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/add_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::add_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_sub_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/sub_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sub_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_mul_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/mul_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::mul_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_div_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/div_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::div_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_rem_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/rem_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::rem_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_pow_scalar(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/pow_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::pow_scalar(&ctx, &a, &b, &c);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_gemm(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/gemm");
    group.warm_up_time(Duration::from_millis(3000));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &size in SIZES {
        if size >= 4096 {
            group.sample_size(20);
        }

        let len = size * size;
        let a = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let b = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::gemm(&ctx, &a, &b, &c, size, size, size);
                kernel::sync(&ctx);
            });
        });
    }

    group.finish();
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

fn bench_sum(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();

    let mut group = c.benchmark_group("kernel/sum");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let input = ctx.create_buffer_from_slice(&random_vec(len)).unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sum(&ctx, &input, &output);
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
    bench_fill,
    bench_add,
    bench_sub,
    bench_mul,
    bench_div,
    bench_rem,
    bench_pow,
    bench_add_scalar,
    bench_sub_scalar,
    bench_mul_scalar,
    bench_div_scalar,
    bench_rem_scalar,
    bench_pow_scalar,
    bench_gemm,
    bench_transpose,
    bench_sum,
    bench_relu,
    bench_sigmoid,
    bench_broadcast_rows
);
criterion::criterion_main!(benches);
