//! Kernel benchmarks.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion};
use xnn::{GpuContext, kernel};

const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096];

fn configure(group: &mut BenchmarkGroup<WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(100);
}

fn bench_fill(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/fill");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let buf = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::fill(&ctx, &buf, 42.0f32).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/add");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 1.0f32).unwrap();
        kernel::fill(&ctx, &b, 2.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::add(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/sub");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 5.0f32).unwrap();
        kernel::fill(&ctx, &b, 2.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sub(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/mul");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 2.0f32).unwrap();
        kernel::fill(&ctx, &b, 3.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::mul(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_div(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/div");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 6.0f32).unwrap();
        kernel::fill(&ctx, &b, 2.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::div(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rem(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/rem");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 7.0f32).unwrap();
        kernel::fill(&ctx, &b, 3.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::rem(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_pow(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/pow");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 2.0f32).unwrap();
        kernel::fill(&ctx, &b, 3.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::pow(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_add_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/add_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[2.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 1.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::add_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_sub_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/sub_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[3.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 10.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sub_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_mul_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/mul_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[3.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 2.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::mul_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_div_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/div_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[3.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 12.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::div_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rem_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/rem_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[3.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 10.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::rem_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_pow_scalar(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/pow_scalar");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer_from_slice(&[3.0f32]).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 2.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::pow_scalar(&ctx, &a, &b, &c).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gemm(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/gemm");
    group.warm_up_time(Duration::from_millis(3000));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &size in SIZES {
        if size >= 4096 {
            group.sample_size(20);
        }

        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();
        let c = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 1.0f32).unwrap();
        kernel::fill(&ctx, &b, 1.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::gemm(&ctx, &a, &b, &c, size, size, size).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/sum");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let input = ctx.create_buffer::<f32>(len).unwrap();
        let output = ctx.create_buffer::<f32>(1).unwrap();

        kernel::fill(&ctx, &input, 1.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sum(&ctx, &input, &output).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/relu");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, -1.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::relu(&ctx, &a, &b).unwrap();
                kernel::sync(&ctx).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/sigmoid");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let a = ctx.create_buffer::<f32>(len).unwrap();
        let b = ctx.create_buffer::<f32>(len).unwrap();

        kernel::fill(&ctx, &a, 0.0f32).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                kernel::sigmoid(&ctx, &a, &b).unwrap();
                kernel::sync(&ctx).unwrap();
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
    bench_sum,
    bench_relu,
    bench_sigmoid
);
criterion::criterion_main!(benches);
