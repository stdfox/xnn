//! Sum reduction operation benchmarks.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use xnn::{Context, Tensor};

fn configure<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.warm_up_time(Duration::from_millis(1000));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);
    group
}

const REDUCE_SIZES: &[(&str, usize, usize)] = &[
    ("256x256", 256, 256),
    ("512x512", 512, 512),
    ("1024x1024", 1024, 1024),
    ("2048x2048", 2048, 2048),
    ("4096x4096", 4096, 4096),
];

fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| rng.random::<f32>()).collect()
}

pub(crate) fn bench_sum_reduce(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/sum_reduce");

    for &(name, rows, cols) in REDUCE_SIZES {
        let len = rows * cols;
        let data = random_vec(len);
        let t = Tensor::<f32>::from_shape_slice(&ctx, &[rows, cols], &data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), &t, |bencher, t| {
            bencher.iter(|| {
                let _ = t.sum_reduce(&[0, 1], false).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}

pub(crate) fn bench_sum_reduce_axis0(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/sum_reduce_axis0");

    for &(name, rows, cols) in REDUCE_SIZES {
        let len = rows * cols;
        let data = random_vec(len);
        let t = Tensor::<f32>::from_shape_slice(&ctx, &[rows, cols], &data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), &t, |bencher, t| {
            bencher.iter(|| {
                let _ = t.sum_reduce(&[0], false).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}

pub(crate) fn bench_sum_reduce_axis1(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/sum_reduce_axis1");

    for &(name, rows, cols) in REDUCE_SIZES {
        let len = rows * cols;
        let data = random_vec(len);
        let t = Tensor::<f32>::from_shape_slice(&ctx, &[rows, cols], &data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), &t, |bencher, t| {
            bencher.iter(|| {
                let _ = t.sum_reduce(&[1], false).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}
