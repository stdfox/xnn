//! Tensor benchmarks.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use xnn::{Context, Tensor};

const SIZES: &[(&str, &[usize])] = &[
    ("1048576", &[1048576]),
    ("2048x2048", &[2048, 2048]),
    ("256x256x128", &[256, 256, 128]),
    ("128x64x64x32", &[128, 64, 64, 32]),
];

fn configure<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.warm_up_time(Duration::from_millis(1000));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);
    group
}

fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| rng.random()).collect()
}

pub fn bench_constant(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/constant");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), dims, |bencher, dims| {
            bencher.iter(|| {
                let _ = Tensor::<f32>::constant(&ctx, dims, &[42.0]).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}

pub fn bench_copy(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/copy");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let tensor = Tensor::<f32>::constant(&ctx, dims, &random_vec(len)).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &tensor,
            |bencher, tensor| {
                bencher.iter(|| {
                    let _ = tensor.copy().unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group!(benches, bench_constant, bench_copy);
criterion::criterion_main!(benches);
