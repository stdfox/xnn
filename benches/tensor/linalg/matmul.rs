//! Matrix multiplication benchmarks.

use std::mem::size_of;
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
    group.sample_size(50);
    group
}

const MATMUL_SIZES: &[(&str, usize, usize, usize)] = &[
    ("64x64x64", 64, 64, 64),
    ("128x128x128", 128, 128, 128),
    ("256x256x256", 256, 256, 256),
    ("512x512x512", 512, 512, 512),
    ("1024x1024x1024", 1024, 1024, 1024),
    ("2048x2048x2048", 2048, 2048, 2048),
];

const BATCHED_SIZES: &[(&str, usize, usize, usize, usize)] = &[
    ("8x64x64x64", 8, 64, 64, 64),
    ("16x128x128x128", 16, 128, 128, 128),
    ("32x64x64x64", 32, 64, 64, 64),
];

fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| rng.random::<f32>()).collect()
}

pub(crate) fn bench_matmul(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/matmul");

    for &(name, m, k, n) in MATMUL_SIZES {
        let a_data = random_vec(m * k);
        let b_data = random_vec(k * n);

        let a = Tensor::<f32>::from_shape_slice(&ctx, &[m, k], &a_data).unwrap();
        let b = Tensor::<f32>::from_shape_slice(&ctx, &[k, n], &b_data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: (2 * m * k * n) as u64,
            bytes: ((m * k + k * n + m * n) * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let _ = a.matmul(b, false, false).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}

pub(crate) fn bench_matmul_transpose(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/matmul_transpose");

    let m = 512;
    let k = 512;
    let n = 512;

    #[allow(clippy::type_complexity)]
    let cases: &[(&str, bool, bool, [usize; 2], [usize; 2])] = &[
        ("NN", false, false, [m, k], [k, n]),
        ("TN", true, false, [k, m], [k, n]),
        ("NT", false, true, [m, k], [n, k]),
        ("TT", true, true, [k, m], [n, k]),
    ];

    for &(name, ta, tb, a_shape, b_shape) in cases {
        let a_data = random_vec(a_shape[0] * a_shape[1]);
        let b_data = random_vec(b_shape[0] * b_shape[1]);

        let a = Tensor::<f32>::from_shape_slice(&ctx, &a_shape, &a_data).unwrap();
        let b = Tensor::<f32>::from_shape_slice(&ctx, &b_shape, &b_data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: (2 * m * k * n) as u64,
            bytes: ((m * k + k * n + m * n) * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&a, &b, ta, tb),
            |bencher, (a, b, ta, tb)| {
                bencher.iter(|| {
                    let _ = a.matmul(b, *ta, *tb).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}

pub(crate) fn bench_matmul_batched(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/matmul_batched");

    for &(name, batch, m, k, n) in BATCHED_SIZES {
        let a_data = random_vec(batch * m * k);
        let b_data = random_vec(batch * k * n);

        let a = Tensor::<f32>::from_shape_slice(&ctx, &[batch, m, k], &a_data).unwrap();
        let b = Tensor::<f32>::from_shape_slice(&ctx, &[batch, k, n], &b_data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: (2 * batch * m * k * n) as u64,
            bytes: (batch * (m * k + k * n + m * n) * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let _ = a.matmul(b, false, false).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}
