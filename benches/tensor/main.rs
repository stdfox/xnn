//! Tensor benchmarks.

mod binary;
mod matrix;
mod reduce;
mod unary;

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use xnn::{Context, Tensor};

const SIZES: &[(&str, &[usize])] = &[
    ("1048576", &[1_048_576]),
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

trait RandomValue {
    fn random_value(rng: &mut StdRng) -> Self;
    fn random_nonzero(rng: &mut StdRng) -> Self;
}

impl RandomValue for f32 {
    fn random_value(rng: &mut StdRng) -> Self {
        rng.random()
    }
    fn random_nonzero(rng: &mut StdRng) -> Self {
        rng.random_range(0.1..1.0)
    }
}

impl RandomValue for i32 {
    fn random_value(rng: &mut StdRng) -> Self {
        rng.random_range(1..1000)
    }
    fn random_nonzero(rng: &mut StdRng) -> Self {
        rng.random_range(1..100)
    }
}

impl RandomValue for u32 {
    fn random_value(rng: &mut StdRng) -> Self {
        rng.random_range(1..1000)
    }
    fn random_nonzero(rng: &mut StdRng) -> Self {
        rng.random_range(1..100)
    }
}

impl RandomValue for bool {
    fn random_value(rng: &mut StdRng) -> Self {
        rng.random()
    }
    fn random_nonzero(rng: &mut StdRng) -> Self {
        rng.random()
    }
}

fn random_vec<T: RandomValue>(len: usize) -> Vec<T> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| T::random_value(&mut rng)).collect()
}

fn random_vec_nonzero<T: RandomValue>(len: usize) -> Vec<T> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len).map(|_| T::random_nonzero(&mut rng)).collect()
}

fn bench_constant(c: &mut Criterion) {
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

fn bench_copy(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/copy");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let data: Vec<f32> = random_vec(len);
        let t = Tensor::<f32>::from_shape_slice(&ctx, dims, &data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), dims, |bencher, _| {
            bencher.iter(|| {
                let _ = t.copy().unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}

criterion::criterion_group!(
    benches,
    bench_constant,
    bench_copy,
    // Binary arithmetic
    binary::arithmetic::bench_add,
    binary::arithmetic::bench_sub,
    binary::arithmetic::bench_mul,
    binary::arithmetic::bench_div,
    binary::arithmetic::bench_rem,
    binary::arithmetic::bench_pow,
    // Binary comparison
    binary::comparison::bench_lt,
    binary::comparison::bench_gt,
    binary::comparison::bench_le,
    binary::comparison::bench_ge,
    binary::comparison::bench_eq,
    binary::comparison::bench_ne,
    // Binary logical
    binary::logical::bench_and,
    binary::logical::bench_or,
    // Matrix matmul
    matrix::matmul::bench_matmul,
    matrix::matmul::bench_matmul_transpose,
    matrix::matmul::bench_matmul_batched,
    // Reduce max_reduce
    reduce::max::bench_max_reduce,
    reduce::max::bench_max_reduce_axis0,
    reduce::max::bench_max_reduce_axis1,
    // Reduce min_reduce
    reduce::min::bench_min_reduce,
    reduce::min::bench_min_reduce_axis0,
    reduce::min::bench_min_reduce_axis1,
    // Reduce sum_reduce
    reduce::sum::bench_sum_reduce,
    reduce::sum::bench_sum_reduce_axis0,
    reduce::sum::bench_sum_reduce_axis1,
    // Unary arithmetic
    unary::arithmetic::bench_abs,
    unary::arithmetic::bench_acos,
    unary::arithmetic::bench_acosh,
    unary::arithmetic::bench_asin,
    unary::arithmetic::bench_asinh,
    unary::arithmetic::bench_atan,
    unary::arithmetic::bench_atanh,
    unary::arithmetic::bench_cos,
    unary::arithmetic::bench_cosh,
    unary::arithmetic::bench_exp,
    unary::arithmetic::bench_log,
    unary::arithmetic::bench_neg,
    unary::arithmetic::bench_rcp,
    unary::arithmetic::bench_sign,
    unary::arithmetic::bench_sin,
    unary::arithmetic::bench_sinh,
    unary::arithmetic::bench_tan,
    unary::arithmetic::bench_tanh,
    // Unary logical
    unary::logical::bench_not,
    // Unary rounding
    unary::rounding::bench_ceil,
    unary::rounding::bench_floor,
    unary::rounding::bench_round,
);
criterion::criterion_main!(benches);
