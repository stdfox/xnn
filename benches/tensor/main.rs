//! Tensor benchmarks.

mod constant;
mod copy;
mod linalg;
mod math;
mod nn;
mod random;
mod reduction;

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

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

criterion::criterion_group!(
    benches,
    // Constant/copy
    constant::bench_constant,
    copy::bench_copy,
    // Random
    random::bench_random_normal,
    random::bench_random_uniform,
    // Linalg: matmul
    linalg::matmul::bench_matmul,
    linalg::matmul::bench_matmul_transpose,
    linalg::matmul::bench_matmul_batched,
    // Math: clamp, select
    math::clamp::bench_clamp,
    math::select::bench_select,
    // Math: binary
    math::binary::bench_add,
    math::binary::bench_div,
    math::binary::bench_max,
    math::binary::bench_min,
    math::binary::bench_mul,
    math::binary::bench_pow,
    math::binary::bench_rem,
    math::binary::bench_sub,
    math::binary::bench_lt,
    math::binary::bench_gt,
    math::binary::bench_le,
    math::binary::bench_ge,
    math::binary::bench_eq,
    math::binary::bench_ne,
    math::binary::bench_and,
    math::binary::bench_or,
    // Math: unary
    math::unary::bench_abs,
    math::unary::bench_acos,
    math::unary::bench_acosh,
    math::unary::bench_asin,
    math::unary::bench_asinh,
    math::unary::bench_atan,
    math::unary::bench_atanh,
    math::unary::bench_cos,
    math::unary::bench_cosh,
    math::unary::bench_exp,
    math::unary::bench_log,
    math::unary::bench_log2,
    math::unary::bench_neg,
    math::unary::bench_rcp,
    math::unary::bench_rsqr,
    math::unary::bench_rsqrt,
    math::unary::bench_sign,
    math::unary::bench_sin,
    math::unary::bench_sinh,
    math::unary::bench_sqr,
    math::unary::bench_sqrt,
    math::unary::bench_tan,
    math::unary::bench_tanh,
    math::unary::bench_ceil,
    math::unary::bench_floor,
    math::unary::bench_round,
    math::unary::bench_not,
    // NN: activation
    nn::activation::bench_elu,
    nn::activation::bench_gelu,
    nn::activation::bench_leaky_relu,
    nn::activation::bench_prelu,
    nn::activation::bench_relu,
    nn::activation::bench_selu,
    nn::activation::bench_sigmoid,
    nn::activation::bench_silu,
    nn::activation::bench_softplus,
    // Reduction
    reduction::max::bench_max_reduce,
    reduction::max::bench_max_reduce_axis0,
    reduction::max::bench_max_reduce_axis1,
    reduction::mean::bench_mean_reduce,
    reduction::mean::bench_mean_reduce_axis0,
    reduction::mean::bench_mean_reduce_axis1,
    reduction::min::bench_min_reduce,
    reduction::min::bench_min_reduce_axis0,
    reduction::min::bench_min_reduce_axis1,
    reduction::sum::bench_sum_reduce,
    reduction::sum::bench_sum_reduce_axis0,
    reduction::sum::bench_sum_reduce_axis1,
);
criterion::criterion_main!(benches);
