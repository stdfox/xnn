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

macro_rules! bench_unary_op {
    ($name:ident, $op:ident) => {
        fn $name(c: &mut Criterion) {
            let ctx = Context::try_default().unwrap();
            let mut group = configure(c, concat!("tensor/", stringify!($op)));

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
                            let _ = tensor.$op().unwrap();
                            ctx.poll().unwrap();
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

bench_unary_op!(bench_abs, abs);
bench_unary_op!(bench_acos, acos);
bench_unary_op!(bench_acosh, acosh);
bench_unary_op!(bench_asin, asin);
bench_unary_op!(bench_asinh, asinh);
bench_unary_op!(bench_atan, atan);
bench_unary_op!(bench_atanh, atanh);
bench_unary_op!(bench_copy, copy);
bench_unary_op!(bench_cos, cos);
bench_unary_op!(bench_cosh, cosh);
bench_unary_op!(bench_exp, exp);
bench_unary_op!(bench_log, log);
bench_unary_op!(bench_neg, neg);
bench_unary_op!(bench_rcp, rcp);
bench_unary_op!(bench_sign, sign);
bench_unary_op!(bench_sin, sin);
bench_unary_op!(bench_sinh, sinh);
bench_unary_op!(bench_tan, tan);
bench_unary_op!(bench_tanh, tanh);

criterion::criterion_group!(
    benches,
    bench_constant,
    bench_abs,
    bench_acos,
    bench_acosh,
    bench_asin,
    bench_asinh,
    bench_atan,
    bench_atanh,
    bench_copy,
    bench_cos,
    bench_cosh,
    bench_exp,
    bench_log,
    bench_neg,
    bench_rcp,
    bench_sign,
    bench_sin,
    bench_sinh,
    bench_tan,
    bench_tanh,
);
criterion::criterion_main!(benches);
