//! Unary operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure};

macro_rules! bench_unary_op {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(c: &mut Criterion) {
            let ctx = Context::new().unwrap();
            let mut group = configure(c, concat!("tensor/", stringify!($op)));

            for &(name, dims) in SIZES {
                let len: usize = dims.iter().product();
                let tensor = Tensor::<f32>::constant(&ctx, dims, &crate::random_vec(len)).unwrap();

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

// Arithmetic
bench_unary_op!(bench_abs, abs);
bench_unary_op!(bench_acos, acos);
bench_unary_op!(bench_acosh, acosh);
bench_unary_op!(bench_asin, asin);
bench_unary_op!(bench_asinh, asinh);
bench_unary_op!(bench_atan, atan);
bench_unary_op!(bench_atanh, atanh);
bench_unary_op!(bench_cos, cos);
bench_unary_op!(bench_cosh, cosh);
bench_unary_op!(bench_exp, exp);
bench_unary_op!(bench_log, log);
bench_unary_op!(bench_log2, log2);
bench_unary_op!(bench_neg, neg);
bench_unary_op!(bench_rcp, rcp);
bench_unary_op!(bench_rsqr, rsqr);
bench_unary_op!(bench_rsqrt, rsqrt);
bench_unary_op!(bench_sign, sign);
bench_unary_op!(bench_sin, sin);
bench_unary_op!(bench_sinh, sinh);
bench_unary_op!(bench_sqr, sqr);
bench_unary_op!(bench_sqrt, sqrt);
bench_unary_op!(bench_tan, tan);
bench_unary_op!(bench_tanh, tanh);

// Rounding
bench_unary_op!(bench_ceil, ceil);
bench_unary_op!(bench_floor, floor);
bench_unary_op!(bench_round, round);

// Logical
pub(crate) fn bench_not(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = configure(c, "tensor/not");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let tensor = Tensor::<bool>::constant(&ctx, dims, &[true]).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<u32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &tensor,
            |bencher, tensor| {
                bencher.iter(|| {
                    let _ = tensor.not().unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}
