//! Activation function benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure, random_vec};

macro_rules! bench_activation {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(c: &mut Criterion) {
            let ctx = Context::new().unwrap();
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

bench_activation!(bench_gelu, gelu);
bench_activation!(bench_relu, relu);
bench_activation!(bench_sigmoid, sigmoid);
bench_activation!(bench_silu, silu);
bench_activation!(bench_softplus, softplus);

macro_rules! bench_activation_with_alpha {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(c: &mut Criterion) {
            let ctx = Context::new().unwrap();
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
                            let _ = tensor.$op(None).unwrap();
                            ctx.poll().unwrap();
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

bench_activation_with_alpha!(bench_elu, elu);
bench_activation_with_alpha!(bench_leaky_relu, leaky_relu);

pub(crate) fn bench_prelu(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = configure(c, "tensor/prelu");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let tensor = Tensor::<f32>::constant(&ctx, dims, &random_vec(len)).unwrap();
        let alpha = Tensor::<f32>::constant(&ctx, dims, &random_vec(len)).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &alpha),
            |bencher, (tensor, alpha)| {
                bencher.iter(|| {
                    let _ = tensor.prelu(alpha).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}

pub(crate) fn bench_selu(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = configure(c, "tensor/selu");

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
                    let _ = tensor.selu(None, None).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}
