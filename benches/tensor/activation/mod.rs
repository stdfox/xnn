//! Activation function benchmarks.

use criterion::{BenchmarkId, Throughput};
use xnn::{Context, Tensor};

macro_rules! bench_activation {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(c: &mut criterion::Criterion) {
            let ctx = Context::try_default().unwrap();
            let mut group = crate::configure(c, concat!("tensor/", stringify!($op)));

            for &(name, dims) in crate::SIZES {
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

bench_activation!(bench_gelu, gelu);
bench_activation!(bench_relu, relu);
bench_activation!(bench_sigmoid, sigmoid);
bench_activation!(bench_silu, silu);
bench_activation!(bench_softplus, softplus);
