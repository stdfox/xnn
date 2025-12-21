//! Unary operation benchmarks.

pub(crate) mod arithmetic;
pub(crate) mod logical;
pub(crate) mod rounding;

macro_rules! bench_unary_op {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(c: &mut criterion::Criterion) {
            let ctx = xnn::Context::try_default().unwrap();
            let mut group = crate::configure(c, concat!("tensor/", stringify!($op)));

            for &(name, dims) in crate::SIZES {
                let len: usize = dims.iter().product();
                let tensor =
                    xnn::Tensor::<f32>::constant(&ctx, dims, &crate::random_vec(len)).unwrap();

                group.throughput(criterion::Throughput::ElementsAndBytes {
                    elements: len as u64,
                    bytes: (len * size_of::<f32>()) as u64,
                });

                group.bench_with_input(
                    criterion::BenchmarkId::from_parameter(name),
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

pub(crate) use bench_unary_op;
