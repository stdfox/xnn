//! Binary operation benchmarks.

pub(crate) mod arithmetic;
pub(crate) mod comparison;

macro_rules! bench_binary_op {
    ($name:ident, $op:ident, $ty:ty) => {
        pub(crate) fn $name(c: &mut criterion::Criterion) {
            let ctx = xnn::Context::try_default().unwrap();
            let mut group = crate::configure(c, concat!("tensor/", stringify!($op)));

            for &(name, dims) in crate::SIZES {
                let len: usize = dims.iter().product();
                let a = xnn::Tensor::<$ty>::constant(&ctx, dims, &crate::random_vec::<$ty>(len))
                    .unwrap();
                let b = xnn::Tensor::<$ty>::constant(
                    &ctx,
                    dims,
                    &crate::random_vec_nonzero::<$ty>(len),
                )
                .unwrap();

                group.throughput(criterion::Throughput::ElementsAndBytes {
                    elements: len as u64,
                    bytes: (len * size_of::<$ty>()) as u64,
                });

                group.bench_with_input(
                    criterion::BenchmarkId::from_parameter(name),
                    &(&a, &b),
                    |bencher, (a, b)| {
                        bencher.iter(|| {
                            let _ = a.$op(b).unwrap();
                            ctx.poll().unwrap();
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

pub(crate) use bench_binary_op;
