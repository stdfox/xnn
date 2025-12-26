//! Binary operation benchmarks.

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
                    bytes: (len * size_of::<<$ty as xnn::Element>::Native>()) as u64,
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

// Arithmetic
bench_binary_op!(bench_add, add, f32);
bench_binary_op!(bench_div, div, f32);
bench_binary_op!(bench_max, max, f32);
bench_binary_op!(bench_min, min, f32);
bench_binary_op!(bench_mul, mul, f32);
bench_binary_op!(bench_pow, pow, f32);
bench_binary_op!(bench_rem, rem, i32);
bench_binary_op!(bench_sub, sub, f32);

// Comparison
bench_binary_op!(bench_lt, lt, f32);
bench_binary_op!(bench_gt, gt, f32);
bench_binary_op!(bench_le, le, f32);
bench_binary_op!(bench_ge, ge, f32);
bench_binary_op!(bench_eq, eq, f32);
bench_binary_op!(bench_ne, ne, f32);

// Logical
bench_binary_op!(bench_and, and, bool);
bench_binary_op!(bench_or, or, bool);
