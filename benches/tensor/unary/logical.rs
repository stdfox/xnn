//! Logical unary operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure};

pub(crate) fn bench_not(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
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
