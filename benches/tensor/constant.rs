//! Constant operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure};

pub(crate) fn bench_constant(c: &mut Criterion) {
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
