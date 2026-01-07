//! Copy operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure, random_vec};

pub(crate) fn bench_copy(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = configure(c, "tensor/copy");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let data: Vec<f32> = random_vec(len);
        let t = Tensor::<f32>::from_shape_slice(&ctx, dims, &data).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), dims, |bencher, _| {
            bencher.iter(|| {
                let _ = t.copy().unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}
