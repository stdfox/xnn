//! Clamp operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure, random_vec};

pub(crate) fn bench_clamp(c: &mut Criterion) {
    let ctx = Context::try_default().unwrap();
    let mut group = configure(c, "tensor/clamp");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let x = Tensor::<f32>::constant(&ctx, dims, &random_vec::<f32>(len)).unwrap();
        let min_val = Tensor::<f32>::constant(&ctx, &[1], &[0.0]).unwrap();
        let max_val = Tensor::<f32>::constant(&ctx, &[1], &[1.0]).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&x, &min_val, &max_val),
            |bencher, (x, min_val, max_val)| {
                bencher.iter(|| {
                    let _ = x.clamp(min_val, max_val).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}
