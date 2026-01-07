//! Select operation benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput};
use xnn::{Context, Tensor};

use crate::{SIZES, configure, random_vec};

pub(crate) fn bench_select(c: &mut Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = configure(c, "tensor/select");

    for &(name, dims) in SIZES {
        let len: usize = dims.iter().product();
        let cond = Tensor::<bool>::constant(&ctx, dims, &random_vec::<bool>(len)).unwrap();
        let true_val = Tensor::<f32>::constant(&ctx, dims, &random_vec::<f32>(len)).unwrap();
        let false_val = Tensor::<f32>::constant(&ctx, dims, &random_vec::<f32>(len)).unwrap();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&cond, &true_val, &false_val),
            |bencher, (cond, true_val, false_val)| {
                bencher.iter(|| {
                    let _ = cond.select(true_val, false_val).unwrap();
                    ctx.poll().unwrap();
                });
            },
        );
    }

    group.finish();
}
