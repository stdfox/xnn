//! Random tensor generation benchmarks.

use core::mem::size_of;

use criterion::{BenchmarkId, Throughput};
use xnn::{Context, Tensor};

pub(crate) fn bench_random_normal(c: &mut criterion::Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = crate::configure(c, "tensor/random_normal");

    for &(name, dims) in crate::SIZES {
        let len: usize = dims.iter().product();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), dims, |bencher, dims| {
            bencher.iter(|| {
                let _ = Tensor::<f32>::random_normal(&ctx, dims, None, None, Some(42.0)).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}

pub(crate) fn bench_random_uniform(c: &mut criterion::Criterion) {
    let ctx = Context::new().unwrap();
    let mut group = crate::configure(c, "tensor/random_uniform");

    for &(name, dims) in crate::SIZES {
        let len: usize = dims.iter().product();

        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        group.bench_with_input(BenchmarkId::from_parameter(name), dims, |bencher, dims| {
            bencher.iter(|| {
                let _ = Tensor::<f32>::random_uniform(&ctx, dims, None, None, Some(42.0)).unwrap();
                ctx.poll().unwrap();
            });
        });
    }

    group.finish();
}
