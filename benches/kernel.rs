//! Kernel benchmarks.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion};
use xnn::{GpuContext, kernel};

const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096];

fn configure(group: &mut BenchmarkGroup<WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(100);
}

fn bench_fill(c: &mut Criterion) {
    let ctx = GpuContext::default();

    let mut group = c.benchmark_group("kernel/fill");
    configure(&mut group);

    for &size in SIZES {
        let len = size * size;
        let buf = ctx.create_buffer::<f32>(len).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = kernel::fill(&ctx, &buf, 42.0f32).unwrap();
            });
        });
    }

    group.finish();
}

criterion::criterion_group!(benches, bench_fill);
criterion::criterion_main!(benches);
