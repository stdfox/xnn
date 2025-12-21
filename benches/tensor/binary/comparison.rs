//! Binary comparison operation benchmarks.

use super::bench_binary_op;

bench_binary_op!(bench_lt, lt, f32);
bench_binary_op!(bench_gt, gt, f32);
bench_binary_op!(bench_le, le, f32);
bench_binary_op!(bench_ge, ge, f32);
bench_binary_op!(bench_eq, eq, f32);
bench_binary_op!(bench_ne, ne, f32);
