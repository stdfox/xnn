//! Binary arithmetic operation benchmarks.

use super::bench_binary_op;

bench_binary_op!(bench_add, add, f32);
bench_binary_op!(bench_div, div, f32);
bench_binary_op!(bench_max, max, f32);
bench_binary_op!(bench_min, min, f32);
bench_binary_op!(bench_mul, mul, f32);
bench_binary_op!(bench_pow, pow, f32);
bench_binary_op!(bench_rem, rem, i32);
bench_binary_op!(bench_sub, sub, f32);
