//! Binary logical operation benchmarks.

use super::bench_binary_op;

bench_binary_op!(bench_and, and, bool);
bench_binary_op!(bench_or, or, bool);
