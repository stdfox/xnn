//! Rounding unary operation benchmarks.

use super::bench_unary_op;

bench_unary_op!(bench_ceil, ceil);
bench_unary_op!(bench_floor, floor);
bench_unary_op!(bench_round, round);
