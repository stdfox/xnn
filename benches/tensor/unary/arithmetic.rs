//! Arithmetic unary operation benchmarks.

use super::bench_unary_op;

bench_unary_op!(bench_abs, abs);
bench_unary_op!(bench_acos, acos);
bench_unary_op!(bench_acosh, acosh);
bench_unary_op!(bench_asin, asin);
bench_unary_op!(bench_asinh, asinh);
bench_unary_op!(bench_atan, atan);
bench_unary_op!(bench_atanh, atanh);
bench_unary_op!(bench_copy, copy);
bench_unary_op!(bench_cos, cos);
bench_unary_op!(bench_cosh, cosh);
bench_unary_op!(bench_exp, exp);
bench_unary_op!(bench_log, log);
bench_unary_op!(bench_neg, neg);
bench_unary_op!(bench_rcp, rcp);
bench_unary_op!(bench_sign, sign);
bench_unary_op!(bench_sin, sin);
bench_unary_op!(bench_sinh, sinh);
bench_unary_op!(bench_tan, tan);
bench_unary_op!(bench_tanh, tanh);
