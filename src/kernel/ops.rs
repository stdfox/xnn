//! Kernel operations.

use crate::element::{FloatElement, IntegerElement, LogicalElement, NumericElement, SignedElement};
use crate::kernel::{constant, copy, linalg, math, nn, reduction};
use crate::{Buffer, Context, Element};

/// Fills buffer with constant value.
pub(crate) fn constant<T: Element>(ctx: &Context, buffer: &Buffer<T>, value: &wgpu::Buffer) {
    constant::execute::<T>(ctx, buffer, value);
}

/// Copies buffer contents from source to destination.
pub(crate) fn copy<T: Element>(ctx: &Context, src: &Buffer<T>, dst: &Buffer<T>) {
    let size_bytes = (src.len() * core::mem::size_of::<T>()) as u64;
    copy::execute(ctx, src.inner(), dst.inner(), size_bytes);
}

/// Batched matrix multiplication: `C = A × B`.
pub(crate) fn matmul<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_dims: &[usize],
    b_dims: &[usize],
    c_dims: &[usize],
    transpose_a: bool,
    transpose_b: bool,
) {
    linalg::matmul::execute::<T>(
        ctx,
        a,
        b,
        c,
        a_dims,
        b_dims,
        c_dims,
        transpose_a,
        transpose_b,
    );
}

/// Element-wise clamp: `y = max(min(x, b), a)`.
pub(crate) fn clamp<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    a: &Buffer<T>,
    b: &Buffer<T>,
    y: &Buffer<T>,
    x_strides: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
    y_strides: &[usize],
) {
    math::clamp::execute::<T>(ctx, x, a, b, y, x_strides, a_strides, b_strides, y_strides);
}

/// Element-wise select: `y = x ? a : b`.
pub(crate) fn select<T: LogicalElement, U: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    a: &Buffer<U>,
    b: &Buffer<U>,
    y: &Buffer<U>,
    x_strides: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
    y_strides: &[usize],
) {
    math::select::execute::<T, U>(ctx, x, a, b, y, x_strides, a_strides, b_strides, y_strides);
}

/// Element-wise addition: `c = a + b`.
pub(crate) fn add<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::add::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise subtraction: `c = a - b`.
pub(crate) fn sub<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::sub::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise multiplication: `c = a * b`.
pub(crate) fn mul<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::mul::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise division: `c = a / b`.
pub(crate) fn div<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::div::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise maximum: `c = max(a, b)`.
pub(crate) fn max<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::max::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise minimum: `c = min(a, b)`.
pub(crate) fn min<T: NumericElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::min::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise remainder: `c = a % b`.
pub(crate) fn rem<T: IntegerElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::rem::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise power: `c = pow(a, b)`.
pub(crate) fn pow<T: FloatElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::pow::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise equality comparison: `c = (a == b)`.
pub(crate) fn eq<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::eq::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise inequality comparison: `c = (a != b)`.
pub(crate) fn ne<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::ne::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise greater-than-or-equal comparison: `c = (a >= b)`.
pub(crate) fn ge<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::ge::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise greater-than comparison: `c = (a > b)`.
pub(crate) fn gt<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::gt::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise less-than-or-equal comparison: `c = (a <= b)`.
pub(crate) fn le<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::le::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise less-than comparison: `c = (a < b)`.
pub(crate) fn lt<T: NumericElement, L: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<L>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::lt::execute::<T, L>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise logical AND: `c = a && b`.
pub(crate) fn and<T: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::and::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise logical OR: `c = a || b`.
pub(crate) fn or<T: LogicalElement>(
    ctx: &Context,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &Buffer<T>,
    a_strides: &[usize],
    b_strides: &[usize],
    c_strides: &[usize],
) {
    math::or::execute::<T, T>(ctx, a, b, c, a_strides, b_strides, c_strides);
}

/// Element-wise absolute value: `b = abs(a)`.
pub(crate) fn abs<T: SignedElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::abs::execute::<T>(ctx, a, b);
}

/// Element-wise negation: `b = -a`.
pub(crate) fn neg<T: SignedElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::neg::execute::<T>(ctx, a, b);
}

/// Element-wise sign: `b = sign(a)`.
pub(crate) fn sign<T: SignedElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::sign::execute::<T>(ctx, a, b);
}

/// Element-wise sine: `b = sin(a)`.
pub(crate) fn sin<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::sin::execute::<T>(ctx, a, b);
}

/// Element-wise cosine: `b = cos(a)`.
pub(crate) fn cos<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::cos::execute::<T>(ctx, a, b);
}

/// Element-wise tangent: `b = tan(a)`.
pub(crate) fn tan<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::tan::execute::<T>(ctx, a, b);
}

/// Element-wise arc sine: `b = asin(a)`.
pub(crate) fn asin<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::asin::execute::<T>(ctx, a, b);
}

/// Element-wise arc cosine: `b = acos(a)`.
pub(crate) fn acos<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::acos::execute::<T>(ctx, a, b);
}

/// Element-wise arc tangent: `b = atan(a)`.
pub(crate) fn atan<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::atan::execute::<T>(ctx, a, b);
}

/// Element-wise hyperbolic sine: `b = sinh(a)`.
pub(crate) fn sinh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::sinh::execute::<T>(ctx, a, b);
}

/// Element-wise hyperbolic cosine: `b = cosh(a)`.
pub(crate) fn cosh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::cosh::execute::<T>(ctx, a, b);
}

/// Element-wise hyperbolic tangent: `b = tanh(a)`.
pub(crate) fn tanh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::tanh::execute::<T>(ctx, a, b);
}

/// Element-wise inverse hyperbolic sine: `b = asinh(a)`.
pub(crate) fn asinh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::asinh::execute::<T>(ctx, a, b);
}

/// Element-wise inverse hyperbolic cosine: `b = acosh(a)`.
pub(crate) fn acosh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::acosh::execute::<T>(ctx, a, b);
}

/// Element-wise inverse hyperbolic tangent: `b = atanh(a)`.
pub(crate) fn atanh<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::atanh::execute::<T>(ctx, a, b);
}

/// Element-wise exponential: `b = exp(a)`.
pub(crate) fn exp<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::exp::execute::<T>(ctx, a, b);
}

/// Element-wise natural logarithm: `b = log(a)`.
pub(crate) fn log<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::log::execute::<T>(ctx, a, b);
}

/// Element-wise base-2 logarithm: `b = log2(a)`.
pub(crate) fn log2<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::log2::execute::<T>(ctx, a, b);
}

/// Element-wise square: `b = a * a`.
pub(crate) fn sqr<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::sqr::execute::<T>(ctx, a, b);
}

/// Element-wise square root: `b = sqrt(a)`.
pub(crate) fn sqrt<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::sqrt::execute::<T>(ctx, a, b);
}

/// Element-wise reciprocal square: `b = 1 / (a * a)`.
pub(crate) fn rsqr<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::rsqr::execute::<T>(ctx, a, b);
}

/// Element-wise reciprocal square root: `b = 1 / sqrt(a)`.
pub(crate) fn rsqrt<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::rsqrt::execute::<T>(ctx, a, b);
}

/// Element-wise reciprocal: `b = 1 / a`.
pub(crate) fn rcp<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::rcp::execute::<T>(ctx, a, b);
}

/// Element-wise ceiling: `b = ceil(a)`.
pub(crate) fn ceil<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::ceil::execute::<T>(ctx, a, b);
}

/// Element-wise floor: `b = floor(a)`.
pub(crate) fn floor<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::floor::execute::<T>(ctx, a, b);
}

/// Element-wise rounding: `b = round(a)`.
pub(crate) fn round<T: FloatElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::round::execute::<T>(ctx, a, b);
}

/// Element-wise logical NOT: `b = !a`.
pub(crate) fn not<T: LogicalElement>(ctx: &Context, a: &Buffer<T>, b: &Buffer<T>) {
    math::not::execute::<T>(ctx, a, b);
}

/// `ELU` activation: `y = x < 0 ? α(eˣ - 1) : x`.
pub(crate) fn elu<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>, alpha: f32) {
    nn::activation::elu::execute(ctx, x, y, alpha, 0.0);
}

/// `GELU` activation: `y = x · σ(1.702x)`.
pub(crate) fn gelu<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    nn::activation::gelu::execute(ctx, x, y, 0.0, 0.0);
}

/// `Leaky ReLU` activation: `y = x < 0 ? αx : x`.
pub(crate) fn leaky_relu<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>, alpha: f32) {
    nn::activation::leaky_relu::execute(ctx, x, y, alpha, 0.0);
}

/// `PReLU` activation: `y = x < 0 ? αx : x` (learned α per element).
pub(crate) fn prelu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: &Buffer<T>,
) {
    nn::activation::prelu::execute(ctx, x, y, alpha);
}

/// `ReLU` activation: `y = max(x, 0)`.
pub(crate) fn relu<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    nn::activation::relu::execute(ctx, x, y, 0.0, 0.0);
}

/// `SELU` activation: `y = λ(x < 0 ? α(eˣ - 1) : x)`.
pub(crate) fn selu<T: FloatElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    alpha: f32,
    lambda: f32,
) {
    nn::activation::selu::execute(ctx, x, y, alpha, lambda);
}

/// `Sigmoid` activation: `y = 1/(1 + e⁻ˣ)`.
pub(crate) fn sigmoid<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    nn::activation::sigmoid::execute(ctx, x, y, 0.0, 0.0);
}

/// `SiLU` activation: `y = x · σ(x)`.
pub(crate) fn silu<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    nn::activation::silu::execute(ctx, x, y, 0.0, 0.0);
}

/// `Softplus` activation: `y = ln(eˣ + 1)`.
pub(crate) fn softplus<T: FloatElement>(ctx: &Context, x: &Buffer<T>, y: &Buffer<T>) {
    nn::activation::softplus::execute(ctx, x, y, 0.0, 0.0);
}

/// Max reduction along specified axes: `y = max(x, axes)`.
pub(crate) fn max_reduce<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    x_dimensions: &[usize],
    x_strides: &[usize],
    y_strides: &[usize],
    axes: &[usize],
) {
    reduction::execute::<reduction::MaxReduce<T>, T>(
        ctx,
        x,
        y,
        x_dimensions,
        x_strides,
        y_strides,
        axes,
    );
}

/// Min reduction along specified axes: `y = min(x, axes)`.
pub(crate) fn min_reduce<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    x_dimensions: &[usize],
    x_strides: &[usize],
    y_strides: &[usize],
    axes: &[usize],
) {
    reduction::execute::<reduction::MinReduce<T>, T>(
        ctx,
        x,
        y,
        x_dimensions,
        x_strides,
        y_strides,
        axes,
    );
}

/// Sum reduction along specified axes: `y = sum(x, axes)`.
pub(crate) fn sum_reduce<T: NumericElement>(
    ctx: &Context,
    x: &Buffer<T>,
    y: &Buffer<T>,
    x_dimensions: &[usize],
    x_strides: &[usize],
    y_strides: &[usize],
    axes: &[usize],
    normalize: bool,
) {
    reduction::sum::execute::<T>(
        ctx,
        x,
        y,
        x_dimensions,
        x_strides,
        y_strides,
        axes,
        normalize,
    );
}
