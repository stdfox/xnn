# Changelog

## 0.2.0 - 2026-01-01

### Highlights

- Full WebGPU/WASM support for running in browsers
- Complete tensor operations API with broadcasting
- MNIST training and web inference examples

### New Features

#### Tensor Operations
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqr`, `sqrt`, `rsqr`, `rsqrt`, `log2`, `max`, `min`, `clamp`
- **Rounding**: `ceil`, `floor`, `round`, `trunc`
- **Transcendental**: `exp`, `log`, `sin`, `cos`, `tan`
- **Comparison**: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- **Logical**: `and`, `or`, `xor`, `not`
- **Reduction**: `sum`, `max`, `min`, `mean`
- **Selection**: `select`
- **Linear Algebra**: batched matrix multiplication with transpose options

#### Activation Functions
- `relu`, `gelu`, `sigmoid`, `silu`, `softplus`
- `elu`, `leaky_relu`, `prelu`, `selu`

#### Platform Support
- WebGPU backend for browsers via WASM
- `no_std` support enabled by default

### Examples

- **mnist-train**: CLI tool for training on MNIST dataset
- **mnist-web**: Interactive digit recognition in browser

### Breaking Changes

- Removed unstable `kernel` module — use `Tensor` API instead
- Element trait hierarchy reorganized with new marker traits

## 0.1.0 - 2025-12-07

- Initial release
