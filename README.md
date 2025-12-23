# xnn

[![Crates.io](https://img.shields.io/crates/v/xnn.svg)](https://crates.io/crates/xnn)
[![Documentation](https://docs.rs/xnn/badge.svg)](https://docs.rs/xnn)
[![License](https://img.shields.io/crates/l/xnn.svg)](LICENSE)

A lightweight ML framework built from scratch in Rust with GPU-first architecture.

## Features

- GPU acceleration via [wgpu](https://wgpu.rs/) (Vulkan, Metal, DX12)
- Element types: `f32`, `i32`, `u32`, `bool`
- Cross-platform: Linux, macOS, Windows
- Automatic compute pipeline caching
- No unsafe code

## Tensor

N-dimensional array with GPU-accelerated operations and automatic broadcasting.

## Types

- `f32` - 32-bit floating point
- `i32` - 32-bit signed integer
- `u32` - 32-bit unsigned integer
- `bool` - Boolean

## Examples

### Linear regression

Trains a simple linear model to fit `y = wx + b`.

```sh
cargo run --release --example linreg
```

### XOR

Trains a 2-layer neural network to solve the XOR problem.

```sh
cargo run --release --example xor
```

## License

MIT — see [LICENSE](LICENSE) for details.
