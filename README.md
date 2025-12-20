# xnn

[![Crates.io](https://img.shields.io/crates/v/xnn.svg)](https://crates.io/crates/xnn)
[![Documentation](https://docs.rs/xnn/badge.svg)](https://docs.rs/xnn)
[![License](https://img.shields.io/crates/l/xnn.svg)](LICENSE)

A lightweight ML framework built from scratch in Rust with GPU-first architecture.

## Features

- GPU acceleration via [wgpu](https://wgpu.rs/) (Vulkan, Metal, DX12)
- Element types: `f32`, `i32`, `u32`
- Cross-platform: Linux, macOS, Windows
- Automatic compute pipeline caching
- No unsafe code

## Examples

Examples require the `unstable-kernels` feature flag.

### Linear regression

Trains a simple linear model to fit `y = 2x + 1`.

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
