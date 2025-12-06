# xnn

A lightweight ML framework built from scratch in Rust with GPU-first architecture.

## Features

- GPU acceleration via [wgpu](https://wgpu.rs/) (Vulkan, Metal, DX12)
- Element types: `f32`, `i32`, `u32`
- Cross-platform: Linux, macOS, Windows
- Automatic compute pipeline caching
- No unsafe code

## Feature flags

| Flag | Description |
|------|-------------|
| `unstable-kernels` | Exposes the low-level `kernel` module |

### `unstable-kernels`

The `kernel` module provides raw GPU compute kernels (GEMM, transpose, element-wise ops, etc.). These are **internal building blocks** not intended for direct use:

- Kernels **panic** on invalid input (no `Result` return)
- API may change without notice
- A stable, user-friendly tensor API will be provided in the future

Enable only if you need low-level GPU access and understand the risks:

```toml
[dependencies]
xnn = { version = "0.1", features = ["unstable-kernels"] }
```

## Examples

Examples require the `unstable-kernels` feature flag.

### Linear regression

Trains a simple linear model to fit `y = 2x + 1`.

```sh
cargo run --release --features=unstable-kernels --example linreg
```

### XOR

Trains a 2-layer neural network to solve the XOR problem.

```sh
cargo run --release --features=unstable-kernels --example xor
```

## License

MIT — see [LICENSE](LICENSE) for details.
