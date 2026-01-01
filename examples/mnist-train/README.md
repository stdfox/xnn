# MNIST Training

Train a simple neural network on MNIST dataset using xnn.

## Model

Fully-connected network: 784 → 128 (ReLU) → 10 (Softmax)

## Usage

```bash
cd examples/mnist-train
cargo run --release
```

### Options

```
--data-dir <PATH>   Directory for MNIST dataset (default: ./dataset)
--output <PATH>     Output path for weights (default: ./weights.bin)
--epochs <N>        Number of training epochs (default: 10)
--help              Show help message
```

## Output

Exports trained weights to `weights.bin` (little-endian f32):
- W1: 784 × 128 = 100,352 floats
- B1: 128 floats
- W2: 128 × 10 = 1,280 floats
- B2: 10 floats
- Total: 101,770 floats (407,080 bytes)
