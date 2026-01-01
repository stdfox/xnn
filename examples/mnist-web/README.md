# MNIST Web

Interactive digit recognition in the browser using WebGPU.

## Build

```bash
cd examples/mnist-web
wasm-pack build --target web
```

## Run

```bash
python3 -m http.server
```

Open http://localhost:8000 in a WebGPU-compatible browser (Chrome/Edge).

## Usage

1. Click "Load weights" and select `weights.bin` file
2. Draw a digit (0-9) on the canvas
3. Click "Predict" to see the result
