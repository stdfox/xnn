//! MNIST training example.
//!
//! Downloads MNIST dataset, trains a simple neural network, and exports weights.
//!
//! Usage:
//!   cargo run -p mnist-train --release [OPTIONS]
//!
//! Options:
//!   --data-dir <PATH>   Directory for MNIST dataset (default: ./dataset)
//!   --output <PATH>     Output path for weights (default: ./weights.bin)
//!   --epochs <N>        Number of training epochs (default: 10)
//!   --help              Show this help message

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use rand::Rng;
use xnn::{Context, Error, Tensor};

const MNIST_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 10;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f32 = 0.1;

/// CLI arguments.
struct Args {
    data_dir: PathBuf,
    output: PathBuf,
    epochs: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./dataset"),
            output: PathBuf::from("./weights.bin"),
            epochs: 10,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--data-dir" => {
                args.data_dir = PathBuf::from(iter.next().ok_or("--data-dir requires a value")?);
            }
            "--output" => {
                args.output = PathBuf::from(iter.next().ok_or("--output requires a value")?);
            }
            "--epochs" => {
                let val = iter.next().ok_or("--epochs requires a value")?;
                args.epochs = val.parse().map_err(|_| "invalid epochs number")?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(args)
}

fn print_help() {
    println!("MNIST training example");
    println!();
    println!("Usage: mnist-train [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --data-dir <PATH>   Directory for MNIST dataset (default: ./dataset)");
    println!("  --output <PATH>     Output path for weights (default: ./weights.bin)");
    println!("  --epochs <N>        Number of training epochs (default: 10)");
    println!("  --help              Show this help message");
}

/// MNIST dataset.
struct Dataset {
    images: Vec<f32>,
    labels: Vec<u8>,
    count: usize,
}

impl Dataset {
    fn load(images_path: &Path, labels_path: &Path) -> std::io::Result<Self> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;
        let count = labels.len();

        Ok(Self {
            images,
            labels,
            count,
        })
    }

    fn load_images(path: &Path) -> std::io::Result<Vec<f32>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(flate2::read::GzDecoder::new(file));

        let mut header = [0u8; 16];
        reader.read_exact(&mut header)?;

        let count = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

        assert_eq!(rows, 28);
        assert_eq!(cols, 28);

        let mut data = vec![0u8; count * 784];
        reader.read_exact(&mut data)?;

        Ok(data.into_iter().map(|x| f32::from(x) / 255.0).collect())
    }

    fn load_labels(path: &Path) -> std::io::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(flate2::read::GzDecoder::new(file));

        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;

        let count = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let mut labels = vec![0u8; count];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }

    fn get_batch(&self, indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let mut images = Vec::with_capacity(indices.len() * INPUT_SIZE);
        let mut labels = Vec::with_capacity(indices.len() * OUTPUT_SIZE);

        for &idx in indices {
            let start = idx * INPUT_SIZE;
            images.extend_from_slice(&self.images[start..start + INPUT_SIZE]);

            let mut one_hot = [0.0f32; OUTPUT_SIZE];
            one_hot[self.labels[idx] as usize] = 1.0;
            labels.extend_from_slice(&one_hot);
        }

        (images, labels)
    }
}

/// Neural network model.
struct Model {
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
}

impl Model {
    fn new(ctx: &Context) -> Result<Self, Error> {
        // He initialization: uniform(-scale, scale) where scale = sqrt(2/fan_in)
        let w1_scale = (2.0 / INPUT_SIZE as f32).sqrt();
        let w2_scale = (2.0 / HIDDEN_SIZE as f32).sqrt();

        Ok(Self {
            w1: Tensor::random_uniform(
                ctx,
                &[INPUT_SIZE, HIDDEN_SIZE],
                Some(-w1_scale),
                Some(w1_scale),
                None,
            )?,
            b1: Tensor::constant(ctx, &[1, HIDDEN_SIZE], &[0.0])?,
            w2: Tensor::random_uniform(
                ctx,
                &[HIDDEN_SIZE, OUTPUT_SIZE],
                Some(-w2_scale),
                Some(w2_scale),
                None,
            )?,
            b2: Tensor::constant(ctx, &[1, OUTPUT_SIZE], &[0.0])?,
        })
    }

    fn forward(&self, x: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), Error> {
        // Layer 1: ReLU(x @ W1 + b1)
        let z1 = x.matmul(&self.w1, false, false)?;
        let z1_bias = z1.add(&self.b1)?;
        let a1 = z1_bias.relu()?;

        // Layer 2: x @ W2 + b2
        let z2 = a1.matmul(&self.w2, false, false)?;
        let logits = z2.add(&self.b2)?;

        // Softmax
        let probs = softmax(&logits)?;

        Ok((a1, probs))
    }

    fn backward(
        &mut self,
        ctx: &Context,
        x: &Tensor<f32>,
        y: &Tensor<f32>,
        a1: &Tensor<f32>,
        probs: &Tensor<f32>,
        lr: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Error> {
        let batch_size = x.dimensions()[0];

        // Output error: d2 = probs - y (cross-entropy derivative with softmax)
        let d2 = probs.sub(y)?;

        // Gradient W2: a1.T @ d2
        let dw2 = a1.matmul(&d2, true, false)?;

        // Gradient b2: sum(d2, axis=0)
        let db2 = d2.sum_reduce(&[0], false)?;

        // Hidden error: d1 = (d2 @ W2.T) * relu'(a1)
        let d2_w2t = d2.matmul(&self.w2, false, true)?;
        let zero = Tensor::constant(ctx, &[batch_size, HIDDEN_SIZE], &[0.0])?;
        let relu_mask = a1.gt(&zero)?;
        let d1 = relu_mask.select(&d2_w2t, &zero)?;

        // Gradient W1: x.T @ d1
        let dw1 = x.matmul(&d1, true, false)?;

        // Gradient b1: sum(d1, axis=0)
        let db1 = d1.sum_reduce(&[0], false)?;

        // Update weights: w -= lr * grad
        self.w2 = self.w2.sub(&dw2.mul(lr)?)?;
        self.b2 = self.b2.sub(&db2.mul(lr)?)?;
        self.w1 = self.w1.sub(&dw1.mul(lr)?)?;
        self.b1 = self.b1.sub(&db1.mul(lr)?)?;

        Ok(d2)
    }

    fn export_weights(&self, path: &Path) -> Result<(), Error> {
        let w1 = self.w1.to_vec()?;
        let b1 = self.b1.to_vec()?;
        let w2 = self.w2.to_vec()?;
        let b2 = self.b2.to_vec()?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Device(format!("create dir: {e}")))?;
        }

        let mut file =
            File::create(path).map_err(|e| Error::Device(format!("create file: {e}")))?;

        // Write as little-endian f32
        for &val in w1.iter().chain(b1.iter()).chain(w2.iter()).chain(b2.iter()) {
            file.write_all(&val.to_le_bytes())
                .map_err(|e| Error::Device(format!("write: {e}")))?;
        }

        Ok(())
    }
}

fn softmax(x: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
    let max_val = x.max_reduce(&[1])?;
    let shifted = x.sub(&max_val)?;
    let exp_vals = shifted.exp()?;
    let sum_exp = exp_vals.sum_reduce(&[1], false)?;
    exp_vals.div(&sum_exp)
}

fn compute_accuracy(probs: &[f32], labels: &[u8]) -> f32 {
    let batch_size = labels.len();
    let mut correct = 0;

    for (i, &label) in labels.iter().enumerate() {
        let start = i * OUTPUT_SIZE;
        let pred = probs[start..start + OUTPUT_SIZE]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if pred == label as usize {
            correct += 1;
        }
    }

    correct as f32 / batch_size as f32
}

fn download_mnist(data_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;

    for filename in [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS] {
        let path = data_dir.join(filename);
        if path.exists() {
            println!("  {filename} already exists");
            continue;
        }

        println!("  Downloading {filename}...");
        let url = format!("{MNIST_URL}{filename}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| std::io::Error::other(format!("HTTP error: {e}")))?;

        let mut file = File::create(&path)?;
        std::io::copy(&mut resp.into_body().into_reader(), &mut file)?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!("Try --help for usage");
        std::process::exit(1);
    });

    println!(
        "Downloading MNIST dataset to {}...",
        args.data_dir.display()
    );
    download_mnist(&args.data_dir)?;

    println!("\nLoading dataset...");
    let train = Dataset::load(
        &args.data_dir.join(TRAIN_IMAGES),
        &args.data_dir.join(TRAIN_LABELS),
    )?;
    let test = Dataset::load(
        &args.data_dir.join(TEST_IMAGES),
        &args.data_dir.join(TEST_LABELS),
    )?;
    println!("  Train: {} images", train.count);
    println!("  Test: {} images", test.count);

    println!("\nInitializing GPU...");
    let ctx = Context::new()?;

    println!("Creating model...");
    let mut model = Model::new(&ctx)?;

    let lr = Tensor::constant(&ctx, &[1], &[LEARNING_RATE / BATCH_SIZE as f32])?;
    let batches_per_epoch = train.count / BATCH_SIZE;

    println!("\nTraining for {} epochs...", args.epochs);
    println!("  Batch size: {BATCH_SIZE}");
    println!("  Learning rate: {LEARNING_RATE}");
    println!();

    let mut rng = rand::rng();
    let mut indices: Vec<usize> = (0..train.count).collect();

    for epoch in 0..args.epochs {
        // Shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }

        let mut total_loss = 0.0;

        for batch in 0..batches_per_epoch {
            let batch_indices = &indices[batch * BATCH_SIZE..(batch + 1) * BATCH_SIZE];
            let (images, labels) = train.get_batch(batch_indices);

            let x = Tensor::from_shape_slice(&ctx, &[BATCH_SIZE, INPUT_SIZE], &images)?;
            let y = Tensor::from_shape_slice(&ctx, &[BATCH_SIZE, OUTPUT_SIZE], &labels)?;

            let (a1, probs) = model.forward(&x)?;
            let d2 = model.backward(&ctx, &x, &y, &a1, &probs, &lr)?;

            // Compute batch loss (cross-entropy approximation)
            let batch_loss: f32 =
                d2.to_vec()?.iter().map(|x| x * x).sum::<f32>() / (BATCH_SIZE * OUTPUT_SIZE) as f32;
            total_loss += batch_loss;
        }

        // Evaluate on test set
        let test_batch = 1000;
        let test_indices: Vec<usize> = (0..test_batch).collect();
        let (test_images, _) = test.get_batch(&test_indices);
        let x_test = Tensor::from_shape_slice(&ctx, &[test_batch, INPUT_SIZE], &test_images)?;
        let (_, test_probs) = model.forward(&x_test)?;
        let test_probs_vec = test_probs.to_vec()?;
        let accuracy = compute_accuracy(&test_probs_vec, &test.labels[..test_batch]);

        let avg_loss = total_loss / batches_per_epoch as f32;
        println!(
            "Epoch {:2}/{}: loss = {:.4}, test_accuracy = {:.2}%",
            epoch + 1,
            args.epochs,
            avg_loss,
            accuracy * 100.0
        );
    }

    println!("\nExporting weights to {}...", args.output.display());
    model.export_weights(&args.output)?;

    let w1_size = INPUT_SIZE * HIDDEN_SIZE;
    let b1_size = HIDDEN_SIZE;
    let w2_size = HIDDEN_SIZE * OUTPUT_SIZE;
    let b2_size = OUTPUT_SIZE;
    let total = w1_size + b1_size + w2_size + b2_size;
    println!("  w1: {w1_size} floats");
    println!("  b1: {b1_size} floats");
    println!("  w2: {w2_size} floats");
    println!("  b2: {b2_size} floats");
    println!("  Total: {total} floats ({} bytes)", total * 4);

    println!("\nDone!");

    Ok(())
}
