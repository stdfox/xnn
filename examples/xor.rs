//! XOR neural network example.
//!
//! Learns XOR function using a 2-layer neural network (2 -> 2 -> 1).

use xnn::{Context, Error, Tensor};

/// Training hyperparameters.
struct Config {
    learning_rate: f32,
    epochs: usize,
}

/// Neural network model (2 -> 2 -> 1).
struct Model {
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
}

impl Model {
    fn new(ctx: &Context) -> Result<Self, Error> {
        Ok(Self {
            w1: Tensor::from_shape_slice(ctx, &[2, 2], &[0.5, -0.5, -0.5, 0.5])?,
            b1: Tensor::constant(ctx, &[1, 2], &[0.0])?,
            w2: Tensor::from_shape_slice(ctx, &[2, 1], &[1.0, 1.0])?,
            b2: Tensor::constant(ctx, &[1, 1], &[0.0])?,
        })
    }

    /// Forward pass, returns (a1, a2) for use in backward pass.
    fn forward(&self, x: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), Error> {
        // Layer 1: a1 = sigmoid(x @ w1 + b1)
        let z1 = x.matmul(&self.w1, false, false)?;
        let z1_bias = z1.add(&self.b1)?;
        let a1 = z1_bias.sigmoid()?;

        // Layer 2: a2 = sigmoid(a1 @ w2 + b2)
        let z2 = a1.matmul(&self.w2, false, false)?;
        let z2_bias = z2.add(&self.b2)?;
        let a2 = z2_bias.sigmoid()?;

        Ok((a1, a2))
    }

    /// Backward pass: compute gradients and update weights.
    fn backward(
        &mut self,
        ctx: &Context,
        x: &Tensor<f32>,
        y: &Tensor<f32>,
        a1: &Tensor<f32>,
        a2: &Tensor<f32>,
        lr: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Error> {
        let ones = Tensor::constant(ctx, &[4, 2], &[1.0])?;

        // Output error: d2 = a2 - y
        let d2 = a2.sub(y)?;

        // Gradient W2: a1.T @ d2
        let dw2 = a1.matmul(&d2, true, false)?;

        // Gradient b2: sum(d2, axis=0)
        let db2 = d2.sum_reduce(&[0], false)?;

        // Hidden error: d1 = (d2 @ w2.T) * a1 * (1 - a1)
        let d2_w2t = d2.matmul(&self.w2, false, true)?;
        let one_minus_a1 = ones.sub(a1)?;
        let sigmoid_deriv = a1.mul(&one_minus_a1)?;
        let d1 = d2_w2t.mul(&sigmoid_deriv)?;

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
}

/// Compute MSE loss.
fn compute_loss(diff: &Tensor<f32>) -> Result<f32, Error> {
    let sq = diff.mul(diff)?;
    let sum = sq.sum_reduce(&[0], false)?;
    Ok(sum.to_vec()?[0] / 4.0)
}

/// Print inference results.
fn print_inference(preds: &[f32]) {
    println!("\nFinal predictions:");
    println!(
        "{:>5} | {:>5} | {:>10} | {:>8} | {:>8}",
        "x1", "x2", "predicted", "expected", "error"
    );
    println!(
        "{:-<5}-+-{:-<5}-+-{:-<10}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", ""
    );

    let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let expected: [u8; 4] = [0, 1, 1, 0];

    for (i, &(x1, x2)) in inputs.iter().enumerate() {
        let pred = preds[i];
        let exp = f32::from(expected[i]);
        let error = (pred - exp).abs();
        let error_str = format_error(error);
        println!(
            "{x1:5} | {x2:5} | {pred:10.4} | {:>8} | {error_str:>8}",
            expected[i]
        );
    }
}

fn format_error(e: f32) -> String {
    match e {
        e if e < 1e-6 => "<1e-6".into(),
        e if e < 1e-5 => "<1e-5".into(),
        e if e < 1e-4 => "<1e-4".into(),
        e if e < 1e-3 => "<1e-3".into(),
        e if e < 1e-2 => "<1e-2".into(),
        _ => format!("{e:.1e}"),
    }
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let ctx = Context::try_default()?;

    let cfg = Config {
        learning_rate: 0.5,
        epochs: 10000,
    };

    // XOR dataset
    let x = Tensor::from_shape_slice(&ctx, &[4, 2], &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = Tensor::from_shape_slice(&ctx, &[4, 1], &[0.0, 1.0, 1.0, 0.0])?;

    let mut model = Model::new(&ctx)?;
    let lr = Tensor::constant(&ctx, &[1], &[cfg.learning_rate / 4.0])?;

    println!("Training XOR neural network: 2 -> 2 -> 1");
    println!("Learning rate: {}\n", cfg.learning_rate);

    for epoch in 0..cfg.epochs {
        let (a1, a2) = model.forward(&x)?;
        let diff = model.backward(&ctx, &x, &y, &a1, &a2, &lr)?;

        if epoch % 1000 == 0 || epoch == cfg.epochs - 1 {
            let loss = compute_loss(&diff)?;
            let preds = a2.to_vec()?;
            println!(
                "Epoch {epoch:5}: loss = {loss:.6}, preds = [{:.3}, {:.3}, {:.3}, {:.3}]",
                preds[0], preds[1], preds[2], preds[3]
            );
        }
    }

    let (_, a2) = model.forward(&x)?;
    print_inference(&a2.to_vec()?);

    Ok(())
}
