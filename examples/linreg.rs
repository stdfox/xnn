//! Linear regression example using gradient descent.
//!
//! Learns y = 2x + 1 from synthetic data.

use xnn::{Context, Error, Tensor};

/// Training hyperparameters.
struct Config {
    samples: u16,
    learning_rate: f32,
    epochs: usize,
}

/// Linear model: y = w*x + b.
struct Model {
    w: Tensor<f32>,
    b: Tensor<f32>,
}

impl Model {
    fn new(ctx: &Context) -> Result<Self, Error> {
        let w = Tensor::random_uniform(ctx, &[1], Some(-1.0), Some(1.0), None)?;
        let b = Tensor::random_uniform(ctx, &[1], Some(-1.0), Some(1.0), None)?;
        Ok(Self { w, b })
    }

    fn weights(&self) -> Result<(f32, f32), Error> {
        let w = self.w.to_vec()?[0];
        let b = self.b.to_vec()?[0];
        Ok((w, b))
    }

    /// Forward pass: pred = w * x + b.
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
        let wx = x.mul(&self.w)?;
        wx.add(&self.b)
    }

    /// Gradient descent step.
    fn step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<f32>,
        lr: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Error> {
        // Forward: pred = w * x + b
        let pred = self.forward(x)?;

        // diff = pred - y
        let diff = pred.sub(y)?;

        // grad_w = sum(diff * x)
        let diff_x = diff.mul(x)?;
        let grad_w = diff_x.sum_reduce(&[0], false)?;

        // grad_b = sum(diff)
        let grad_b = diff.sum_reduce(&[0], false)?;

        // w = w - lr * grad_w
        let w_update = grad_w.mul(lr)?;
        self.w = self.w.sub(&w_update)?;

        // b = b - lr * grad_b
        let b_update = grad_b.mul(lr)?;
        self.b = self.b.sub(&b_update)?;

        Ok(diff)
    }
}

/// Compute MSE loss from diff tensor.
fn compute_loss(diff: &Tensor<f32>) -> Result<f32, Error> {
    let mse = diff.sqr()?.mean_reduce(&[0])?;
    Ok(mse.to_vec()?[0])
}

/// Generate synthetic training data: y = 2x + 1.
fn generate_data(ctx: &Context, n: u16) -> Result<(Tensor<f32>, Tensor<f32>), Error> {
    let n_f32 = f32::from(n);
    let x_data: Vec<f32> = (0..n).map(|i| f32::from(i) / n_f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();
    let x = Tensor::from_slice(ctx, &x_data)?;
    let y = Tensor::from_slice(ctx, &y_data)?;
    Ok((x, y))
}

/// Print inference results.
fn print_inference(w: f32, b: f32) {
    println!("\nInference:");
    println!(
        "{:>6} | {:>10} | {:>10} | {:>8}",
        "x", "predicted", "expected", "error"
    );
    println!("{:-<6}-+-{:-<10}-+-{:-<10}-+-{:-<8}", "", "", "", "");

    for test_x in [0.0f32, 0.25, 0.5, 0.75, 1.0, 2.0] {
        let predicted = w * test_x + b;
        let expected = 2.0 * test_x + 1.0;
        let error = (predicted - expected).abs();
        let error_str = format_error(error);
        println!("{test_x:6.2} | {predicted:10.4} | {expected:10.4} | {error_str:>8}");
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
    let ctx = Context::new()?;

    let cfg = Config {
        samples: 2048,
        learning_rate: 0.1,
        epochs: 500,
    };

    let (x, y) = generate_data(&ctx, cfg.samples)?;

    let mut model = Model::new(&ctx)?;

    // Learning rate scaled by 2/n for gradient descent
    let n = f32::from(cfg.samples);
    let lr = Tensor::constant(&ctx, &[1], &[2.0 * cfg.learning_rate / n])?;

    println!("Training linear regression: y = wx + b");
    println!("Target: w = 2.0, b = 1.0\n");

    for epoch in 0..cfg.epochs {
        let diff = model.step(&x, &y, &lr)?;

        if epoch % 100 == 0 || epoch == cfg.epochs - 1 {
            let loss = compute_loss(&diff)?;
            let (w, b) = model.weights()?;
            println!("Epoch {epoch:3}: loss = {loss:.6}, w = {w:.4}, b = {b:.4}");
        }
    }

    let (w, b) = model.weights()?;
    println!("\nTarget: w = 2.0000, b = 1.0000");
    println!("Final:  w = {w:.4}, b = {b:.4}");

    print_inference(w, b);

    Ok(())
}
