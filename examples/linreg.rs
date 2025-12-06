//! Linear regression example using gradient descent.
//!
//! Learns y = 2x + 1 from synthetic data.
//! All computations are performed on GPU using kernels.

use xnn::GpuContext;
use xnn::kernel::{add_scalar, fill, mul, mul_scalar, sub, sum};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let ctx = GpuContext::default();

    // Hyperparameters
    let n = 2048;
    let lr = 0.1f32;
    let epochs = 500;

    // Precomputed constant: 2 * lr / n
    let grad_scale = ctx.create_buffer_from_slice(&[2.0 * lr / n as f32])?;

    // Generate synthetic data: y = 2x + 1
    let x_data: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    // Upload data to GPU
    let x = ctx.create_buffer_from_slice(&x_data)?;
    let y = ctx.create_buffer_from_slice(&y_data)?;

    // Double-buffered weights for in-place updates
    let w_bufs = [ctx.create_buffer::<f32>(1)?, ctx.create_buffer::<f32>(1)?];
    let b_bufs = [ctx.create_buffer::<f32>(1)?, ctx.create_buffer::<f32>(1)?];
    fill(&ctx, &w_bufs[0], 0.0f32);
    fill(&ctx, &b_bufs[0], 0.0f32);

    // Allocate working buffers
    let wx = ctx.create_buffer::<f32>(n)?;
    let pred = ctx.create_buffer::<f32>(n)?;
    let diff = ctx.create_buffer::<f32>(n)?;
    let tmp = ctx.create_buffer::<f32>(n)?;
    let sum_out = ctx.create_buffer::<f32>(1)?;
    let grad_buf = ctx.create_buffer::<f32>(1)?;

    println!("Training linear regression: y = wx + b");
    println!("Target: w = 2.0, b = 1.0\n");

    for epoch in 0..epochs {
        let cur = epoch % 2;
        let nxt = 1 - cur;

        // Forward pass: pred = w * x + b
        mul_scalar(&ctx, &x, &w_bufs[cur], &wx);
        add_scalar(&ctx, &wx, &b_bufs[cur], &pred);

        // Compute diff = pred - y
        sub(&ctx, &pred, &y, &diff);

        // Gradient for w: 2/n * sum(diff * x) -> scaled by lr
        mul(&ctx, &diff, &x, &tmp);
        sum(&ctx, &tmp, &sum_out);
        mul_scalar(&ctx, &sum_out, &grad_scale, &grad_buf);
        sub(&ctx, &w_bufs[cur], &grad_buf, &w_bufs[nxt]);

        // Gradient for b: 2/n * sum(diff) -> scaled by lr
        sum(&ctx, &diff, &sum_out);
        mul_scalar(&ctx, &sum_out, &grad_scale, &grad_buf);
        sub(&ctx, &b_bufs[cur], &grad_buf, &b_bufs[nxt]);

        if epoch % 100 == 0 || epoch == epochs - 1 {
            // MSE loss: sum(diff^2) / n
            mul(&ctx, &diff, &diff, &tmp);
            sum(&ctx, &tmp, &sum_out);
            let loss = ctx.read_buffer(&sum_out)?[0] / n as f32;
            let w = ctx.read_buffer(&w_bufs[nxt])?[0];
            let b = ctx.read_buffer(&b_bufs[nxt])?[0];
            println!(
                "Epoch {:3}: loss = {:.6}, w = {:.4}, b = {:.4}",
                epoch, loss, w, b
            );
        }
    }

    let final_idx = epochs % 2;
    let w = ctx.read_buffer(&w_bufs[final_idx])?[0];
    let b = ctx.read_buffer(&b_bufs[final_idx])?[0];

    println!("\nTarget: w = 2.0000, b = 1.0000");
    println!("Final:  w = {:.4}, b = {:.4}", w, b);

    // Inference: verify model predictions
    println!("\nInference:");
    println!(
        "{:>6} | {:>10} | {:>10} | {:>8}",
        "x", "predicted", "expected", "error"
    );
    println!("{:-<6}-+-{:-<10}-+-{:-<10}-+-{:-<8}", "", "", "", "");

    for &test_x in &[0.0f32, 0.25, 0.5, 0.75, 1.0, 2.0] {
        let predicted = w * test_x + b;
        let expected = 2.0 * test_x + 1.0;
        let error = (predicted - expected).abs();
        let error_str = if error < 1e-6 {
            "<1e-6".to_string()
        } else if error < 1e-5 {
            "<1e-5".to_string()
        } else if error < 1e-4 {
            "<1e-4".to_string()
        } else if error < 1e-3 {
            "<1e-3".to_string()
        } else if error < 1e-2 {
            "<1e-2".to_string()
        } else {
            format!("{error:.1e}")
        };
        println!(
            "{:6.2} | {:10.4} | {:10.4} | {:>8}",
            test_x, predicted, expected, error_str
        );
    }

    Ok(())
}
