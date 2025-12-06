//! XOR neural network example.
//!
//! Learns XOR function using a 2-layer neural network (2 -> 2 -> 1).
//! All computations are performed on GPU using kernels.

use xnn::GpuContext;
use xnn::kernel::{add, broadcast_rows, fill, gemm, mul, mul_scalar, sigmoid, sub, sum, transpose};

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let ctx = GpuContext::default();

    // Hyperparameters
    let lr = 0.5f32;
    let epochs = 10000;

    // XOR dataset (4 samples, 2 features)
    // Input: [[0,0], [0,1], [1,0], [1,1]]
    // Output: [0, 1, 1, 0]
    let x_data: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Upload data to GPU (X: 4x2, Y: 4x1)
    let x = ctx.create_buffer_from_slice(&x_data)?; // 4x2
    let y = ctx.create_buffer_from_slice(&y_data)?; // 4x1

    // Network: 2 -> 2 -> 1
    // Layer 1: W1 (2x2), b1 (2)
    // Layer 2: W2 (2x1), b2 (1)

    // Initialize weights with small random-ish values (deterministic for reproducibility)
    let w1_init: Vec<f32> = vec![0.5, -0.5, -0.5, 0.5];
    let w2_init: Vec<f32> = vec![1.0, 1.0];

    // Double-buffered weights
    let w1_bufs = [
        ctx.create_buffer_from_slice(&w1_init)?,
        ctx.create_buffer::<f32>(4)?,
    ];
    let w2_bufs = [
        ctx.create_buffer_from_slice(&w2_init)?,
        ctx.create_buffer::<f32>(2)?,
    ];
    let b1_bufs = [ctx.create_buffer::<f32>(2)?, ctx.create_buffer::<f32>(2)?];
    let b2_bufs = [ctx.create_buffer::<f32>(1)?, ctx.create_buffer::<f32>(1)?];
    fill(&ctx, &b1_bufs[0], 0.0f32);
    fill(&ctx, &b2_bufs[0], 0.0f32);

    // Gradient scale buffer
    let lr_div_n = ctx.create_buffer_from_slice(&[lr / 4.0])?;

    // Working buffers
    let z1 = ctx.create_buffer::<f32>(8)?; // 4x2 (pre-activation)
    let z1_bias = ctx.create_buffer::<f32>(8)?; // 4x2 (after bias add)
    let a1 = ctx.create_buffer::<f32>(8)?; // 4x2
    let z2 = ctx.create_buffer::<f32>(4)?; // 4x1 (pre-activation)
    let z2_bias = ctx.create_buffer::<f32>(4)?; // 4x1 (after bias add)
    let a2 = ctx.create_buffer::<f32>(4)?; // 4x1 (output)

    // Bias broadcast buffers (for adding bias to each row)
    let b1_broadcast = ctx.create_buffer::<f32>(8)?; // 4x2
    let b2_broadcast = ctx.create_buffer::<f32>(4)?; // 4x1

    // Gradient buffers
    let d2 = ctx.create_buffer::<f32>(4)?; // 4x1, output error
    let d1 = ctx.create_buffer::<f32>(8)?; // 4x2, hidden error
    let a1_t = ctx.create_buffer::<f32>(8)?; // 2x4, transposed
    let x_t = ctx.create_buffer::<f32>(8)?; // 2x4, transposed
    let w2_t = ctx.create_buffer::<f32>(2)?; // 1x2, transposed
    let sigmoid_deriv = ctx.create_buffer::<f32>(8)?; // 4x2

    let grad_w2 = ctx.create_buffer::<f32>(2)?; // 2x1
    let grad_w1 = ctx.create_buffer::<f32>(4)?; // 2x2
    let grad_b2 = ctx.create_buffer::<f32>(1)?;
    let grad_b1 = ctx.create_buffer::<f32>(2)?;

    let tmp_sum = ctx.create_buffer::<f32>(1)?;
    let scaled_grad_w1 = ctx.create_buffer::<f32>(4)?; // 2x2
    let scaled_grad_w2 = ctx.create_buffer::<f32>(2)?; // 2x1
    let scaled_grad_b1 = ctx.create_buffer::<f32>(2)?; // 2

    // Ones buffer for bias gradient computation
    let ones = ctx.create_buffer_from_slice(&[1.0f32; 4])?;

    // One buffer for sigmoid derivative
    let ones_hidden = ctx.create_buffer_from_slice(&[1.0f32; 8])?;

    // Precompute X transpose (constant across epochs)
    transpose(&ctx, &x, &x_t, 4, 2);

    println!("Training XOR neural network: 2 -> 2 -> 1");
    println!("Learning rate: {lr}\n");

    for epoch in 0..epochs {
        let cur = epoch % 2;
        let nxt = 1 - cur;

        // === Forward pass ===

        // Layer 1: z1 = X @ W1, a1 = sigmoid(z1 + b1)
        gemm(&ctx, &x, &w1_bufs[cur], &z1, 4, 2, 2);
        broadcast_rows(&ctx, &b1_bufs[cur], &b1_broadcast, 4, 2);
        add(&ctx, &z1, &b1_broadcast, &z1_bias);
        sigmoid(&ctx, &z1_bias, &a1);

        // Layer 2: z2 = a1 @ W2, a2 = sigmoid(z2 + b2)
        gemm(&ctx, &a1, &w2_bufs[cur], &z2, 4, 2, 1);
        broadcast_rows(&ctx, &b2_bufs[cur], &b2_broadcast, 4, 1);
        add(&ctx, &z2, &b2_broadcast, &z2_bias);
        sigmoid(&ctx, &z2_bias, &a2);

        // === Backward pass ===

        // Output layer error: d2 = a2 - y (using MSE derivative, sigmoid already applied)
        // For sigmoid + MSE: d2 = (a2 - y) * a2 * (1 - a2)
        // Simplified: d2 = a2 - y (common approximation that works well)
        sub(&ctx, &a2, &y, &d2);

        // Gradient for W2: grad_W2 = a1.T @ d2
        transpose(&ctx, &a1, &a1_t, 4, 2);
        gemm(&ctx, &a1_t, &d2, &grad_w2, 2, 4, 1);

        // Gradient for b2: grad_b2 = sum(d2)
        sum(&ctx, &d2, &grad_b2);

        // Hidden layer error: d1 = (d2 @ W2.T) * a1 * (1 - a1)
        transpose(&ctx, &w2_bufs[cur], &w2_t, 2, 1);
        gemm(&ctx, &d2, &w2_t, &b1_broadcast, 4, 1, 2);
        // sigmoid derivative: a1 * (1 - a1)
        sub(&ctx, &ones_hidden, &a1, &z1_bias);
        mul(&ctx, &a1, &z1_bias, &sigmoid_deriv);
        mul(&ctx, &b1_broadcast, &sigmoid_deriv, &d1);

        // Gradient for W1: grad_W1 = X.T @ d1
        gemm(&ctx, &x_t, &d1, &grad_w1, 2, 4, 2);

        // Gradient for b1: grad_b1 = sum(d1) per column
        // Sum columns: ones.T @ d1
        gemm(&ctx, &ones, &d1, &grad_b1, 1, 4, 2);

        // === Update weights ===

        // W2 -= lr * grad_W2 / n
        mul_scalar(&ctx, &grad_w2, &lr_div_n, &scaled_grad_w2);
        sub(&ctx, &w2_bufs[cur], &scaled_grad_w2, &w2_bufs[nxt]);

        // b2 -= lr * grad_b2 / n
        mul_scalar(&ctx, &grad_b2, &lr_div_n, &tmp_sum);
        sub(&ctx, &b2_bufs[cur], &tmp_sum, &b2_bufs[nxt]);

        // W1 -= lr * grad_W1 / n
        mul_scalar(&ctx, &grad_w1, &lr_div_n, &scaled_grad_w1);
        sub(&ctx, &w1_bufs[cur], &scaled_grad_w1, &w1_bufs[nxt]);

        // b1 -= lr * grad_b1 / n
        mul_scalar(&ctx, &grad_b1, &lr_div_n, &scaled_grad_b1);
        sub(&ctx, &b1_bufs[cur], &scaled_grad_b1, &b1_bufs[nxt]);

        if epoch % 1000 == 0 || epoch == epochs - 1 {
            // MSE loss: sum((a2 - y)^2) / n
            mul(&ctx, &d2, &d2, &z2);
            sum(&ctx, &z2, &tmp_sum);
            let loss = ctx.read_buffer(&tmp_sum)?[0] / 4.0;
            let preds = ctx.read_buffer(&a2)?;
            println!(
                "Epoch {:5}: loss = {:.6}, preds = [{:.3}, {:.3}, {:.3}, {:.3}]",
                epoch, loss, preds[0], preds[1], preds[2], preds[3]
            );
        }
    }

    // Final predictions
    println!("\nFinal predictions:");
    println!(
        "{:>5} | {:>5} | {:>10} | {:>8} | {:>8}",
        "x1", "x2", "predicted", "expected", "error"
    );
    println!(
        "{:-<5}-+-{:-<5}-+-{:-<10}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", ""
    );

    let final_idx = epochs % 2;

    // Run final forward pass
    gemm(&ctx, &x, &w1_bufs[final_idx], &z1, 4, 2, 2);
    broadcast_rows(&ctx, &b1_bufs[final_idx], &b1_broadcast, 4, 2);
    add(&ctx, &z1, &b1_broadcast, &z1_bias);
    sigmoid(&ctx, &z1_bias, &a1);
    gemm(&ctx, &a1, &w2_bufs[final_idx], &z2, 4, 2, 1);
    broadcast_rows(&ctx, &b2_bufs[final_idx], &b2_broadcast, 4, 1);
    add(&ctx, &z2, &b2_broadcast, &z2_bias);
    sigmoid(&ctx, &z2_bias, &a2);

    let preds = ctx.read_buffer(&a2)?;
    let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let expected = [0, 1, 1, 0];

    for (i, &(x1, x2)) in inputs.iter().enumerate() {
        let pred = preds[i];
        let exp = expected[i] as f32;
        let error = (pred - exp).abs();
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
            "{:5} | {:5} | {:10.4} | {:>8} | {:>8}",
            x1, x2, pred, expected[i], error_str
        );
    }

    Ok(())
}
