//! MNIST digit recognition web example.
//!
//! Run with: `wasm-pack build --target web && python3 -m http.server`

use std::cell::RefCell;

use js_sys::Float32Array;
use wasm_bindgen::prelude::*;
use xnn::{Context, Tensor};

/// Logs a message to the browser console.
macro_rules! log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into())
    };
}

thread_local! {
    static CTX: RefCell<Option<Context>> = const { RefCell::new(None) };
    static MODEL: RefCell<Option<MnistModel>> = const { RefCell::new(None) };
}

/// Weight sizes.
const W1_SIZE: usize = 784 * 128;
const B1_SIZE: usize = 128;
const W2_SIZE: usize = 128 * 10;
const B2_SIZE: usize = 10;

/// MNIST neural network model (784 -> 128 -> 10).
struct MnistModel {
    ctx: Context,
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
}

impl MnistModel {
    /// Creates a new model from weight data.
    fn from_weights(ctx: Context, weights: &[f32]) -> Result<Self, JsValue> {
        let expected = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE;
        if weights.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Invalid weights size: {} (expected {expected})",
                weights.len(),
            )));
        }

        let w1_end = W1_SIZE;
        let b1_end = w1_end + B1_SIZE;
        let w2_end = b1_end + W2_SIZE;
        let b2_end = w2_end + B2_SIZE;

        let w1 = Tensor::from_shape_slice(&ctx, &[784, 128], &weights[..w1_end])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let b1 = Tensor::from_shape_slice(&ctx, &[1, 128], &weights[w1_end..b1_end])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let w2 = Tensor::from_shape_slice(&ctx, &[128, 10], &weights[b1_end..w2_end])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let b2 = Tensor::from_shape_slice(&ctx, &[1, 10], &weights[w2_end..b2_end])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            ctx,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Runs forward pass, returns output tensor.
    fn forward(&self, pixels: &[f32]) -> Result<Tensor<f32>, xnn::Error> {
        let x = Tensor::from_shape_slice(&self.ctx, &[1, 784], pixels)?;

        // Layer 1: ReLU(x @ W1 + b1)
        let h = x.matmul(&self.w1, false, false)?.add(&self.b1)?.relu()?;

        // Layer 2: x @ W2 + b2
        let logits = h.matmul(&self.w2, false, false)?.add(&self.b2)?;

        // Softmax
        softmax(&logits)
    }
}

/// Computes softmax: exp(x - max(x)) / sum(exp(x - max(x))).
fn softmax(x: &Tensor<f32>) -> Result<Tensor<f32>, xnn::Error> {
    let max_val = x.max_reduce(&[1])?;
    let shifted = x.sub(&max_val)?;
    let exp_vals = shifted.exp()?;
    let sum_exp = exp_vals.sum_reduce(&[1], false)?;
    exp_vals.div(&sum_exp)
}

/// Entry point: initializes WebGPU context.
#[wasm_bindgen(start)]
pub async fn init() {
    console_error_panic_hook::set_once();

    log!("Initializing WebGPU...");

    match Context::new_async().await {
        Ok(ctx) => {
            CTX.with(|c| *c.borrow_mut() = Some(ctx));
            log!("WebGPU initialized! Please load weights file.");
        }
        Err(e) => {
            log!("Failed to initialize WebGPU: {:?}", e);
        }
    }
}

/// Loads model weights from binary data (little-endian f32).
///
/// # Errors
///
/// Returns error if data size is not a multiple of 4, WebGPU is not initialized,
/// or weights have invalid size.
#[wasm_bindgen]
pub fn load_weights(data: &[u8]) -> Result<(), JsValue> {
    if !data.len().is_multiple_of(4) {
        return Err(JsValue::from_str(
            "Invalid weights data: size must be multiple of 4",
        ));
    }

    // Convert bytes to f32 (little-endian)
    let weights: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    log!("Loading {} weights...", weights.len());

    CTX.with(|c| {
        let ctx = c.borrow();
        let ctx = ctx
            .as_ref()
            .ok_or_else(|| JsValue::from_str("WebGPU not initialized"))?;

        let model = MnistModel::from_weights(ctx.clone(), &weights)?;
        MODEL.with(|m| *m.borrow_mut() = Some(model));

        log!("Model loaded successfully!");
        Ok(())
    })
}

/// Returns true if model is loaded and ready.
#[must_use]
#[wasm_bindgen]
pub fn is_ready() -> bool {
    MODEL.with(|m| m.borrow().is_some())
}

/// Predicts digit probabilities from canvas pixel data.
///
/// # Errors
///
/// Returns error if model is not initialized or prediction fails.
#[wasm_bindgen]
pub async fn predict(pixels: &[f32]) -> Result<Float32Array, JsValue> {
    // Run forward pass synchronously (GPU operations are queued)
    let probs = MODEL.with(|m| {
        let model = m.borrow();
        let model = model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not initialized"))?;

        model
            .forward(pixels)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    })?;

    // Read results async
    let data = probs
        .to_vec_async()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(Float32Array::from(data.as_slice()))
}
