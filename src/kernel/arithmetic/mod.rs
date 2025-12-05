//! Element-wise arithmetic kernels.
//!
//! Provides kernels for basic arithmetic operations on GPU buffers.
//!
//! # Operations
//!
//! - [`add`]: Element-wise addition (c = a + b).
//! - [`sub`]: Element-wise subtraction (c = a - b).
//! - [`mul`]: Element-wise multiplication (c = a * b).
//! - [`div`]: Element-wise division (c = a / b).
//! - [`rem`]: Element-wise remainder (c = a % b).
//! - [`pow`]: Element-wise power (c = a ^ b), f32 only.

mod add;
mod div;
mod mul;
mod pow;
mod rem;
mod sub;

pub use add::add;
pub use div::div;
pub use mul::mul;
pub use pow::pow;
pub use rem::rem;
pub use sub::sub;
