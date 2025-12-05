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
//! - [`add_scalar`]: Scalar addition (c = a + scalar).
//! - [`sub_scalar`]: Scalar subtraction (c = a - scalar).
//! - [`mul_scalar`]: Scalar multiplication (c = a * scalar).
//! - [`div_scalar`]: Scalar division (c = a / scalar).
//! - [`rem_scalar`]: Scalar remainder (c = a % scalar).
//! - [`pow_scalar`]: Scalar power (c = a ^ scalar), f32 only.

mod add;
mod add_scalar;
mod div;
mod div_scalar;
mod mul;
mod mul_scalar;
mod pow;
mod pow_scalar;
mod rem;
mod rem_scalar;
mod sub;
mod sub_scalar;

pub use add::add;
pub use add_scalar::add_scalar;
pub use div::div;
pub use div_scalar::div_scalar;
pub use mul::mul;
pub use mul_scalar::mul_scalar;
pub use pow::pow;
pub use pow_scalar::pow_scalar;
pub use rem::rem;
pub use rem_scalar::rem_scalar;
pub use sub::sub;
pub use sub_scalar::sub_scalar;
