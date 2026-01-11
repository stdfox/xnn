//! Tensor integration tests.

mod constant;
mod copy;
mod from_shape_slice;
mod from_slice;
mod linalg;
mod math;
mod nn;
mod random;
mod reduction;

use core::fmt::Debug;

use xnn::{Element, Tensor};

/// Asserts two float tensors are approximately equal.
#[track_caller]
pub(crate) fn assert_tensor_relative_eq<T>(result: &Tensor<T>, expected: &Tensor<T>)
where
    T: Element + Debug + approx::RelativeEq,
{
    let a = result.to_vec().unwrap();
    let b = expected.to_vec().unwrap();
    assert_eq!(result.dimensions(), expected.dimensions());
    for (a, b) in a.iter().zip(b.iter()) {
        approx::assert_relative_eq!(a, b);
    }
}

/// Asserts two tensors are equal.
#[track_caller]
pub(crate) fn assert_tensor_eq<T: Element + PartialEq + Debug>(
    result: &Tensor<T>,
    expected: &Tensor<T>,
) {
    assert_eq!(result.dimensions(), expected.dimensions());
    assert_eq!(result.to_vec().unwrap(), expected.to_vec().unwrap());
}

/// Asserts two float slices are approximately equal.
#[track_caller]
pub(crate) fn assert_vec_relative_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (a, e) in actual.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(a, e, epsilon = epsilon);
    }
}
