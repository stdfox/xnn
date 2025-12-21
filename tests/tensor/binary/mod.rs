//! Binary operation tests.

mod arithmetic;
mod comparison;

/// Test vector A: `[1.0, 2.0, 3.0, 4.0]`
pub const VECTOR_A: &[f32] = &[1.0, 2.0, 3.0, 4.0];
/// Test vector B: `[5.0, 6.0, 7.0, 8.0]`
pub const VECTOR_B: &[f32] = &[5.0, 6.0, 7.0, 8.0];

/// Test matrix A (2x3): `[[1, 2, 3], [4, 5, 6]]`
pub const MATRIX_A: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// Test matrix B (2x3): `[[10, 20, 30], [40, 50, 60]]`
pub const MATRIX_B: &[f32] = &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
/// Matrix shape: `[2, 3]`
pub const MATRIX_SHAPE: &[usize] = &[2, 3];

/// Scalar value for broadcast tests.
pub const SCALAR: f32 = 10.0;

/// Column vector (3x1): `[[1], [2], [3]]`
pub const COLUMN: &[f32] = &[1.0, 2.0, 3.0];
/// Column shape: `[3, 1]`
pub const COLUMN_SHAPE: &[usize] = &[3, 1];

/// Row vector (1x4): `[[10, 20, 30, 40]]`
pub const ROW: &[f32] = &[10.0, 20.0, 30.0, 40.0];
/// Row shape: `[1, 4]`
pub const ROW_SHAPE: &[usize] = &[1, 4];

/// Trailing broadcast vector: `[10.0, 20.0, 30.0]`
pub const TRAILING: &[f32] = &[10.0, 20.0, 30.0];

/// Integer vector A: `[1, 2, 3, 4]`
pub const VECTOR_I32_A: &[i32] = &[1, 2, 3, 4];
/// Integer vector B: `[10, 20, 30, 40]`
pub const VECTOR_I32_B: &[i32] = &[10, 20, 30, 40];

/// Unsigned integer vector A: `[1, 2, 3, 4]`
pub const VECTOR_U32_A: &[u32] = &[1, 2, 3, 4];
/// Unsigned integer vector B: `[10, 20, 30, 40]`
pub const VECTOR_U32_B: &[u32] = &[10, 20, 30, 40];

/// Asserts that two f32 slices are approximately equal.
#[track_caller]
pub fn assert_vec_relative_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (a, e) in actual.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(a, e, epsilon = 1e-4);
    }
}
