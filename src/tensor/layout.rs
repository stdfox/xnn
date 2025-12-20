//! Tensor memory layout with dimensions, strides, and offset.

use crate::Error;
use crate::error::TensorError;

/// Tensor memory layout descriptor.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    dimensions: Box<[usize]>,
    strides: Box<[usize]>,
    offset: usize,
}

impl Layout {
    /// Creates a new contiguous layout from shape.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`](crate::error::TensorError::InvalidShape) if any dimension is zero.
    pub(crate) fn from_shape(shape: &[usize]) -> Result<Self, Error> {
        if shape.contains(&0) {
            return Err(TensorError::InvalidShape("dimensions must be non-zero".into()).into());
        }

        Ok(Self {
            dimensions: shape.into(),
            strides: Self::compute_strides(shape),
            offset: 0,
        })
    }

    /// Returns the dimensions as a slice.
    #[allow(dead_code)]
    pub(crate) fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the strides as a slice.
    #[allow(dead_code)]
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the memory offset.
    #[allow(dead_code)]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the total number of elements.
    ///
    /// Returns 1 for scalars.
    pub(crate) fn size(&self) -> usize {
        self.dimensions.iter().product::<usize>().max(1)
    }

    /// Computes row-major (C-contiguous) strides for the given dimensions.
    fn compute_strides(dimensions: &[usize]) -> Box<[usize]> {
        let mut strides = vec![1; dimensions.len()];
        for i in (0..dimensions.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }
        strides.into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_shape() {
        // valid
        assert!(Layout::from_shape(&[1, 2, 3, 4]).is_ok());
        assert!(Layout::from_shape(&[2, 2]).is_ok());
        assert!(Layout::from_shape(&[4]).is_ok());
        assert!(Layout::from_shape(&[]).is_ok());

        // zero dimension
        assert!(Layout::from_shape(&[0, 1, 1]).is_err());
        assert!(Layout::from_shape(&[1, 0, 1]).is_err());
        assert!(Layout::from_shape(&[1, 1, 0]).is_err());
        assert!(Layout::from_shape(&[0]).is_err());
    }

    #[test]
    fn test_dimensions() {
        let l = Layout::from_shape(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.dimensions(), &[1, 2, 3, 4]);

        let l = Layout::from_shape(&[2, 2]).unwrap();
        assert_eq!(l.dimensions(), &[2, 2]);

        let l = Layout::from_shape(&[4]).unwrap();
        assert_eq!(l.dimensions(), &[4]);

        let l = Layout::from_shape(&[]).unwrap();
        assert_eq!(l.dimensions(), &[] as &[usize]);
    }

    #[test]
    fn test_strides() {
        let l = Layout::from_shape(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.strides(), &[24, 12, 4, 1]);

        let l = Layout::from_shape(&[2, 2]).unwrap();
        assert_eq!(l.strides(), &[2, 1]);

        let l = Layout::from_shape(&[4]).unwrap();
        assert_eq!(l.strides(), &[1]);

        let l = Layout::from_shape(&[]).unwrap();
        assert_eq!(l.strides(), &[] as &[usize]);
    }

    #[test]
    fn test_offset() {
        let l = Layout::from_shape(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_shape(&[2, 2]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_shape(&[4]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_shape(&[]).unwrap();
        assert_eq!(l.offset(), 0);
    }

    #[test]
    fn test_size() {
        let l = Layout::from_shape(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.size(), 24);

        let l = Layout::from_shape(&[2, 2]).unwrap();
        assert_eq!(l.size(), 4);

        let l = Layout::from_shape(&[4]).unwrap();
        assert_eq!(l.size(), 4);

        let l = Layout::from_shape(&[]).unwrap();
        assert_eq!(l.size(), 1);
    }
}
