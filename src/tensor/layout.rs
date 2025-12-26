//! Tensor memory layout with dimensions, strides, and offset.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

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
    /// Creates a new contiguous layout from dimensions.
    ///
    /// # Errors
    ///
    /// - [`TensorError::InvalidShape`] if any dimension is zero.
    pub(crate) fn from_dimensions(dimensions: &[usize]) -> Result<Self, Error> {
        if dimensions.contains(&0) {
            return Err(TensorError::InvalidShape("dimensions must be non-zero".into()).into());
        }

        Ok(Self {
            dimensions: dimensions.into(),
            strides: Self::compute_strides(dimensions),
            offset: 0,
        })
    }

    /// Returns the dimensions as a slice.
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

    /// Computes broadcast dimensions and strides for multiple layouts.
    ///
    /// Returns output dimensions and strides for each input layout,
    /// or `None` if layouts are not broadcast-compatible.
    #[allow(clippy::type_complexity)]
    pub(crate) fn broadcast(layouts: &[&Layout]) -> Option<(Box<[usize]>, Vec<Box<[usize]>>)> {
        if layouts.is_empty() {
            return Some((Box::new([]), Vec::new()));
        }

        if layouts.len() == 1 {
            return Some((
                layouts[0].dimensions.clone(),
                vec![layouts[0].strides.clone()],
            ));
        }

        let mut out_dims = layouts[0].dimensions.clone();
        for layout in &layouts[1..] {
            out_dims = Self::broadcast_dimensions(&out_dims, &layout.dimensions)?;
        }

        let strides = layouts
            .iter()
            .map(|l| l.broadcast_strides(&out_dims))
            .collect();

        Some((out_dims, strides))
    }

    /// Computes broadcast dimensions for two dimension slices.
    fn broadcast_dimensions(a: &[usize], b: &[usize]) -> Option<Box<[usize]>> {
        let mut result: Vec<usize> = a
            .iter()
            .rev()
            .copied()
            .chain(core::iter::repeat(1))
            .zip(b.iter().rev().copied().chain(core::iter::repeat(1)))
            .take(a.len().max(b.len()))
            .map(|(a, b)| match (a, b) {
                (a, b) if a == b => Some(a),
                (1, b) => Some(b),
                (a, 1) => Some(a),
                _ => None,
            })
            .collect::<Option<_>>()?;

        result.reverse();

        Some(result.into_boxed_slice())
    }

    /// Computes strides for broadcasting this layout to target shape.
    ///
    /// Broadcast dimensions have stride 0.
    fn broadcast_strides(&self, target: &[usize]) -> Box<[usize]> {
        let dimensions = &self.dimensions;
        let strides = &self.strides;

        let mut result: Vec<usize> =
            core::iter::repeat_n(0, target.len().saturating_sub(dimensions.len()))
                .chain(
                    dimensions
                        .iter()
                        .zip(strides)
                        .zip(
                            target
                                .iter()
                                .skip(target.len().saturating_sub(dimensions.len())),
                        )
                        .map(|((&dim, &stride), &t)| if dim == t { stride } else { 0 }),
                )
                .collect();

        if result.len() < target.len() {
            result.resize(target.len(), 0);
        }

        result.into_boxed_slice()
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
    fn test_from_dimensions() {
        // valid
        assert!(Layout::from_dimensions(&[1, 2, 3, 4]).is_ok());
        assert!(Layout::from_dimensions(&[2, 2]).is_ok());
        assert!(Layout::from_dimensions(&[4]).is_ok());
        assert!(Layout::from_dimensions(&[]).is_ok());

        // zero dimension
        assert!(Layout::from_dimensions(&[0, 1, 1]).is_err());
        assert!(Layout::from_dimensions(&[1, 0, 1]).is_err());
        assert!(Layout::from_dimensions(&[1, 1, 0]).is_err());
        assert!(Layout::from_dimensions(&[0]).is_err());
    }

    #[test]
    fn test_dimensions() {
        let l = Layout::from_dimensions(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.dimensions(), &[1, 2, 3, 4]);

        let l = Layout::from_dimensions(&[2, 2]).unwrap();
        assert_eq!(l.dimensions(), &[2, 2]);

        let l = Layout::from_dimensions(&[4]).unwrap();
        assert_eq!(l.dimensions(), &[4]);

        let l = Layout::from_dimensions(&[]).unwrap();
        assert_eq!(l.dimensions(), &[] as &[usize]);
    }

    #[test]
    fn test_strides() {
        let l = Layout::from_dimensions(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.strides(), &[24, 12, 4, 1]);

        let l = Layout::from_dimensions(&[2, 2]).unwrap();
        assert_eq!(l.strides(), &[2, 1]);

        let l = Layout::from_dimensions(&[4]).unwrap();
        assert_eq!(l.strides(), &[1]);

        let l = Layout::from_dimensions(&[]).unwrap();
        assert_eq!(l.strides(), &[] as &[usize]);
    }

    #[test]
    fn test_offset() {
        let l = Layout::from_dimensions(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_dimensions(&[2, 2]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_dimensions(&[4]).unwrap();
        assert_eq!(l.offset(), 0);

        let l = Layout::from_dimensions(&[]).unwrap();
        assert_eq!(l.offset(), 0);
    }

    #[test]
    fn test_size() {
        let l = Layout::from_dimensions(&[1, 2, 3, 4]).unwrap();
        assert_eq!(l.size(), 24);

        let l = Layout::from_dimensions(&[2, 2]).unwrap();
        assert_eq!(l.size(), 4);

        let l = Layout::from_dimensions(&[4]).unwrap();
        assert_eq!(l.size(), 4);

        let l = Layout::from_dimensions(&[]).unwrap();
        assert_eq!(l.size(), 1);
    }

    #[test]
    fn test_broadcast_empty() {
        let (dims, strides) = Layout::broadcast(&[]).unwrap();
        assert_eq!(dims.as_ref(), &[] as &[usize]);
        assert!(strides.is_empty());
    }

    #[test]
    fn test_broadcast_single() {
        let a = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides.len(), 1);
        assert_eq!(strides[0].as_ref(), &[12, 4, 1]);
    }

    #[test]
    fn test_broadcast_two_same() {
        let a = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let b = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides[0].as_ref(), &[12, 4, 1]);
        assert_eq!(strides[1].as_ref(), &[12, 4, 1]);
    }

    #[test]
    fn test_broadcast_two_scalar() {
        let a = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let b = Layout::from_dimensions(&[]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides[0].as_ref(), &[12, 4, 1]);
        assert_eq!(strides[1].as_ref(), &[0, 0, 0]);
    }

    #[test]
    fn test_broadcast_two_trailing() {
        let a = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let b = Layout::from_dimensions(&[4]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides[0].as_ref(), &[12, 4, 1]);
        assert_eq!(strides[1].as_ref(), &[0, 0, 1]);
    }

    #[test]
    fn test_broadcast_two_expand() {
        let a = Layout::from_dimensions(&[3, 1]).unwrap();
        let b = Layout::from_dimensions(&[1, 4]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b]).unwrap();
        assert_eq!(dims.as_ref(), &[3, 4]);
        assert_eq!(strides[0].as_ref(), &[1, 0]);
        assert_eq!(strides[1].as_ref(), &[0, 1]);
    }

    #[test]
    fn test_broadcast_two_multi_expand() {
        let a = Layout::from_dimensions(&[2, 1, 4]).unwrap();
        let b = Layout::from_dimensions(&[3, 1]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides[0].as_ref(), &[4, 0, 1]);
        assert_eq!(strides[1].as_ref(), &[0, 1, 0]);
    }

    #[test]
    fn test_broadcast_three() {
        let a = Layout::from_dimensions(&[2, 1, 4]).unwrap();
        let b = Layout::from_dimensions(&[3, 1]).unwrap();
        let c = Layout::from_dimensions(&[1]).unwrap();
        let (dims, strides) = Layout::broadcast(&[&a, &b, &c]).unwrap();
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
        assert_eq!(strides[0].as_ref(), &[4, 0, 1]);
        assert_eq!(strides[1].as_ref(), &[0, 1, 0]);
        assert_eq!(strides[2].as_ref(), &[0, 0, 0]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let a = Layout::from_dimensions(&[3]).unwrap();
        let b = Layout::from_dimensions(&[4]).unwrap();
        assert!(Layout::broadcast(&[&a, &b]).is_none());

        let a = Layout::from_dimensions(&[2, 3]).unwrap();
        let b = Layout::from_dimensions(&[3, 2]).unwrap();
        assert!(Layout::broadcast(&[&a, &b]).is_none());
    }

    #[test]
    fn test_broadcast_three_incompatible() {
        let a = Layout::from_dimensions(&[2, 3]).unwrap();
        let b = Layout::from_dimensions(&[3]).unwrap();
        let c = Layout::from_dimensions(&[4]).unwrap();
        assert!(Layout::broadcast(&[&a, &b, &c]).is_none());
    }

    #[test]
    fn test_broadcast_strides_same() {
        let a = Layout::from_dimensions(&[2, 3, 4]).unwrap();
        let target = [2, 3, 4];
        assert_eq!(a.broadcast_strides(&target).as_ref(), &[12, 4, 1]);
    }

    #[test]
    fn test_broadcast_strides_scalar() {
        let a = Layout::from_dimensions(&[]).unwrap();
        let target = [2, 3, 4];
        assert_eq!(a.broadcast_strides(&target).as_ref(), &[0, 0, 0]);
    }

    #[test]
    fn test_broadcast_strides_trailing() {
        let a = Layout::from_dimensions(&[4]).unwrap();
        let target = [2, 3, 4];
        assert_eq!(a.broadcast_strides(&target).as_ref(), &[0, 0, 1]);
    }

    #[test]
    fn test_broadcast_strides_expand() {
        let a = Layout::from_dimensions(&[3, 1]).unwrap();
        let target = [3, 4];
        assert_eq!(a.broadcast_strides(&target).as_ref(), &[1, 0]);

        let b = Layout::from_dimensions(&[1, 4]).unwrap();
        assert_eq!(b.broadcast_strides(&target).as_ref(), &[0, 1]);
    }

    #[test]
    fn test_broadcast_strides_multi_expand() {
        let a = Layout::from_dimensions(&[2, 1, 4]).unwrap();
        let target = [2, 3, 4];
        assert_eq!(a.broadcast_strides(&target).as_ref(), &[4, 0, 1]);

        let b = Layout::from_dimensions(&[3, 1]).unwrap();
        assert_eq!(b.broadcast_strides(&target).as_ref(), &[0, 1, 0]);
    }
}
