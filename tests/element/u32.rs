//! u32 element tests.

use xnn::Element;

#[test]
fn test_native_size() {
    assert_eq!(<u32 as Element>::NATIVE_SIZE, 4);
}

#[test]
fn test_wgsl_type() {
    assert_eq!(u32::wgsl_type(), "u32");
}

#[test]
fn test_wgsl_zero() {
    assert_eq!(u32::wgsl_zero(), "0u");
}

#[test]
fn test_wgsl_one() {
    assert_eq!(u32::wgsl_one(), "1u");
}

#[test]
fn test_wgsl_max() {
    assert_eq!(u32::wgsl_max(), "0xffffffffu");
}

#[test]
fn test_wgsl_min() {
    assert_eq!(u32::wgsl_min(), "0u");
}

#[test]
fn test_from_native() {
    assert_eq!(u32::from_native(42), 42);
    assert_eq!(u32::from_native(0), 0);
    assert_eq!(u32::from_native(u32::MAX), u32::MAX);
}

#[test]
fn test_to_native() {
    assert_eq!(42_u32.to_native(), 42);
    assert_eq!(0_u32.to_native(), 0);
    assert_eq!(u32::MAX.to_native(), u32::MAX);
}

#[test]
fn test_native_roundtrip() {
    let values = [0_u32, 1, 42, u32::MAX];
    for v in values {
        assert_eq!(u32::from_native(v.to_native()), v);
    }
}
