//! bool element tests.

use xnn::Element;

#[test]
fn test_native_size() {
    assert_eq!(<bool as Element>::NATIVE_SIZE, 4);
}

#[test]
fn test_wgsl_type() {
    assert_eq!(bool::wgsl_type(), "u32");
}

#[test]
fn test_wgsl_zero() {
    assert_eq!(bool::wgsl_zero(), "0u");
}

#[test]
fn test_wgsl_one() {
    assert_eq!(bool::wgsl_one(), "1u");
}

#[test]
fn test_wgsl_max() {
    assert_eq!(bool::wgsl_max(), "0xffffffffu");
}

#[test]
fn test_wgsl_min() {
    assert_eq!(bool::wgsl_min(), "0u");
}

#[test]
fn test_from_native() {
    assert!(!bool::from_native(0));

    assert!(bool::from_native(1));
    assert!(bool::from_native(42));
    assert!(bool::from_native(u32::MAX));
}

#[test]
fn test_to_native() {
    assert_eq!(false.to_native(), 0);
    assert_eq!(true.to_native(), 1);
}

#[test]
fn test_native_roundtrip() {
    assert!(!bool::from_native(false.to_native()));
    assert!(bool::from_native(true.to_native()));
}
