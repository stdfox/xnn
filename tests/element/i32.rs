//! i32 element tests.

use xnn::Element;

#[test]
fn test_native_size() {
    assert_eq!(<i32 as Element>::NATIVE_SIZE, 4);
}

#[test]
fn test_wgsl_type() {
    assert_eq!(i32::wgsl_type(), "i32");
}

#[test]
fn test_wgsl_zero() {
    assert_eq!(i32::wgsl_zero(), "0i");
}

#[test]
fn test_wgsl_one() {
    assert_eq!(i32::wgsl_one(), "1i");
}

#[test]
fn test_wgsl_max() {
    assert_eq!(i32::wgsl_max(), "0x7fffffffi");
}

#[test]
fn test_wgsl_min() {
    assert_eq!(i32::wgsl_min(), "(-0x7fffffffi - 1i)");
}

#[test]
fn test_from_native() {
    assert_eq!(i32::from_native(42), 42);
    assert_eq!(i32::from_native(-1), -1);
    assert_eq!(i32::from_native(0), 0);
}

#[test]
fn test_to_native() {
    assert_eq!(42_i32.to_native(), 42);
    assert_eq!((-1_i32).to_native(), -1);
    assert_eq!(0_i32.to_native(), 0);
}

#[test]
fn test_native_roundtrip() {
    let values = [0_i32, 1, -1, 42, i32::MAX, i32::MIN];
    for v in values {
        assert_eq!(i32::from_native(v.to_native()), v);
    }
}
