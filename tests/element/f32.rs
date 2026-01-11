//! f32 element tests.

use approx::assert_relative_eq;
use xnn::Element;
use xnn::element::FloatElement;

#[test]
fn test_native_size() {
    assert_eq!(<f32 as Element>::NATIVE_SIZE, 4);
}

#[test]
fn test_wgsl_type() {
    assert_eq!(f32::wgsl_type(), "f32");
}

#[test]
fn test_wgsl_zero() {
    assert_eq!(f32::wgsl_zero(), "0.0");
}

#[test]
fn test_wgsl_one() {
    assert_eq!(f32::wgsl_one(), "1.0");
}

#[test]
fn test_wgsl_max() {
    assert_eq!(f32::wgsl_max(), "3.402823466e+38");
}

#[test]
fn test_wgsl_min() {
    assert_eq!(f32::wgsl_min(), "-3.402823466e+38");
}

#[test]
fn test_from_native() {
    assert_relative_eq!(f32::from_native(42.5), 42.5);
    assert_relative_eq!(f32::from_native(-1.0), -1.0);
    assert_relative_eq!(f32::from_native(0.0), 0.0);
}

#[test]
fn test_to_native() {
    assert_relative_eq!((42.5_f32).to_native(), 42.5);
    assert_relative_eq!((-1.0_f32).to_native(), -1.0);
    assert_relative_eq!((0.0_f32).to_native(), 0.0);
}

#[test]
fn test_native_roundtrip() {
    let values = [0.0_f32, 1.0, -1.0, 42.5, f32::MAX, f32::MIN];
    for v in values {
        assert_relative_eq!(f32::from_native(v.to_native()), v);
    }
}

#[test]
fn test_from_f64() {
    assert_relative_eq!(f32::from_f64(42.5), 42.5_f32);
    assert_relative_eq!(f32::from_f64(-1.0), -1.0_f32);
    assert_relative_eq!(f32::from_f64(0.0), 0.0_f32);
}

#[test]
fn test_to_f64() {
    assert_relative_eq!((42.5_f32).to_f64(), 42.5_f64);
    assert_relative_eq!((-1.0_f32).to_f64(), -1.0_f64);
    assert_relative_eq!((0.0_f32).to_f64(), 0.0_f64);
}

#[test]
fn test_f64_roundtrip() {
    let values = [0.0_f32, 1.0, -1.0, 42.5, 1e10, -1e10];
    for v in values {
        let roundtrip = f32::from_f64(v.to_f64());
        assert_relative_eq!(roundtrip, v);
    }
}
