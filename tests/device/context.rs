//! Context tests.

use xnn::Context;

#[test]
fn test_try_default() {
    let ctx = Context::try_default();
    assert!(ctx.is_ok());
}

#[test]
fn test_from_adapter_index() {
    let ctx = Context::from_adapter_index(0);
    assert!(ctx.is_ok());
}

#[test]
fn test_poll() {
    let ctx = Context::try_default().unwrap();
    ctx.poll().unwrap();
}

#[test]
fn test_clone() {
    let ctx1 = Context::try_default().unwrap();
    let ctx2 = ctx1.clone();
    let debug1 = format!("{ctx1:?}");
    let debug2 = format!("{ctx2:?}");
    assert_eq!(debug1, debug2);
}

#[test]
fn test_debug() {
    let ctx = Context::try_default().unwrap();
    let debug = format!("{ctx:?}");
    assert!(debug.contains("Context"));
}
