//! Context integration tests.

use xnn::Context;

#[test]
fn test_new() {
    let ctx = Context::new();
    assert!(ctx.is_ok());
}

#[test]
fn test_new_async() {
    let ctx = pollster::block_on(Context::new_async());
    assert!(ctx.is_ok());
}

#[test]
fn test_with_adapter_index() {
    let ctx = Context::with_adapter_index(0);
    assert!(ctx.is_ok());
}

#[test]
fn test_with_adapter_index_invalid() {
    let ctx = Context::with_adapter_index(999);
    assert!(ctx.is_err());
}

#[test]
fn test_with_adapter_async() {
    let result = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        Context::with_adapter_async(&adapter).await
    });
    assert!(result.is_ok());
}

#[test]
fn test_with_adapter() {
    let result = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        Context::with_adapter(&adapter)
    });
    assert!(result.is_ok());
}

#[test]
fn test_poll() {
    let ctx = Context::new().unwrap();
    assert!(ctx.poll().is_ok());
}

#[test]
fn test_clone() {
    let ctx1 = Context::new().unwrap();
    let ctx2 = ctx1.clone();
    assert_eq!(format!("{ctx1:?}"), format!("{ctx2:?}"));
}

#[test]
fn test_debug() {
    let ctx = Context::new().unwrap();
    let debug = format!("{ctx:?}");
    assert!(debug.contains("Context"));
    assert!(debug.contains("allocator"));
    assert!(debug.contains("pipelines"));
}
