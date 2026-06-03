//! # gravity (C ABI)
//!
//! A stable, flat **C ABI** over the pure-Rust Gravity core. The boundary is
//! JSON strings, byte-compatible with the contract the previous `_ffi_bridge.py`
//! served, so existing C / Go (cgo) / Python (ctypes) callers link and run
//! unchanged against `include/gravity.h`.
//!
//! All `char*` returns are heap-allocated UTF-8 and **must** be released with
//! [`gravity_free_string`]; [`gravity_version`] returns a static pointer that
//! must not be freed. Functions acquire a shared multi-thread Tokio runtime
//! built on [`gravity_init`].
//!
//! The safe, unit-tested JSON conversion lives in the `convert` module; this
//! file is the thin `unsafe` shell that manages C strings, the runtime, and
//! streaming.

mod convert;

use gravity_client::AsyncGravityClient;
use gravity_core::{Conversation, StreamEvent};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::mpsc;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

/// Process-global runtime + client, built lazily by [`gravity_init`].
struct Engine {
    runtime: Runtime,
    client: AsyncGravityClient,
}

static ENGINE: OnceLock<Engine> = OnceLock::new();

fn engine() -> &'static Engine {
    ENGINE.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build Gravity runtime");
        Engine {
            runtime,
            client: AsyncGravityClient::new(),
        }
    })
}

/// Allocate a C string from a Rust string. Caller frees via [`gravity_free_string`].
fn into_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new("{\"error\":\"response contained interior NUL\"}")
            .unwrap()
            .into_raw(),
    }
}

fn json_c_string(v: &serde_json::Value) -> *mut c_char {
    into_c_string(v.to_string())
}

/// Borrow an input C string as `&str`. Returns `None` for null/invalid UTF-8.
///
/// # Safety
/// `ptr` must be null or a valid NUL-terminated C string.
unsafe fn c_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

/// Initialize the library. Returns `0` on success. Idempotent.
///
/// # Safety
/// Safe to call from any thread; the first call builds the runtime.
#[no_mangle]
pub extern "C" fn gravity_init() -> c_int {
    let _ = engine();
    0
}

/// Return the library version as a static C string (do **not** free).
#[no_mangle]
pub extern "C" fn gravity_version() -> *const c_char {
    static VERSION: &[u8] = b"gravity/2.0.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Free a string previously returned by a `gravity_*` function.
///
/// # Safety
/// `ptr` must be null or a pointer returned by this library and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn gravity_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

// ── Blocking chat ───────────────────────────────────────────────────────────

/// Perform a blocking chat. Input/return are JSON; caller frees the result.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn gravity_chat(request_json: *const c_char) -> *mut c_char {
    let Some(body) = c_str(request_json) else {
        return json_c_string(&serde_json::json!({"error": "null or invalid request_json"}));
    };
    let eng = engine();
    let registry = gravity_providers::builtin();
    let value = match convert::parse_chat_request(registry, body) {
        Ok((account, request)) => {
            let conv = Conversation::new("");
            match eng
                .runtime
                .block_on(eng.client.chat_over(&request, std::slice::from_ref(&account), &conv))
            {
                Ok(resp) => convert::response_to_json(&resp),
                Err(e) => convert::error_json(&e),
            }
        }
        Err(e) => convert::error_json(&e),
    };
    json_c_string(&value)
}

/// List registered providers as a JSON array. Caller frees the result.
#[no_mangle]
pub extern "C" fn gravity_list_providers() -> *mut c_char {
    let _ = engine();
    json_c_string(&convert::providers_list_json(gravity_providers::builtin()))
}

/// Generate embeddings (JSON in/out). Currently returns a structured error for
/// providers without embedding support; the ABI shape is preserved.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn gravity_embeddings(request_json: *const c_char) -> *mut c_char {
    let Some(_body) = c_str(request_json) else {
        return json_c_string(&serde_json::json!({"error": "null or invalid request_json"}));
    };
    json_c_string(&serde_json::json!({"error": "embeddings not yet supported for this provider"}))
}

// ── Callback streaming ──────────────────────────────────────────────────────

/// Token callback: invoked once per text delta. `token` is valid only during
/// the call.
pub type TokenCallback = extern "C" fn(token: *const c_char, userdata: *mut c_void);

/// Stream a chat, invoking `callback` per token. Blocks until complete.
/// Returns `0` on success, `-1` on error.
///
/// # Safety
/// `request_json` must be valid UTF-8; `callback` must be a valid function
/// pointer; `userdata` is passed back opaquely.
#[no_mangle]
pub unsafe extern "C" fn gravity_chat_stream(
    request_json: *const c_char,
    callback: TokenCallback,
    userdata: *mut c_void,
) -> c_int {
    let Some(body) = c_str(request_json) else {
        return -1;
    };
    let eng = engine();
    let registry = gravity_providers::builtin();
    let Ok((account, request)) = convert::parse_chat_request(registry, body) else {
        return -1;
    };
    let userdata = SendPtr(userdata);
    let result = eng.runtime.block_on(async move {
        use futures::StreamExt;
        let conv = Conversation::new("");
        let mut stream = eng.client.chat_stream_over(&request, &account, &conv).await?;
        let mut emitted = false;
        while let Some(ev) = stream.next().await {
            match ev? {
                StreamEvent::TextDelta(bytes) => {
                    emitted = true;
                    if let Ok(text) = CString::new(bytes.to_vec()) {
                        callback(text.as_ptr(), userdata.0);
                    }
                }
                // Providers that don't stream incrementally only emit Done with
                // the full text — surface it so the callback isn't starved.
                StreamEvent::Done(resp) if !emitted => {
                    if let Ok(text) = CString::new(resp.text().as_bytes().to_vec()) {
                        callback(text.as_ptr(), userdata.0);
                    }
                }
                _ => {}
            }
        }
        Ok::<(), gravity_core::Error>(())
    });
    if result.is_ok() {
        0
    } else {
        -1
    }
}

// ── Pull streaming ──────────────────────────────────────────────────────────

/// Opaque handle for a pull-style stream.
struct StreamHandle {
    rx: mpsc::Receiver<Result<String, String>>,
}

/// Open a pull-style stream. Returns an opaque handle, or null on error.
/// A background runtime task feeds tokens into a channel.
///
/// # Safety
/// `request_json` must be a valid NUL-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn gravity_stream_open(request_json: *const c_char) -> *mut c_void {
    let Some(body) = c_str(request_json) else {
        return std::ptr::null_mut();
    };
    let eng = engine();
    let registry = gravity_providers::builtin();
    let Ok((account, request)) = convert::parse_chat_request(registry, body) else {
        return std::ptr::null_mut();
    };
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    eng.runtime.spawn(async move {
        use futures::StreamExt;
        let conv = Conversation::new("");
        match eng.client.chat_stream_over(&request, &account, &conv).await {
            Ok(mut stream) => {
                let mut emitted = false;
                while let Some(ev) = stream.next().await {
                    match ev {
                        Ok(StreamEvent::TextDelta(bytes)) => {
                            emitted = true;
                            if tx.send(Ok(String::from_utf8_lossy(&bytes).into_owned())).is_err() {
                                break; // consumer closed
                            }
                        }
                        Ok(StreamEvent::Done(resp)) if !emitted => {
                            let _ = tx.send(Ok(resp.text().to_owned()));
                        }
                        Ok(_) => {}
                        Err(e) => {
                            let _ = tx.send(Err(e.to_string()));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
            }
        }
    });
    Box::into_raw(Box::new(StreamHandle { rx })) as *mut c_void
}

/// Pull the next token. Returns a heap string (caller frees), or null at end of
/// stream. Errors arrive as a `{"error":...}` JSON token.
///
/// # Safety
/// `handle` must be a handle from [`gravity_stream_open`] not yet closed.
#[no_mangle]
pub unsafe extern "C" fn gravity_stream_next(handle: *mut c_void) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    let h = &*(handle as *const StreamHandle);
    match h.rx.recv() {
        Ok(Ok(token)) => into_c_string(token),
        Ok(Err(err)) => json_c_string(&serde_json::json!({ "error": err })),
        Err(_) => std::ptr::null_mut(), // channel closed → end of stream
    }
}

/// Close a pull-style stream and free its handle.
///
/// # Safety
/// `handle` must be a handle from [`gravity_stream_open`], closed at most once.
#[no_mangle]
pub unsafe extern "C" fn gravity_stream_close(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut StreamHandle));
    }
}

/// Wrapper to move a raw pointer across an async boundary. The callback runs on
/// the runtime thread while the calling thread is blocked in `block_on`, so the
/// userdata pointer remains valid for the duration.
struct SendPtr(*mut c_void);
unsafe impl Send for SendPtr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_static_and_correct() {
        let p = gravity_version();
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
        assert_eq!(s, "gravity/2.0.0");
    }

    #[test]
    fn list_providers_via_abi() {
        assert_eq!(gravity_init(), 0);
        let ptr = gravity_list_providers();
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
        unsafe { gravity_free_string(ptr) };
        assert!(s.contains("claude"));
    }

    #[test]
    fn chat_with_bad_json_returns_error_envelope() {
        let req = CString::new("not json").unwrap();
        let ptr = unsafe { gravity_chat(req.as_ptr()) };
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
        unsafe { gravity_free_string(ptr) };
        assert!(s.contains("error"));
    }
}
