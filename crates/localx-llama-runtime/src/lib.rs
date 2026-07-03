//! Process and network side of the llama.cpp stack, shared across LocalX.
//!
//! Everything with I/O sits behind traits/thin shells so the crate stays
//! cross-platform and the domain logic stays unit-testable:
//!
//! - [`probe`] — hardware probe (nvidia-smi VRAM/GPU names, logical cores),
//!   implementing `localx_llama_core::HardwareProbe`.
//! - [`nothink`] — the no-think filter (streaming `<think>` strip, root-only key
//!   strip, system-message merge, `[no output]` fallback) — the in-process
//!   replacement for the python sidecar (axum was spike-proven on Rust 1.82).
//! - [`proxy`] — the axum no-think proxy that composes the `nothink` transforms
//!   into a method/header-faithful forwarder with per-delta SSE stripping.
//! - [`server`] — server lifecycle decision logic (readiness vs listening).
//! - [`download`] — pin-verify + asset-selection *decision logic*; the HTTP
//!   fetch/install shell lives in the consuming app (e.g. LocalBox `update.rs`).
//!
//! Socket→PID reaping and source-build orchestration remain in the app layer.

#![forbid(unsafe_code)]

pub mod download;
pub mod error;
pub mod health;
pub mod net;
pub mod nothink;
pub mod probe;
pub mod proxy;
pub mod server;
pub mod spawn;

pub use error::RuntimeError;
pub use health::{HealthState, ProxyAction, ProxyTarget};
pub use net::{free_port, is_port_free, is_port_listening};
pub use nothink::{
    fallback_if_empty, strip_think, strip_think_json_response, SseThinkFilter, ThinkStripper,
    EMPTY_AFTER_THINK,
};
pub use probe::SystemProbe;
pub use proxy::{
    ByteStream, ForwardRequest, ForwardResponse, ProxyConfig, ProxyState, ReqwestUpstream, Upstream,
};
pub use server::{embed_server_args, resolve_server_binary, wait_for_port};
pub use spawn::{simplify_cwd, spawn_detached};
