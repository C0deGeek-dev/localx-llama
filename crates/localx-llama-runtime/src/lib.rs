//! Process and network side of the llama.cpp stack, shared across LocalX.
//!
//! Everything with I/O sits behind traits/thin shells so the crate stays
//! cross-platform and the domain logic stays unit-testable:
//!
//! - [`probe`] — hardware probe (nvidia-smi VRAM/GPU names, logical cores),
//!   implementing `localx_llama_core::HardwareProbe`.
//! - [`nothink`] — the no-think filter (streaming `<think>` strip, root-only key
//!   strip, system-message merge, `[no output]` fallback) — the in-process
//!   replacement for the python sidecar (subject 00.5 confirmed axum on 1.82).
//!
//! Server lifecycle, verified download/install, embed-serve, and the axum proxy
//! wiring land in later boxes of this subject.

#![forbid(unsafe_code)]

pub mod nothink;
pub mod probe;

pub use nothink::{strip_think, ThinkStripper, EMPTY_AFTER_THINK};
pub use probe::SystemProbe;
