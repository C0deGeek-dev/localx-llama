//! Pure, host-neutral llama.cpp domain primitives shared across the LocalX stack.
//!
//! No process spawning, no console, no OS-specific I/O in the public API — those
//! live in `localx-llama-runtime`. This crate is the data half of the launcher
//! contract:
//!
//! - [`model`] — model definitions, context/quant key resolution, load-time validation.
//! - [`args`] — the `llama-server` argv builder + KV/spec-type gating + parser→sampler mapping.
//! - [`vram`] — VRAM detection abstraction, quant-fit classification, KV-cache context math.
//!
//! Every carried behaviour is pinned by a golden test (plan §6.16).

#![forbid(unsafe_code)]

pub mod args;
pub mod error;
pub mod model;
pub mod vram;

pub use error::CoreError;
pub use model::{Mode, ModelDef, QuantEntry};
pub use vram::{FitClass, HardwareProbe, VramInfo, VramSource};
