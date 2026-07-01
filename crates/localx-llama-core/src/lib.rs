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
pub mod config;
pub mod error;
pub mod launcher;
pub mod model;
pub mod quant;
pub mod tuner;
pub mod vram;

pub use error::CoreError;
pub use launcher::{
    assert_compatible, discover_root, BackendSession, KvTypes, Launcher, LauncherError,
    LauncherVersion, RUNTIME_LLAMACPP, TARGET_LOCALBOX,
};
pub use model::{Mode, ModelDef, QuantEntry};
pub use tuner::{Overrides, TunerBestConfig, TunerEntry, TUNER_SCHEMA};
pub use vram::{FitClass, HardwareProbe, VramInfo, VramSource};
