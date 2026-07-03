//! Typed errors for the runtime crate.

/// Errors from download verification, install, and server lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeError {
    /// A downloaded file's SHA-256 did not match its pin.
    #[error("SHA-256 mismatch: expected {expected}, got {got} — refusing to use the download")]
    ShaMismatch { expected: String, got: String },

    /// Pins are required but the asset has none.
    #[error("download pin required but none is configured; computed SHA-256 is {computed}")]
    PinRequired { computed: String },

    /// A `llama-server` binary could not be located and none was provided.
    #[error("no llama-server binary available; provide one on PATH or in config (bring-your-own)")]
    NoServerBinary,
}
