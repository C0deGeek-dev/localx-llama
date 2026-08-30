//! The launcher contract: the interface a benchmark/tuner depends on for
//! everything it does not own — the model catalog, hardware detection,
//! llama.cpp binary resolution, and server lifecycle.
//!
//! The tuner is coupled to this *interface*, never to a launcher's internals;
//! any host that implements [`Launcher`] — including a test mock — can drive
//! it. Both ends are versioned: a consumer gates on the version triple
//! ([`assert_compatible`]) before trusting an implementation, and the
//! supported target/runtime pair is declared, not assumed.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::model::{Mode, ModelDef};

/// The default target a benchmark consumes.
pub const TARGET_LOCALBOX: &str = "LocalBox";
/// The only runtime the contract currently speaks.
pub const RUNTIME_LLAMACPP: &str = "llamacpp";

/// The version envelope both ends gate on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LauncherVersion {
    /// Product version (semver-ish string).
    pub version: String,
    /// Contract API version; a consumer requires `>= 1`.
    pub api_version: u32,
    /// Best-config export schema version; a consumer requires `>= 1`.
    pub launcher_export_version: u32,
    /// Launch targets this implementation can drive.
    pub supported_targets: Vec<String>,
    /// Server runtimes this implementation can drive.
    pub supported_runtimes: Vec<String>,
}

/// A launcher-contract failure.
#[derive(Debug, thiserror::Error)]
pub enum LauncherError {
    /// The version triple does not satisfy the consumer's floor.
    #[error("launcher contract not satisfied: {0}")]
    Incompatible(String),
    /// A model key the catalog does not know.
    #[error("unknown model key: {0}")]
    UnknownModel(String),
    /// A capability could not be resolved (binary, path, port, ...).
    #[error("{0}")]
    Unavailable(String),
}

/// Gate a launcher/benchmark pairing on the version triple and the
/// target/runtime declaration: `api_version >= 1`,
/// `launcher_export_version >= 1`, and the requested target and runtime must
/// both be declared supported.
///
/// # Errors
/// Returns [`LauncherError::Incompatible`] naming exactly what failed.
pub fn assert_compatible(
    version: &LauncherVersion,
    target: &str,
    runtime: &str,
) -> Result<(), LauncherError> {
    if version.api_version < 1 {
        return Err(LauncherError::Incompatible(format!(
            "API version {} is below required 1",
            version.api_version
        )));
    }
    if version.launcher_export_version < 1 {
        return Err(LauncherError::Incompatible(format!(
            "launcher export version {} is below required 1",
            version.launcher_export_version
        )));
    }
    if !target.is_empty() && !version.supported_targets.iter().any(|t| t == target) {
        return Err(LauncherError::Incompatible(format!(
            "target '{target}' is not supported (supported: {})",
            version.supported_targets.join(", ")
        )));
    }
    if !runtime.is_empty() && !version.supported_runtimes.iter().any(|r| r == runtime) {
        return Err(LauncherError::Incompatible(format!(
            "runtime '{runtime}' is not supported (supported: {})",
            version.supported_runtimes.join(", ")
        )));
    }
    Ok(())
}

/// The baseline KV cache types a launcher reports for a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTypes {
    pub k: String,
    pub v: String,
}

/// The active backend session a launcher records after a launch, so a later
/// stop/reap targets the right process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSession {
    /// Model key the session serves.
    pub key: String,
    /// Backend mode.
    pub mode: Mode,
    /// TCP port the server listens on.
    pub port: u16,
    /// Server process id, when known.
    pub pid: Option<u32>,
}

/// Everything a tuner needs from a launcher. Implemented by the launcher
/// product (and by test mocks); consumed by the benchmark. The argv itself is
/// built by the shared args builder — the launcher supplies what only it
/// knows: the catalog, the hardware, the binaries, and the lifecycle.
pub trait Launcher {
    /// The implementation's version envelope, for [`assert_compatible`].
    fn version(&self) -> LauncherVersion;

    // --- model resolution ------------------------------------------------
    /// Resolve a model key to its catalog definition.
    ///
    /// # Errors
    /// [`LauncherError::UnknownModel`] for a key the catalog does not know.
    fn model_def(&self, key: &str) -> Result<ModelDef, LauncherError>;
    /// The GGUF path for a definition (and optional quant override).
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] when no GGUF resolves.
    fn gguf_path(&self, def: &ModelDef, quant: Option<&str>) -> Result<PathBuf, LauncherError>;
    /// Context tokens for a context key.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] for an unknown context key.
    fn context_value(&self, def: &ModelDef, context_key: &str) -> Result<u32, LauncherError>;
    /// Canonicalize a context key.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] for an unknown context key.
    fn resolve_context_key(
        &self,
        def: &ModelDef,
        context_key: &str,
    ) -> Result<String, LauncherError>;
    /// The vision projector path, when the model has one.
    fn vision_module_path(&self, key: &str, def: &ModelDef) -> Option<PathBuf>;
    /// Canonicalize a quant key.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] for an unknown quant.
    fn resolve_quant_key(&self, def: &ModelDef, quant: &str) -> Result<String, LauncherError>;

    // --- hardware ----------------------------------------------------------
    /// Total device VRAM in GB (0 = unknown).
    fn vram_gb(&self) -> u32;

    // --- llama.cpp binary resolution ---------------------------------------
    /// The `llama-server` binary for a mode, installing when permitted.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] when the binary cannot be resolved.
    fn server_binary(&self, mode: Mode, non_interactive: bool) -> Result<PathBuf, LauncherError>;
    /// The `llama-bench` binary, when present.
    fn bench_binary(&self, non_interactive: bool) -> Option<PathBuf>;
    /// The `llama-perplexity` binary for a mode, when present.
    fn perplexity_binary(&self, non_interactive: bool, mode: Mode) -> Option<PathBuf>;
    /// The install root for a mode.
    fn install_root(&self, mode: Mode) -> PathBuf;

    // --- KV capability -----------------------------------------------------
    /// The baseline KV cache types for a model.
    fn kv_types(&self, def: &ModelDef) -> KvTypes;
    /// Whether a KV cache type is supported under a mode.
    fn kv_type_supported(&self, kv_type: &str, mode: Mode) -> bool;

    // --- server lifecycle ----------------------------------------------------
    /// A free TCP port at/above `start`.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] when no port can be found.
    fn free_port(&self, start: u16) -> Result<u16, LauncherError>;
    /// Block until the server on `port` answers, or time out.
    ///
    /// # Errors
    /// [`LauncherError::Unavailable`] on timeout.
    fn wait_server(&self, port: u16, timeout_secs: u32) -> Result<(), LauncherError>;
    /// Stop the launched server.
    fn stop_server(&self, quiet: bool);
    /// Record the active backend session for later stop/reap.
    fn set_backend_session(&self, session: &BackendSession);

    // --- paths ---------------------------------------------------------------
    /// Expand `%VAR%` / `~` style path spellings.
    fn expand_path(&self, path: &str) -> PathBuf;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn version() -> LauncherVersion {
        LauncherVersion {
            version: "1.2.1".to_string(),
            api_version: 1,
            launcher_export_version: 3,
            supported_targets: vec!["LocalBox".to_string(), "LocalLLMLauncher".to_string()],
            supported_runtimes: vec!["llamacpp".to_string()],
        }
    }

    #[test]
    fn the_version_triple_gates_compatibility() {
        assert!(assert_compatible(&version(), TARGET_LOCALBOX, RUNTIME_LLAMACPP).is_ok());

        let mut old_api = version();
        old_api.api_version = 0;
        let err = assert_compatible(&old_api, TARGET_LOCALBOX, RUNTIME_LLAMACPP).unwrap_err();
        assert!(err.to_string().contains("API version 0"));

        let mut old_export = version();
        old_export.launcher_export_version = 0;
        assert!(assert_compatible(&old_export, TARGET_LOCALBOX, RUNTIME_LLAMACPP).is_err());
    }

    #[test]
    fn undeclared_target_or_runtime_is_refused() {
        let err = assert_compatible(&version(), "SomethingElse", RUNTIME_LLAMACPP).unwrap_err();
        assert!(err.to_string().contains("SomethingElse"));
        let err = assert_compatible(&version(), TARGET_LOCALBOX, "vllm").unwrap_err();
        assert!(err.to_string().contains("vllm"));
        // Blank target/runtime means "no preference" and always passes.
        assert!(assert_compatible(&version(), "", "").is_ok());
    }

    #[test]
    fn the_version_envelope_round_trips_the_wire_shape() {
        // The JSON keys are the cross-product contract (snake_case, exactly as
        // the existing bridge reads them).
        let json = serde_json::to_value(version()).unwrap();
        assert_eq!(json["api_version"], 1);
        assert_eq!(json["launcher_export_version"], 3);
        assert_eq!(json["supported_targets"][0], "LocalBox");
        assert_eq!(json["supported_runtimes"][0], "llamacpp");
    }
}
