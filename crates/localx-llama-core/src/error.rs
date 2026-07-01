//! Typed errors for the core domain.

/// Errors from model resolution, validation, and argv construction.
///
/// Messages mirror the launcher's originals so operators see the same guidance
/// they do today.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    /// A `turbo3`/`turbo4` KV cache type was requested on a mode that lacks the
    /// turboquant-aware fork.
    #[error("KV cache type '{ty}' requires a turboquant-aware fork. Pick a mainline type ({mainline}) or switch to llama.cpp turboquant or mtpturbo mode.")]
    KvTypeNeedsFork { ty: String, mainline: String },

    /// A KV cache type outside both the mainline and turbo sets.
    #[error("Unknown KV cache type '{ty}'. Mainline: {mainline}; turbo (turboquant/mtpturbo only): {turbo}.")]
    UnknownKvType {
        ty: String,
        mainline: String,
        turbo: String,
    },

    /// An MTP spec-type requested in plain turboquant mode, which has no MTP path.
    #[error("Spec-type '{spec}' (MTP) is not supported by the turboquant fork. Switch to native (mainline MTP) or mtpturbo (combined build).")]
    SpecTypeUnsupported { spec: String },

    /// A parser name with no known sampler/template mapping.
    #[error("Unknown parser: {0}")]
    UnknownParser(String),

    /// A context key that resolves to no entry in the model's `contexts` map.
    #[error("Unknown context '{key}'. Available: {available}")]
    UnknownContext { key: String, available: String },

    /// A quant key not present in the model's `quants` map.
    #[error("Unknown quant '{key}'. Available: {available}")]
    UnknownQuant { key: String, available: String },

    /// Consolidated model-definition validation failure (all field errors at once).
    #[error("model definition invalid:\n{0}")]
    Validation(String),

    /// An attempt to set a catalog-only key (`Models`/`CommandAliases`) in settings.
    #[error("'{0}' comes from the catalog and cannot be set as a per-machine setting")]
    CatalogOnly(String),
}
