//! Model definitions, key resolution, and load-time validation.
//!
//! Ported from the launcher's catalog + model helpers. The `ModelDef` schema is
//! the heart of the system: the data half of the launcher contract.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// The runtime backend a launch targets. `native` = mainline llama.cpp;
/// `turboquant`/`mtpturbo` are the C0deGeek-dev forks with extra KV/spec support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Mainline llama.cpp.
    Native,
    /// turboquant fork (turbo3/turbo4 KV, no MTP).
    Turboquant,
    /// mtpturbo fork (turbo KV + MTP spec-types).
    Mtpturbo,
}

impl Mode {
    /// The lowercase wire name used in schemas and CLI flags.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Native => "native",
            Mode::Turboquant => "turboquant",
            Mode::Mtpturbo => "mtpturbo",
        }
    }
}

/// A single quant variant of a model (one GGUF file).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct QuantEntry {
    /// GGUF filename within the model repo.
    #[serde(default)]
    pub file: String,
    /// On-disk size in GB, when known (feeds quant-fit classification).
    #[serde(rename = "SizeGB", default, skip_serializing_if = "Option::is_none")]
    pub size_gb: Option<f64>,
    /// Optional human note shown in the picker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A model definition as stored in the catalog.
///
/// Optional-but-typed fields mirror the launcher's `Contains`-guarded reads:
/// absent is meaningful and never defaulted to a benign value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ModelDef {
    /// HuggingFace repo id (`owner/name`).
    pub repo: String,
    /// Quant key -> variant. May be empty for models that don't support switching.
    #[serde(default)]
    pub quants: BTreeMap<String, QuantEntry>,
    /// Default quant key; must exist in `quants` when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Context key -> `num_ctx`. The empty key `""` is the default context.
    #[serde(default)]
    pub contexts: BTreeMap<String, i64>,
    /// Parser family driving sampler + chat-template mapping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser: Option<String>,
    /// Display tier (e.g. `flagship`), catalog-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Whether the strict sampler overlay is on by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// KV cache type for keys (default `q8_0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_k: Option<String>,
    /// KV cache type for values (defaults to the key type).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_v: Option<String>,
    /// GPU layers to offload (default 999 = all).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i64>,
    /// MoE expert layers to keep on CPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_cpu_moe: Option<i64>,
    /// Lock model in RAM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlock: Option<bool>,
    /// Disable mmap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_mmap: Option<bool>,
    /// Flash attention on/off (omitted leaves the server default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash_attn: Option<bool>,
    /// Explicit chat template (file path or inline), overrides the parser mapping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    /// `strip` (default) or `keep` reasoning routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_policy: Option<String>,
    /// Speculative-decoding spec-type (mainline canonical, e.g. `draft-mtp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_type: Option<String>,
    /// Extra raw llama-server args appended after everything else.
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Multimodal projector module id/path (enables `--mmproj` when resolved).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_module: Option<String>,
    /// Human description shown in the picker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The parser families the launcher knows how to map to samplers/templates.
pub const KNOWN_PARSERS: &[&str] = &["none", "qwen3coder", "qwen36", "qwen36-think"];

/// Resolve a (possibly aliased or blank) context key to a real key in `contexts`.
///
/// Order matches the launcher: blank/`default` -> `""`; exact hit; case-insensitive
/// hit; legacy aliases `fast->32k`, `deep->64k`, `128->128k`; else error.
pub fn resolve_context_key(def: &ModelDef, context_key: &str) -> Result<String, CoreError> {
    let key = if context_key.trim().is_empty() || context_key.eq_ignore_ascii_case("default") {
        ""
    } else {
        context_key
    };

    if def.contexts.contains_key(key) {
        return Ok(key.to_string());
    }
    for existing in def.contexts.keys() {
        if existing.eq_ignore_ascii_case(key) {
            return Ok(existing.clone());
        }
    }

    let alias = match key.to_ascii_lowercase().as_str() {
        "fast" => Some("32k"),
        "deep" => Some("64k"),
        "128" => Some("128k"),
        _ => None,
    };
    if let Some(target) = alias {
        if def.contexts.contains_key(target) {
            return Ok(target.to_string());
        }
    }

    let available = def
        .contexts
        .keys()
        .map(|k| {
            if k.is_empty() {
                "default".to_string()
            } else {
                k.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(CoreError::UnknownContext {
        key: key.to_string(),
        available,
    })
}

/// The `num_ctx` for a context key, or `None` when the resolved key has no value.
pub fn context_value(def: &ModelDef, context_key: &str) -> Result<Option<i64>, CoreError> {
    let key = resolve_context_key(def, context_key)?;
    Ok(def.contexts.get(&key).copied())
}

/// Resolve a quant key case-insensitively against `quants`.
pub fn resolve_quant_key(def: &ModelDef, quant: &str) -> Result<String, CoreError> {
    for key in def.quants.keys() {
        if key.eq_ignore_ascii_case(quant) {
            return Ok(key.clone());
        }
    }
    let available = def.quants.keys().cloned().collect::<Vec<_>>().join(", ");
    Err(CoreError::UnknownQuant {
        key: quant.to_string(),
        available,
    })
}

/// Validate a model definition, collecting every field error into one message.
///
/// A typo fails at load, not at the eventual call site (mirrors the launcher's
/// consolidated validator).
pub fn validate_model_def(name: &str, def: &ModelDef) -> Result<(), CoreError> {
    let mut errors: Vec<String> = Vec::new();

    if def.repo.trim().is_empty() {
        errors.push(format!("{name}.Repo must not be empty"));
    }

    if let Some(q) = &def.quant {
        let hit = def.quants.keys().any(|k| k.eq_ignore_ascii_case(q));
        if !hit {
            let available = def.quants.keys().cloned().collect::<Vec<_>>().join(", ");
            errors.push(format!(
                "{name}.Quant '{q}' is not a key in {name}.Quants ({available})"
            ));
        }
    }

    if let Some(p) = &def.parser {
        if !KNOWN_PARSERS.iter().any(|k| k.eq_ignore_ascii_case(p)) {
            errors.push(format!(
                "{name}.Parser '{p}' is unknown (expected one of {})",
                KNOWN_PARSERS.join(", ")
            ));
        }
    }

    for (ck, v) in &def.contexts {
        if *v <= 0 {
            let shown = if ck.is_empty() { "default" } else { ck };
            errors.push(format!(
                "{name}.Contexts['{shown}'] must be a positive integer, got {v}"
            ));
        }
    }

    for (qk, qe) in &def.quants {
        if let Some(sz) = qe.size_gb {
            if sz <= 0.0 {
                errors.push(format!(
                    "{name}.Quants['{qk}'].SizeGB must be positive, got {sz}"
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(CoreError::Validation(errors.join("\n")))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn def_with_contexts(pairs: &[(&str, i64)]) -> ModelDef {
        let mut d = ModelDef {
            repo: "owner/model".into(),
            ..Default::default()
        };
        for (k, v) in pairs {
            d.contexts.insert((*k).to_string(), *v);
        }
        d
    }

    #[test]
    fn context_blank_and_default_map_to_empty_key() {
        let d = def_with_contexts(&[("", 65536), ("128k", 131072)]);
        assert_eq!(resolve_context_key(&d, "").unwrap(), "");
        assert_eq!(resolve_context_key(&d, "default").unwrap(), "");
        assert_eq!(resolve_context_key(&d, "DEFAULT").unwrap(), "");
        assert_eq!(context_value(&d, "").unwrap(), Some(65536));
    }

    #[test]
    fn context_legacy_aliases_resolve() {
        let d = def_with_contexts(&[("32k", 32768), ("64k", 65536), ("128k", 131072)]);
        assert_eq!(resolve_context_key(&d, "fast").unwrap(), "32k");
        assert_eq!(resolve_context_key(&d, "deep").unwrap(), "64k");
        assert_eq!(resolve_context_key(&d, "128").unwrap(), "128k");
    }

    #[test]
    fn context_case_insensitive_and_unknown_errors() {
        let d = def_with_contexts(&[("128K", 131072)]);
        assert_eq!(resolve_context_key(&d, "128k").unwrap(), "128K");
        let err = resolve_context_key(&d, "huge").unwrap_err();
        assert!(matches!(err, CoreError::UnknownContext { .. }));
    }

    #[test]
    fn quant_resolves_case_insensitive() {
        let mut d = ModelDef {
            repo: "o/m".into(),
            ..Default::default()
        };
        d.quants.insert("Q4_K_M".into(), QuantEntry::default());
        assert_eq!(resolve_quant_key(&d, "q4_k_m").unwrap(), "Q4_K_M");
        assert!(resolve_quant_key(&d, "q8").is_err());
    }

    #[test]
    fn validation_collects_all_errors() {
        let mut d = ModelDef {
            repo: "".into(),
            quant: Some("nope".into()),
            parser: Some("bogus".into()),
            ..Default::default()
        };
        d.quants.insert("q4".into(), QuantEntry::default());
        d.contexts.insert("128k".into(), 0);
        let err = validate_model_def("acme", &d).unwrap_err();
        let CoreError::Validation(msg) = err else {
            panic!("expected Validation");
        };
        assert!(msg.contains("acme.Repo"));
        assert!(msg.contains("acme.Quant 'nope'"));
        assert!(msg.contains("acme.Parser 'bogus'"));
        assert!(msg.contains("acme.Contexts['128k']"));
        // four distinct errors, one per line
        assert_eq!(msg.lines().count(), 4);
    }

    #[test]
    fn validation_passes_for_good_def() {
        let mut d = ModelDef {
            repo: "o/m".into(),
            quant: Some("q4".into()),
            parser: Some("qwen36".into()),
            ..Default::default()
        };
        d.quants.insert(
            "q4".into(),
            QuantEntry {
                file: "m.gguf".into(),
                size_gb: Some(12.0),
                note: None,
            },
        );
        d.contexts.insert("".into(), 65536);
        assert!(validate_model_def("m", &d).is_ok());
    }

    #[test]
    fn deserializes_pascal_case_catalog_json() {
        let json = r#"{
            "Repo": "owner/model",
            "Quant": "q4",
            "Quants": { "q4": { "File": "m.gguf", "SizeGB": 12.5 } },
            "Contexts": { "": 65536, "128k": 131072 },
            "Parser": "qwen36",
            "KvCacheK": "q8_0",
            "NGpuLayers": 999,
            "VisionModule": "proj"
        }"#;
        let d: ModelDef = serde_json::from_str(json).unwrap();
        assert_eq!(d.repo, "owner/model");
        assert_eq!(d.quants["q4"].size_gb, Some(12.5));
        assert_eq!(d.contexts[""], 65536);
        assert_eq!(d.kv_cache_k.as_deref(), Some("q8_0"));
        assert_eq!(d.vision_module.as_deref(), Some("proj"));
    }
}
