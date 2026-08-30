//! Config precedence engine.
//!
//! The pure merge behind the launcher's three-layer config load:
//! `defaults` (lowest) < legacy catalog scalars < per-machine `settings`
//! (highest), with `Models`/`CommandAliases` sourced only from the catalog and
//! never overridable by settings.
//!
//! File I/O, UTF-8-no-BOM serialization, and `%USERPROFILE%`-style path expansion
//! are the app/runtime's job — this crate stays pure. The false-vs-unset footgun
//! that bit the PowerShell version simply doesn't exist here: an absent key is
//! `None`/missing, an explicit `false` is `Value::Bool(false)` — distinct by type.

use serde_json::{Map, Value};

use crate::error::CoreError;

/// Keys owned exclusively by the catalog; per-machine settings cannot override them.
pub const CATALOG_ONLY_KEYS: &[&str] = &["Models", "CommandAliases"];

fn is_catalog_only(key: &str) -> bool {
    CATALOG_ONLY_KEYS.contains(&key)
}

/// Assemble the effective config from its layers.
///
/// Order (later wins): `defaults` -> `legacy_scalars` -> catalog `Models`/
/// `CommandAliases` -> `settings` (excluding the catalog-only keys). Every
/// present key — including an explicit `false` — overrides; absent keys pass
/// through untouched.
pub fn assemble_config(
    defaults: &Map<String, Value>,
    legacy_scalars: &Map<String, Value>,
    catalog: &Map<String, Value>,
    settings: &Map<String, Value>,
) -> Map<String, Value> {
    let mut cfg = defaults.clone();

    for (k, v) in legacy_scalars {
        cfg.insert(k.clone(), v.clone());
    }

    for locked in CATALOG_ONLY_KEYS {
        if let Some(v) = catalog.get(*locked) {
            cfg.insert((*locked).to_string(), v.clone());
        }
    }

    for (k, v) in settings {
        if is_catalog_only(k) {
            continue;
        }
        cfg.insert(k.clone(), v.clone());
    }

    cfg
}

/// Set a per-machine setting, refusing the catalog-only keys.
///
/// A `false` value is stored literally (`Value::Bool(false)`), never coerced to
/// "unset" — the reason the launcher persists booleans explicitly.
pub fn set_setting(
    settings: &mut Map<String, Value>,
    key: &str,
    value: Value,
) -> Result<(), CoreError> {
    if is_catalog_only(key) {
        return Err(CoreError::CatalogOnly(key.to_string()));
    }
    settings.insert(key.to_string(), value);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => Map::new(),
        }
    }

    #[test]
    fn precedence_defaults_legacy_settings() {
        let defaults = map(json!({ "A": 1, "B": 1, "Port": 8080 }));
        let legacy = map(json!({ "B": 2, "C": 2 }));
        let catalog = map(json!({ "Models": { "m": {} }, "Ignored": 1 }));
        let settings = map(json!({ "A": 9 }));

        let cfg = assemble_config(&defaults, &legacy, &catalog, &settings);
        assert_eq!(cfg["A"], json!(9)); // settings win
        assert_eq!(cfg["B"], json!(2)); // legacy overlays defaults
        assert_eq!(cfg["C"], json!(2)); // legacy-only
        assert_eq!(cfg["Port"], json!(8080)); // defaults survive
        assert_eq!(cfg["Models"], json!({ "m": {} })); // catalog-only key
        assert!(!cfg.contains_key("Ignored")); // non-locked catalog keys not merged
    }

    #[test]
    fn settings_cannot_override_catalog_only_keys() {
        let defaults = Map::new();
        let legacy = Map::new();
        let catalog = map(json!({ "Models": { "real": {} }, "CommandAliases": { "x": "y" } }));
        let settings = map(json!({ "Models": { "hijack": {} }, "CommandAliases": {} }));

        let cfg = assemble_config(&defaults, &legacy, &catalog, &settings);
        assert_eq!(cfg["Models"], json!({ "real": {} }));
        assert_eq!(cfg["CommandAliases"], json!({ "x": "y" }));
    }

    #[test]
    fn explicit_false_overrides_and_is_distinct_from_absent() {
        let defaults = map(json!({ "Flag": true, "Other": true }));
        let settings = map(json!({ "Flag": false }));
        let cfg = assemble_config(&defaults, &Map::new(), &Map::new(), &settings);
        assert_eq!(cfg["Flag"], json!(false)); // false present -> overrides true
        assert_eq!(cfg["Other"], json!(true)); // absent in settings -> untouched
        assert!(cfg["Flag"].is_boolean());
    }

    #[test]
    fn set_setting_persists_false_and_guards_catalog_keys() {
        let mut settings = Map::new();
        assert!(set_setting(&mut settings, "Bypass", json!(false)).is_ok());
        assert_eq!(settings["Bypass"], json!(false));
        assert!(settings["Bypass"].is_boolean()); // literal false, not unset

        let err = set_setting(&mut settings, "Models", json!({})).unwrap_err();
        assert!(matches!(err, CoreError::CatalogOnly(k) if k == "Models"));
    }
}
