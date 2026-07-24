//! The tuner best-config store schema (`best-<key>.json`).
//!
//! Written by LocalBench, read back by LocalBox for AutoBest launches. Both sides
//! gate on `schema == 1` and per-entry `tuner_version`. The `overrides` object's
//! keys map 1:1 to the argv-builder parameters, so a stored profile re-hydrates
//! directly into [`LaunchParams`] — the contract that keeps the two tools in step.

use serde::{Deserialize, Serialize};

use crate::args::LaunchParams;
use crate::model::Mode;

/// The store schema version both writer and reader gate on.
pub const TUNER_SCHEMA: u32 = 1;

/// Prompt-length band a tuned entry was measured at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptLength {
    /// Short-prompt band.
    Short,
    /// Long-prompt band.
    Long,
}

/// The tuning profile a stored entry targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    /// Max-throughput profile.
    Pure,
    /// Balanced throughput/quality profile.
    Balanced,
}

/// Search strategy that produced an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchStrategy {
    /// Greedy hill-climb.
    Greedy,
    /// Beam search.
    Beam,
}

/// The tunable overrides for a stored profile. Keys are PascalCase to match the
/// launcher's re-hydration list; every field maps to one argv-builder parameter.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct Overrides {
    /// GPU layers to offload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i64>,
    /// MoE layers on CPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_cpu_moe: Option<i64>,
    /// `--ubatch-size`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<i64>,
    /// `--batch-size`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<i64>,
    /// `--threads`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<i64>,
    /// `--threads-batch`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<i64>,
    /// Lock in RAM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlock: Option<bool>,
    /// Disable mmap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_mmap: Option<bool>,
    /// Flash attention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn: Option<bool>,
    /// Multi-GPU split mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    /// `--swa-full`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swa_full: Option<bool>,
    /// `--cache-prompt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    /// `--cache-reuse`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_reuse: Option<i64>,
    /// KV key cache type.
    #[serde(rename = "KvK", skip_serializing_if = "Option::is_none")]
    pub kv_k: Option<String>,
    /// KV value cache type.
    #[serde(rename = "KvV", skip_serializing_if = "Option::is_none")]
    pub kv_v: Option<String>,
    /// Spec-type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_type: Option<String>,
    /// Max draft tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_draft_n_max: Option<i64>,
}

impl Overrides {
    /// Fold the stored overrides into per-call [`LaunchParams`] for an AutoBest launch.
    pub fn to_launch_params(&self) -> LaunchParams {
        LaunchParams {
            parallel: None,
            cache_reuse: self.cache_reuse,
            kv_k: self.kv_k.clone(),
            kv_v: self.kv_v.clone(),
            n_gpu_layers: self.n_gpu_layers,
            n_cpu_moe: self.n_cpu_moe,
            mlock: self.mlock,
            no_mmap: self.no_mmap,
            ubatch_size: self.ubatch_size,
            batch_size: self.batch_size,
            threads: self.threads,
            threads_batch: self.threads_batch,
            flash_attn: self.flash_attn,
            swa_full: self.swa_full.unwrap_or(false),
            cache_prompt: self.cache_prompt,
            split_mode: self.split_mode.clone(),
            chat_template_override: None,
            thinking_policy: None,
            strict: None,
            vision_module_path: None,
            draft_module_path: None,
            spec_type: self.spec_type.clone(),
            spec_draft_n_max: self.spec_draft_n_max,
            extra_args: Vec::new(),
        }
    }
}

/// One tuned entry in the store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunerEntry {
    /// Quant key this entry was tuned for.
    pub quant: String,
    /// Context key this entry was tuned for.
    #[serde(rename = "contextKey")]
    pub context_key: String,
    /// Concrete context tokens, when recorded.
    #[serde(
        rename = "contextTokens",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub context_tokens: Option<i64>,
    /// Backend mode.
    pub mode: Mode,
    /// VRAM the entry was measured at.
    #[serde(rename = "vramGB")]
    pub vram_gb: i64,
    /// Prompt-length band.
    pub prompt_length: PromptLength,
    /// Tuning profile.
    pub profile: Profile,
    /// Search strategy, when recorded.
    #[serde(
        rename = "searchStrategy",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub search_strategy: Option<SearchStrategy>,
    /// Beam width, when beam search.
    #[serde(rename = "beamWidth", default, skip_serializing_if = "Option::is_none")]
    pub beam_width: Option<i64>,
    /// Winning score.
    pub score: f64,
    /// Unit for `score`.
    #[serde(rename = "scoreUnit")]
    pub score_unit: String,
    /// Pure-profile score, when recorded.
    #[serde(rename = "pureScore", default, skip_serializing_if = "Option::is_none")]
    pub pure_score: Option<f64>,
    /// The full argv the winner ran with.
    pub args: Vec<String>,
    /// Tunable overrides that re-hydrate into [`LaunchParams`].
    pub overrides: Overrides,
    /// When measured (RFC3339-ish string).
    pub measured_at: String,
    /// Per-entry tuner version both sides gate on.
    pub tuner_version: i64,
    /// Number of trials behind this entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_count: Option<i64>,
    /// GPU names present during measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_names: Option<Vec<String>>,
    /// llama.cpp build id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llamacpp_build: Option<String>,
}

/// The tuner best-config store (`best-<key>.json`).
///
/// Unknown fields (e.g. the `localbox-autobest-v1` provenance keys) are tolerated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunerBestConfig {
    /// Store schema version (must equal [`TUNER_SCHEMA`]).
    pub schema: u32,
    /// Model key this store is for.
    pub key: String,
    /// VRAM the store was tuned at, when recorded.
    #[serde(rename = "vramGB", default, skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<i64>,
    /// Tuned entries.
    pub entries: Vec<TunerEntry>,
}

impl TunerBestConfig {
    /// Whether the store's schema version is one this build understands.
    pub fn schema_supported(&self) -> bool {
        self.schema == TUNER_SCHEMA
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/best-q3635ba3b.json");

    #[test]
    fn deserializes_real_autobest_fixture() {
        let store: TunerBestConfig = serde_json::from_str(FIXTURE).unwrap();
        assert!(store.schema_supported());
        assert_eq!(store.key, "q3635ba3b");
        assert_eq!(store.entries.len(), 1);
        let e = &store.entries[0];
        assert_eq!(e.quant, "iq2m");
        assert_eq!(e.mode, Mode::Native);
        assert_eq!(e.prompt_length, PromptLength::Short);
        assert_eq!(e.profile, Profile::Pure);
        assert_eq!(e.tuner_version, 4);
    }

    #[test]
    fn overrides_rehydrate_into_launch_params() {
        let store: TunerBestConfig = serde_json::from_str(FIXTURE).unwrap();
        let o = &store.entries[0].overrides;
        assert_eq!(o.n_gpu_layers, Some(999));
        assert_eq!(o.n_cpu_moe, Some(35));
        assert_eq!(o.mlock, Some(true));
        assert_eq!(o.kv_k.as_deref(), Some("q8_0"));

        let lp = o.to_launch_params();
        assert_eq!(lp.n_gpu_layers, Some(999));
        assert_eq!(lp.n_cpu_moe, Some(35));
        assert_eq!(lp.mlock, Some(true));
        assert_eq!(lp.ubatch_size, Some(512));
        assert_eq!(lp.batch_size, Some(1024));
        assert_eq!(lp.flash_attn, Some(true));
        assert_eq!(lp.kv_k.as_deref(), Some("q8_0"));
    }

    #[test]
    fn overrides_round_trip_preserves_pascal_keys() {
        let o = Overrides {
            n_gpu_layers: Some(999),
            n_cpu_moe: Some(35),
            kv_k: Some("q8_0".into()),
            flash_attn: Some(true),
            ..Default::default()
        };
        let json = serde_json::to_string(&o).unwrap();
        assert!(json.contains("\"NGpuLayers\":999"));
        assert!(json.contains("\"NCpuMoe\":35"));
        assert!(json.contains("\"KvK\":\"q8_0\""));
        let back: Overrides = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let store = TunerBestConfig {
            schema: 2,
            key: "k".into(),
            vram_gb: None,
            entries: vec![],
        };
        assert!(!store.schema_supported());
    }
}
