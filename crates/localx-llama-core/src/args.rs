//! Pure `llama-server` argv builder + KV/spec-type gating + parser→sampler mapping.
//!
//! Side-effect free: given a resolved [`ModelDef`], a mode, a GGUF path, and
//! per-call [`LaunchParams`], emit the exact argument vector to splat. The
//! emission order matches the launcher byte-for-byte (pinned by golden tests).
//!
//! `ub <= b` is a hard llama-server invariant deliberately NOT enforced here —
//! that is the tuner's responsibility.

use crate::error::CoreError;
use crate::model::{context_value, Mode, ModelDef};

/// KV cache types the mainline build accepts.
pub const MAINLINE_KV_TYPES: &[&str] = &[
    "f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
];
/// KV cache types only the turbo forks register.
pub const TURBO_KV_TYPES: &[&str] = &["turbo3", "turbo4"];
/// MTP spec-type names (mainline canonical + fork aliases).
pub const MTP_SPEC_TYPES: &[&str] = &["draft-mtp", "mtp", "nextn"];

fn mode_supports_turbo_kv(mode: Mode) -> bool {
    matches!(mode, Mode::Turboquant | Mode::Mtpturbo)
}

/// Validate a KV cache type against the active mode.
pub fn validate_kv_type(ty: &str, mode: Mode) -> Result<(), CoreError> {
    let t = ty.to_ascii_lowercase();
    if MAINLINE_KV_TYPES.contains(&t.as_str()) {
        return Ok(());
    }
    if TURBO_KV_TYPES.contains(&t.as_str()) {
        if !mode_supports_turbo_kv(mode) {
            return Err(CoreError::KvTypeNeedsFork {
                ty: t,
                mainline: MAINLINE_KV_TYPES.join(", "),
            });
        }
        return Ok(());
    }
    Err(CoreError::UnknownKvType {
        ty: t,
        mainline: MAINLINE_KV_TYPES.join(", "),
        turbo: TURBO_KV_TYPES.join(", "),
    })
}

/// Reject MTP spec-types in plain turboquant mode (the fork has no MTP path).
pub fn validate_spec_type(spec: &str, mode: Mode) -> Result<(), CoreError> {
    if spec.trim().is_empty() {
        return Ok(());
    }
    let s = spec.to_ascii_lowercase();
    if mode == Mode::Turboquant && MTP_SPEC_TYPES.contains(&s.as_str()) {
        return Err(CoreError::SpecTypeUnsupported { spec: s });
    }
    Ok(())
}

/// Translate the catalog's canonical spec-type to the fork's name at emit time.
///
/// mtpturbo renames `draft-mtp` to bare `mtp`; everything else passes through.
pub fn spec_type_for_mode(spec: &str, mode: Mode) -> String {
    if mode != Mode::Mtpturbo {
        return spec.to_string();
    }
    if spec.eq_ignore_ascii_case("draft-mtp") {
        "mtp".to_string()
    } else {
        spec.to_string()
    }
}

/// Resolve active KV types: explicit -> per-model -> defaults (`q8_0`; V follows K).
pub fn resolve_kv_types(
    def: &ModelDef,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
) -> (String, String) {
    let non_blank = |s: Option<&str>| s.filter(|v| !v.trim().is_empty()).map(str::to_string);
    let k = non_blank(kv_k)
        .or_else(|| non_blank(def.kv_cache_k.as_deref()))
        .unwrap_or_else(|| "q8_0".to_string());
    let v = non_blank(kv_v)
        .or_else(|| non_blank(def.kv_cache_v.as_deref()))
        .unwrap_or_else(|| k.clone());
    (k, v)
}

/// A chat-template override, pre-resolved by the caller (file-vs-inline needs I/O).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateOverride {
    /// A template file on disk -> `--chat-template-file`.
    File(String),
    /// An inline template string -> `--chat-template`.
    Inline(String),
}

/// Chat-template args: an override wins, else a parser-based mapping.
pub fn chat_template_args(parser: &str, over: Option<&ChatTemplateOverride>) -> Vec<String> {
    match over {
        Some(ChatTemplateOverride::File(p)) => {
            vec!["--chat-template-file".to_string(), p.clone()]
        }
        Some(ChatTemplateOverride::Inline(s)) => vec!["--chat-template".to_string(), s.clone()],
        None => match parser {
            "qwen3coder" | "qwen36" | "qwen36-think" => vec!["--jinja".to_string()],
            _ => Vec::new(),
        },
    }
}

/// Reasoning routing flags. `strip` disables generation (not just wire hiding).
pub fn reasoning_args(thinking_policy: &str) -> Vec<String> {
    let policy = if thinking_policy.trim().is_empty() {
        "strip"
    } else {
        thinking_policy
    };
    if policy == "keep" {
        vec![
            "--reasoning".into(),
            "on".into(),
            "--reasoning-format".into(),
            "deepseek".into(),
        ]
    } else {
        vec![
            "--reasoning".into(),
            "off".into(),
            "--reasoning-budget".into(),
            "0".into(),
            "--reasoning-format".into(),
            "none".into(),
        ]
    }
}

/// The constrained-decoding capability a mode's runtime supports.
///
/// Every llama.cpp mode launches llama-server, whose completion endpoint accepts
/// a `json_schema` constraint. Host-neutral: reports what the runtime can do.
pub fn constrained_decoding(_mode: Mode) -> &'static str {
    "json_schema"
}

/// Modelfile PARAMETER/RENDERER/PARSER lines for a parser family.
fn parser_lines(parser: &str) -> Result<Vec<String>, CoreError> {
    let v: Vec<&str> = match parser {
        "none" => vec![],
        "qwen3coder" => vec![
            "RENDERER qwen3-coder",
            "PARSER qwen3-coder",
            "PARAMETER temperature 0.7",
            "PARAMETER top_k 20",
            "PARAMETER top_p 0.8",
            "PARAMETER repeat_penalty 1.05",
            "PARAMETER stop \"<|im_end|>\"",
            "PARAMETER stop \"<|im_start|>\"",
            "PARAMETER stop \"<|endoftext|>\"",
        ],
        "qwen36" => vec![
            "RENDERER qwen3-coder",
            "PARSER qwen3-coder",
            "PARAMETER temperature 0.7",
            "PARAMETER top_k 20",
            "PARAMETER top_p 0.8",
            "PARAMETER min_p 0",
            "PARAMETER presence_penalty 0",
            "PARAMETER repeat_penalty 1.05",
            "PARAMETER stop \"<|im_end|>\"",
            "PARAMETER stop \"<|im_start|>\"",
        ],
        "qwen36-think" => vec![
            "RENDERER qwen3-coder",
            "PARSER qwen3-coder",
            "PARAMETER temperature 0.6",
            "PARAMETER top_k 20",
            "PARAMETER top_p 0.95",
            "PARAMETER stop \"<|im_end|>\"",
            "PARAMETER stop \"<|im_start|>\"",
        ],
        other => return Err(CoreError::UnknownParser(other.to_string())),
    };
    Ok(v.into_iter().map(str::to_string).collect())
}

/// The strict overlay's Modelfile lines (sampler PARAMETERs + a SYSTEM block).
///
/// Only the PARAMETER lines become argv; the SYSTEM block is filtered out by
/// [`ollama_parameter_args`] and injected elsewhere.
fn strict_modelfile_lines() -> Vec<String> {
    [
        "PARAMETER temperature 0.2",
        "PARAMETER top_p 0.8",
        "PARAMETER top_k 20",
        "PARAMETER min_p 0.05",
        "PARAMETER presence_penalty 0",
        "PARAMETER repeat_penalty 1.15",
        "PARAMETER repeat_last_n 4096",
        "SYSTEM \"\"\"",
        "You are a strict senior software engineer working inside a real repository.",
        "\"\"\"",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

/// Translate Modelfile PARAMETER lines to llama-server sampler flags.
///
/// Unknown PARAMETER names (and non-PARAMETER lines) are skipped silently.
pub fn ollama_parameter_args<I, S>(lines: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = Vec::new();
    for line in lines {
        let text = line.as_ref().trim();
        let Some(rest) = text.strip_prefix("PARAMETER") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some((name, value)) = rest.split_once(char::is_whitespace) else {
            continue;
        };
        let mut value = value.trim().to_string();
        // Unwrap one layer of matching quotes.
        let bytes = value.as_bytes();
        if bytes.len() >= 2 {
            let first = bytes[0];
            let last = bytes[bytes.len() - 1];
            if (first == b'"' || first == b'\'') && last == first {
                value = value[1..value.len() - 1].to_string();
            }
        }
        let flag = match name.to_ascii_lowercase().as_str() {
            "temperature" => "--temp",
            "top_k" => "--top-k",
            "top_p" => "--top-p",
            "min_p" => "--min-p",
            "repeat_penalty" => "--repeat-penalty",
            "repeat_last_n" => "--repeat-last-n",
            "presence_penalty" => "--presence-penalty",
            "frequency_penalty" => "--frequency-penalty",
            "tfs_z" => "--tfs",
            "typical_p" => "--typical",
            "mirostat" => "--mirostat",
            "mirostat_tau" => "--mirostat-ent",
            "mirostat_eta" => "--mirostat-lr",
            "seed" => "--seed",
            _ => continue,
        };
        out.push(flag.to_string());
        out.push(value);
    }
    out
}

/// The strict sampler overlay translated for llama-server.
pub fn strict_sampler_args() -> Vec<String> {
    ollama_parameter_args(strict_modelfile_lines())
}

/// Per-call launch parameters. All fields are overrides; `None`/`0`/`false`
/// means "not passed" and falls back to the model def then a hard default,
/// exactly as the launcher resolves them.
#[derive(Debug, Clone, Default)]
pub struct LaunchParams {
    /// `--parallel` slot count (emitted when > 0).
    pub parallel: Option<i64>,
    /// `--cache-reuse` (emitted when > 0).
    pub cache_reuse: Option<i64>,
    /// KV cache key type override.
    pub kv_k: Option<String>,
    /// KV cache value type override.
    pub kv_v: Option<String>,
    /// GPU layers to offload.
    pub n_gpu_layers: Option<i64>,
    /// MoE layers to keep on CPU.
    pub n_cpu_moe: Option<i64>,
    /// Lock in RAM.
    pub mlock: Option<bool>,
    /// Disable mmap.
    pub no_mmap: Option<bool>,
    /// `--ubatch-size` (emitted when > 0).
    pub ubatch_size: Option<i64>,
    /// `--batch-size` (emitted when > 0).
    pub batch_size: Option<i64>,
    /// `--threads` (emitted when > 0).
    pub threads: Option<i64>,
    /// `--threads-batch` (emitted when > 0).
    pub threads_batch: Option<i64>,
    /// Flash attention on/off (emitted only when set).
    pub flash_attn: Option<bool>,
    /// Emit `--swa-full`.
    pub swa_full: bool,
    /// Emit `--cache-prompt` when `Some(true)`.
    pub cache_prompt: Option<bool>,
    /// Multi-GPU split mode.
    pub split_mode: Option<String>,
    /// Pre-resolved chat-template override (caller does the file-vs-inline check).
    pub chat_template_override: Option<ChatTemplateOverride>,
    /// Reasoning policy override (`strip`/`keep`).
    pub thinking_policy: Option<String>,
    /// Force the strict sampler overlay.
    pub strict: Option<bool>,
    /// Resolved multimodal projector path (enables `--mmproj`).
    pub vision_module_path: Option<String>,
    /// Resolved draft-model path for classic speculative decoding (enables
    /// `--spec-draft-model` as `draft-simple`).
    pub draft_module_path: Option<String>,
    /// Spec-type override.
    pub spec_type: Option<String>,
    /// Max draft tokens for speculative decoding.
    pub spec_draft_n_max: Option<i64>,
    /// Extra raw args appended last (after the def's own extra args).
    pub extra_args: Vec<String>,
}

/// Build the `llama-server` argument vector for a launch.
pub fn build_llama_server_args(
    def: &ModelDef,
    context_key: &str,
    mode: Mode,
    model_arg_path: &str,
    port: i64,
    p: &LaunchParams,
) -> Result<Vec<String>, CoreError> {
    let mut a: Vec<String> = Vec::new();
    let push2 = |x: &str, y: String, a: &mut Vec<String>| {
        a.push(x.to_string());
        a.push(y);
    };

    // -m <model>
    push2("-m", model_arg_path.to_string(), &mut a);

    // -c <num_ctx> when > 0
    if let Some(n) = context_value(def, context_key)? {
        if n > 0 {
            push2("-c", n.to_string(), &mut a);
        }
    }

    push2("--host", "127.0.0.1".to_string(), &mut a);
    push2("--port", port.to_string(), &mut a);

    if let Some(n) = p.parallel {
        if n > 0 {
            push2("--parallel", n.to_string(), &mut a);
        }
    }
    if let Some(n) = p.cache_reuse {
        if n > 0 {
            push2("--cache-reuse", n.to_string(), &mut a);
        }
    }

    // GPU layers: per-call (>0) -> per-model -> 999. Always emitted.
    let ngl = p
        .n_gpu_layers
        .filter(|v| *v > 0)
        .or(def.n_gpu_layers)
        .unwrap_or(999);
    push2("-ngl", ngl.to_string(), &mut a);

    // MoE CPU offload.
    let n_cpu_moe = p.n_cpu_moe.or(def.n_cpu_moe).unwrap_or(0);
    if n_cpu_moe > 0 {
        push2("--n-cpu-moe", n_cpu_moe.to_string(), &mut a);
    }

    if p.mlock.or(def.mlock).unwrap_or(false) {
        a.push("--mlock".to_string());
    }
    if p.no_mmap.or(def.no_mmap).unwrap_or(false) {
        a.push("--no-mmap".to_string());
    }

    if let Some(n) = p.ubatch_size {
        if n > 0 {
            push2("--ubatch-size", n.to_string(), &mut a);
        }
    }
    if let Some(n) = p.batch_size {
        if n > 0 {
            push2("--batch-size", n.to_string(), &mut a);
        }
    }
    if let Some(n) = p.threads {
        if n > 0 {
            push2("--threads", n.to_string(), &mut a);
        }
    }
    if let Some(n) = p.threads_batch {
        if n > 0 {
            push2("--threads-batch", n.to_string(), &mut a);
        }
    }

    if let Some(fa) = p.flash_attn.or(def.flash_attn) {
        push2(
            "--flash-attn",
            if fa { "on" } else { "off" }.to_string(),
            &mut a,
        );
    }
    if p.swa_full {
        a.push("--swa-full".to_string());
    }
    if p.cache_prompt == Some(true) {
        a.push("--cache-prompt".to_string());
    }
    if let Some(sm) = p.split_mode.as_deref() {
        if !sm.trim().is_empty() {
            push2("--split-mode", sm.to_string(), &mut a);
        }
    }

    // KV cache types (validated against mode). llama.cpp's --cache-type-k/v
    // matcher is exact-match against lowercase type names, so emit the lowercased
    // form we validated — otherwise an uppercase catalog spelling (e.g. "Q8_0")
    // passes validation here and is then rejected by the server at startup.
    let (kv_k, kv_v) = resolve_kv_types(def, p.kv_k.as_deref(), p.kv_v.as_deref());
    validate_kv_type(&kv_k, mode)?;
    validate_kv_type(&kv_v, mode)?;
    push2("--cache-type-k", kv_k.to_ascii_lowercase(), &mut a);
    push2("--cache-type-v", kv_v.to_ascii_lowercase(), &mut a);

    // Chat template.
    let parser = def.parser.as_deref().unwrap_or("none");
    a.extend(chat_template_args(
        parser,
        p.chat_template_override.as_ref(),
    ));

    // Vision / multimodal.
    if let Some(vp) = p.vision_module_path.as_deref() {
        if !vp.trim().is_empty() {
            push2("--mmproj", vp.to_string(), &mut a);
        }
    }

    // Reasoning routing.
    let policy = p
        .thinking_policy
        .as_deref()
        .or(def.thinking_policy.as_deref())
        .unwrap_or("strip");
    a.extend(reasoning_args(policy));

    // Parser sampler flags, then the strict overlay.
    a.extend(ollama_parameter_args(parser_lines(parser)?));
    if p.strict.or(def.strict).unwrap_or(false) {
        a.extend(strict_sampler_args());
    }

    // Speculative decoding. A resolved draft model runs as `draft-simple`;
    // any explicit spec-type other than that conflicts with it (one
    // speculation engine per launch). Without a drafter, the MTP/spec path
    // is unchanged (both fields required).
    let spec = p.spec_type.as_deref().or(def.spec_type.as_deref());
    let draft_path = p
        .draft_module_path
        .as_deref()
        .filter(|path| !path.trim().is_empty());
    if let Some(path) = draft_path {
        if let Some(spec) = spec.filter(|s| !s.trim().is_empty()) {
            if !spec.eq_ignore_ascii_case("draft-simple") {
                return Err(CoreError::SpecTypeConflictsWithDraft {
                    spec: spec.to_ascii_lowercase(),
                });
            }
        }
        push2("--spec-type", "draft-simple".to_string(), &mut a);
        push2("--spec-draft-model", path.to_string(), &mut a);
        if let Some(n) = p.spec_draft_n_max.filter(|n| *n > 0) {
            push2("--spec-draft-n-max", n.to_string(), &mut a);
        }
    } else if let (Some(spec), Some(n)) = (spec, p.spec_draft_n_max) {
        if !spec.trim().is_empty() && n > 0 {
            validate_spec_type(spec, mode)?;
            let emitted = spec_type_for_mode(spec, mode);
            push2("--spec-type", emitted, &mut a);
            push2("--spec-draft-n-max", n.to_string(), &mut a);
        }
    }

    // Extra args: per-model first, then per-call (call wins, last on the line).
    a.extend(def.extra_args.iter().cloned());
    a.extend(p.extra_args.iter().cloned());

    Ok(a)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::model::QuantEntry;

    fn base_def() -> ModelDef {
        let mut d = ModelDef {
            repo: "owner/model".into(),
            parser: Some("qwen36".into()),
            ..Default::default()
        };
        d.quants.insert("q4".into(), QuantEntry::default());
        d.contexts.insert("".into(), 65536);
        d.contexts.insert("128k".into(), 131072);
        d
    }

    #[test]
    fn minimal_argv_is_byte_exact() {
        let d = base_def();
        let args = build_llama_server_args(
            &d,
            "",
            Mode::Native,
            "/models/m.gguf",
            8080,
            &LaunchParams::default(),
        )
        .unwrap();
        assert_eq!(
            args,
            vec![
                "-m",
                "/models/m.gguf",
                "-c",
                "65536",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "-ngl",
                "999",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--jinja",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--reasoning-format",
                "none",
                "--temp",
                "0.7",
                "--top-k",
                "20",
                "--top-p",
                "0.8",
                "--min-p",
                "0",
                "--presence-penalty",
                "0",
                "--repeat-penalty",
                "1.05",
            ]
        );
    }

    #[test]
    fn strict_overlay_appends_after_parser_samplers() {
        let d = base_def();
        let p = LaunchParams {
            strict: Some(true),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "128k", Mode::Native, "m.gguf", 9090, &p).unwrap();
        // strict sampler flags present, appended after the parser samplers.
        let joined = args.join(" ");
        assert!(joined.contains("--temp 0.2"));
        assert!(joined.contains("--repeat-last-n 4096"));
        // context 128k resolved.
        assert!(joined.contains("-c 131072"));
        // strict SYSTEM block must NOT leak into argv.
        assert!(!joined.contains("senior software engineer"));
    }

    #[test]
    fn tuner_knobs_and_extra_args_emit_in_order() {
        let mut d = base_def();
        d.extra_args = vec!["--from-def".into()];
        let p = LaunchParams {
            parallel: Some(1),
            n_gpu_layers: Some(20),
            n_cpu_moe: Some(16),
            ubatch_size: Some(512),
            batch_size: Some(512),
            flash_attn: Some(true),
            extra_args: vec!["--from-call".into()],
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        let j = args.join(" ");
        assert!(j.contains("--parallel 1"));
        assert!(j.contains("-ngl 20"));
        assert!(j.contains("--n-cpu-moe 16"));
        assert!(j.contains("--ubatch-size 512 --batch-size 512"));
        assert!(j.contains("--flash-attn on"));
        // per-model extra args before per-call.
        assert!(j.ends_with("--from-def --from-call"));
        // ub<=b NOT enforced: both emitted verbatim even if equal.
    }

    #[test]
    fn kv_type_is_emitted_lowercase_even_from_an_uppercase_catalog_spelling() {
        // llama.cpp matches --cache-type-k exact-lowercase; an uppercase catalog
        // value must not pass validation and then get rejected by the server.
        let d = base_def();
        let p = LaunchParams {
            kv_k: Some("Q8_0".into()),
            kv_v: Some("Q8_0".into()),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        let j = args.join(" ");
        assert!(j.contains("--cache-type-k q8_0"), "got: {j}");
        assert!(j.contains("--cache-type-v q8_0"), "got: {j}");
        assert!(!j.contains("Q8_0"), "uppercase leaked to argv: {j}");
    }

    #[test]
    fn turbo_kv_rejected_on_native_accepted_on_fork() {
        let d = base_def();
        let p = LaunchParams {
            kv_k: Some("turbo4".into()),
            ..Default::default()
        };
        let err = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap_err();
        assert!(matches!(err, CoreError::KvTypeNeedsFork { .. }));
        assert!(build_llama_server_args(&d, "", Mode::Turboquant, "m.gguf", 8080, &p).is_ok());
    }

    #[test]
    fn mtp_spec_translated_on_mtpturbo_rejected_on_turboquant() {
        let d = base_def();
        let p = LaunchParams {
            spec_type: Some("draft-mtp".into()),
            spec_draft_n_max: Some(4),
            ..Default::default()
        };
        // turboquant rejects MTP.
        assert!(matches!(
            build_llama_server_args(&d, "", Mode::Turboquant, "m.gguf", 8080, &p).unwrap_err(),
            CoreError::SpecTypeUnsupported { .. }
        ));
        // mtpturbo renames draft-mtp -> mtp.
        let args = build_llama_server_args(&d, "", Mode::Mtpturbo, "m.gguf", 8080, &p).unwrap();
        let j = args.join(" ");
        assert!(j.contains("--spec-type mtp --spec-draft-n-max 4"));
        assert!(!j.contains("draft-mtp"));
    }

    #[test]
    fn keep_thinking_emits_deepseek() {
        assert_eq!(
            reasoning_args("keep"),
            vec!["--reasoning", "on", "--reasoning-format", "deepseek"]
        );
    }

    #[test]
    fn chat_template_override_file_vs_inline() {
        assert_eq!(
            chat_template_args("none", Some(&ChatTemplateOverride::File("t.jinja".into()))),
            vec!["--chat-template-file", "t.jinja"]
        );
        assert_eq!(
            chat_template_args("none", Some(&ChatTemplateOverride::Inline("{{x}}".into()))),
            vec!["--chat-template", "{{x}}"]
        );
        assert!(chat_template_args("none", None).is_empty());
        assert_eq!(chat_template_args("qwen36", None), vec!["--jinja"]);
    }

    #[test]
    fn mmproj_only_for_nonblank_vision_path() {
        let d = base_def();
        let p = LaunchParams {
            vision_module_path: Some("proj.gguf".into()),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        assert!(args.join(" ").contains("--mmproj proj.gguf"));
    }

    #[test]
    fn draft_module_emits_draft_simple_speculation() {
        let d = base_def();
        // A resolved drafter alone runs as draft-simple, byte-exact tail.
        let p = LaunchParams {
            draft_module_path: Some("drafter.gguf".into()),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        let j = args.join(" ");
        assert!(j.contains("--spec-type draft-simple --spec-draft-model drafter.gguf"));
        assert!(!j.contains("--spec-draft-n-max"));
        // Draft depth rides along when set.
        let p = LaunchParams {
            draft_module_path: Some("drafter.gguf".into()),
            spec_draft_n_max: Some(6),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        assert!(args.join(" ").contains(
            "--spec-type draft-simple --spec-draft-model drafter.gguf --spec-draft-n-max 6"
        ));
        // An explicit draft-simple spec-type is redundant but compatible.
        let p = LaunchParams {
            draft_module_path: Some("drafter.gguf".into()),
            spec_type: Some("draft-simple".into()),
            ..Default::default()
        };
        assert!(build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).is_ok());
        // The drafter works in prism mode too (no MTP there, but classic
        // drafting is plain llama-server surface).
        let p = LaunchParams {
            draft_module_path: Some("drafter.gguf".into()),
            ..Default::default()
        };
        assert!(build_llama_server_args(&d, "", Mode::PrismMl, "m.gguf", 8080, &p).is_ok());
        // A blank path emits nothing draft-related.
        let p = LaunchParams {
            draft_module_path: Some("  ".into()),
            ..Default::default()
        };
        let args = build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap();
        assert!(!args.join(" ").contains("spec-draft-model"));
    }

    #[test]
    fn draft_module_refuses_a_conflicting_spec_type() {
        let mut d = base_def();
        d.spec_type = Some("draft-mtp".into());
        let p = LaunchParams {
            draft_module_path: Some("drafter.gguf".into()),
            spec_draft_n_max: Some(4),
            ..Default::default()
        };
        // One speculation engine per launch: an MTP spec-type (from the model
        // def or the call) cannot combine with a drafter.
        assert!(matches!(
            build_llama_server_args(&d, "", Mode::Native, "m.gguf", 8080, &p).unwrap_err(),
            CoreError::SpecTypeConflictsWithDraft { .. }
        ));
    }
}
