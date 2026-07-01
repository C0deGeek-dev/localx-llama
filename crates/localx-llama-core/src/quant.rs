//! Quant taxonomy + HuggingFace filename parsing.
//!
//! Ported from the launcher's HF helpers: quant-code extraction from a GGUF
//! filename, the semantics table (family + tier note), quant-key normalization,
//! the default context ladder, display-name formatting, and parser suggestion.

use std::sync::OnceLock;

use regex::Regex;

/// `(code, family, tier-note)` for known quants. Tier note may be empty.
pub const QUANT_SEMANTICS: &[(&str, &str, &str)] = &[
    ("IQ1_S", "1-bit imatrix", "for the desperate"),
    ("IQ1_M", "1-bit imatrix", "mostly desperate"),
    ("IQ2_XXS", "2-bit imatrix", "very low quality"),
    ("IQ2_XS", "2-bit imatrix", "very low quality"),
    ("IQ2_S", "2-bit imatrix", "low quality"),
    ("IQ2_M", "2-bit imatrix", "low quality, long-context only"),
    ("Q2_K_S", "2-bit k-quant small", "very low quality"),
    ("Q2_K", "2-bit k-quant", "IQ3_XXS often better"),
    ("IQ3_XXS", "3-bit imatrix", "lower quality"),
    ("IQ3_XS", "3-bit imatrix", "lower quality"),
    ("Q3_K_S", "3-bit k-quant small", "IQ3_XS often better"),
    ("IQ3_S", "3-bit imatrix", "beats Q3_K*"),
    ("IQ3_M", "3-bit imatrix", "good 3-bit baseline"),
    ("Q3_K_M", "3-bit k-quant medium", "IQ3_S often better"),
    ("Q3_K_L", "3-bit k-quant large", "IQ3_M often better"),
    (
        "IQ4_XS",
        "4-bit imatrix",
        "good 4-bit, smallest 4-bit option",
    ),
    ("IQ4_NL", "4-bit imatrix non-linear", "good 4-bit baseline"),
    ("Q4_0", "4-bit legacy", "fast, low quality"),
    ("Q4_1", "4-bit legacy", ""),
    (
        "Q4_K_S",
        "4-bit k-quant small",
        "optimal size/speed/quality",
    ),
    (
        "Q4_K_M",
        "4-bit k-quant medium",
        "fast, recommended sweet spot",
    ),
    ("Q4_K_P", "4-bit k-quant", "similar to Q4_K_M"),
    ("MXFP4", "4-bit MoE-aware", "similar to IQ4_NL"),
    ("MXFP4_MOE", "4-bit MoE-aware", "similar to IQ4_NL"),
    ("Q5_K_S", "5-bit k-quant small", "noticeable quality bump"),
    ("Q5_K_M", "5-bit k-quant medium", "noticeable quality bump"),
    ("Q6_K", "6-bit k-quant", "high quality"),
    ("Q6_K_P", "6-bit k-quant", "high quality"),
    ("Q8_0", "8-bit", "highest practical quality"),
    ("BF16", "bfloat16 full precision", "expect partial offload"),
    ("F16", "float16 full precision", "expect partial offload"),
    (
        "F32",
        "float32 full precision",
        "almost certainly partial offload",
    ),
];

/// Family + tier note for a quant code (uppercased lookup), if known.
pub fn quant_semantics(code: &str) -> Option<(&'static str, &'static str)> {
    let upper = code.to_ascii_uppercase();
    QUANT_SEMANTICS
        .iter()
        .find(|(c, _, _)| *c == upper)
        .map(|(_, family, tier)| (*family, *tier))
}

fn quant_code_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            r"(IQ\d_[A-Z]+(?:_[A-Z0-9]+)?)$",
            r"(MXFP\d_[A-Z]+)$",
            r"(Q\d_K_[A-Z])$",
            r"(Q\d_K)$",
            r"(Q\d_\d)$",
            r"(APEX(?:-I)?-[A-Z][A-Za-z]+)$",
            r"(BF16|F16|F32)$",
        ]
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect()
    })
}

/// Extract the quant code from a GGUF filename (leaf, extension stripped).
///
/// Returns `None` when no known quant pattern matches the tail.
pub fn hf_quant_code(file_name: &str) -> Option<String> {
    let leaf = file_name.rsplit('/').next().unwrap_or(file_name);
    // Strip only the final extension, like Path::GetFileNameWithoutExtension.
    let stem = leaf.rsplit_once('.').map(|(a, _)| a).unwrap_or(leaf);
    for re in quant_code_patterns() {
        if let Some(caps) = re.captures(stem) {
            if let Some(m) = caps.get(1) {
                return Some(m.as_str().to_string());
            }
        }
    }
    None
}

/// Normalize a quant code to a catalog key: drop underscores, lowercase.
pub fn quant_code_to_key(code: &str) -> String {
    code.replace('_', "").to_ascii_lowercase()
}

/// Baseline picker note: `<CODE> · <family> · ~<size> GB · <tier>` (missing parts omitted).
pub fn quant_note_text(code: &str, size_gb: Option<f64>) -> String {
    let upper = code.to_ascii_uppercase();
    let mut parts: Vec<String> = vec![upper.clone()];
    let sem = quant_semantics(&upper);
    if let Some((family, _)) = sem {
        if !family.is_empty() {
            parts.push(family.to_string());
        }
    }
    if let Some(sz) = size_gb {
        if sz > 0.0 {
            parts.push(format!("~{} GB", (sz * 10.0).round() / 10.0));
        }
    }
    if let Some((_, tier)) = sem {
        if !tier.is_empty() {
            parts.push(tier.to_string());
        }
    }
    parts.join(" · ")
}

/// The default context ladder used when importing a model from HuggingFace.
pub fn default_contexts() -> [(&'static str, i64); 5] {
    [
        ("", 65_536),
        ("32k", 32_768),
        ("64k", 65_536),
        ("128k", 131_072),
        ("256k", 262_144),
    ]
}

/// Lazily compile a static pattern, degrading to `None` on the (impossible for a
/// literal) compile error rather than panicking — the crate forbids `expect`.
fn cached_regex(cell: &'static OnceLock<Option<Regex>>, pattern: &str) -> Option<&'static Regex> {
    cell.get_or_init(|| Regex::new(pattern).ok()).as_ref()
}

/// Format a repo id into a display name: tail after `/`, `-`->space, `Qwen<d>`->`Qwen <d>`.
pub fn format_display_name(repo: &str) -> String {
    static QWEN: OnceLock<Option<Regex>> = OnceLock::new();
    let tail = repo.rsplit('/').next().unwrap_or(repo);
    let spaced = tail.replace('-', " ");
    match cached_regex(&QWEN, r"(?i)Qwen(\d)") {
        Some(re) => re.replace_all(&spaced, "Qwen $1").into_owned(),
        None => spaced,
    }
}

/// Suggest a parser family from a repo id.
pub fn suggest_parser(repo: &str) -> &'static str {
    static QWEN35: OnceLock<Option<Regex>> = OnceLock::new();
    static THINK: OnceLock<Option<Regex>> = OnceLock::new();
    let name = repo.to_ascii_lowercase();
    if name.contains("coder") {
        return "qwen3coder";
    }
    let is_qwen35 = cached_regex(&QWEN35, r"qwen3\.?[56]").is_some_and(|re| re.is_match(&name));
    if is_qwen35 {
        let is_think = cached_regex(&THINK, r"thinking|reasoning|opus|sonnet|haiku|claude")
            .is_some_and(|re| re.is_match(&name));
        if is_think {
            return "qwen36-think";
        }
        return "qwen36";
    }
    "none"
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn extracts_quant_code_from_filenames() {
        assert_eq!(
            hf_quant_code("model-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            hf_quant_code("foo/bar-IQ2_M.gguf").as_deref(),
            Some("IQ2_M")
        );
        assert_eq!(hf_quant_code("m.Q6_K.gguf").as_deref(), Some("Q6_K"));
        assert_eq!(hf_quant_code("x-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(hf_quant_code("x-Q4_0.gguf").as_deref(), Some("Q4_0"));
        assert_eq!(hf_quant_code("no-quant-here.gguf"), None);
    }

    #[test]
    fn quant_key_normalization() {
        assert_eq!(quant_code_to_key("Q4_K_M"), "q4km");
        assert_eq!(quant_code_to_key("IQ2_M"), "iq2m");
        assert_eq!(quant_code_to_key("BF16"), "bf16");
    }

    #[test]
    fn quant_note_joins_known_parts() {
        assert_eq!(
            quant_note_text("Q4_K_M", Some(12.34)),
            "Q4_K_M · 4-bit k-quant medium · ~12.3 GB · fast, recommended sweet spot"
        );
        // unknown code: only the code, plus size if given.
        assert_eq!(quant_note_text("ZZZ", None), "ZZZ");
        assert_eq!(quant_note_text("ZZZ", Some(3.0)), "ZZZ · ~3 GB");
        // Q4_1 has an empty tier note -> omitted.
        assert_eq!(quant_note_text("Q4_1", None), "Q4_1 · 4-bit legacy");
    }

    #[test]
    fn default_context_ladder() {
        let ladder = default_contexts();
        assert_eq!(ladder[0], ("", 65_536));
        assert_eq!(ladder[3], ("128k", 131_072));
        assert_eq!(ladder[4], ("256k", 262_144));
    }

    #[test]
    fn display_name_formatting() {
        assert_eq!(
            format_display_name("owner/Qwen3-Coder-30B"),
            "Qwen 3 Coder 30B"
        );
        assert_eq!(format_display_name("just-a-name"), "just a name");
    }

    #[test]
    fn parser_suggestion() {
        assert_eq!(suggest_parser("owner/Qwen3-Coder-30B"), "qwen3coder");
        assert_eq!(suggest_parser("owner/Qwen3.6-32B"), "qwen36");
        assert_eq!(suggest_parser("owner/Qwen3.6-Thinking-32B"), "qwen36-think");
        assert_eq!(suggest_parser("owner/Llama-3-8B"), "none");
    }
}
