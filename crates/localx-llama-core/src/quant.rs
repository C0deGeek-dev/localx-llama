//! Quant taxonomy + HuggingFace filename parsing.
//!
//! Ported from the launcher's HF helpers: quant-code extraction from a GGUF
//! filename, the semantics table (family + tier note), quant-key normalization,
//! the default context ladder, display-name formatting, and parser suggestion.

use std::collections::BTreeMap;
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

/// A quant name reduced to its comparable form: the imatrix marker dropped,
/// separators removed, uppercased. `Q4_K_M`, `q4km` and `i1-q4km` all reduce to
/// `Q4KM`, so the taxonomy can be looked up by the catalog key the rest of the
/// system carries as readily as by the canonical code.
fn comparable(code: &str) -> String {
    let lower = code.to_ascii_lowercase();
    let body = lower
        .strip_prefix("i1-")
        .or_else(|| lower.strip_prefix("imat-"))
        .or_else(|| lower.strip_prefix("imatrix-"))
        .unwrap_or(&lower);
    body.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// The taxonomy row for a quant code or catalog key: its canonical code, its
/// family, and its tier. `None` for a name the taxonomy does not know.
#[must_use]
pub fn quant_semantics(code: &str) -> Option<(&'static str, &'static str, &'static str)> {
    let wanted = comparable(code);
    QUANT_SEMANTICS
        .iter()
        .find(|(candidate, _, _)| comparable(candidate) == wanted)
        .map(|(code, family, tier)| (*code, *family, *tier))
}

/// One GGUF file from a Hugging Face repository listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantFile {
    pub name: String,
    pub size: Option<u64>,
}

/// One selectable quant variant: a stable key, its ordered file(s), and their
/// combined size when every shard reported one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantCandidate {
    pub key: String,
    pub files: Vec<String>,
    pub total_size: Option<u64>,
}

/// Why a quant could not be selected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantSelectError {
    #[error("no GGUF quant variants were found")]
    NoneAvailable,
    #[error("quant '{requested}' not found; available: {available}")]
    Unknown {
        requested: String,
        available: String,
    },
}

fn compact(token: &str) -> String {
    token
        .to_ascii_lowercase()
        .chars()
        .filter(|character| !matches!(character, '_' | '-'))
        .collect()
}

fn is_quant_token(segment: &str) -> bool {
    let segment = segment.to_ascii_lowercase();
    if matches!(segment.as_str(), "f16" | "f32" | "bf16") {
        return true;
    }
    if let Some(rest) = segment.strip_prefix("iq") {
        return rest.starts_with(|character: char| character.is_ascii_digit());
    }
    if let Some(rest) = segment.strip_prefix('q') {
        return rest.starts_with(|character: char| character.is_ascii_digit());
    }
    false
}

fn split_shard(stem: &str) -> (&str, Option<(u32, u32)>) {
    let Some(of_position) = stem.rfind("-of-") else {
        return (stem, None);
    };
    let before = &stem[..of_position];
    let total = &stem[of_position + "-of-".len()..];
    let Some(dash) = before.rfind('-') else {
        return (stem, None);
    };
    let index = &before[dash + 1..];
    let is_digits = |value: &str| {
        !value.is_empty() && value.bytes().all(|character| character.is_ascii_digit())
    };
    if is_digits(index) && is_digits(total) {
        let index = index.parse().unwrap_or(0);
        let total = total.parse().unwrap_or(0);
        (&before[..dash], Some((index, total)))
    } else {
        (stem, None)
    }
}

/// The stable quant key denoted by a GGUF filename. Imatrix markers are kept
/// so static and imatrix builds remain distinct candidates.
#[must_use]
pub fn quant_key_from_filename(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let (base, _) = split_shard(stem);
    let segments: Vec<&str> = base
        .split(['.', '-'])
        .filter(|segment| !segment.is_empty())
        .collect();
    let index = segments
        .iter()
        .rposition(|segment| is_quant_token(segment))?;
    let key = compact(segments[index]);
    let imatrix = index > 0 && matches!(segments[index - 1], "i1" | "imat" | "imatrix");
    Some(if imatrix { format!("i1-{key}") } else { key })
}

fn shard_index(name: &str) -> u32 {
    let lower = name.to_ascii_lowercase();
    let stem = lower.strip_suffix(".gguf").unwrap_or(&lower);
    split_shard(stem).1.map_or(1, |(index, _)| index)
}

const MAX_SHARD_COUNT: u32 = 1024;

/// Expand a standard first split-GGUF filename into all of its shard names.
/// Non-primary, malformed, or unreasonable shard suffixes remain unchanged.
#[must_use]
pub fn shard_files_from_primary(primary: &str) -> Vec<String> {
    let lower = primary.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return vec![primary.to_string()];
    };
    let Some(of_position) = stem.rfind("-of-") else {
        return vec![primary.to_string()];
    };
    let before = &stem[..of_position];
    let Some(index_dash) = before.rfind('-') else {
        return vec![primary.to_string()];
    };
    let index_text = &stem[index_dash + 1..of_position];
    let total_text = &stem[of_position + "-of-".len()..];
    let is_digits = |value: &str| {
        !value.is_empty() && value.bytes().all(|character| character.is_ascii_digit())
    };
    if !is_digits(index_text) || !is_digits(total_text) {
        return vec![primary.to_string()];
    }
    let (Ok(index), Ok(total)) = (index_text.parse::<u32>(), total_text.parse::<u32>()) else {
        return vec![primary.to_string()];
    };
    if index != 1 || !(2..=MAX_SHARD_COUNT).contains(&total) {
        return vec![primary.to_string()];
    }

    let index_start = index_dash + 1;
    let index_width = of_position - index_start;
    let prefix = &primary[..index_start];
    let suffix = &primary[of_position..];
    (1..=total)
        .map(|part| format!("{prefix}{part:0index_width$}{suffix}"))
        .collect()
}

/// Group GGUF files into stable quant candidates with ordered shards and a
/// total size only when every file reported one.
#[must_use]
pub fn quant_candidates(files: &[QuantFile]) -> Vec<QuantCandidate> {
    let mut groups: BTreeMap<String, Vec<&QuantFile>> = BTreeMap::new();
    for file in files {
        if let Some(key) = quant_key_from_filename(&file.name) {
            groups.entry(key).or_default().push(file);
        }
    }
    groups
        .into_iter()
        .map(|(key, mut parts)| {
            parts.sort_by_key(|file| shard_index(&file.name));
            let total_size = parts
                .iter()
                .try_fold(0_u64, |total, file| file.size.map(|size| total + size));
            let files = parts.iter().map(|file| file.name.clone()).collect();
            QuantCandidate {
                key,
                files,
                total_size,
            }
        })
        .collect()
}

fn without_imatrix(compact_key: &str) -> &str {
    compact_key.strip_prefix("i1").unwrap_or(compact_key)
}

fn default_index(candidates: &[QuantCandidate]) -> usize {
    if let Some(index) = candidates
        .iter()
        .position(|candidate| without_imatrix(&compact(&candidate.key)) == "q4km")
    {
        return index;
    }
    if candidates
        .iter()
        .all(|candidate| candidate.total_size.is_some())
    {
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by_key(|&index| candidates[index].total_size);
        order[order.len() / 2]
    } else {
        candidates.len() / 2
    }
}

fn match_quant(candidates: &[QuantCandidate], requested: &str) -> Option<usize> {
    let wanted = compact(requested);
    if let Some(index) = candidates.iter().position(|candidate| {
        candidate.key.eq_ignore_ascii_case(requested) || compact(&candidate.key) == wanted
    }) {
        return Some(index);
    }
    let wanted_without_imatrix = without_imatrix(&wanted);
    let fuzzy: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            without_imatrix(&compact(&candidate.key)) == wanted_without_imatrix
        })
        .map(|(index, _)| index)
        .collect();
    (fuzzy.len() == 1).then(|| fuzzy[0])
}

/// Select an exact/compact quant request, with an unambiguous
/// imatrix-insensitive fallback, or choose the established Q4_K_M/median
/// default when no request was supplied.
///
/// # Errors
/// A typed error when no candidates exist or the requested quant is unknown.
pub fn select_quant(
    mut candidates: Vec<QuantCandidate>,
    requested: Option<&str>,
) -> Result<QuantCandidate, QuantSelectError> {
    if candidates.is_empty() {
        return Err(QuantSelectError::NoneAvailable);
    }
    match requested {
        Some(requested) => {
            let index =
                match_quant(&candidates, requested).ok_or_else(|| QuantSelectError::Unknown {
                    requested: requested.to_string(),
                    available: candidates
                        .iter()
                        .map(|candidate| candidate.key.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                })?;
            Ok(candidates.swap_remove(index))
        }
        None => {
            let index = default_index(&candidates);
            Ok(candidates.swap_remove(index))
        }
    }
}

/// Baseline picker note: `<CODE> · <family> · ~<size> GB · <tier>` (missing parts omitted).
pub fn quant_note_text(code: &str, size_gb: Option<f64>) -> String {
    let entry = quant_semantics(code);
    let head = entry.map_or_else(
        || code.to_ascii_uppercase(),
        |(code, _, _)| code.to_string(),
    );
    let mut parts: Vec<String> = vec![head];
    let sem = entry.map(|(_, family, tier)| (family, tier));
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

    fn file(name: &str, size: Option<u64>) -> QuantFile {
        QuantFile {
            name: name.to_string(),
            size,
        }
    }

    #[test]
    fn discovery_quant_keys_cover_real_naming_conventions() {
        assert_eq!(
            quant_key_from_filename("Llama-3.2-1B-Instruct-Q4_K_M.gguf").as_deref(),
            Some("q4km")
        );
        assert_eq!(
            quant_key_from_filename("Some-Model.Q4_K_M.gguf").as_deref(),
            Some("q4km")
        );
        assert_eq!(
            quant_key_from_filename("Some-Model.i1-Q4_K_M.gguf").as_deref(),
            Some("i1-q4km")
        );
        assert_eq!(
            quant_key_from_filename("Model-IQ4_XS.gguf").as_deref(),
            Some("iq4xs")
        );
        assert_eq!(
            quant_key_from_filename("Model-Q8_0.gguf").as_deref(),
            Some("q80")
        );
        assert_eq!(
            quant_key_from_filename("Model-f16.gguf").as_deref(),
            Some("f16")
        );
        assert_eq!(quant_key_from_filename("Qwen3-27B.gguf"), None);
        assert_eq!(quant_key_from_filename("README.md"), None);
    }

    #[test]
    fn multipart_shards_group_into_one_candidate_with_summed_size() {
        let files = vec![
            file("Big-Model-Q6_K-00002-of-00003.gguf", Some(20)),
            file("Big-Model-Q6_K-00001-of-00003.gguf", Some(10)),
            file("Big-Model-Q6_K-00003-of-00003.gguf", Some(30)),
            file("Big-Model-Q4_K_M.gguf", Some(5)),
        ];
        let candidates = quant_candidates(&files);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].key, "q4km");
        assert_eq!(candidates[0].files, vec!["Big-Model-Q4_K_M.gguf"]);
        assert_eq!(candidates[0].total_size, Some(5));

        let q6k = &candidates[1];
        assert_eq!(q6k.key, "q6k");
        assert_eq!(
            q6k.files,
            vec![
                "Big-Model-Q6_K-00001-of-00003.gguf",
                "Big-Model-Q6_K-00002-of-00003.gguf",
                "Big-Model-Q6_K-00003-of-00003.gguf",
            ]
        );
        assert_eq!(q6k.total_size, Some(60));
    }

    #[test]
    fn a_primary_shard_expands_without_changing_its_filename_format() {
        assert_eq!(
            shard_files_from_primary("sub/Model-Q6_K-00001-of-00003.GGUF"),
            vec![
                "sub/Model-Q6_K-00001-of-00003.GGUF",
                "sub/Model-Q6_K-00002-of-00003.GGUF",
                "sub/Model-Q6_K-00003-of-00003.GGUF",
            ]
        );
        assert_eq!(
            shard_files_from_primary("Model-Q6_K-1-of-12.gguf")[11],
            "Model-Q6_K-12-of-12.gguf"
        );
    }

    #[test]
    fn non_primary_malformed_and_unreasonable_shards_stay_single_file() {
        for name in [
            "Model-Q4_K_M.gguf",
            "Model-Q6_K-00002-of-00003.gguf",
            "Model-Q6_K-00001-of-00001.gguf",
            "Model-Q6_K-00001-of-01025.gguf",
            "Model-Q6_K-first-of-three.gguf",
            "Model-Q6_K-00001-of-00003.bin",
        ] {
            assert_eq!(shard_files_from_primary(name), vec![name.to_string()]);
        }
    }

    #[test]
    fn a_missing_shard_size_makes_the_total_unknown() {
        let candidates = quant_candidates(&[
            file("M-Q6_K-00001-of-00002.gguf", Some(10)),
            file("M-Q6_K-00002-of-00002.gguf", None),
        ]);
        assert_eq!(candidates[0].total_size, None);
        assert_eq!(candidates[0].files.len(), 2);
    }

    #[test]
    fn a_requested_quant_matches_by_key_or_compact_form() {
        let candidates = quant_candidates(&[
            file("M-Q4_K_M.gguf", Some(4)),
            file("M-IQ4_XS.gguf", Some(3)),
        ]);
        assert_eq!(
            select_quant(candidates.clone(), Some("iq4xs")).unwrap().key,
            "iq4xs"
        );
        assert_eq!(
            select_quant(candidates.clone(), Some("Q4_K_M"))
                .unwrap()
                .key,
            "q4km"
        );
        assert!(matches!(
            select_quant(candidates, Some("q2k")).unwrap_err(),
            QuantSelectError::Unknown { .. }
        ));
    }

    #[test]
    fn the_default_prefers_q4km_then_median_size() {
        let with_q4km = quant_candidates(&[
            file("M-Q2_K.gguf", Some(2)),
            file("M-Q4_K_M.gguf", Some(4)),
            file("M-Q8_0.gguf", Some(8)),
        ]);
        assert_eq!(select_quant(with_q4km, None).unwrap().key, "q4km");

        let without_q4km = quant_candidates(&[
            file("M-IQ2_XS.gguf", Some(1)),
            file("M-Q5_K_M.gguf", Some(5)),
            file("M-Q8_0.gguf", Some(8)),
        ]);
        assert_eq!(select_quant(without_q4km, None).unwrap().key, "q5km");
    }

    #[test]
    fn selecting_from_an_empty_list_is_an_error() {
        assert_eq!(
            select_quant(Vec::new(), None).unwrap_err(),
            QuantSelectError::NoneAvailable
        );
    }

    #[test]
    fn an_imatrix_only_repo_resolves_like_a_real_listing() {
        let base = "Qwen3.8-27B-Uncensored-Heretic-Abliterated";
        let files = vec![
            file(&format!("{base}.i1-IQ4_XS.gguf"), Some(15_082_507_808)),
            file(&format!("{base}.i1-Q4_K_M.gguf"), Some(16_547_401_248)),
            file(&format!("{base}.i1-Q6_K.gguf"), Some(22_082_530_848)),
            file(&format!("{base}.i1-IQ1_S.gguf"), Some(7_149_825_568)),
            file(&format!("{base}.imatrix.gguf"), Some(13_642_624)),
        ];
        let candidates = quant_candidates(&files);
        let keys: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.key.as_str())
            .collect();
        assert_eq!(keys, vec!["i1-iq1s", "i1-iq4xs", "i1-q4km", "i1-q6k"]);

        for request in ["q4km", "Q4_K_M", "i1-q4km", "i1-Q4_K_M"] {
            assert_eq!(
                select_quant(candidates.clone(), Some(request)).unwrap().key,
                "i1-q4km",
                "request {request:?}"
            );
        }
        assert_eq!(select_quant(candidates, None).unwrap().key, "i1-q4km");
    }

    /// The taxonomy is looked up with whatever spelling the caller has: the
    /// canonical code, the compact catalog key, or that key still carrying its
    /// imatrix marker. Discovery produces the last two, so a lookup that only
    /// accepted the first left every imported model without a note.
    #[test]
    fn the_taxonomy_accepts_the_catalog_key_and_the_canonical_code() {
        let canonical = quant_semantics("Q4_K_M");
        assert_eq!(canonical.map(|(code, _, _)| code), Some("Q4_K_M"));
        assert_eq!(quant_semantics("q4km"), canonical);
        assert_eq!(quant_semantics("i1-q4km"), canonical);
        assert_eq!(quant_semantics("not-a-quant"), None);
        // The note heads with the canonical spelling however it was asked for.
        assert!(quant_note_text("i1-q4km", None).starts_with("Q4_K_M"));
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
