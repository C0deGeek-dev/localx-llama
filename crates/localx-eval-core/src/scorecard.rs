//! The machine-readable capability scorecard: the cross-corpus contract a
//! headless harness run emits so a benchmark can grade the *harness* on three
//! layers — results, code quality, and process — rather than a single
//! pass/fail bit.
//!
//! The shape here is the wire contract both ends honour: the producer
//! (LocalPilot's headless runner) emits one [`Scorecard`] per task run as JSON,
//! and the consumer (an external benchmark runner) deserializes and ranks it.
//! Only the deterministic, trace-independent pieces live here:
//! [`QualityBlock::from_signals`] assembles the `quality` block from a captured
//! diff, check outcomes, and the diff-derived helpers below; the `results` and
//! `speed` blocks are graded/measured by the runner, so they are plain data the
//! caller fills in; the `process` block is derived by the producer from its own
//! session trace.

use serde::{Deserialize, Serialize};

use crate::check::CheckOutcome;
use crate::discipline::DisciplineMetrics;
use crate::judge::JudgeBlock;

/// The scorecard contract version. Bump on any breaking shape change (a removed
/// or renamed field); additive fields keep the version.
pub const SCORECARD_SCHEMA: u32 = 1;

/// One task run, graded on three layers plus a reported speed guardrail.
///
/// `speed` is a guardrail, never the headline metric — correctness gates, then
/// quality and process rank (the composite lives in [`crate::ablation`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scorecard {
    /// Contract version ([`SCORECARD_SCHEMA`]).
    pub schema: u32,
    /// Task identifier (corpus-local, e.g. a first-party task name or an
    /// external instance id).
    pub task: String,
    /// The harness arm this run used (e.g. `full`, `baseline`, `no-retrieval`),
    /// so an ablation can group runs by configuration.
    pub arm: String,
    /// The model id the run used, or `fake` for the offline deterministic path.
    pub model: String,
    /// Did the change resolve the task, and is it regression-safe?
    pub results: ResultsBlock,
    /// Static code-quality signals on the produced diff.
    pub quality: QualityBlock,
    /// How the agent worked: tool economy, discipline, retrieval, recovery.
    pub process: ProcessBlock,
    /// Reported speed/cost guardrail. Never the headline score.
    pub speed: SpeedBlock,
    /// Optional LLM-as-judge scores for the quality dimensions static signals
    /// cannot see. `null` when no judge ran (the offline static path).
    pub judge: Option<JudgeBlock>,
}

impl Scorecard {
    /// Serialize the scorecard to its canonical JSON string (the wire contract).
    ///
    /// # Errors
    /// Returns the `serde_json` error if serialization fails (it does not for
    /// this all-owned, finite type, but the contract is fallible by signature).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// The results layer: did the work get done, safely?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultsBlock {
    /// The task's own test(s) passed after the change.
    pub passed: bool,
    /// No previously-passing test regressed (the `PASS_TO_PASS`/regression set
    /// still passes). Vacuously `true` when a corpus carries no regression set.
    pub regression_safe: bool,
    /// Fractional credit in `0.0..=1.0` for a partially-solved task (e.g. the
    /// fraction of target tests flipped). `1.0` on a full pass, `0.0` on no
    /// progress.
    pub partial_credit: f64,
    /// Target tests the task graded against.
    pub tests_total: u32,
    /// Of those, how many passed after the change.
    pub tests_passed: u32,
}

/// The code-quality layer: static signals on the produced diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBlock {
    /// Added lines in the produced diff.
    pub diff_added: u32,
    /// Removed lines in the produced diff.
    pub diff_removed: u32,
    /// Files the diff touches.
    pub diff_files: u32,
    /// Candidate churn relative to the gold patch (`(added+removed) /
    /// gold(added+removed)`); `null` when there is no gold patch or the gold
    /// patch is empty. A ratio near `1.0` is minimal; large is bloated.
    pub vs_gold_ratio: Option<f64>,
    /// `cargo fmt --check` (or the stack's formatter check) passed.
    pub format_clean: bool,
    /// The linter (clippy / equivalent) reported no findings.
    pub lint_clean: bool,
    /// The type/compile check passed.
    pub typecheck_clean: bool,
    /// Added cyclomatic-ish complexity, a best-effort diff-derived proxy;
    /// `null` when not computed.
    pub complexity_delta: Option<i64>,
    /// The diff added at least one test.
    pub tests_added: bool,
}

impl QualityBlock {
    /// Assemble the quality block from a captured diff, an optional gold diff,
    /// the gate's check outcomes, and the diff-derived complexity/tests signals.
    ///
    /// The `format_clean` / `lint_clean` / `typecheck_clean` flags are read from
    /// `checks` by conventional name (`fmt`/`format`, `clippy`/`lint`,
    /// `check`/`typecheck`/`build`); an absent check is treated as clean
    /// (it did not report a finding), so a corpus that runs only a subset of the
    /// gate still produces a well-formed block.
    #[must_use]
    pub fn from_signals(
        diff: &DiffStat,
        gold: Option<&DiffStat>,
        checks: &[CheckOutcome],
        complexity_delta: Option<i64>,
        tests_added: bool,
    ) -> Self {
        let candidate_churn = diff.added + diff.removed;
        let vs_gold_ratio = gold.and_then(|g| {
            let gold_churn = g.added + g.removed;
            if gold_churn == 0 {
                None
            } else {
                Some(f64::from(candidate_churn) / f64::from(gold_churn))
            }
        });
        Self {
            diff_added: diff.added,
            diff_removed: diff.removed,
            diff_files: diff.files,
            vs_gold_ratio,
            format_clean: check_clean(checks, &["fmt", "format"]),
            lint_clean: check_clean(checks, &["clippy", "lint"]),
            typecheck_clean: check_clean(checks, &["check", "typecheck", "build"]),
            complexity_delta,
            tests_added,
        }
    }
}

/// The process layer: how the agent worked, derived by the producer from its
/// session trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessBlock {
    /// Total tool calls the run made.
    pub tool_calls: u32,
    /// Calls that repeated an earlier identical `(tool, arguments)` call.
    pub redundant_calls: u32,
    /// An observation call (read/search/run/status) preceded the first mutating
    /// call — the agent looked before it edited. Vacuously `true` with no edit.
    pub reproduce_before_fix: bool,
    /// A test-like call appears in the trace before the final claim.
    pub test_before_done: bool,
    /// Retrieval contributed to the run (memories surfaced, or a retrieval tool
    /// was called).
    pub retrieval_used: bool,
    /// How many memories/knowledge chunks were surfaced and used across the run.
    pub retrieval_count: u32,
    /// The recorded turn stop label (e.g. `Done`, `BudgetExceeded`,
    /// `NoProgress`), or `unknown` when none was recorded.
    pub exit_reason: String,
    /// After a failed call, a later grounded success followed — the agent
    /// recovered rather than giving up or claiming on the failure.
    pub recovered_after_failure: bool,
    /// The per-capability discipline rates, when a rollup is attached (a
    /// producer that computes none leaves this `null`).
    pub discipline: Option<DisciplineMetrics>,
}

/// The reported speed/cost guardrail. Never the headline metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedBlock {
    /// Wall-clock duration of the run, in milliseconds (runner-measured).
    pub wall_ms: u64,
    /// Input tokens reported across the run.
    pub input_tokens: u64,
    /// Output tokens reported across the run.
    pub output_tokens: u64,
}

/// Line/file counts of a unified diff, the diff-size + blast-radius signal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffStat {
    /// Added content lines (`+`, excluding the `+++` file header).
    pub added: u32,
    /// Removed content lines (`-`, excluding the `---` file header).
    pub removed: u32,
    /// Distinct files the diff touches.
    pub files: u32,
}

impl DiffStat {
    /// Parse the line/file counts out of a unified diff (`git diff` output).
    #[must_use]
    pub fn from_unified(diff: &str) -> Self {
        let mut added = 0u32;
        let mut removed = 0u32;
        let mut files = 0u32;
        for line in diff.lines() {
            if line.starts_with("diff --git ") {
                files += 1;
            } else if line.starts_with("+++") || line.starts_with("---") {
                // File headers, not content.
            } else if line.starts_with('+') {
                added += 1;
            } else if line.starts_with('-') {
                removed += 1;
            }
        }
        Self {
            added,
            removed,
            files,
        }
    }
}

/// Whether the diff adds a test: an added line declares a test, or touches a
/// conventional test path. A best-effort, language-agnostic proxy. Markers that
/// are short common substrings (`it(`, `describe(`, `test(`) are matched only at
/// the start of a trimmed added line, so a token like `digit()` does not falsely
/// register as a JavaScript test.
#[must_use]
pub fn tests_added_in_diff(diff: &str) -> bool {
    diff.lines().any(|line| {
        let Some(added) = line.strip_prefix('+') else {
            return false;
        };
        if added.starts_with("++") {
            return false; // the `+++` file header
        }
        let lower = added.to_lowercase();
        let trimmed = lower.trim_start();
        lower.contains("#[test]")
            || lower.contains("@test")
            || trimmed.starts_with("def test_")
            || trimmed.starts_with("describe(")
            || trimmed.starts_with("it(")
            || trimmed.starts_with("test(")
            || (lower.contains("fn ") && lower.contains("test"))
    }) || diff.lines().any(|line| {
        line.starts_with("diff --git")
            && (line.contains("/tests/") || line.contains("test_") || line.contains(".test."))
    })
}

/// A best-effort added-complexity proxy: net branch/decision keywords introduced
/// by the diff (added minus removed). Not a real cyclomatic count, but a
/// deterministic, language-agnostic signal that tracks branchiness.
#[must_use]
pub fn complexity_delta_in_diff(diff: &str) -> i64 {
    const KEYWORDS: &[&str] = &[
        " if ", " for ", " while ", " match ", " case ", "&&", "||", "?", " elif ", " when ",
        " catch", " switch",
    ];
    let mut delta: i64 = 0;
    for line in diff.lines() {
        let (sign, body) = if let Some(body) = line.strip_prefix('+') {
            if body.starts_with('+') {
                continue;
            }
            (1i64, body)
        } else if let Some(body) = line.strip_prefix('-') {
            if body.starts_with('-') {
                continue;
            }
            (-1i64, body)
        } else {
            continue;
        };
        let padded = format!(" {body} ");
        let hits: i64 = KEYWORDS
            .iter()
            .map(|kw| padded.matches(kw).count() as i64)
            .sum();
        delta += sign * hits;
    }
    delta
}

/// Whether a gate check of one of the given conventional names reported a clean
/// pass. An absent check is treated as clean (it raised no finding).
fn check_clean(checks: &[CheckOutcome], names: &[&str]) -> bool {
    checks
        .iter()
        .filter(|c| names.iter().any(|n| c.name.contains(n)))
        .all(CheckOutcome::passed)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::check::CheckStatus;

    fn sample_scorecard() -> Scorecard {
        Scorecard {
            schema: SCORECARD_SCHEMA,
            task: "fix-off-by-one".to_string(),
            arm: "full".to_string(),
            model: "fake".to_string(),
            results: ResultsBlock {
                passed: true,
                regression_safe: true,
                partial_credit: 1.0,
                tests_total: 3,
                tests_passed: 3,
            },
            quality: QualityBlock {
                diff_added: 4,
                diff_removed: 2,
                diff_files: 1,
                vs_gold_ratio: Some(1.5),
                format_clean: true,
                lint_clean: true,
                typecheck_clean: true,
                complexity_delta: Some(0),
                tests_added: false,
            },
            process: ProcessBlock {
                tool_calls: 3,
                redundant_calls: 0,
                reproduce_before_fix: true,
                test_before_done: true,
                retrieval_used: true,
                retrieval_count: 2,
                exit_reason: "Done".to_string(),
                recovered_after_failure: false,
                discipline: None,
            },
            speed: SpeedBlock {
                wall_ms: 1200,
                input_tokens: 500,
                output_tokens: 200,
            },
            judge: None,
        }
    }

    #[test]
    fn scorecard_contract_round_trips_through_json() {
        let card = sample_scorecard();
        let json = card.to_json().expect("serialize");
        let back: Scorecard = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(card, back);
    }

    #[test]
    fn scorecard_json_carries_all_three_layers_and_speed() {
        let json = sample_scorecard().to_json().expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse");
        for key in [
            "schema", "task", "arm", "model", "results", "quality", "process", "speed", "judge",
        ] {
            assert!(value.get(key).is_some(), "scorecard must carry `{key}`");
        }
        assert!(value["judge"].is_null(), "no judge ran for this sample");
        // Nullable contract fields serialize as present (null), not omitted.
        let mut minimal = sample_scorecard();
        minimal.quality.vs_gold_ratio = None;
        minimal.quality.complexity_delta = None;
        minimal.process.discipline = None;
        let value: serde_json::Value =
            serde_json::from_str(&minimal.to_json().expect("serialize")).expect("parse");
        assert!(value["quality"]["vs_gold_ratio"].is_null());
        assert!(value["quality"]["complexity_delta"].is_null());
        assert!(value["process"]["discipline"].is_null());
    }

    #[test]
    fn diff_stat_parses_unified_diff() {
        let diff = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,2 +1,3 @@
 pub fn two() -> i32 {
-    1
+    2
+    // fixed
 }
";
        let stat = DiffStat::from_unified(diff);
        assert_eq!(stat.files, 1);
        assert_eq!(stat.added, 2);
        assert_eq!(stat.removed, 1);
    }

    #[test]
    fn quality_block_computes_vs_gold_ratio_and_reads_checks() {
        let diff = DiffStat {
            added: 6,
            removed: 0,
            files: 1,
        };
        let gold = DiffStat {
            added: 3,
            removed: 0,
            files: 1,
        };
        let checks = vec![
            CheckOutcome {
                name: "fmt".into(),
                status: CheckStatus::Passed,
                detail: String::new(),
                fixed: false,
                severity: None,
            },
            CheckOutcome {
                name: "clippy".into(),
                status: CheckStatus::Failed,
                detail: "1 warning".into(),
                fixed: false,
                severity: None,
            },
        ];
        let quality = QualityBlock::from_signals(&diff, Some(&gold), &checks, Some(2), true);
        assert_eq!(quality.vs_gold_ratio, Some(2.0));
        assert!(quality.format_clean);
        assert!(!quality.lint_clean, "the failing clippy check is a finding");
        assert!(
            quality.typecheck_clean,
            "an absent typecheck check counts as clean"
        );
        assert!(quality.tests_added);
    }

    #[test]
    fn tests_added_and_complexity_proxies_read_the_diff() {
        let with_test = "\
diff --git a/tests/feature.rs b/tests/feature.rs
+++ b/tests/feature.rs
+#[test]
+fn it_works() {
+    if cond { assert!(true) }
+}
";
        assert!(tests_added_in_diff(with_test));
        assert!(
            complexity_delta_in_diff(with_test) >= 1,
            "the added `if` raises the complexity proxy"
        );

        let no_test = "\
diff --git a/src/lib.rs b/src/lib.rs
+++ b/src/lib.rs
+pub const N: u32 = 3;
";
        assert!(!tests_added_in_diff(no_test));
        assert_eq!(complexity_delta_in_diff(no_test), 0);
    }
}
