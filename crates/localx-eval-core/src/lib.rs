//! Host-neutral evaluation primitives shared by LocalPilot and LocalBench.
//!
//! The shared home for the eval contract LocalBench reuses without depending on
//! LocalPilot's agent loop:
//!
//! - [`scorecard`] — the three-layer capability-scorecard wire contract
//!   (results / quality / process, plus the speed guardrail) and its
//!   deterministic diff-derived signals.
//! - [`discipline`] — the per-capability tool-discipline rates and the
//!   provisional composite Tool Discipline Score.
//! - [`judge`] — the blinded, cache-deterministic LLM-as-judge (rubric,
//!   prompts, parsing, ranking self-test, kappa calibration). The live model
//!   call is the host's to supply.
//! - [`ablation`] — arm matrix, per-feature attribution, and the
//!   correctness-gated composite ranking.
//! - [`check`] — gate-mediated check execution ([`check::CheckRunner`]) with
//!   fix-and-re-run orchestration; the host injects its command policy through
//!   [`check::CommandGate`].
//! - [`verify`] — stack detection for the verify-before-done command.
//! - [`grade`] — grade fidelity (exit 0 AND tests_run > 0; Rust sums every
//!   `test result:` line), fail-closed per language.
//! - [`gate`] — provenance gating (offline artifacts can't satisfy a live
//!   gate) and safety-as-a-gate / capability-as-a-delta.

#![forbid(unsafe_code)]

pub mod ablation;
pub mod check;
pub mod discipline;
pub mod gate;
pub mod grade;
pub mod judge;
pub mod scorecard;
pub mod verify;

pub use ablation::{
    ablation_matrix, attribute, composite_score, feature_signal, mean_std, rank, signal_value,
    AblationArm, AttributionRow, CompositeOutcome, FeatureToggles,
};
pub use check::{
    AllowAll, CheckCommand, CheckOutcome, CheckRunner, CheckSeverity, CheckSpec, CheckStatus,
    CommandGate,
};
pub use discipline::DisciplineMetrics;
pub use gate::{parse_provenance, GateCard, Provenance, Safety, Verdict};
pub use grade::{count_tests, count_tests_generic, grade, GradeOutcome, Lang};
pub use judge::{
    blind, cohens_kappa, judge_prompt, parse_judge_block, parse_preference, preference_prompt,
    ranking_verdict, resolve_preference, BlindedPair, Judge, JudgeBlock, JudgeCache, JudgeError,
    JudgeInput, Preferred, RankingFixture, RankingTrust, RANKING_FIXTURES, RUBRIC,
};
pub use scorecard::{
    complexity_delta_in_diff, tests_added_in_diff, DiffStat, ProcessBlock, QualityBlock,
    ResultsBlock, Scorecard, SpeedBlock, SCORECARD_SCHEMA,
};
pub use verify::{detect_verify_command, resolve_verify_command, VERIFY_CHECK_NAME};
