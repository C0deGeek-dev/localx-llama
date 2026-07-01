//! Host-neutral evaluation primitives shared by LocalPilot and LocalBench.
//!
//! The shared home (decision O2) for the eval contract LocalBench reuses without
//! depending on LocalPilot's agent loop:
//!
//! - [`grade`] — grade fidelity (exit 0 AND tests_run > 0; Rust sums every
//!   `test result:` line), fail-closed per language.
//! - [`scorecard`] — provenance gating (offline artifacts can't satisfy a live
//!   gate) and safety-as-a-gate / capability-as-a-delta.
//!
//! The full scorecard/judge/ablation move out of `localpilot-harness` (the
//! coordinated LocalPilot refactor) builds on this foundation.

#![forbid(unsafe_code)]

pub mod grade;
pub mod scorecard;

pub use grade::{count_tests, grade, GradeOutcome, Lang};
pub use scorecard::{parse_provenance, Provenance, Safety, Scorecard, Verdict};
