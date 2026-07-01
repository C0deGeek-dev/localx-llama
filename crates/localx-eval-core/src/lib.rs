//! Host-neutral evaluation primitives shared by LocalPilot and LocalBench.
//!
//! Extracted (subject 03, decision O2) from `localpilot-harness` so a benchmark
//! tool can reuse the scorecard/judge/ablation contract and the stack-detected
//! build/test grader without depending on LocalPilot's agent loop.
//!
//! Invariants carried here are load-bearing for benchmark integrity: scorecard
//! provenance (an offline artifact must not satisfy a live gate), grade fidelity
//! (exit 0 AND tests_run > 0), and arm-as-serialized-config.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        assert_eq!(2 + 2, 4);
    }
}
