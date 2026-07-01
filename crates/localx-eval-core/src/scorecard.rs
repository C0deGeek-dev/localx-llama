//! Scorecard provenance + safety gating.
//!
//! Two benchmark-integrity invariants:
//! 1. **Provenance.** A scorecard is tagged `Live` or `Offline`. An offline
//!    artifact must be *structurally* unable to satisfy a live gate — the fix for
//!    the TdsScorer masquerade where an offline FakeProvider scorecard was parsed
//!    as a live result because both shared a regex.
//! 2. **Safety is a gate; capability is a delta.** A safety breach reports FAILED
//!    regardless of the composite score; the composite is only ever a delta
//!    between arms, never a trusted absolute.

use serde::{Deserialize, Serialize};

/// Where a scorecard came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provenance {
    /// Produced by a real live model run.
    Live,
    /// Produced offline (fixtures/FakeProvider) — cannot satisfy a live gate.
    Offline,
}

/// Determine provenance from a scorecard line: a `live ` prefix marks a live run.
///
/// An offline artifact has no such prefix, so it can never be mistaken for live.
pub fn parse_provenance(line: &str) -> Provenance {
    if line.trim_start().to_ascii_lowercase().starts_with("live ") {
        Provenance::Live
    } else {
        Provenance::Offline
    }
}

/// Safety outcome — a gate, not a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Safety {
    /// No safety breach.
    Passed,
    /// A safety breach occurred.
    Breached,
}

/// Overall pass/fail verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Passed.
    Passed,
    /// Failed.
    Failed,
}

/// A minimal scorecard with the gating fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scorecard {
    /// Where this scorecard came from.
    pub provenance: Provenance,
    /// Safety gate outcome.
    pub safety: Safety,
    /// Capability composite — a **delta between arms**, never a trusted absolute.
    pub composite_delta: f64,
}

impl Scorecard {
    /// Whether this scorecard may satisfy a gate that requires a live run.
    ///
    /// Offline artifacts are structurally rejected.
    pub fn satisfies_live_gate(&self) -> bool {
        matches!(self.provenance, Provenance::Live)
    }

    /// Overall verdict. A safety breach is FAILED regardless of the composite.
    pub fn verdict(&self) -> Verdict {
        match self.safety {
            Safety::Breached => Verdict::Failed,
            Safety::Passed => Verdict::Passed,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn provenance_requires_live_prefix() {
        assert_eq!(parse_provenance("live scorecard: {...}"), Provenance::Live);
        assert_eq!(
            parse_provenance("  LIVE scorecard: {...}"),
            Provenance::Live
        );
        // no prefix -> offline; the offline FakeProvider line cannot masquerade.
        assert_eq!(parse_provenance("scorecard: {...}"), Provenance::Offline);
        assert_eq!(
            parse_provenance("offline scorecard: {...}"),
            Provenance::Offline
        );
    }

    #[test]
    fn offline_cannot_satisfy_a_live_gate() {
        let offline = Scorecard {
            provenance: Provenance::Offline,
            safety: Safety::Passed,
            composite_delta: 42.0,
        };
        assert!(!offline.satisfies_live_gate());
        let live = Scorecard {
            provenance: Provenance::Live,
            ..offline.clone()
        };
        assert!(live.satisfies_live_gate());
    }

    #[test]
    fn safety_breach_fails_regardless_of_score() {
        let breached = Scorecard {
            provenance: Provenance::Live,
            safety: Safety::Breached,
            composite_delta: 999.0, // high score is irrelevant
        };
        assert_eq!(breached.verdict(), Verdict::Failed);
        let clean = Scorecard {
            safety: Safety::Passed,
            ..breached
        };
        assert_eq!(clean.verdict(), Verdict::Passed);
    }
}
