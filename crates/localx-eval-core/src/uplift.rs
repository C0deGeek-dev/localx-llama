//! The identity a lesson-off/on uplift run is bound to, shared by the tool that
//! produces a receipt and the host that requested it.
//!
//! An uplift result is only evidence about the run that was asked for. A
//! receipt naming just a task set's *name* could be any run that happened to
//! share it, so the producer attests content instead: the task set by digest,
//! each arm by its configuration and injection inputs, and the whole request by
//! an opaque `binding` the requester chose. The requester recomputes what it
//! expected and compares with [`UpliftIdentity::mismatches`]; anything else is a
//! different run.
//!
//! The types carry no statistics and no grading: those stay with the producer.
//! `binding` is deliberately opaque here — what a host binds a run to (a
//! candidate, an assignment, a source revision) is the host's own lineage.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema tag of one arm's result file.
pub const UPLIFT_ARM_SCHEMA: &str = "localbench-uplift-arm-v1";
/// Schema tag of the combined receipt that carries an [`UpliftIdentity`].
pub const UPLIFT_RECEIPT_SCHEMA: &str = "localbench-uplift-v2";

/// `sha256:<hex>` over a value's canonical JSON. Struct fields serialize in
/// declaration order, so the same value always gives the same digest.
///
/// A value that cannot be serialized digests as its error text, which matches
/// nothing a well-formed value produces.
#[must_use]
pub fn content_digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|error| error.to_string().into_bytes());
    text_digest(&bytes)
}

/// `sha256:<hex>` over raw bytes.
#[must_use]
pub fn text_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(7 + digest.len() * 2);
    hex.push_str("sha256:");
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// A task set, by content rather than by name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSetIdentity {
    /// The task set's own name. Informational: never matched on.
    pub name: String,
    /// Digest of the task set's tasks and expectations.
    pub digest: String,
    pub task_count: usize,
}

/// How the lesson was meant to reach the lesson arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InjectionMode {
    /// Placed in context directly.
    Forced,
    /// Expected to arrive through normal retrieval, so a null result is
    /// confounded with retrieval quality.
    Retrieved,
}

/// What an arm was meant to inject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectionIdentity {
    pub mode: InjectionMode,
    /// The memory ids the arm must show it used, sorted. Empty for a baseline,
    /// which must show none.
    pub intended: Vec<String>,
    /// Digest of the seed pack staged for the arm; `None` for a baseline.
    pub seed_pack_digest: Option<String>,
}

impl InjectionIdentity {
    /// A baseline arm's injection: nothing intended, nothing seeded.
    #[must_use]
    pub fn none() -> Self {
        Self {
            mode: InjectionMode::Retrieved,
            intended: Vec::new(),
            seed_pack_digest: None,
        }
    }

    /// A lesson arm's injection, with the intended ids sorted and de-duplicated.
    #[must_use]
    pub fn lessons(
        mode: InjectionMode,
        mut intended: Vec<String>,
        seed_pack_digest: String,
    ) -> Self {
        intended.sort();
        intended.dedup();
        Self {
            mode,
            intended,
            seed_pack_digest: Some(seed_pack_digest),
        }
    }
}

/// One arm, by everything that configures it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmIdentity {
    pub arm: String,
    pub is_lesson_arm: bool,
    /// Digest of the memory configuration the arm ran with.
    pub config_digest: String,
    pub model: String,
    pub trials: u32,
    /// Per-turn timeout, in seconds.
    pub timeout_secs: u64,
    pub injection: InjectionIdentity,
}

/// What one arm's result file attests: the request it belongs to, the task set
/// it ran, and the arm itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmRunIdentity {
    pub binding: String,
    pub task_set: TaskSetIdentity,
    pub arm: ArmIdentity,
}

/// Everything a combined uplift receipt is bound to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpliftIdentity {
    /// The requester's own binding for this run, carried through unchanged.
    pub binding: String,
    pub task_set: TaskSetIdentity,
    pub baseline: ArmIdentity,
    pub lessons: ArmIdentity,
}

/// Why two arm files cannot be combined, or a receipt is not the requested run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdentityMismatch {
    /// The two sides name different requests.
    #[error("binding differs: expected {expected}, got {actual}")]
    Binding { expected: String, actual: String },
    /// The task set's content differs.
    #[error("task set differs: expected {expected}, got {actual}")]
    TaskSet { expected: String, actual: String },
    /// A field of one arm differs.
    #[error("{arm} arm differs in {field}: expected {expected}, got {actual}")]
    Arm {
        arm: String,
        field: &'static str,
        expected: String,
        actual: String,
    },
    /// The arms were not one baseline and one lesson arm.
    #[error("the pair is not one baseline arm and one lesson arm")]
    NotAPair,
    /// The arms disagree on what must be equal between them.
    #[error("the arms differ in {0}, which must be the same for both")]
    ArmsDisagree(&'static str),
}

impl UpliftIdentity {
    /// Join two arm identities into the identity of their pair. The arms must
    /// be one baseline and one lesson arm of the same request, over the same
    /// task set, model, trial count and timeout — the lesson is the only thing
    /// allowed to differ.
    ///
    /// # Errors
    /// Every [`IdentityMismatch`] that makes the two files not a pair.
    pub fn pair(
        first: &ArmRunIdentity,
        second: &ArmRunIdentity,
    ) -> Result<Self, Vec<IdentityMismatch>> {
        let (baseline, lessons) = match (first.arm.is_lesson_arm, second.arm.is_lesson_arm) {
            (false, true) => (first, second),
            (true, false) => (second, first),
            _ => return Err(vec![IdentityMismatch::NotAPair]),
        };
        let mut problems = Vec::new();
        if baseline.binding != lessons.binding {
            problems.push(IdentityMismatch::ArmsDisagree("binding"));
        }
        if baseline.task_set != lessons.task_set {
            problems.push(IdentityMismatch::ArmsDisagree("task set"));
        }
        if baseline.arm.model != lessons.arm.model {
            problems.push(IdentityMismatch::ArmsDisagree("model"));
        }
        if baseline.arm.trials != lessons.arm.trials {
            problems.push(IdentityMismatch::ArmsDisagree("trials"));
        }
        if baseline.arm.timeout_secs != lessons.arm.timeout_secs {
            problems.push(IdentityMismatch::ArmsDisagree("timeout"));
        }
        if !problems.is_empty() {
            return Err(problems);
        }
        Ok(Self {
            binding: baseline.binding.clone(),
            task_set: baseline.task_set.clone(),
            baseline: baseline.arm.clone(),
            lessons: lessons.arm.clone(),
        })
    }

    /// The run's deterministic id: a digest of everything it is bound to.
    #[must_use]
    pub fn run_id(&self) -> String {
        content_digest(self)
    }

    /// How this identity differs from the one the requester `expected`. Empty
    /// means it is the requested run. Task-set names are never compared.
    #[must_use]
    pub fn mismatches(&self, expected: &Self) -> Vec<IdentityMismatch> {
        let mut problems = Vec::new();
        if self.binding != expected.binding {
            problems.push(IdentityMismatch::Binding {
                expected: expected.binding.clone(),
                actual: self.binding.clone(),
            });
        }
        if self.task_set.digest != expected.task_set.digest
            || self.task_set.task_count != expected.task_set.task_count
        {
            problems.push(IdentityMismatch::TaskSet {
                expected: expected.task_set.digest.clone(),
                actual: self.task_set.digest.clone(),
            });
        }
        for (actual, wanted) in [
            (&self.baseline, &expected.baseline),
            (&self.lessons, &expected.lessons),
        ] {
            let mut differ = |field: &'static str, wanted: String, got: String| {
                if wanted != got {
                    problems.push(IdentityMismatch::Arm {
                        arm: actual.arm.clone(),
                        field,
                        expected: wanted,
                        actual: got,
                    });
                }
            };
            differ(
                "role",
                wanted.is_lesson_arm.to_string(),
                actual.is_lesson_arm.to_string(),
            );
            differ(
                "configuration",
                wanted.config_digest.clone(),
                actual.config_digest.clone(),
            );
            differ("model", wanted.model.clone(), actual.model.clone());
            differ(
                "trials",
                wanted.trials.to_string(),
                actual.trials.to_string(),
            );
            differ(
                "timeout",
                wanted.timeout_secs.to_string(),
                actual.timeout_secs.to_string(),
            );
            differ(
                "injection",
                content_digest(&wanted.injection),
                content_digest(&actual.injection),
            );
        }
        problems
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn task_set(digest: &str) -> TaskSetIdentity {
        TaskSetIdentity {
            name: "headroom".to_string(),
            digest: digest.to_string(),
            task_count: 3,
        }
    }

    fn arm(lesson: bool) -> ArmIdentity {
        ArmIdentity {
            arm: if lesson { "lessons" } else { "baseline" }.to_string(),
            is_lesson_arm: lesson,
            config_digest: format!("sha256:config-{lesson}"),
            model: "m".to_string(),
            trials: 3,
            timeout_secs: 600,
            injection: if lesson {
                InjectionIdentity::lessons(
                    InjectionMode::Retrieved,
                    vec![
                        "mem-b".to_string(),
                        "mem-a".to_string(),
                        "mem-a".to_string(),
                    ],
                    "sha256:pack".to_string(),
                )
            } else {
                InjectionIdentity::none()
            },
        }
    }

    fn run(lesson: bool) -> ArmRunIdentity {
        ArmRunIdentity {
            binding: "bind-1".to_string(),
            task_set: task_set("sha256:tasks"),
            arm: arm(lesson),
        }
    }

    #[test]
    fn a_digest_is_stable_and_content_bound() {
        let one = content_digest(&arm(true));
        assert_eq!(one, content_digest(&arm(true)));
        assert!(one.starts_with("sha256:") && one.len() == 71);
        assert_ne!(one, content_digest(&arm(false)));
    }

    #[test]
    fn intended_ids_are_sorted_and_deduplicated() {
        assert_eq!(arm(true).injection.intended, ["mem-a", "mem-b"]);
    }

    #[test]
    fn two_arms_pair_in_either_order_into_one_run() {
        let forward = UpliftIdentity::pair(&run(false), &run(true)).unwrap();
        let backward = UpliftIdentity::pair(&run(true), &run(false)).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(forward.run_id(), backward.run_id());
        assert!(!forward.baseline.is_lesson_arm && forward.lessons.is_lesson_arm);
    }

    #[test]
    fn two_of_the_same_arm_are_not_a_pair() {
        assert_eq!(
            UpliftIdentity::pair(&run(true), &run(true)).unwrap_err(),
            vec![IdentityMismatch::NotAPair]
        );
        assert_eq!(
            UpliftIdentity::pair(&run(false), &run(false)).unwrap_err(),
            vec![IdentityMismatch::NotAPair]
        );
    }

    #[test]
    fn arms_of_different_requests_or_settings_do_not_combine() {
        let mut other = run(true);
        other.binding = "bind-2".to_string();
        other.task_set = task_set("sha256:other");
        other.arm.model = "n".to_string();
        other.arm.trials = 5;
        other.arm.timeout_secs = 60;
        assert_eq!(
            UpliftIdentity::pair(&run(false), &other).unwrap_err(),
            vec![
                IdentityMismatch::ArmsDisagree("binding"),
                IdentityMismatch::ArmsDisagree("task set"),
                IdentityMismatch::ArmsDisagree("model"),
                IdentityMismatch::ArmsDisagree("trials"),
                IdentityMismatch::ArmsDisagree("timeout"),
            ]
        );
    }

    #[test]
    fn a_receipt_matches_only_the_run_that_was_requested() {
        let expected = UpliftIdentity::pair(&run(false), &run(true)).unwrap();
        assert!(expected.mismatches(&expected).is_empty());

        // The same name over different tasks is a different run.
        let mut renamed = expected.clone();
        renamed.task_set.name = "another name".to_string();
        assert!(
            renamed.mismatches(&expected).is_empty(),
            "names are not matched"
        );
        let mut other_tasks = expected.clone();
        other_tasks.task_set.digest = "sha256:other".to_string();
        assert!(matches!(
            other_tasks.mismatches(&expected)[..],
            [IdentityMismatch::TaskSet { .. }]
        ));

        let mut other_request = expected.clone();
        other_request.binding = "bind-9".to_string();
        assert!(matches!(
            other_request.mismatches(&expected)[..],
            [IdentityMismatch::Binding { .. }]
        ));

        // A different lesson was injected.
        let mut other_lesson = expected.clone();
        other_lesson.lessons.injection.intended = vec!["mem-z".to_string()];
        let problems = other_lesson.mismatches(&expected);
        assert!(
            matches!(&problems[..], [IdentityMismatch::Arm { arm, field: "injection", .. }] if arm == "lessons"),
            "{problems:?}"
        );
        assert_ne!(other_lesson.run_id(), expected.run_id());

        // The baseline ran with another configuration.
        let mut misstaged = expected.clone();
        misstaged.baseline.config_digest = "sha256:learning-on".to_string();
        assert!(matches!(
            &misstaged.mismatches(&expected)[..],
            [IdentityMismatch::Arm {
                field: "configuration",
                ..
            }]
        ));
    }

    #[test]
    fn the_identity_round_trips_through_json() {
        let identity = UpliftIdentity::pair(&run(false), &run(true)).unwrap();
        let json = serde_json::to_string(&identity).unwrap();
        assert!(json.contains("\"mode\":\"retrieved\""));
        assert_eq!(
            serde_json::from_str::<UpliftIdentity>(&json).unwrap(),
            identity
        );
    }
}
