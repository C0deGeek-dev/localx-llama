//! Host-neutral check execution: run a named build/test/lint command, capture a
//! bounded outcome, and orchestrate an optional fix-and-re-run — with the *policy*
//! (may this command run at all? how is output sanitized?) injected by the host
//! through [`CommandGate`].
//!
//! LocalPilot's quality gate implements the gate with its permission engine, so
//! every check still routes through the same decision path as any other command;
//! a benchmark grader implements it with its own (typically permissive, sandboxed)
//! policy. The execution and fix-orchestration semantics are identical for both —
//! that is the point of sharing them.

use std::future::Future;
use std::path::Path;
use std::time::Duration;

/// Cap on captured check output before truncation.
const MAX_OUTPUT_BYTES: usize = 16 * 1024;

/// Default per-check timeout. Full checks (test suites) can be slow.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// A program plus its argument list — no shell interpretation anywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckCommand {
    /// The program to run.
    pub program: String,
    /// Arguments passed as a list, not a shell string.
    pub args: Vec<String>,
}

impl CheckCommand {
    /// A command from a program and its arguments.
    #[must_use]
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    /// Split a single command line on whitespace into a program and arguments
    /// (no shell interpretation). Returns `None` for a blank line.
    #[must_use]
    pub fn from_command_line(command: &str) -> Option<Self> {
        let mut parts = command.split_whitespace();
        let program = parts.next()?.to_string();
        let args = parts.map(str::to_string).collect();
        Some(Self { program, args })
    }
}

/// The severity a failing check reports with, applied by the host's gating
/// layer. `None` on a [`CheckOutcome`] means the host decides (its default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckSeverity {
    /// Findings are ignored.
    Off,
    /// Findings warn but do not block.
    Warn,
    /// Findings block.
    Block,
}

/// One runnable check: the command that answers a named question ("does it
/// format/lint/build/test?"), an optional already-authorized fixer to run on
/// failure, and the severity its findings carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckSpec {
    /// Stable check name (`fmt`, `clippy`, `test`, `verify`, ...).
    pub name: String,
    /// The check command.
    pub command: CheckCommand,
    /// Fixer run when the check fails, after which the check re-runs once.
    /// `None` means findings are reported as-is. The caller's policy decides
    /// whether a configured fixer is offered here at all.
    pub fixer: Option<CheckCommand>,
    /// Per-check severity carried onto the outcome for the host's gating layer.
    pub severity: Option<CheckSeverity>,
}

/// What happened when a check ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// The check command exited successfully.
    Passed,
    /// The check ran and reported findings (non-zero exit).
    Failed,
    /// The gate refused the command.
    Denied,
    /// The command could not be started or timed out.
    Errored,
}

/// The result of running one check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckOutcome {
    /// The check's name.
    pub name: String,
    /// What happened.
    pub status: CheckStatus,
    /// Bounded, gate-sanitized detail (exit code + captured output). Empty on a
    /// clean pass.
    pub detail: String,
    /// Whether a fixer ran and the check was re-run.
    pub fixed: bool,
    /// The check's configured severity, carried so the host's gating layer can
    /// apply a per-check override.
    pub severity: Option<CheckSeverity>,
}

impl CheckOutcome {
    /// Whether the check passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.status == CheckStatus::Passed
    }
}

/// The host's command policy: whether a command may run, and how captured
/// output is sanitized before it becomes finding detail. There is no path
/// around [`allow`](CommandGate::allow) — the runner asks before every spawn,
/// including fixers and re-runs.
pub trait CommandGate: Sync {
    /// Decide whether the command may run. The future carries no `Send` bound so
    /// a host whose approval flow is single-threaded (an interactive prompt) can
    /// implement the gate; drive the runner on a current-thread or local context
    /// when the gate needs it.
    fn allow(&self, command: &CheckCommand) -> impl Future<Output = bool>;

    /// Sanitize captured output before it becomes finding detail (e.g. secret
    /// redaction). The default keeps it as-is.
    fn sanitize(&self, text: String) -> String {
        text
    }
}

/// A gate that allows every command unchanged — for graders that run inside an
/// already-sandboxed environment where the sandbox is the policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

impl CommandGate for AllowAll {
    fn allow(&self, _command: &CheckCommand) -> impl Future<Output = bool> {
        std::future::ready(true)
    }
}

/// The outcome of a single command invocation, before fix orchestration.
enum RunResult {
    /// Allowed and run; `success` is the exit-code verdict.
    Ran { success: bool, detail: String },
    /// The gate refused the command.
    Denied,
    /// The command could not be started or timed out.
    Errored(String),
}

/// Runs checks through a [`CommandGate`] in a working directory, with bounded
/// output capture, a per-check timeout, and fix-and-re-run orchestration.
pub struct CheckRunner<'a, G> {
    gate: &'a G,
    root: &'a Path,
    timeout: Duration,
}

impl<'a, G: CommandGate> CheckRunner<'a, G> {
    /// A runner that asks `gate` before every spawn and runs allowed commands
    /// in `root`.
    #[must_use]
    pub fn new(gate: &'a G, root: &'a Path) -> Self {
        Self {
            gate,
            root,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Override the per-check timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Run a check; when it fails and a fixer is supplied, run the fixer and
    /// re-run the check once. Every command goes through the gate.
    pub async fn run(&self, spec: &CheckSpec) -> CheckOutcome {
        match self.run_command(&spec.command).await {
            RunResult::Ran { success: true, .. } => {
                outcome(spec, CheckStatus::Passed, String::new(), false)
            }
            RunResult::Denied => outcome(
                spec,
                CheckStatus::Denied,
                "the gate refused the check command".to_string(),
                false,
            ),
            RunResult::Errored(detail) => outcome(spec, CheckStatus::Errored, detail, false),
            RunResult::Ran {
                success: false,
                detail,
            } => self.maybe_fix(spec, detail).await,
        }
    }

    /// On a failing check, run the fixer (if one is supplied) and re-run the
    /// check once; otherwise report the failure as-is.
    async fn maybe_fix(&self, spec: &CheckSpec, first_detail: String) -> CheckOutcome {
        let Some(fixer) = &spec.fixer else {
            return outcome(spec, CheckStatus::Failed, first_detail, false);
        };
        // The fixer is itself a gate-checked command; its own result does not
        // decide the outcome — the re-run of the check does.
        let _ = self.run_command(fixer).await;
        match self.run_command(&spec.command).await {
            RunResult::Ran { success: true, .. } => {
                outcome(spec, CheckStatus::Passed, String::new(), true)
            }
            RunResult::Ran {
                success: false,
                detail,
            } => outcome(spec, CheckStatus::Failed, detail, true),
            RunResult::Denied => outcome(
                spec,
                CheckStatus::Denied,
                "the gate refused the check re-run".to_string(),
                true,
            ),
            RunResult::Errored(detail) => outcome(spec, CheckStatus::Errored, detail, true),
        }
    }

    /// Ask the gate and — only if allowed — spawn the command in the working
    /// directory, capturing a bounded, sanitized result.
    async fn run_command(&self, command: &CheckCommand) -> RunResult {
        if !self.gate.allow(command).await {
            return RunResult::Denied;
        }

        let mut process = tokio::process::Command::new(&command.program);
        process
            .args(&command.args)
            .current_dir(self.root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let child = match process.spawn() {
            Ok(child) => child,
            Err(error) => {
                return RunResult::Errored(format!("failed to start {}: {error}", command.program))
            }
        };
        match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = bound(self.gate.sanitize(format!(
                    "exit: {code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )));
                RunResult::Ran {
                    success: output.status.success(),
                    detail,
                }
            }
            Ok(Err(error)) => RunResult::Errored(error.to_string()),
            Err(_) => {
                RunResult::Errored(format!("check timed out after {}s", self.timeout.as_secs()))
            }
        }
    }
}

fn outcome(spec: &CheckSpec, status: CheckStatus, detail: String, fixed: bool) -> CheckOutcome {
    CheckOutcome {
        name: spec.name.clone(),
        status,
        detail,
        fixed,
        severity: spec.severity,
    }
}

/// Truncate `text` to the output cap on a char boundary.
fn bound(mut text: String) -> String {
    if text.len() <= MAX_OUTPUT_BYTES {
        return text;
    }
    let mut end = MAX_OUTPUT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str("\n... [output truncated]");
    text
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A gate that refuses everything.
    struct DenyAll;
    impl CommandGate for DenyAll {
        fn allow(&self, _command: &CheckCommand) -> impl Future<Output = bool> {
            std::future::ready(false)
        }
    }

    /// A gate that stamps sanitized output, proving sanitize runs pre-bound.
    struct Stamping;
    impl CommandGate for Stamping {
        fn allow(&self, _command: &CheckCommand) -> impl Future<Output = bool> {
            std::future::ready(true)
        }
        fn sanitize(&self, text: String) -> String {
            text.replace("exit:", "exit-code:")
        }
    }

    fn spec(command: CheckCommand, fixer: Option<CheckCommand>) -> CheckSpec {
        CheckSpec {
            name: "t".to_string(),
            command,
            fixer,
            severity: None,
        }
    }

    // Cross-platform command builders (no shell assumptions baked in).
    #[cfg(windows)]
    fn exit_with(code: i32) -> CheckCommand {
        CheckCommand::new("cmd", vec!["/C".to_string(), format!("exit {code}")])
    }
    #[cfg(not(windows))]
    fn exit_with(code: i32) -> CheckCommand {
        CheckCommand::new("sh", vec!["-c".to_string(), format!("exit {code}")])
    }

    #[cfg(windows)]
    fn require_marker() -> CheckCommand {
        CheckCommand::new("cmd", vec!["/C".to_string(), "dir marker.txt".to_string()])
    }
    #[cfg(not(windows))]
    fn require_marker() -> CheckCommand {
        CheckCommand::new("ls", vec!["marker.txt".to_string()])
    }

    #[cfg(windows)]
    fn create_marker() -> CheckCommand {
        CheckCommand::new(
            "cmd",
            vec!["/C".to_string(), "type nul > marker.txt".to_string()],
        )
    }
    #[cfg(not(windows))]
    fn create_marker() -> CheckCommand {
        CheckCommand::new("touch", vec!["marker.txt".to_string()])
    }

    #[tokio::test]
    async fn a_passing_command_is_reported_passed() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let outcome = runner.run(&spec(exit_with(0), None)).await;
        assert_eq!(outcome.status, CheckStatus::Passed);
        assert!(outcome.passed());
        assert!(!outcome.fixed);
    }

    #[tokio::test]
    async fn a_failing_command_is_reported_failed_with_detail() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let outcome = runner.run(&spec(exit_with(1), None)).await;
        assert_eq!(outcome.status, CheckStatus::Failed);
        assert!(outcome.detail.contains("exit: 1"));
    }

    #[tokio::test]
    async fn a_denied_command_is_not_spawned() {
        // A nonexistent program under a denying gate reports Denied, not
        // Errored — proof it was never spawned.
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&DenyAll, dir.path());
        let outcome = runner
            .run(&spec(
                CheckCommand::new("definitely-not-a-real-program-xyzzy", Vec::new()),
                None,
            ))
            .await;
        assert_eq!(outcome.status, CheckStatus::Denied);
    }

    #[tokio::test]
    async fn a_fixer_runs_and_the_check_re_runs_to_pass() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let outcome = runner
            .run(&spec(require_marker(), Some(create_marker())))
            .await;
        assert_eq!(outcome.status, CheckStatus::Passed);
        assert!(outcome.fixed);
        assert!(dir.path().join("marker.txt").is_file());
    }

    #[tokio::test]
    async fn no_fixer_means_the_failure_is_reported_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let outcome = runner.run(&spec(require_marker(), None)).await;
        assert_eq!(outcome.status, CheckStatus::Failed);
        assert!(!outcome.fixed);
        assert!(!dir.path().join("marker.txt").is_file());
    }

    #[tokio::test]
    async fn output_is_sanitized_by_the_gate() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&Stamping, dir.path());
        let outcome = runner.run(&spec(exit_with(3), None)).await;
        assert!(outcome.detail.contains("exit-code: 3"));
    }

    #[test]
    fn command_line_splits_on_whitespace_and_rejects_blank() {
        let cmd = CheckCommand::from_command_line("ctest --output-on-failure").unwrap();
        assert_eq!(cmd.program, "ctest");
        assert_eq!(cmd.args, vec!["--output-on-failure".to_string()]);
        assert!(CheckCommand::from_command_line("   ").is_none());
    }
}
