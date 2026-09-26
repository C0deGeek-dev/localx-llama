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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::AsyncReadExt;
use tokio::sync::Notify;

/// Cap on captured check output before truncation.
const MAX_OUTPUT_BYTES: usize = 16 * 1024;

/// Windows `CREATE_NO_WINDOW`: the check gets its own console (with no
/// window) instead of attaching to the host's — see `run_command`.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

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
pub trait CommandGate {
    /// Decide whether the command may run. Neither the gate nor its future
    /// carries a `Send`/`Sync` bound, so a host whose approval flow is
    /// single-threaded (an interactive prompt behind a non-thread-safe handle)
    /// can implement it; drive the runner on a current-thread or local context
    /// when the gate needs it.
    fn allow(&self, command: &CheckCommand) -> impl Future<Output = bool>;

    /// Sanitize captured output before it becomes finding detail (e.g. secret
    /// redaction). The default keeps it as-is.
    fn sanitize(&self, text: String) -> String {
        text
    }

    /// Stop everything a timed-out or cancelled command started, before the
    /// runner kills the command itself. `pid` is the command's process id; on
    /// Unix it also leads its own process group. The default does nothing, so
    /// only the direct child is killed — a host whose checks start their own
    /// children (a test runner does) should reap the whole tree here.
    fn reap(&self, pid: u32) -> impl Future<Output = ()> {
        let _ = pid;
        std::future::ready(())
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

/// Stops a running check from outside. Cloning shares the signal: cancelling
/// any clone cancels every run that holds one.
#[derive(Debug, Clone, Default)]
pub struct CancelSignal {
    inner: Arc<CancelInner>,
}

#[derive(Debug, Default)]
struct CancelInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl CancelSignal {
    /// A signal nobody has cancelled yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel every run holding this signal. Idempotent.
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
    }

    /// Whether [`cancel`](Self::cancel) has been called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Resolves once the signal is cancelled; at once if it already is.
    pub async fn cancelled(&self) {
        loop {
            // Registered before the check, so a cancel landing in between is
            // not missed.
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Which environment a spawned command sees.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EnvPolicy {
    /// The host's whole environment, unchanged.
    #[default]
    Inherit,
    /// Only the named host variables that are set, plus explicit additions.
    Only {
        /// Host variables passed through when present.
        keep: Vec<String>,
        /// Variables set for the command, after `keep`.
        set: Vec<(String, String)>,
    },
}

/// How one command invocation ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEnd {
    /// It ran to completion. `code` is `None` when a signal ended it.
    Exited { code: Option<i32>, success: bool },
    /// The gate refused it; nothing was spawned.
    Denied,
    /// It could not be started (a missing program, for one).
    NotStarted(String),
    /// It outlived the runner's timeout and was stopped.
    TimedOut,
    /// The runner's [`CancelSignal`] fired and it was stopped.
    Cancelled,
    /// Waiting on it failed.
    Failed(String),
}

/// One command invocation, as [`CheckRunner::execute`] observed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandRun {
    /// How it ended.
    pub end: CommandEnd,
    /// Captured standard output, sanitized by the gate and bounded.
    pub stdout: String,
    /// Captured standard error, sanitized by the gate and bounded.
    pub stderr: String,
    /// Output went past the capture limit and the rest was discarded.
    pub truncated: bool,
    /// Both output pipes closed. `false` means something the command started
    /// still held them after it ended or was stopped, and was still running
    /// when the runner stopped waiting — an empty capture is then unproven,
    /// not clean.
    pub pipes_closed: bool,
    /// The spawned process id, when it started.
    pub pid: Option<u32>,
    /// Wall-clock time from the gate's decision to the end.
    pub elapsed: Duration,
}

impl CommandRun {
    fn without_process(end: CommandEnd, started: Instant) -> Self {
        Self {
            end,
            stdout: String::new(),
            stderr: String::new(),
            truncated: false,
            pipes_closed: true,
            pid: None,
            elapsed: started.elapsed(),
        }
    }
}

/// Runs checks through a [`CommandGate`] in a working directory, with bounded
/// output capture, a per-check timeout, and fix-and-re-run orchestration.
pub struct CheckRunner<'a, G> {
    gate: &'a G,
    root: &'a Path,
    timeout: Duration,
    env: EnvPolicy,
    cancel: Option<CancelSignal>,
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
            env: EnvPolicy::Inherit,
            cancel: None,
        }
    }

    /// Override the per-check timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The environment spawned commands see. [`EnvPolicy::Inherit`] by default.
    #[must_use]
    pub fn with_env(mut self, env: EnvPolicy) -> Self {
        self.env = env;
        self
    }

    /// Stop a running command, and refuse to start one, once `cancel` fires.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelSignal) -> Self {
        self.cancel = Some(cancel);
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

    /// One command through the gate, folded into the fix orchestration's
    /// terms. The detail keeps its long-standing shape.
    async fn run_command(&self, command: &CheckCommand) -> RunResult {
        let run = self.execute(command).await;
        let captured = || {
            format!(
                "\n--- stdout ---\n{}\n--- stderr ---\n{}",
                run.stdout, run.stderr
            )
        };
        match &run.end {
            CommandEnd::Denied => RunResult::Denied,
            CommandEnd::Exited { code, success } => RunResult::Ran {
                success: *success,
                detail: bound(self.gate.sanitize(format!(
                    "exit: {}{}",
                    code.unwrap_or(-1),
                    captured()
                ))),
            },
            CommandEnd::NotStarted(detail) | CommandEnd::Failed(detail) => {
                RunResult::Errored(detail.clone())
            }
            CommandEnd::TimedOut => RunResult::Errored(bound(self.gate.sanitize(format!(
                "check timed out after {}s{}",
                self.timeout.as_secs(),
                captured()
            )))),
            CommandEnd::Cancelled => RunResult::Errored(bound(
                self.gate
                    .sanitize(format!("check was cancelled{}", captured())),
            )),
        }
    }

    /// Ask the gate and — only if allowed — run one command in the working
    /// directory, with no fixer. Output is captured as it arrives and bounded
    /// while reading, so a command that floods its output cannot grow the
    /// host's memory. A command that outlives the timeout or is cancelled is
    /// stopped: the gate's [`reap`](CommandGate::reap) runs first, so a host
    /// can take down everything the command started, then the direct child is
    /// killed. What was captured before the stop is kept.
    pub async fn execute(&self, command: &CheckCommand) -> CommandRun {
        let started = Instant::now();
        if !self.gate.allow(command).await {
            return CommandRun::without_process(CommandEnd::Denied, started);
        }
        if self.cancel.as_ref().is_some_and(CancelSignal::is_cancelled) {
            return CommandRun::without_process(CommandEnd::Cancelled, started);
        }

        let mut process = tokio::process::Command::new(&command.program);
        process
            .args(&command.args)
            .current_dir(self.root)
            // Never inherit the host's stdin: checks run next to interactive
            // TUIs (LocalPilot's chat REPL), and a check that reads stdin
            // would consume the terminal's keystrokes — including the Ctrl+C
            // key event raw mode relies on. A check that wants input gets
            // immediate EOF instead of hanging until the timeout.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let EnvPolicy::Only { keep, set } = &self.env {
            process.env_clear();
            for name in keep {
                if let Some(value) = std::env::var_os(name) {
                    process.env(name, value);
                }
            }
            for (name, value) in set {
                process.env(name, value);
            }
        }
        // Isolate the check from the host's terminal: on Windows a shared
        // console would let the child read `CONIN$` or re-cook the console
        // mode with `SetConsoleMode`; on Unix a new (never-foreground)
        // process group means a direct `/dev/tty` read gets SIGTTIN instead
        // of the host's input, and the whole group can be signalled at once.
        #[cfg(windows)]
        process.creation_flags(CREATE_NO_WINDOW);
        #[cfg(unix)]
        process.process_group(0);

        let mut child = match process.spawn() {
            Ok(child) => child,
            Err(error) => {
                return CommandRun::without_process(
                    CommandEnd::NotStarted(format!("failed to start {}: {error}", command.program)),
                    started,
                )
            }
        };
        let pid = child.id();
        let stdout = Captured::start(child.stdout.take());
        let stderr = Captured::start(child.stderr.take());

        let cancelled = async {
            match &self.cancel {
                Some(cancel) => cancel.cancelled().await,
                None => std::future::pending().await,
            }
        };
        let end = tokio::select! {
            status = child.wait() => match status {
                Ok(status) => CommandEnd::Exited {
                    code: status.code(),
                    success: status.success(),
                },
                Err(error) => CommandEnd::Failed(error.to_string()),
            },
            () = tokio::time::sleep(self.timeout) => CommandEnd::TimedOut,
            () = cancelled => CommandEnd::Cancelled,
        };
        let stopped = matches!(end, CommandEnd::TimedOut | CommandEnd::Cancelled);
        if stopped {
            // The tree first, while the child still exists to be walked from;
            // then the child itself, whatever the reap managed.
            if let Some(pid) = pid {
                self.gate.reap(pid).await;
            }
            let _ = child.start_kill();
            let _ = tokio::time::timeout(PIPE_GRACE, child.wait()).await;
        }

        // A grandchild that inherited a pipe keeps it open after the child is
        // gone, so the readers are given a grace, not an unbounded wait.
        let stdout_closed = stdout.finish(PIPE_GRACE).await;
        let stderr_closed = stderr.finish(PIPE_GRACE).await;
        let pipes_closed = stdout_closed && stderr_closed;
        if !pipes_closed && !stopped {
            // The command exited but something it started is still writing.
            if let Some(pid) = pid {
                self.gate.reap(pid).await;
            }
        }
        let (out, out_truncated) = stdout.take();
        let (err, err_truncated) = stderr.take();
        CommandRun {
            end,
            stdout: bound(self.gate.sanitize(out)),
            stderr: bound(self.gate.sanitize(err)),
            truncated: out_truncated || err_truncated,
            pipes_closed,
            pid,
            elapsed: started.elapsed(),
        }
    }
}

/// How long the readers are given to see a pipe close once the command has
/// ended or been stopped.
const PIPE_GRACE: Duration = Duration::from_secs(2);

/// Bytes kept from each output stream; the rest is read and discarded so the
/// command never blocks on a full pipe.
const CAPTURE_LIMIT: usize = 64 * 1024;

/// One output stream, read on its own task into a bounded buffer that stays
/// readable if the task has to be abandoned.
struct Captured {
    buffer: Arc<Mutex<(Vec<u8>, bool)>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Captured {
    fn start<R>(reader: Option<R>) -> Self
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        let buffer = Arc::new(Mutex::new((Vec::new(), false)));
        let task = reader.map(|mut reader| {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move {
                let mut chunk = [0u8; 8192];
                loop {
                    match reader.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if let Ok(mut held) = buffer.lock() {
                                let room = CAPTURE_LIMIT.saturating_sub(held.0.len());
                                if n > room {
                                    held.1 = true;
                                }
                                let keep = n.min(room);
                                held.0.extend_from_slice(&chunk[..keep]);
                            }
                        }
                    }
                }
            })
        });
        Self { buffer, task }
    }

    /// Wait up to `grace` for the stream to close. `true` when it did; `false`
    /// when the reader had to be abandoned with the pipe still open.
    async fn finish(&self, grace: Duration) -> bool {
        let Some(task) = &self.task else {
            return true;
        };
        let deadline = Instant::now() + grace;
        while !task.is_finished() {
            if Instant::now() >= deadline {
                task.abort();
                return false;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        true
    }

    fn take(&self) -> (String, bool) {
        match self.buffer.lock() {
            Ok(held) => (String::from_utf8_lossy(&held.0).into_owned(), held.1),
            Err(_) => (String::new(), true),
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

    /// A gate that allows everything and records every reap it is asked for.
    #[derive(Default)]
    struct Recording {
        reaped: Mutex<Vec<u32>>,
    }
    impl CommandGate for Recording {
        fn allow(&self, _command: &CheckCommand) -> impl Future<Output = bool> {
            std::future::ready(true)
        }
        fn reap(&self, pid: u32) -> impl Future<Output = ()> {
            self.reaped.lock().unwrap().push(pid);
            std::future::ready(())
        }
    }

    /// Prints `started`, then waits well past the tests' timeouts. Kept short:
    /// the recording gate reaps nothing, so the runtime waits it out at exit.
    #[cfg(windows)]
    fn start_then_hang() -> CheckCommand {
        CheckCommand::new(
            "cmd",
            vec![
                "/C".to_string(),
                "echo started& ping -n 8 127.0.0.1 >nul".to_string(),
            ],
        )
    }
    #[cfg(not(windows))]
    fn start_then_hang() -> CheckCommand {
        CheckCommand::new(
            "sh",
            vec!["-c".to_string(), "echo started; sleep 8".to_string()],
        )
    }

    /// Prints a file's contents.
    #[cfg(windows)]
    fn print_file(name: &str) -> CheckCommand {
        CheckCommand::new("cmd", vec!["/C".to_string(), format!("type {name}")])
    }
    #[cfg(not(windows))]
    fn print_file(name: &str) -> CheckCommand {
        CheckCommand::new("cat", vec![name.to_string()])
    }

    /// Prints the environment the command sees.
    #[cfg(windows)]
    fn print_env() -> CheckCommand {
        CheckCommand::new("cmd", vec!["/C".to_string(), "set".to_string()])
    }
    #[cfg(not(windows))]
    fn print_env() -> CheckCommand {
        CheckCommand::new("env", Vec::new())
    }

    /// Exits at once, leaving a background process holding its output pipe.
    #[cfg(windows)]
    fn exit_leaving_a_writer() -> CheckCommand {
        CheckCommand::new(
            "cmd",
            vec!["/C".to_string(), "start /b ping -n 8 127.0.0.1".to_string()],
        )
    }
    #[cfg(not(windows))]
    fn exit_leaving_a_writer() -> CheckCommand {
        CheckCommand::new("sh", vec!["-c".to_string(), "sleep 8 &".to_string()])
    }

    #[tokio::test]
    async fn execute_reports_the_exit_and_the_captured_output() {
        let dir = tempfile::tempdir().unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let run = runner.execute(&exit_with(3)).await;
        assert_eq!(
            run.end,
            CommandEnd::Exited {
                code: Some(3),
                success: false
            }
        );
        assert!(run.pid.is_some());
        assert!(run.pipes_closed);
        assert!(!run.truncated);
    }

    #[tokio::test]
    async fn a_missing_program_is_not_started_and_nothing_is_reaped() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Recording::default();
        let runner = CheckRunner::new(&gate, dir.path());
        let run = runner
            .execute(&CheckCommand::new(
                "definitely-not-a-real-program-xyzzy",
                Vec::new(),
            ))
            .await;
        assert!(matches!(run.end, CommandEnd::NotStarted(_)), "{run:?}");
        assert_eq!(run.pid, None);
        assert!(gate.reaped.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_timeout_reaps_before_killing_and_keeps_what_was_captured() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Recording::default();
        let runner = CheckRunner::new(&gate, dir.path()).with_timeout(Duration::from_millis(1500));
        let run = runner.execute(&start_then_hang()).await;
        assert_eq!(run.end, CommandEnd::TimedOut);
        assert!(run.stdout.contains("started"), "{run:?}");
        assert_eq!(*gate.reaped.lock().unwrap(), vec![run.pid.unwrap()]);
        assert!(run.elapsed < Duration::from_secs(20), "{:?}", run.elapsed);

        // The long-standing fold keeps the message and now carries the output.
        let outcome = runner.run(&spec(start_then_hang(), None)).await;
        assert_eq!(outcome.status, CheckStatus::Errored);
        assert!(outcome.detail.starts_with("check timed out after 1s"));
        assert!(outcome.detail.contains("started"), "{}", outcome.detail);
    }

    #[tokio::test]
    async fn a_cancel_stops_a_running_command_and_prevents_the_next() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Recording::default();
        let cancel = CancelSignal::new();
        let runner = CheckRunner::new(&gate, dir.path()).with_cancel(cancel.clone());
        let trigger = {
            let cancel = cancel.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(700)).await;
                cancel.cancel();
            })
        };
        let run = runner.execute(&start_then_hang()).await;
        trigger.await.unwrap();
        assert_eq!(run.end, CommandEnd::Cancelled);
        assert_eq!(gate.reaped.lock().unwrap().len(), 1);
        assert!(run.elapsed < Duration::from_secs(20), "{:?}", run.elapsed);

        let next = runner.execute(&exit_with(0)).await;
        assert_eq!(next.end, CommandEnd::Cancelled);
        assert_eq!(next.pid, None, "nothing starts once cancelled");
    }

    #[tokio::test]
    async fn a_flood_of_output_is_bounded_while_it_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let line = "x".repeat(99) + "\n";
        std::fs::write(dir.path().join("big.txt"), line.repeat(40_000)).unwrap();
        let runner = CheckRunner::new(&AllowAll, dir.path());
        let run = runner.execute(&print_file("big.txt")).await;
        assert!(
            matches!(run.end, CommandEnd::Exited { success: true, .. }),
            "{:?}",
            run.end
        );
        assert!(run.truncated, "4 MB went past the capture limit");
        assert!(run.stdout.len() <= MAX_OUTPUT_BYTES + 64);
        assert!(run.pipes_closed);
    }

    #[tokio::test]
    async fn an_environment_allowlist_passes_only_what_it_names() {
        let dir = tempfile::tempdir().unwrap();
        let keep: Vec<String> = ["PATH", "SystemRoot", "ComSpec"]
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let runner = CheckRunner::new(&AllowAll, dir.path()).with_env(EnvPolicy::Only {
            keep,
            set: vec![("LOCALX_EVAL_MARK".to_string(), "on".to_string())],
        });
        let run = runner.execute(&print_env()).await;
        assert!(
            matches!(run.end, CommandEnd::Exited { success: true, .. }),
            "{run:?}"
        );
        assert!(run.stdout.contains("LOCALX_EVAL_MARK=on"), "{}", run.stdout);
        // Cargo sets this for the test process; the allowlist drops it.
        assert!(std::env::var_os("CARGO_PKG_NAME").is_some());
        assert!(!run.stdout.contains("CARGO_PKG_NAME"), "{}", run.stdout);

        let inherit = CheckRunner::new(&AllowAll, dir.path())
            .execute(&print_env())
            .await;
        assert!(inherit.stdout.contains("CARGO_PKG_NAME"));
    }

    #[tokio::test]
    async fn a_process_left_holding_the_pipe_is_reported_and_reaped() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Recording::default();
        let runner = CheckRunner::new(&gate, dir.path());
        let run = runner.execute(&exit_leaving_a_writer()).await;
        assert!(
            matches!(run.end, CommandEnd::Exited { success: true, .. }),
            "{run:?}"
        );
        assert!(
            !run.pipes_closed,
            "a background writer still holds the pipe"
        );
        assert_eq!(*gate.reaped.lock().unwrap(), vec![run.pid.unwrap()]);
    }

    #[tokio::test]
    async fn a_cancel_signal_is_shared_and_resolves_once_fired() {
        let signal = CancelSignal::new();
        let clone = signal.clone();
        assert!(!clone.is_cancelled());
        signal.cancel();
        assert!(clone.is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), clone.cancelled())
            .await
            .expect("an already-cancelled signal resolves at once");
    }

    #[test]
    fn command_line_splits_on_whitespace_and_rejects_blank() {
        let cmd = CheckCommand::from_command_line("ctest --output-on-failure").unwrap();
        assert_eq!(cmd.program, "ctest");
        assert_eq!(cmd.args, vec!["--output-on-failure".to_string()]);
        assert!(CheckCommand::from_command_line("   ").is_none());
    }
}
