//! Run a short-lived llama.cpp helper tool (`--help`, `llama-fit-params`,
//! `llama-bench`) and collect what it printed, within a time limit.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// What a finished tool printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    /// Whether it exited with status zero.
    pub success: bool,
    /// Standard output.
    pub stdout: String,
    /// Standard error (llama.cpp logs go here).
    pub stderr: String,
}

/// Run `binary args…` and return its output once it exits.
///
/// Returns `None` when the binary cannot be started or has not exited within
/// `timeout` (it is then killed). Standard input is closed so a tool never
/// waits on the caller's terminal; both output streams are drained while it
/// runs so a large report cannot block it on a full pipe.
#[must_use]
pub fn run_tool(binary: &Path, args: &[String], timeout: Duration) -> Option<ToolOutput> {
    let mut child = Command::new(binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    Some(ToolOutput {
        success: status.success(),
        stdout: stdout.recv_timeout(remaining).unwrap_or_default(),
        stderr: stderr.recv_timeout(remaining).unwrap_or_default(),
    })
}

/// Read a pipe to the end on its own thread.
fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    if let Some(mut pipe) = pipe {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            let _ = sender.send(String::from_utf8_lossy(&bytes).into_owned());
        });
    } else {
        let _ = sender.send(String::new());
    }
    receiver
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn cargo() -> std::path::PathBuf {
        std::env::var_os("CARGO")
            .expect("cargo sets CARGO for tests")
            .into()
    }

    #[test]
    fn output_and_exit_status_are_captured() {
        let ok = run_tool(
            &cargo(),
            &["--version".to_string()],
            Duration::from_secs(30),
        )
        .unwrap();
        assert!(ok.success);
        assert!(ok.stdout.starts_with("cargo "), "{}", ok.stdout);
        let failed = run_tool(
            &cargo(),
            &["--no-such-cargo-flag".to_string()],
            Duration::from_secs(30),
        )
        .unwrap();
        assert!(!failed.success);
        assert!(!failed.stderr.is_empty());
    }

    #[test]
    fn a_missing_binary_is_none() {
        let missing = std::env::temp_dir()
            .join(format!("localx-no-such-tool-{}", std::process::id()))
            .join("llama-fit-params");
        assert_eq!(run_tool(&missing, &[], Duration::from_secs(5)), None);
    }
}
