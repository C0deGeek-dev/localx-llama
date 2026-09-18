//! Read a llama.cpp tool's `--help` output.
//!
//! Builds disagree about which flags exist, so callers decide what to pass by
//! asking the binary that will run (see
//! [`localx_llama_core::capabilities::ServerCapabilities::from_help`]). The
//! read is bounded: a binary that hangs, cannot start, or prints nothing yields
//! `None`, and the caller falls back to the long-standing flags.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Run `<binary> --help` and return its standard output.
///
/// Returns `None` when the binary cannot be started, does not finish within
/// `timeout` (it is killed), or prints nothing. Standard input is closed so a
/// tool can never wait on the caller's terminal.
#[must_use]
pub fn read_help_output(binary: &Path, timeout: Duration) -> Option<String> {
    let mut child = Command::new(binary)
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    // Drain stdout on a thread: help text is larger than a pipe buffer, so the
    // child would block on write if nobody read while we wait for it to exit.
    let (sender, receiver) = mpsc::channel();
    if let Some(mut stdout) = child.stdout.take() {
        std::thread::spawn(move || {
            let mut text = String::new();
            let read = stdout.read_to_string(&mut text).map(|_| text);
            let _ = sender.send(read);
        });
    }
    let text = receiver.recv_timeout(timeout).ok().and_then(Result::ok);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    text.filter(|text| !text.trim().is_empty())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_binary_has_no_help() {
        let missing = std::env::temp_dir()
            .join(format!("localx-no-such-tool-{}", std::process::id()))
            .join("llama-server");
        assert_eq!(read_help_output(&missing, Duration::from_secs(5)), None);
    }

    #[test]
    fn a_real_tool_help_is_read_in_full() {
        // `cargo` is always present where these tests run and prints help to stdout.
        let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
        let help = read_help_output(Path::new(&cargo), Duration::from_secs(30)).unwrap();
        assert!(help.contains("Usage"), "{help}");
    }
}
