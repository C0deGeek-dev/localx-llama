//! Read a llama.cpp tool's `--help` output.
//!
//! Builds disagree about which flags exist, so callers decide what to pass by
//! asking the binary that will run (see
//! [`localx_llama_core::capabilities::ServerCapabilities::from_help`]). The
//! read is bounded: a binary that hangs, cannot start, or prints nothing yields
//! `None`, and the caller falls back to the long-standing flags.

use std::path::Path;
use std::time::Duration;

use crate::tool::run_tool;

/// Run `<binary> --help` and return its standard output.
///
/// Returns `None` when the binary cannot be started, does not finish within
/// `timeout` (it is killed), or prints nothing.
#[must_use]
pub fn read_help_output(binary: &Path, timeout: Duration) -> Option<String> {
    run_tool(binary, &["--help".to_string()], timeout)
        .map(|output| output.stdout)
        .filter(|text| !text.trim().is_empty())
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
