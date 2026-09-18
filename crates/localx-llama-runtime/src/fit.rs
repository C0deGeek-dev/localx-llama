//! Run llama.cpp's memory fitter (`llama-fit-params`) as an oracle.
//!
//! The fitter loads only metadata and answers in seconds; see
//! [`localx_llama_core::fit`] for what it is given and what its answer means.

use std::path::{Path, PathBuf};
use std::time::Duration;

use localx_llama_core::fit::{parse_fit_output, FitError, FitPlacement};

use crate::tool::run_tool;

/// The `llama-fit-params` executable name for this platform.
#[must_use]
pub fn fit_params_exe_name() -> &'static str {
    if cfg!(windows) {
        "llama-fit-params.exe"
    } else {
        "llama-fit-params"
    }
}

/// The `llama-fit-params` that ships beside a `llama-server`, when present.
#[must_use]
pub fn fit_params_beside(server_binary: &Path) -> Option<PathBuf> {
    let candidate = server_binary.with_file_name(fit_params_exe_name());
    candidate.is_file().then_some(candidate)
}

/// Why the fitter gave no placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitRunError {
    /// It could not be started or did not finish in time.
    DidNotRun,
    /// It ran but its answer was a failure or unreadable.
    Fit(FitError),
}

impl std::fmt::Display for FitRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DidNotRun => f.write_str("llama-fit-params did not run to completion"),
            Self::Fit(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for FitRunError {}

/// Ask `binary` (a `llama-fit-params`) where a launch fits.
///
/// `args` come from [`localx_llama_core::fit::fit_params_args`]; `-lv 4` is
/// added so the per-device summary is logged.
///
/// # Errors
/// [`FitRunError::DidNotRun`] when the tool could not start or timed out;
/// [`FitRunError::Fit`] when it reported a failure or printed no placement.
pub fn run_fit_params(
    binary: &Path,
    args: &[String],
    timeout: Duration,
) -> Result<FitPlacement, FitRunError> {
    let mut full = args.to_vec();
    full.extend(["-lv".to_string(), "4".to_string()]);
    let output = run_tool(binary, &full, timeout).ok_or(FitRunError::DidNotRun)?;
    parse_fit_output(&output.stdout, &output.stderr).map_err(FitRunError::Fit)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_fitter_does_not_run() {
        let missing = std::env::temp_dir()
            .join(format!("localx-no-such-tool-{}", std::process::id()))
            .join(fit_params_exe_name());
        assert_eq!(
            run_fit_params(&missing, &[], Duration::from_secs(5)),
            Err(FitRunError::DidNotRun)
        );
        assert_eq!(
            fit_params_beside(&missing.with_file_name("llama-server")),
            None
        );
    }

    /// Live check against a real build: set `LOCALX_FIT_PARAMS` to a
    /// `llama-fit-params` binary and `LOCALX_FIT_MODEL` to a GGUF, then run
    /// `cargo test -p localx-llama-runtime -- --ignored live_fit`.
    #[test]
    #[ignore = "needs a llama.cpp build and a model on disk"]
    fn live_fit_reports_a_placement() {
        let binary = std::env::var_os("LOCALX_FIT_PARAMS").expect("LOCALX_FIT_PARAMS");
        let model = std::env::var("LOCALX_FIT_MODEL").expect("LOCALX_FIT_MODEL");
        let context = std::env::var("LOCALX_FIT_CTX").unwrap_or_else(|_| "65536".to_string());
        let server = [
            "-m",
            &model,
            "-c",
            &context,
            "-ngl",
            "999",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
        ]
        .map(str::to_string);
        let args = localx_llama_core::fit::fit_params_args(&server, 1536);
        let fit = run_fit_params(Path::new(&binary), &args, Duration::from_secs(120)).unwrap();
        println!(
            "gpu_layers={} n_cpu_moe={} devices={:?}",
            fit.gpu_layers,
            fit.n_cpu_moe(),
            fit.devices
        );
        assert!(fit.gpu_layers != 0);
    }
}
