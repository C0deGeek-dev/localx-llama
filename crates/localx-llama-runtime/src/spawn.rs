//! Detached child-process spawning with the Windows verbatim-path fix.
//!
//! A Windows verbatim `\\?\` path is a *containment* form, not a *spawn* form —
//! a child (llama-server, a grader) cannot `cd` into it. `simplify_cwd` de-
//! verbatims the working directory via `dunce::simplified` before spawning, while
//! callers keep the verbatim spelling for boundary checks.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// De-verbatim a working directory so a spawned child can enter it.
pub fn simplify_cwd(path: &Path) -> PathBuf {
    dunce::simplified(path).to_path_buf()
}

/// Spawn a detached child, optionally in a working directory and with stdout+
/// stderr redirected to a log file.
pub fn spawn_detached(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    log_path: Option<&Path>,
) -> std::io::Result<Child> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(simplify_cwd(dir));
    }
    if let Some(lp) = log_path {
        let out = File::create(lp)?;
        let err = out.try_clone()?;
        cmd.stdout(Stdio::from(out)).stderr(Stdio::from(err));
    }
    cmd.spawn()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn normal_path_round_trips() {
        let p = Path::new("some/relative/dir");
        assert_eq!(simplify_cwd(p), PathBuf::from("some/relative/dir"));
    }

    #[cfg(windows)]
    #[test]
    fn strips_verbatim_prefix_on_windows() {
        // The containment form must not reach a child's cwd.
        assert_eq!(
            simplify_cwd(Path::new(r"\\?\C:\work")),
            PathBuf::from(r"C:\work")
        );
    }

    #[test]
    fn spawns_and_waits_a_trivial_process() {
        #[cfg(windows)]
        let (prog, args) = (
            "cmd",
            vec!["/c".to_string(), "exit".to_string(), "0".to_string()],
        );
        #[cfg(not(windows))]
        let (prog, args) = ("sh", vec!["-c".to_string(), "exit 0".to_string()]);
        let mut child = spawn_detached(prog, &args, None, None).unwrap();
        assert!(child.wait().unwrap().success());
    }
}
