//! Workspace verification-target detection for a verify-before-done gate.
//!
//! Given a workspace root, resolve the command that answers "does this code
//! build / do its tests pass?" — the signal a gate runs before work is allowed
//! to finalize. Resolution is: an explicit override (a single command line,
//! split on whitespace — no shell) wins; otherwise a conventional command is
//! detected from the stack's marker files; otherwise `None` (no detectable
//! target, so the gate is a no-op).
//!
//! This is detection only — running the command is the host's job (through its
//! own gate/runner), so there is no second command engine. It deliberately
//! covers a broad language set: the gate runs against arbitrary solve
//! workspaces, where the biggest convergence lever is catching a
//! C++/Rust/Go/JS build failure before the loop "submits" code it never
//! compiled.

use std::path::Path;

use crate::check::CheckCommand;

/// The check name a verify gate presents. Stable so a scorecard/report reader
/// can find the verify outcome among gate outcomes.
pub const VERIFY_CHECK_NAME: &str = "verify";

/// Resolve the verification command for `root`: the `override_cmd` (split on
/// whitespace — no shell) when set and non-blank, otherwise the stack-detected
/// command, otherwise `None`.
#[must_use]
pub fn resolve_verify_command(root: &Path, override_cmd: Option<&str>) -> Option<CheckCommand> {
    if let Some(command) = override_cmd {
        if let Some(check) = CheckCommand::from_command_line(command) {
            return Some(check);
        }
    }
    detect_verify_command(root)
}

/// Detect a conventional verify command from `root`'s marker files, or `None`
/// when no supported stack is present. Marker files only — no execution. The
/// first matching stack in priority order wins.
#[must_use]
pub fn detect_verify_command(root: &Path) -> Option<CheckCommand> {
    // Priority order: a language-native test command is preferred over a generic
    // `make`, so a project that carries both is verified by its real test suite.
    if has_file(root, "Cargo.toml") {
        return Some(CheckCommand::new(cargo(), vec_of(&["test"])));
    }
    if has_file(root, "go.mod") {
        return Some(CheckCommand::new(go(), vec_of(&["test", "./..."])));
    }
    if has_file(root, "pom.xml") {
        return Some(CheckCommand::new(maven(), vec_of(&["-q", "test"])));
    }
    if has_file(root, "build.gradle") || has_file(root, "build.gradle.kts") {
        return Some(CheckCommand::new(
            gradle(),
            vec_of(&["test", "--console=plain"]),
        ));
    }
    if has_file(root, "package.json") {
        return Some(CheckCommand::new(npm(), vec_of(&["test"])));
    }
    if is_python(root) {
        return Some(CheckCommand::new(python(), vec_of(&["-m", "pytest", "-q"])));
    }
    // C/C++ without a language test runner: a top-level Makefile is the most
    // portable single-command build, so it wins when present.
    if has_file(root, "Makefile") || has_file(root, "makefile") {
        return Some(CheckCommand::new(make(), Vec::new()));
    }
    // Otherwise, when the workspace carries C++ sources (a CMake project or a
    // bare exercise layout), compile-check them with a single artifact-free
    // `g++ -fsyntax-only` over the enumerated translation units. This catches the
    // parse/type/include errors that are the dominant "submitted code that never
    // compiled" failure — the biggest convergence lever — without writing an
    // `a.out`/`*.o` that would pollute a captured diff, and without the
    // three-command CMake configure→build→ctest pipeline that does not fit one
    // program+args call. A full CMake build/test is available via an explicit
    // override.
    if let Some(sources) = cpp_sources(root) {
        let mut args = vec_of(&["-std=c++17", "-I.", "-fsyntax-only"]);
        args.extend(sources);
        return Some(CheckCommand::new(gpp(), args));
    }
    None
}

/// Root-level C++ translation units (`.cpp`/`.cc`/`.cxx`), sorted for a stable
/// command, or `None` when the workspace has none. Marker-free detection: the
/// presence of C++ sources at the workspace root is itself the signal (a CMake
/// project keeps them there, as does the bare exercise layout).
fn cpp_sources(root: &Path) -> Option<Vec<String>> {
    let mut sources: Vec<String> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let lower = name.to_ascii_lowercase();
            (lower.ends_with(".cpp") || lower.ends_with(".cc") || lower.ends_with(".cxx"))
                .then(|| name.to_string())
        })
        .collect();
    if sources.is_empty() {
        return None;
    }
    sources.sort();
    Some(sources)
}

fn vec_of(args: &[&str]) -> Vec<String> {
    args.iter().map(|a| (*a).to_string()).collect()
}

fn has_file(root: &Path, name: &str) -> bool {
    root.join(name).is_file()
}

/// A Python project: a build/test marker file, or any top-level `test_*.py` /
/// `*_test.py`.
fn is_python(root: &Path) -> bool {
    const MARKERS: &[&str] = &[
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "tox.ini",
        "pytest.ini",
        "conftest.py",
    ];
    if MARKERS.iter().any(|m| has_file(root, m)) {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry.file_name().to_str().is_some_and(|name| {
            name.ends_with(".py") && (name.starts_with("test_") || name.ends_with("_test.py"))
        })
    })
}

// Cross-platform program names: a Windows shim is invoked through its `.cmd`
// launcher (Command::new spawns the program directly, no shell), while `cargo`,
// `go`, and `make` resolve the same on every tier-1 platform.
fn cargo() -> String {
    "cargo".to_string()
}
fn go() -> String {
    "go".to_string()
}
fn make() -> String {
    "make".to_string()
}
fn gpp() -> String {
    "g++".to_string()
}
#[cfg(windows)]
fn npm() -> String {
    "npm.cmd".to_string()
}
#[cfg(not(windows))]
fn npm() -> String {
    "npm".to_string()
}
#[cfg(windows)]
fn maven() -> String {
    "mvn.cmd".to_string()
}
#[cfg(not(windows))]
fn maven() -> String {
    "mvn".to_string()
}
#[cfg(windows)]
fn gradle() -> String {
    "gradle.bat".to_string()
}
#[cfg(not(windows))]
fn gradle() -> String {
    "gradle".to_string()
}
#[cfg(windows)]
fn python() -> String {
    "python".to_string()
}
#[cfg(not(windows))]
fn python() -> String {
    "python3".to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn touch(root: &Path, name: &str) {
        std::fs::write(root.join(name), "x").unwrap();
    }

    #[test]
    fn no_target_when_workspace_is_bare() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detect_verify_command(dir.path()).is_none());
    }

    #[test]
    fn detects_each_stack() {
        // (marker file, expected program, expected first arg)
        let cases: &[(&str, &str, &str)] =
            &[("Cargo.toml", "cargo", "test"), ("go.mod", "go", "test")];
        for (marker, program, first_arg) in cases {
            let dir = tempfile::tempdir().unwrap();
            touch(dir.path(), marker);
            let check = detect_verify_command(dir.path()).expect("a target");
            assert_eq!(check.program, *program, "program for {marker}");
            assert_eq!(check.args.first().map(String::as_str), Some(*first_arg));
        }
    }

    #[test]
    fn rust_beats_a_generic_makefile() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "Makefile");
        touch(dir.path(), "Cargo.toml");
        let check = detect_verify_command(dir.path()).unwrap();
        assert_eq!(check.program, "cargo");
    }

    #[test]
    fn makefile_only_is_make() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "Makefile");
        assert_eq!(detect_verify_command(dir.path()).unwrap().program, "make");
    }

    #[test]
    fn detects_cpp_sources_as_a_gpp_compile_check() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "anagram.cpp");
        touch(dir.path(), "anagram_test.cpp");
        let check = detect_verify_command(dir.path()).expect("a C++ target");
        assert_eq!(check.program, "g++");
        // Artifact-free syntax check over the sorted translation units.
        assert_eq!(
            check.args,
            vec_of(&[
                "-std=c++17",
                "-I.",
                "-fsyntax-only",
                "anagram.cpp",
                "anagram_test.cpp",
            ])
        );
    }

    #[test]
    fn a_makefile_beats_the_cpp_compile_check() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "main.cpp");
        touch(dir.path(), "Makefile");
        // The author's real build (Makefile) wins over the g++ syntax fallback.
        assert_eq!(detect_verify_command(dir.path()).unwrap().program, "make");
    }

    #[test]
    fn a_language_test_runner_beats_the_cpp_compile_check() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "binding.cpp");
        touch(dir.path(), "Cargo.toml");
        // A native test runner outranks the C++ compile fallback.
        assert_eq!(detect_verify_command(dir.path()).unwrap().program, "cargo");
    }

    #[test]
    fn detects_python_by_test_file() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "solution_test.py");
        let check = detect_verify_command(dir.path()).expect("python target");
        assert_eq!(check.args, vec_of(&["-m", "pytest", "-q"]));
    }

    #[test]
    fn override_wins_over_detection() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "Cargo.toml");
        let check = resolve_verify_command(dir.path(), Some("ctest --output-on-failure")).unwrap();
        assert_eq!(check.program, "ctest");
        assert_eq!(check.args, vec_of(&["--output-on-failure"]));
    }

    #[test]
    fn blank_override_falls_back_to_detection() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "Cargo.toml");
        let check = resolve_verify_command(dir.path(), Some("   ")).unwrap();
        assert_eq!(check.program, "cargo");
    }

    #[test]
    fn override_resolves_with_no_detected_target() {
        let dir = tempfile::tempdir().unwrap();
        let check = resolve_verify_command(dir.path(), Some("bash run-tests.sh")).unwrap();
        assert_eq!(check.program, "bash");
        assert_eq!(check.args, vec_of(&["run-tests.sh"]));
    }
}
