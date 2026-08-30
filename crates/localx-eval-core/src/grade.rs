//! Grade fidelity — the anti-false-pass core of the benchmark grader.
//!
//! Two load-bearing rules ported from the aider grading fidelity work:
//! 1. A task is graded **passed only when exit code is 0 AND `tests_run > 0`** —
//!    a compile-with-zero-tests must never score as solved.
//! 2. Counts are parsed from the FULL captured output, never a truncated
//!    display tail — Rust prints a `test result:` line per binary (lib / each
//!    integration / doc-tests) and the *last* lines are usually the doc-test
//!    `0 passed`, so the count is the **sum across every result line**, of
//!    tests that RAN (passed + failed).
//!
//! Unrecognized output fails closed to zero tests.

use regex::Regex;
use std::sync::OnceLock;

/// The languages the grader knows how to count tests for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    /// Rust (`cargo test`).
    Rust,
    /// Python (`unittest`/`pytest` "Ran N tests").
    Python,
    /// C++ (catch2).
    Cpp,
    /// JavaScript (jest).
    Javascript,
    /// Go (`go test`).
    Go,
    /// Java (Gradle).
    Java,
}

/// The result of grading a task run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradeOutcome {
    /// Whether the task is graded as solved.
    pub passed: bool,
    /// Number of tests actually executed (0 fails the gate).
    pub tests_run: u32,
}

fn regex_of(cell: &'static OnceLock<Option<Regex>>, pattern: &str) -> &'static Option<Regex> {
    cell.get_or_init(|| Regex::new(pattern).ok())
}

fn rust_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // `test result: ok. N passed; M failed; ...` per binary — sum N+M over all.
    regex_of(
        &RE,
        r"test result:\s+\S+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed",
    )
}

fn python_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    regex_of(&RE, r"Ran\s+(\d+)\s+tests?\b")
}

fn cpp_pass_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // catch2 pass summary: `All tests passed (... in N test cases)`.
    regex_of(&RE, r"in\s+(\d+)\s+test cases?")
}

fn cpp_fail_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // catch2 failure summary: `test cases: N | ...`.
    regex_of(&RE, r"(?m)^test cases:\s+(\d+)")
}

fn javascript_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // jest summary: `Tests: [X failed,] Y passed, Z total`.
    regex_of(&RE, r"Tests:[^\r\n]*?(\d+)\s+total")
}

fn go_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // A package that ran tests is `ok <pkg>` or `FAIL <pkg>`; a package with
    // none is `? <pkg> [no test files]`.
    regex_of(&RE, r"(?m)^(ok|FAIL)\s+\S")
}

fn java_skip_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    regex_of(&RE, r"(?m)^>?\s*Task :test (NO-SOURCE|SKIPPED)")
}

fn generic_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    regex_of(&RE, r"(\d+)\s+passed")
}

fn first_capture(re: &Option<Regex>, output: &str) -> Option<u32> {
    re.as_ref()?
        .captures(output)?
        .get(1)?
        .as_str()
        .parse::<u32>()
        .ok()
}

/// Count tests executed in the tool output for a language. Fails closed to 0.
pub fn count_tests(lang: Lang, output: &str) -> u32 {
    match lang {
        Lang::Rust => rust_re()
            .as_ref()
            .map(|re| {
                // SUM every `test result:` line — not the last (doc-test 0 passed) —
                // counting tests that RAN (passed + failed).
                re.captures_iter(output)
                    .filter_map(|c| {
                        let passed = c.get(1)?.as_str().parse::<u32>().ok()?;
                        let failed = c.get(2)?.as_str().parse::<u32>().ok()?;
                        Some(passed + failed)
                    })
                    .sum()
            })
            .unwrap_or(0),
        Lang::Python => first_capture(python_re(), output).unwrap_or(0),
        Lang::Cpp => first_capture(cpp_pass_re(), output)
            .or_else(|| first_capture(cpp_fail_re(), output))
            .unwrap_or(0),
        Lang::Javascript => first_capture(javascript_re(), output).unwrap_or(0),
        Lang::Go => go_re()
            .as_ref()
            .map(|re| re.find_iter(output).count() as u32)
            .unwrap_or(0),
        Lang::Java => {
            // Gradle prints no test count; `BUILD SUCCESSFUL` with the test task
            // actually executed (not NO-SOURCE / SKIPPED) means tests ran
            // (floor 1); an explicit no-source/skipped test task is the
            // zero-test signal.
            if !output.contains("BUILD SUCCESSFUL") {
                return 0;
            }
            let skipped = java_skip_re()
                .as_ref()
                .is_some_and(|re| re.is_match(output));
            u32::from(!skipped)
        }
    }
}

/// Best-effort test count for an unrecognized language: the sum of every
/// `N passed` occurrence; 0 (fail closed) when none.
pub fn count_tests_generic(output: &str) -> u32 {
    generic_re()
        .as_ref()
        .map(|re| {
            re.captures_iter(output)
                .filter_map(|c| c.get(1)?.as_str().parse::<u32>().ok())
                .sum()
        })
        .unwrap_or(0)
}

/// Grade a task run: passed iff exit code is 0 **and** at least one test ran.
pub fn grade(lang: Lang, exit_code: i32, output: &str) -> GradeOutcome {
    let tests_run = count_tests(lang, output);
    GradeOutcome {
        passed: exit_code == 0 && tests_run > 0,
        tests_run,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn rust_sums_every_result_line_not_the_doctest_tail() {
        // Two real suites (3 + 5) plus the trailing 0-passed doc-test line.
        let out = "\
test result: ok. 3 passed; 0 failed; 0 ignored
test result: ok. 5 passed; 0 failed; 0 ignored
   Doc-tests foo
test result: ok. 0 passed; 0 failed; 0 ignored";
        assert_eq!(count_tests(Lang::Rust, out), 8); // 3 + 5 + 0, not the tail's 0
    }

    #[test]
    fn rust_counts_tests_that_ran_not_only_passes() {
        // 2 passed + 3 failed = 5 tests RAN; the exit code decides the verdict.
        let out = "test result: FAILED. 2 passed; 3 failed; 0 ignored";
        assert_eq!(count_tests(Lang::Rust, out), 5);
        assert!(!grade(Lang::Rust, 101, out).passed);
    }

    #[test]
    fn exit_zero_with_no_tests_is_not_a_pass() {
        // Compiles, exit 0, but ran nothing -> must fail the gate.
        let g = grade(Lang::Rust, 0, "Compiling foo\nFinished");
        assert_eq!(g.tests_run, 0);
        assert!(!g.passed);
    }

    #[test]
    fn passing_run_is_graded_solved() {
        let g = grade(Lang::Rust, 0, "test result: ok. 4 passed; 0 failed");
        assert!(g.passed);
        assert_eq!(g.tests_run, 4);
        // non-zero exit never passes even with tests.
        assert!(!grade(Lang::Rust, 1, "test result: ok. 4 passed; 0 failed").passed);
    }

    #[test]
    fn python_go_java_counts() {
        assert_eq!(count_tests(Lang::Python, "Ran 12 tests in 0.03s"), 12);
        assert_eq!(
            count_tests(Lang::Go, "ok  example/pkg 0.2s\nok  example/two 0.1s"),
            2
        );
        // A FAILing package still RAN its tests.
        assert_eq!(
            count_tests(Lang::Go, "ok  example/pkg 0.2s\nFAIL example/two 0.1s"),
            2
        );
        assert_eq!(count_tests(Lang::Go, "?   example/pkg [no test files]"), 0);
        assert_eq!(count_tests(Lang::Java, "BUILD SUCCESSFUL in 3s"), 1);
        assert_eq!(count_tests(Lang::Java, "BUILD FAILED"), 0);
        assert_eq!(
            count_tests(Lang::Java, "> Task :test NO-SOURCE\nBUILD SUCCESSFUL in 1s"),
            0
        );
        assert_eq!(
            count_tests(Lang::Java, "> Task :test SKIPPED\nBUILD SUCCESSFUL in 1s"),
            0
        );
    }

    #[test]
    fn cpp_and_javascript_counts() {
        assert_eq!(
            count_tests(
                Lang::Cpp,
                "All tests passed (12 assertions in 4 test cases)"
            ),
            4
        );
        assert_eq!(
            count_tests(Lang::Cpp, "test cases: 6 | 4 passed | 2 failed"),
            6
        );
        assert_eq!(
            count_tests(
                Lang::Javascript,
                "Tests:       1 failed, 17 passed, 18 total"
            ),
            18
        );
        assert_eq!(count_tests(Lang::Javascript, "No tests found"), 0);
    }

    #[test]
    fn generic_fallback_sums_passed_counts() {
        assert_eq!(
            count_tests_generic("suite a: 3 passed\nsuite b: 4 passed"),
            7
        );
        assert_eq!(count_tests_generic("nothing recognisable"), 0);
    }

    #[test]
    fn unrecognized_output_fails_closed() {
        assert_eq!(count_tests(Lang::Rust, "gibberish"), 0);
        assert!(!grade(Lang::Python, 0, "no test marker here").passed);
    }
}
