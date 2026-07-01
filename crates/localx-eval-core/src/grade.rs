//! Grade fidelity — the anti-false-pass core of the benchmark grader.
//!
//! Two load-bearing rules ported from the aider grading fidelity work:
//! 1. A task is graded **passed only when exit code is 0 AND `tests_run > 0`** —
//!    a compile-with-zero-tests must never score as solved.
//! 2. Rust counts must **sum every `test result:` line**, not the display tail
//!    (the tail is usually the `0 passed` doc-test line).
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

fn rust_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"test result:\s+\w+\.\s+(\d+)\s+passed").ok())
}

fn python_re() -> &'static Option<Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"Ran\s+(\d+)\s+tests?").ok())
}

/// Count tests executed in the tool output for a language. Fails closed to 0.
pub fn count_tests(lang: Lang, output: &str) -> u32 {
    match lang {
        Lang::Rust => rust_re()
            .as_ref()
            .map(|re| {
                // SUM every `test result:` line — not the last (doc-test 0 passed).
                re.captures_iter(output)
                    .filter_map(|c| c.get(1))
                    .filter_map(|m| m.as_str().parse::<u32>().ok())
                    .sum()
            })
            .unwrap_or(0),
        Lang::Python => python_re()
            .as_ref()
            .and_then(|re| re.captures(output))
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse::<u32>().ok())
            .unwrap_or(0),
        Lang::Go => output
            .lines()
            .filter(|l| {
                let t = l.trim_start();
                t.starts_with("ok ") && !t.contains("[no test files]")
            })
            .count() as u32,
        Lang::Java => {
            let ok = output.contains("BUILD SUCCESSFUL")
                && !output.contains("NO-SOURCE")
                && !output.contains("BUILD FAILED");
            u32::from(ok)
        }
    }
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
        assert_eq!(count_tests(Lang::Go, "?   example/pkg [no test files]"), 0);
        assert_eq!(count_tests(Lang::Java, "BUILD SUCCESSFUL in 3s"), 1);
        assert_eq!(count_tests(Lang::Java, "BUILD FAILED"), 0);
        assert_eq!(count_tests(Lang::Java, "> Task :test NO-SOURCE"), 0);
    }

    #[test]
    fn unrecognized_output_fails_closed() {
        assert_eq!(count_tests(Lang::Rust, "gibberish"), 0);
        assert!(!grade(Lang::Python, 0, "no test marker here").passed);
    }
}
