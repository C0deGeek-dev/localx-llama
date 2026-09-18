//! What a particular `llama-server` build accepts, read from its own `--help`.
//!
//! Builds disagree about launch flags. Mainline replaced `--no-mmap` and
//! `--mlock` with `--load-mode` and now rejects the old spellings; the
//! turboquant fork still has only the old flags; the prism fork carries both.
//! Neither the engine mode nor a build number predicts which, so the answer is
//! read from the help text of the binary that will actually run.

/// The launch options a `llama-server` binary understands.
///
/// The default describes a build that has only the long-standing flags, which
/// is the safe reading when the help text is missing or unrecognised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServerCapabilities {
    /// `--load-mode` exists (the replacement for `--no-mmap` / `--mlock`).
    pub load_mode: bool,
    /// `--fit` exists: the server places layers and MoE experts in free
    /// device memory itself when no placement is given.
    pub fit: bool,
}

impl ServerCapabilities {
    /// Read capabilities from `llama-server --help` output.
    ///
    /// Only option names that introduce a help entry count. A description that
    /// merely mentions a flag (for example "DEPRECATED in favor of
    /// `--load-mode`") is not the flag.
    #[must_use]
    pub fn from_help(help: &str) -> Self {
        Self {
            load_mode: lists_option(help, "--load-mode"),
            fit: lists_option(help, "--fit"),
        }
    }

    /// How this build spells the model-loading choice.
    #[must_use]
    pub fn load_flags(&self) -> LoadFlags {
        if self.load_mode {
            LoadFlags::LoadMode
        } else {
            LoadFlags::Legacy
        }
    }
}

/// How a build spells memory-mapping and memory-locking of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadFlags {
    /// `--no-mmap` and `--mlock`.
    #[default]
    Legacy,
    /// `--load-mode <none|mmap|mlock|mmap+mlock>`.
    LoadMode,
}

/// Whether `option` introduces an entry anywhere in `help`.
fn lists_option(help: &str, option: &str) -> bool {
    help.lines()
        .any(|line| entry_options(line).any(|name| name == option))
}

/// The option names a help line introduces: its leading `-`/`--` tokens,
/// separated by commas or spaces, up to the first token that is not an option
/// (an argument placeholder such as `MODE`, or description text). Continuation
/// lines start with description text and so introduce nothing.
fn entry_options(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|token| !token.is_empty())
        .take_while(|token| token.len() > 1 && token.starts_with('-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Excerpt of mainline `llama-server --help` (build 11034).
    const MAINLINE: &str = "\
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory ('on' or
-lm,   --load-mode MODE                 model loading mode (default: auto)
                                        - auto: mmap, unless a device does not support it
                                        - none: no special loading mode
                                        - mlock: force system to keep model in RAM rather than swapping or
                                        compressing
-lzm,  --lazy-mode MODE                 on-demand reading of certain tensors, for example per-layer embeddings
";

    /// Excerpt of the turboquant fork's help (tqp-v0.3.1): legacy flags only.
    const TURBOQUANT: &str = "\
--mlock                                 force system to keep model in RAM rather than swapping or compressing
                                        (env: LLAMA_ARG_MLOCK)
--mmap, --no-mmap                       whether to memory-map model. (if mmap disabled, slower load but may
                                        reduce pageouts if not using mlock) (default: enabled)
-dio,  --direct-io, -ndio, --no-direct-io
";

    /// Excerpt of the prism fork's help (b10685): both spellings.
    const PRISM: &str = "\
--mlock                                 DEPRECATED in favor of `--load-mode`: force system to keep model in
                                        RAM rather than swapping or compressing
--mmap, --no-mmap                       DEPRECATED in favor of `--load-mode`: whether to memory-map model. (if
-lm,   --load-mode MODE                 model loading mode (default: auto)
";

    #[test]
    fn each_build_reports_its_own_load_flags() {
        assert_eq!(
            ServerCapabilities::from_help(MAINLINE).load_flags(),
            LoadFlags::LoadMode
        );
        assert_eq!(
            ServerCapabilities::from_help(TURBOQUANT).load_flags(),
            LoadFlags::Legacy
        );
        assert_eq!(
            ServerCapabilities::from_help(PRISM).load_flags(),
            LoadFlags::LoadMode
        );
    }

    #[test]
    fn fit_is_read_from_its_own_entry_not_from_fit_target() {
        assert!(ServerCapabilities::from_help(MAINLINE).fit);
        assert!(!ServerCapabilities::from_help(TURBOQUANT).fit);
        let only_target = "-fitt, --fit-target MiB0,MiB1,MiB2,...\n";
        assert!(!ServerCapabilities::from_help(only_target).fit);
    }

    #[test]
    fn a_flag_named_only_in_a_description_is_not_available() {
        let only_mentioned = "\
--mlock                                 DEPRECATED in favor of `--load-mode`: force system to keep model in
                                        see --load-mode for the replacement
";
        assert!(!ServerCapabilities::from_help(only_mentioned).load_mode);
    }

    #[test]
    fn missing_or_unrecognised_help_means_legacy_flags() {
        assert_eq!(
            ServerCapabilities::from_help(""),
            ServerCapabilities::default()
        );
        assert_eq!(
            ServerCapabilities::from_help("error: something went wrong\n").load_flags(),
            LoadFlags::Legacy
        );
    }

    #[test]
    fn entry_options_stop_at_the_first_non_option_token() {
        let names: Vec<&str> =
            entry_options("-fit,  --fit [on|off]                   whether to adjust").collect();
        assert_eq!(names, vec!["-fit", "--fit"]);
        let names: Vec<&str> = entry_options("                    - auto: mmap").collect();
        assert!(names.is_empty());
    }
}
