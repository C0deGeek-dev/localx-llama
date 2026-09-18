//! llama.cpp's own memory fitter as an oracle.
//!
//! `llama-fit-params` computes, without loading any tensor data, how many
//! layers — and which MoE expert tensors — fit in free device memory for a
//! given model and launch shape, and prints the resulting CLI arguments. It is
//! the allocator's own arithmetic, so it answers in seconds what a search
//! would otherwise learn by starting servers until one runs out of memory.
//!
//! This module is the pure half: which launch flags the fitter must see
//! ([`fit_params_args`]) and what its output means ([`parse_fit_output`]).
//! Running the tool is `localx_llama_runtime::fit`.

use std::fmt;

/// Server flags (each followed by a value) that shape device memory and that
/// `llama-fit-params` accepts. Everything else in a server argv is dropped.
const MEMORY_FLAGS_WITH_VALUE: &[&str] = &[
    "-m",
    "--model",
    "-c",
    "--ctx-size",
    "-np",
    "--parallel",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-fa",
    "--flash-attn",
    "-ub",
    "--ubatch-size",
    "-b",
    "--batch-size",
    "-lm",
    "--load-mode",
    "-lzm",
    "--lazy-mode",
    "-sm",
    "--split-mode",
    "-ts",
    "--tensor-split",
    "-mg",
    "--main-gpu",
];

/// Memory-shaping switches (no value) that `llama-fit-params` accepts.
const MEMORY_SWITCHES: &[&str] = &[
    "--swa-full",
    "--no-mmap",
    "--mlock",
    "-kvu",
    "--kv-unified",
    "-nkvo",
    "--no-kv-offload",
];

/// Placement flags the fitter decides itself. Passing any of them makes it
/// keep that value instead of fitting, so they are always removed.
const PLACEMENT_FLAGS: &[&str] = &[
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
    "-ncmoe",
    "--n-cpu-moe",
    "-ot",
    "--override-tensor",
];

/// The `llama-fit-params` arguments for a server launch.
///
/// Built from the exact argv the server would receive, so the fitter sees the
/// same model, context, KV types, batch shape, and slot count. Placement flags
/// (`-ngl`, `--n-cpu-moe`, `-ot`) are removed — the fitter keeps any value it
/// is given rather than fitting it — as are server-only flags it rejects.
/// `margin_mib` is the free VRAM to leave per device (`--fit-target`).
///
/// The fitter cannot account for a vision projector or a draft model (it
/// rejects `--mmproj` and `--spec-*`); callers widen `margin_mib` by their
/// size instead.
#[must_use]
pub fn fit_params_args(server_argv: &[String], margin_mib: u32) -> Vec<String> {
    let mut out = Vec::new();
    let mut tokens = server_argv.iter().peekable();
    while let Some(token) = tokens.next() {
        let flag = token.as_str();
        if MEMORY_FLAGS_WITH_VALUE.contains(&flag) {
            out.push(token.clone());
            if let Some(value) = tokens.next() {
                out.push(value.clone());
            }
        } else if MEMORY_SWITCHES.contains(&flag) {
            out.push(token.clone());
        } else if is_flag(flag) || PLACEMENT_FLAGS.contains(&flag) {
            // Dropped, and so is its value when it has one.
            if tokens.peek().is_some_and(|next| !is_flag(next)) {
                tokens.next();
            }
        }
    }
    out.push("--fit-target".to_string());
    out.push(margin_mib.to_string());
    out
}

/// Whether a token is an option (`-x`, `--xyz`) rather than a value. A
/// negative number such as `-1` is a value.
fn is_flag(token: &str) -> bool {
    let mut chars = token.chars();
    chars.next() == Some('-')
        && chars
            .next()
            .is_some_and(|c| c == '-' || c.is_ascii_alphabetic())
}

/// Where the fitter put the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitPlacement {
    /// The context size it fitted for (the requested one when it was given).
    pub context: Option<i64>,
    /// Layers on the GPU; `-1` means every layer.
    pub gpu_layers: i64,
    /// Blocks whose MoE expert tensors stay in system memory, including a
    /// boundary block the fitter split. Empty when everything fits.
    pub cpu_expert_blocks: Vec<u32>,
    /// The fitter's raw `-ot` value, when it overrode any tensor placement.
    pub tensor_overrides: Option<String>,
    /// Per-device outcome from the fitter's log, when it printed one.
    pub devices: Vec<DeviceFit>,
}

impl FitPlacement {
    /// The equivalent `--n-cpu-moe`: how many blocks' experts live in system
    /// memory. The fitter offloads the *last* blocks and `--n-cpu-moe` the
    /// *first*; the count, which is what memory depends on, is the same.
    #[must_use]
    pub fn n_cpu_moe(&self) -> i64 {
        i64::try_from(self.cpu_expert_blocks.len()).unwrap_or(i64::MAX)
    }
}

/// One device's share of a fitted placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFit {
    /// The device label as the fitter printed it, e.g. `CUDA0 (NVIDIA GeForce RTX 4090)`.
    pub name: String,
    /// Layers placed on this device.
    pub layers: u32,
    /// Of those, layers with tensors spilled to system memory.
    pub overflowing: u32,
    /// Projected device memory in use.
    pub used_mib: u64,
    /// Projected device memory left free.
    pub free_mib: u64,
}

/// Why fitter output could not be read as a placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The fitter reported a failure (for example the model would not load).
    Failed(String),
    /// Standard output did not contain a fitted argument line.
    Unrecognised,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed(reason) => write!(f, "llama-fit-params failed: {reason}"),
            Self::Unrecognised => f.write_str("llama-fit-params output was not recognised"),
        }
    }
}

impl std::error::Error for FitError {}

/// Read `llama-fit-params` output: `stdout` carries the fitted arguments,
/// `log` (its stderr) the per-device summary and any error.
///
/// # Errors
/// [`FitError::Failed`] with the first error line when the fitter failed;
/// [`FitError::Unrecognised`] when no `-ngl` value was printed.
pub fn parse_fit_output(stdout: &str, log: &str) -> Result<FitPlacement, FitError> {
    let tokens = split_args(stdout);
    let value_of = |names: &[&str]| {
        tokens
            .iter()
            .position(|t| names.contains(&t.as_str()))
            .and_then(|i| tokens.get(i + 1))
    };
    let Some(gpu_layers) = value_of(&["-ngl"]).and_then(|v| v.parse::<i64>().ok()) else {
        return Err(first_error(log).map_or(FitError::Unrecognised, FitError::Failed));
    };
    let context = value_of(&["-c"]).and_then(|v| v.parse::<i64>().ok());
    let tensor_overrides = value_of(&["-ot"]).cloned();
    let cpu_expert_blocks = tensor_overrides
        .as_deref()
        .map(cpu_blocks)
        .unwrap_or_default();
    Ok(FitPlacement {
        context,
        gpu_layers,
        cpu_expert_blocks,
        tensor_overrides,
        devices: device_fits(log),
    })
}

/// Split a printed argument line, honouring double quotes around a value.
fn split_args(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for c in line.trim().chars() {
        match c {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Block indices named by `-ot` entries that send tensors to the CPU,
/// e.g. `blk\.14\.ffn_(gate|up|gate_up|down).*=CPU`.
fn cpu_blocks(overrides: &str) -> Vec<u32> {
    let mut blocks: Vec<u32> = overrides
        .split(',')
        .filter(|entry| entry.trim_end().ends_with("=CPU"))
        .filter_map(|entry| {
            let rest = entry.trim().strip_prefix("blk\\.")?;
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    blocks
}

/// The last summary line per device, e.g.
/// `  - CUDA0 (NVIDIA GeForce RTX 4090): 49 layers (35 overflowing),  21092 MiB used,   1781 MiB free`.
fn device_fits(log: &str) -> Vec<DeviceFit> {
    let mut devices: Vec<DeviceFit> = Vec::new();
    for line in log.lines() {
        let Some(fit) = device_fit(line) else {
            continue;
        };
        match devices.iter_mut().find(|d| d.name == fit.name) {
            Some(existing) => *existing = fit,
            None => devices.push(fit),
        }
    }
    devices
}

fn device_fit(line: &str) -> Option<DeviceFit> {
    let (_, rest) = line.split_once(" - ")?;
    let (name, rest) = rest.split_once("): ")?;
    let name = format!("{name})");
    let (layers_part, rest) = rest.split_once(" layers")?;
    let layers = layers_part.trim().parse().ok()?;
    let overflowing = rest
        .split_once('(')
        .and_then(|(_, r)| r.split_once(" overflowing"))
        .and_then(|(n, _)| n.trim().parse().ok())
        .unwrap_or(0);
    let used_mib = number_before(rest, " MiB used")?;
    let free_mib = number_before(rest, " MiB free")?;
    Some(DeviceFit {
        name,
        layers,
        overflowing,
        used_mib,
        free_mib,
    })
}

fn number_before(text: &str, marker: &str) -> Option<u64> {
    let (before, _) = text.split_once(marker)?;
    before
        .rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|n| n.parse().ok())
}

/// The first error line the fitter logged, without its timestamp prefix.
fn first_error(log: &str) -> Option<String> {
    log.lines().find(|line| line.contains(" E ")).map(|line| {
        line.split_once(" E ")
            .map_or(line, |(_, msg)| msg)
            .trim()
            .to_string()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn argv(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn the_fitter_sees_the_memory_shape_but_never_a_placement() {
        let server = argv(
            "-m model.gguf -c 262144 --host 127.0.0.1 --port 8080 --parallel 1 --cache-reuse 256 \
             -ngl 999 --n-cpu-moe 38 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
             --jinja --reasoning off --reasoning-budget 0 --reasoning-format none --temp 0.7 \
             --threads 12 --threads-batch 16 --load-mode none -ot blk\\.1\\.ffn=CPU",
        );
        assert_eq!(
            fit_params_args(&server, 1536),
            argv(
                "-m model.gguf -c 262144 --parallel 1 --flash-attn on --cache-type-k q8_0 \
                 --cache-type-v q8_0 --load-mode none --fit-target 1536"
            )
        );
    }

    #[test]
    fn a_negative_value_is_dropped_with_its_flag() {
        let server = argv("-m m.gguf --parallel -1 -ngl -1 --no-mmap");
        assert_eq!(
            fit_params_args(&server, 1024),
            argv("-m m.gguf --parallel -1 --no-mmap --fit-target 1024")
        );
    }

    /// Stdout of a native build fitting a 48-block MoE at 64k (abridged `-ot`).
    const MOE_STDOUT: &str = r#"-c 65536 -ngl 49 -ot "blk\.14\.ffn_(gate|up|gate_up|down).*=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU""#;

    const MOE_LOG: &str = "\
0.01.942.423 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 4090): 49 layers,   7687 MiB used,  15186 MiB free
0.04.290.154 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 4090): 49 layers (35 overflowing),  21092 MiB used,   1781 MiB free
0.04.290.157 I common_fit_params: successfully fit params to free device memory
";

    #[test]
    fn a_moe_fit_names_the_blocks_whose_experts_stay_on_the_cpu() {
        let fit = parse_fit_output(MOE_STDOUT, MOE_LOG).unwrap();
        assert_eq!(fit.context, Some(65536));
        assert_eq!(fit.gpu_layers, 49);
        assert_eq!(fit.cpu_expert_blocks, vec![14, 15, 16]);
        assert_eq!(fit.n_cpu_moe(), 3);
        assert!(fit.tensor_overrides.unwrap().starts_with("blk\\.14\\."));
        assert_eq!(
            fit.devices,
            vec![DeviceFit {
                name: "CUDA0 (NVIDIA GeForce RTX 4090)".to_string(),
                layers: 49,
                overflowing: 35,
                used_mib: 21092,
                free_mib: 1781,
            }]
        );
    }

    #[test]
    fn a_model_that_fits_whole_offloads_every_layer() {
        let fit = parse_fit_output("-c 65536 -ngl -1\n", "").unwrap();
        assert_eq!(fit.gpu_layers, -1);
        assert!(fit.cpu_expert_blocks.is_empty());
        assert_eq!(fit.n_cpu_moe(), 0);
        assert!(fit.devices.is_empty());
    }

    #[test]
    fn a_failed_fit_reports_the_fitter_error() {
        let log = "\
0.00.105.908 E gguf_init_from_reader: this file matches the legacy Prism Q2_0 layout
0.00.111.928 E llama_fit_params: failed to fit CLI arguments to free memory, exiting...
";
        assert_eq!(
            parse_fit_output("", log),
            Err(FitError::Failed(
                "gguf_init_from_reader: this file matches the legacy Prism Q2_0 layout".to_string()
            ))
        );
        assert_eq!(
            parse_fit_output("nonsense", ""),
            Err(FitError::Unrecognised)
        );
    }
}
