//! Cross-platform hardware probe.
//!
//! Implements `localx_llama_core::HardwareProbe` over nvidia-smi (with defensive
//! parsing), plus GPU names and logical-core count. The subprocess is a thin
//! shell; the parsing is pure and unit-tested. Port→PID and process control land
//! in the server-lifecycle module.

use std::process::Command;

use localx_llama_core::HardwareProbe;

/// Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
/// output to the largest GPU's VRAM in MB.
///
/// Returns `None` on empty/garbage output. Multi-GPU takes the max — llama-server
/// runs on one card.
pub fn parse_nvidia_smi_vram_mb(output: &str) -> Option<i64> {
    let max = output
        .lines()
        .filter_map(|l| l.trim().parse::<i64>().ok())
        .filter(|mb| *mb > 0)
        .max();
    max
}

/// The largest GPU's VRAM in whole GB from nvidia-smi memory output.
pub fn parse_nvidia_smi_vram_gb(output: &str) -> Option<i64> {
    parse_nvidia_smi_vram_mb(output).map(|mb| ((mb as f64) / 1024.0).round() as i64)
}

/// Parse `nvidia-smi --query-gpu=name --format=csv,noheader` into GPU names.
pub fn parse_nvidia_smi_names(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

fn run_nvidia_smi(query: &str) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([query, "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The real system hardware probe. Session callers may cache the VRAM result so
/// repeated dashboard renders don't re-invoke nvidia-smi.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemProbe;

impl SystemProbe {
    /// A new probe.
    pub fn new() -> Self {
        Self
    }

    /// GPU names, or empty when nvidia-smi is unavailable.
    pub fn gpu_names(&self) -> Vec<String> {
        run_nvidia_smi("--query-gpu=name")
            .map(|o| parse_nvidia_smi_names(&o))
            .unwrap_or_default()
    }

    /// Logical CPU count (falls back to 1 if the platform can't report it).
    pub fn logical_cores(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

impl HardwareProbe for SystemProbe {
    fn auto_vram_gb(&self) -> Option<i64> {
        run_nvidia_smi("--query-gpu=memory.total").and_then(|o| parse_nvidia_smi_vram_gb(&o))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use localx_llama_core::vram::{resolve_vram, VramSource};

    #[test]
    fn parses_vram_and_takes_max() {
        assert_eq!(parse_nvidia_smi_vram_mb("24564\n"), Some(24564));
        assert_eq!(parse_nvidia_smi_vram_mb("8192\n16384\n"), Some(16384));
        assert_eq!(parse_nvidia_smi_vram_gb("24564\n"), Some(24));
        assert_eq!(parse_nvidia_smi_vram_gb("49140\n"), Some(48));
    }

    #[test]
    fn empty_or_garbage_is_none() {
        assert_eq!(parse_nvidia_smi_vram_mb(""), None);
        assert_eq!(parse_nvidia_smi_vram_mb("\n\n"), None);
        assert_eq!(parse_nvidia_smi_vram_mb("not a number\n"), None);
        assert_eq!(parse_nvidia_smi_vram_mb("0\n"), None);
    }

    #[test]
    fn parses_gpu_names() {
        assert_eq!(
            parse_nvidia_smi_names("NVIDIA GeForce RTX 4090\n NVIDIA A100 \n\n"),
            vec!["NVIDIA GeForce RTX 4090", "NVIDIA A100"]
        );
    }

    #[test]
    fn probe_feeds_the_core_vram_ladder() {
        // A probe with no GPU falls through the core ladder to the fallback.
        struct NoGpu;
        impl HardwareProbe for NoGpu {
            fn auto_vram_gb(&self) -> Option<i64> {
                None
            }
        }
        let info = resolve_vram(None, &NoGpu);
        assert_eq!(info.source, VramSource::Fallback);
        assert_eq!(info.gb, 24);
    }
}
