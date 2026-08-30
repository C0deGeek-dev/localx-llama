//! Pin-verified download logic (pure).
//!
//! The invariant-carrying core of install/update: SHA-256 verification, the
//! require-pins posture, per-OS release-asset selection (x64 not the arm64 twin),
//! CUDA-major ordering (driver-major-first, to dodge the garbage-flood mismatch),
//! and `.build-stamp` freshness against the *resolved* tag (never `latest`). The
//! actual HTTP fetch is a thin shell over these decisions.

use sha2::{Digest, Sha256};

use crate::error::RuntimeError;

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Verify a byte slice against an expected hex SHA-256 (case-insensitive).
pub fn verify_sha256(data: &[u8], expected_hex: &str) -> Result<(), RuntimeError> {
    let got = sha256_hex(data);
    if got.eq_ignore_ascii_case(expected_hex.trim()) {
        Ok(())
    } else {
        Err(RuntimeError::ShaMismatch {
            expected: expected_hex.trim().to_ascii_lowercase(),
            got,
        })
    }
}

/// Outcome of applying the download-pin posture to a fetched asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinOutcome {
    /// The asset matched its pin.
    Verified,
    /// No pin configured (allowed because require-pins is off); the computed hash
    /// is surfaced so it can be recorded as a new pin.
    Unpinned {
        /// The computed SHA-256 for the operator to pin.
        computed: String,
    },
}

/// Apply the verify-or-abort / trust-on-first-use / require-pins posture.
///
/// - pin present + match -> [`PinOutcome::Verified`]
/// - pin present + mismatch -> error (delete + refuse upstream)
/// - pin absent + require-pins -> error
/// - pin absent + not required -> [`PinOutcome::Unpinned`] with the computed hash
pub fn check_download_pin(
    data: &[u8],
    pin: Option<&str>,
    require_pins: bool,
) -> Result<PinOutcome, RuntimeError> {
    match pin {
        Some(p) => {
            verify_sha256(data, p)?;
            Ok(PinOutcome::Verified)
        }
        None => {
            let computed = sha256_hex(data);
            if require_pins {
                Err(RuntimeError::PinRequired { computed })
            } else {
                Ok(PinOutcome::Unpinned { computed })
            }
        }
    }
}

fn lower(s: &str) -> String {
    s.to_ascii_lowercase()
}

/// Whether an asset name targets arm64 (must be rejected on an x64 host).
pub fn is_arm64_asset(name: &str) -> bool {
    let l = lower(name);
    l.contains("arm64") || l.contains("aarch64")
}

/// Whether an asset name targets x64.
pub fn is_x64_asset(name: &str) -> bool {
    let l = lower(name);
    (l.contains("x64") || l.contains("x86_64") || l.contains("win64")) && !is_arm64_asset(name)
}

/// Architecture family used when filtering a release's asset list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetArch {
    /// Intel/AMD 64-bit (`x64`, `x86_64`, or `win64`).
    X64,
    /// ARM 64-bit (`arm64` or `aarch64`).
    Arm64,
}

/// Whether an asset name is eligible for the requested architecture.
///
/// Some upstream x64 assets carry no architecture token, so the x64 rule keeps
/// accepting an unmarked name while always rejecting an explicit ARM twin.
pub fn asset_arch_matches(name: &str, arch: AssetArch) -> bool {
    match arch {
        AssetArch::X64 => is_x64_asset(name) || !is_arm64_asset(name),
        AssetArch::Arm64 => is_arm64_asset(name),
    }
}

/// Select the CPU release asset for `arch`, tolerating the upstream
/// `-avx2-`→`-cpu-` rename and fork spellings. The explicit priority keeps the
/// broad AVX2 build ahead of narrower/newer instruction-set variants regardless
/// of GitHub's asset ordering.
///
/// Architecture eligibility goes through [`asset_arch_matches`], so this is the
/// one CPU selector for every host: an x64 pick still accepts an unmarked name
/// and never an ARM twin, and an ARM pick requires an explicit ARM token. A
/// rule that hard-coded x64 would force each caller to keep an ARM copy, which
/// is how the duplicate this replaced came about.
pub fn select_cpu_asset<'a>(names: &[&'a str], arch: AssetArch) -> Option<&'a str> {
    for token in ["-avx2-", "-avx512-", "-avx-", "-noavx-", "-cpu-"] {
        if let Some(hit) = names
            .iter()
            .copied()
            .find(|name| lower(name).contains(token) && asset_arch_matches(name, arch))
        {
            return Some(hit);
        }
    }
    names.iter().copied().find(|name| {
        (lower(name).contains("cpu") || lower(name).contains("avx2"))
            && asset_arch_matches(name, arch)
    })
}

/// Order CUDA major versions so the host driver's major is tried first, then the
/// rest descending. A CUDA build whose major exceeds the driver's emits a garbage
/// flood rather than erroring, so matching the driver comes first.
pub fn cuda_major_order(driver_major: u32, available: &[u32]) -> Vec<u32> {
    let mut rest: Vec<u32> = available
        .iter()
        .copied()
        .filter(|m| *m != driver_major)
        .collect();
    rest.sort_unstable_by(|a, b| b.cmp(a));
    let mut out = Vec::with_capacity(available.len());
    if available.contains(&driver_major) {
        out.push(driver_major);
    }
    out.extend(rest);
    out
}

/// Whether a `.build-stamp` is stale against the **resolved** (pinned) tag.
///
/// Compared to the resolved tag, never `latest` — otherwise a pinned build reads
/// as perpetually stale and re-downloads the pin it already has.
pub fn build_stamp_is_stale(stamp: &str, resolved_tag: &str) -> bool {
    stamp.trim() != resolved_tag.trim()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        // SHA-256("abc")
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn pin_verify_and_mismatch() {
        let data = b"hello";
        let good = sha256_hex(data);
        assert!(verify_sha256(data, &good).is_ok());
        assert!(verify_sha256(data, &good.to_uppercase()).is_ok());
        let err = verify_sha256(data, "deadbeef").unwrap_err();
        assert!(matches!(err, RuntimeError::ShaMismatch { .. }));
    }

    #[test]
    fn pin_posture() {
        let data = b"asset-bytes";
        let hash = sha256_hex(data);
        assert_eq!(
            check_download_pin(data, Some(&hash), true).unwrap(),
            PinOutcome::Verified
        );
        // absent pin + require -> error with computed hash.
        assert!(matches!(
            check_download_pin(data, None, true).unwrap_err(),
            RuntimeError::PinRequired { .. }
        ));
        // absent pin + not required -> unpinned with computed hash.
        assert_eq!(
            check_download_pin(data, None, false).unwrap(),
            PinOutcome::Unpinned { computed: hash }
        );
        // wrong pin for the tag can never match.
        assert!(check_download_pin(data, Some("00"), false).is_err());
    }

    #[test]
    fn cpu_asset_selection_picks_x64_not_arm64() {
        let names = [
            "llama-b9596-bin-win-cpu-arm64.zip",
            "llama-b9596-bin-win-cpu-x64.zip",
            "llama-b9596-bin-win-cuda-x64.zip",
        ];
        assert_eq!(
            select_cpu_asset(&names, AssetArch::X64),
            Some("llama-b9596-bin-win-cpu-x64.zip")
        );
        // legacy -avx2- naming still matches.
        assert_eq!(
            select_cpu_asset(&["llama-bin-win-avx2-x64.zip"], AssetArch::X64),
            Some("llama-bin-win-avx2-x64.zip")
        );
        assert!(is_arm64_asset("foo-arm64.zip"));
        assert!(!is_x64_asset("foo-arm64.zip"));
    }

    #[test]
    fn cpu_asset_selection_preserves_priority_and_tolerates_fork_renames() {
        let names = [
            "llama-bin-win-cpu-x64.zip",
            "llama-bin-win-avx512-x64.zip",
            "llama-bin-win-avx2-x64.zip",
        ];
        assert_eq!(
            select_cpu_asset(&names, AssetArch::X64),
            Some("llama-bin-win-avx2-x64.zip")
        );
        assert_eq!(
            select_cpu_asset(&["fork-windows-x64-cpu.zip"], AssetArch::X64),
            Some("fork-windows-x64-cpu.zip")
        );
        assert_eq!(
            select_cpu_asset(&["fork-windows-arm64-cpu.zip"], AssetArch::X64),
            None
        );
    }

    /// The same rule serves ARM hosts. Before this took an architecture, an
    /// x64-only selector left every ARM caller carrying its own copy, and that
    /// copy did not tolerate the fork rename.
    #[test]
    fn the_same_selector_serves_arm_hosts_including_fork_renames() {
        let names = ["llama-bin-win-cpu-arm64.zip", "llama-bin-win-avx2-x64.zip"];
        assert_eq!(
            select_cpu_asset(&names, AssetArch::Arm64),
            Some("llama-bin-win-cpu-arm64.zip")
        );
        assert_eq!(
            select_cpu_asset(&["fork-windows-arm64-cpu.zip"], AssetArch::Arm64),
            Some("fork-windows-arm64-cpu.zip")
        );
        // An unmarked name is an x64 build; it is never an ARM pick.
        assert_eq!(
            select_cpu_asset(&["legacy-cpu.zip"], AssetArch::Arm64),
            None
        );
    }

    #[test]
    fn architecture_matching_keeps_unmarked_x64_compatibility() {
        assert!(asset_arch_matches("legacy-cpu.zip", AssetArch::X64));
        assert!(asset_arch_matches("build-win64.zip", AssetArch::X64));
        assert!(!asset_arch_matches("build-arm64.zip", AssetArch::X64));
        assert!(asset_arch_matches("build-aarch64.tar.gz", AssetArch::Arm64));
        assert!(!asset_arch_matches("legacy-cpu.zip", AssetArch::Arm64));
    }

    #[test]
    fn cuda_order_is_driver_major_first() {
        assert_eq!(cuda_major_order(13, &[11, 12, 13]), vec![13, 12, 11]);
        assert_eq!(cuda_major_order(13, &[11, 12]), vec![12, 11]);
        assert_eq!(cuda_major_order(12, &[12]), vec![12]);
    }

    #[test]
    fn build_stamp_compares_to_resolved_tag() {
        // pinned build present: stamp == resolved -> not stale.
        assert!(!build_stamp_is_stale("b9596", "b9596"));
        // a different resolved tag -> stale (must re-fetch).
        assert!(build_stamp_is_stale("b9596", "b9700"));
        // the key invariant: pinned b9596 must NOT read stale vs latest b9856,
        // because we compare to the resolved (pinned) tag, not latest.
        let resolved = "b9596"; // resolver returned the pin, not latest
        assert!(!build_stamp_is_stale("b9596", resolved));
    }
}
