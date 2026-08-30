//! VRAM detection abstraction, quant-fit classification, and KV-cache context math.
//!
//! Pure heuristics ported from the launcher. Hardware access is injected via the
//! [`HardwareProbe`] trait so this crate stays GPU-free and unit-testable; the
//! real nvidia-smi/NVML probe lives in `localx-llama-runtime`.

/// Injected VRAM source. The runtime crate implements this over NVML/nvidia-smi;
/// tests implement it with a fixed value.
pub trait HardwareProbe {
    /// The largest GPU's VRAM in GB, or `None` if detection is unavailable.
    ///
    /// llama-server runs on one card, so multi-GPU hosts report the max.
    fn auto_vram_gb(&self) -> Option<i64>;
}

/// Where a resolved VRAM figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// From explicit configuration.
    Configured,
    /// Auto-detected via the hardware probe.
    Auto,
    /// The hard fallback used when nothing else is available.
    Fallback,
}

/// A resolved VRAM figure and its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramInfo {
    /// VRAM in GB.
    pub gb: i64,
    /// How it was resolved.
    pub source: VramSource,
}

/// The hard fallback VRAM when neither config nor detection yields a value.
pub const FALLBACK_VRAM_GB: i64 = 24;

/// Resolve VRAM: configured (>0) -> auto-detect -> fallback of 24 GB.
///
/// A non-positive configured value is ignored (treated as unset), matching the
/// launcher's tolerant read.
pub fn resolve_vram(configured: Option<i64>, probe: &dyn HardwareProbe) -> VramInfo {
    if let Some(v) = configured {
        if v > 0 {
            return VramInfo {
                gb: v,
                source: VramSource::Configured,
            };
        }
    }
    if let Some(v) = probe.auto_vram_gb() {
        if v > 0 {
            return VramInfo {
                gb: v,
                source: VramSource::Auto,
            };
        }
    }
    VramInfo {
        gb: FALLBACK_VRAM_GB,
        source: VramSource::Fallback,
    }
}

/// How well a quant fits available VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitClass {
    /// >= 7 GB headroom (room for KV cache + overhead).
    Fits,
    /// >= 2 GB headroom (fine at short context).
    Tight,
    /// Won't fit.
    Over,
    /// Size unknown — never guessed.
    Unknown,
}

impl FitClass {
    /// Lowercase label used in pickers.
    pub fn as_str(self) -> &'static str {
        match self {
            FitClass::Fits => "fits",
            FitClass::Tight => "tight",
            FitClass::Over => "over",
            FitClass::Unknown => "",
        }
    }
}

/// Classify a quant's on-disk size against VRAM.
///
/// `fits` needs >= 7 GB headroom, `tight` >= 2 GB, else `over`. Unknown size
/// yields [`FitClass::Unknown`] — never a guess.
pub fn quant_fit_class(size_gb: Option<f64>, vram_gb: i64) -> FitClass {
    let Some(size) = size_gb else {
        return FitClass::Unknown;
    };
    let vram = vram_gb as f64;
    if size <= vram - 7.0 {
        FitClass::Fits
    } else if size <= vram - 2.0 {
        FitClass::Tight
    } else {
        FitClass::Over
    }
}

/// The largest `num_ctx` safe to combine with a q8_0 KV cache.
///
/// Each GB above ~16 is worth ~16k q8 tokens; floors at 64k.
///
/// Library-only / not yet wired: no consumer computes KV-cache pressure yet
/// (LocalBox's fit hint is weight-only). Kept for a future KV-aware fit; wire a
/// consumer or remove it if that never lands.
pub fn q8_kv_max_context(vram_gb: i64) -> i64 {
    const FLOOR: i64 = 65_536;
    const PER_GB: i64 = 16_384;
    let scaled = (vram_gb - 16) * PER_GB;
    scaled.max(FLOOR)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    struct FixedProbe(Option<i64>);
    impl HardwareProbe for FixedProbe {
        fn auto_vram_gb(&self) -> Option<i64> {
            self.0
        }
    }

    #[test]
    fn fit_boundaries_on_24gb() {
        // fits needs <= 24-7 = 17; tight <= 24-2 = 22; else over.
        assert_eq!(quant_fit_class(Some(17.0), 24), FitClass::Fits);
        assert_eq!(quant_fit_class(Some(17.1), 24), FitClass::Tight);
        assert_eq!(quant_fit_class(Some(22.0), 24), FitClass::Tight);
        assert_eq!(quant_fit_class(Some(22.1), 24), FitClass::Over);
    }

    #[test]
    fn fit_flips_to_fits_on_bigger_card() {
        // A 30 GB quant is 'over' on 24 GB but 'fits' on 48 GB.
        assert_eq!(quant_fit_class(Some(30.0), 24), FitClass::Over);
        assert_eq!(quant_fit_class(Some(30.0), 48), FitClass::Fits);
    }

    #[test]
    fn unknown_size_is_unknown_not_a_guess() {
        assert_eq!(quant_fit_class(None, 24), FitClass::Unknown);
        assert_eq!(FitClass::Unknown.as_str(), "");
    }

    #[test]
    fn q8_kv_context_scales_and_floors() {
        assert_eq!(q8_kv_max_context(16), 65_536);
        assert_eq!(q8_kv_max_context(24), 131_072);
        assert_eq!(q8_kv_max_context(48), 524_288);
        // below 16 GB still floors, never negative.
        assert_eq!(q8_kv_max_context(8), 65_536);
    }

    #[test]
    fn vram_resolution_ladder() {
        assert_eq!(
            resolve_vram(Some(48), &FixedProbe(Some(24))),
            VramInfo {
                gb: 48,
                source: VramSource::Configured
            }
        );
        assert_eq!(
            resolve_vram(None, &FixedProbe(Some(24))),
            VramInfo {
                gb: 24,
                source: VramSource::Auto
            }
        );
        assert_eq!(
            resolve_vram(None, &FixedProbe(None)),
            VramInfo {
                gb: 24,
                source: VramSource::Fallback
            }
        );
        // non-positive configured is ignored -> falls through to auto.
        assert_eq!(
            resolve_vram(Some(0), &FixedProbe(Some(12))),
            VramInfo {
                gb: 12,
                source: VramSource::Auto
            }
        );
    }
}
