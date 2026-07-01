//! Process and network side of the llama.cpp stack, shared across LocalX.
//!
//! Everything with I/O sits behind traits so the crate stays cross-platform:
//! `llama-server` lifecycle, pin-verified install/download with CUDA-driver
//! matching, the CPU-only embed-serve sibling, and the no-think streaming filter
//! (an in-process replacement for the python sidecar — see subject 00.5, which
//! confirmed axum/hyper build on MSRV 1.82).
//!
//! Subject 02 fills this in; hardware/process/port access is injected per-OS.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        assert_eq!(2 + 2, 4);
    }
}
