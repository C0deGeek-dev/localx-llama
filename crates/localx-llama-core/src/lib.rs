//! Pure, host-neutral llama.cpp domain primitives shared across the LocalX stack.
//!
//! No process spawning, no console, no OS-specific I/O in the public API — those
//! live in `localx-llama-runtime`. This crate is the data half of the launcher
//! contract: model definitions, the `llama-server` argv builder, VRAM/quant-fit
//! math, the config-precedence engine, and the tuner/AutoBest schema.
//!
//! Subject 01 fills this in, invariant-first: every carried behaviour lands as a
//! golden test before its logic is considered ported.

#![forbid(unsafe_code)]

/// Placeholder module set — replaced by the real domain in subject 01.
#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        // Readiness smoke: the crate compiles and tests run under MSRV 1.82.
        assert_eq!(2 + 2, 4);
    }
}
