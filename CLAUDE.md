# CLAUDE.md — localx-llama

Shared Rust crate tier reused by LocalBox, LocalBench, and LocalPilot via
**rev-pinned git dependencies** (not a submodule). Planning lives in the private
LocalHub repo; this repo holds only its README and code.

## Non-negotiables

- **Pure vs I/O crate split is mandatory.** `localx-llama-core` is pure domain
  (no I/O — argv builder, VRAM math, config precedence, schemas). All process/
  network I/O lives in `localx-llama-runtime` behind traits (`HardwareProbe`,
  `Upstream`, `Launcher`, `CommandGate`) so the domain stays unit-testable. Do
  not add I/O to `-core`; do not duplicate domain logic into `-runtime`.
- **Every carried behaviour is pinned by a golden/behaviour test.** The argv
  builder has a byte-exact golden test; the catalog and AutoBest schemas round-
  trip real fixtures; the proxy has method/header/SSE contract tests. A change to
  a carried behaviour updates its golden test in the same commit — never delete a
  golden assertion to make a diff pass.
- **Engineering rules:** MSRV 1.82, edition 2021, exact-pinned workspace deps,
  `#![forbid(unsafe_code)]` (use safe platform APIs like `creation_flags` /
  `process_group`, never `pre_exec`), no `unwrap`/`expect`/`todo`/`dbg` outside
  `#[cfg(test)]`. Windows / Linux / macOS are equal tier-1 (ADR-0007).
- **Wire contracts are versioned.** The `LauncherVersion` JSON envelope is a
  cross-product contract; any key rename is breaking — bump and update every
  consumer's conformance test in the same train.

## Consumption / re-pin ceremony

Crate versions stay `0.1.0`; consumption is rev-pinned. When a consumer needs a
new behaviour, advance its `rev` at a checkpoint and re-run that consumer's suite
(and the launcher-envelope conformance test). Do not publish to crates.io.

## Local gate (mirror CI)

```
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --workspace
```
