# localx-llama

Shared Rust crate tier for the LocalX stack — the primitives reused by
**LocalBox**, **LocalBench**, and **LocalPilot**.

| Crate | Responsibility |
|---|---|
| `localx-llama-core` | Pure domain: model definitions, `llama-server` argv builder, VRAM/quant-fit math, config precedence, tuner/AutoBest schema. No I/O. |
| `localx-llama-runtime` | Process/network side behind cross-platform traits: server lifecycle, pin-verified install/download, CPU-only embed-serve, in-process no-think streaming filter. |
| `localx-eval-core` | Evaluation primitives extracted from LocalPilot's harness: scorecard, blind judge, ablation, stack-detected grader. Shared by LocalPilot and LocalBench. |

## Consuming this repo

Product repos depend on these crates via a **rev-pinned Cargo git dependency**
(not a submodule):

```toml
[dependencies]
localx-llama-core = { git = "https://github.com/C0deGeek-dev/localx-llama", rev = "<pinned-sha>" }
```

During active development, a local `[patch]` / path override is used for
velocity; the rev is pinned at each checkpoint.

## Toolchain

MSRV **1.82**, edition **2021**, exact-pinned workspace deps, `#![forbid(unsafe_code)]`.
Windows / Linux / macOS are equal tier-1 (matches the stack's ADR-0007).

```
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --workspace
```

Part of the LocalX ecosystem. Planning lives in the private LocalHub repo.
