# localx-llama

Shared Rust crate tier for the LocalX stack — the primitives reused by
**LocalBox**, **LocalBench**, and **LocalPilot**.

| Crate | Responsibility |
|---|---|
| `localx-llama-core` | Pure domain: model definitions, `llama-server` argv builder, per-build launch capabilities (read from a binary's `--help`), VRAM/quant-fit math, config precedence, tuner/AutoBest schema. No I/O. |
| `localx-llama-runtime` | Process/network side behind cross-platform traits: server lifecycle, a bounded `--help` read for capability detection, a `llama-fit-params` runner, pin-verify + asset-selection *decision logic* (the HTTP fetch/install shell lives in the consuming app), CPU-only embed-serve, and the in-process no-think proxy (method/header-faithful forwarding + per-delta SSE `<think>` stripping). |
| `localx-eval-core` | Evaluation primitives extracted from LocalPilot's harness: scorecard, blind judge, ablation, stack-detected grader. Shared by LocalPilot and LocalBench. |

llama.cpp builds disagree about launch flags: mainline replaced `--no-mmap` and
`--mlock` with `--load-mode` and rejects the old spellings, while the forks keep
them. The argv builder therefore takes the *intent* (`mlock`, `no_mmap`) plus
the target build's `LoadFlags`, read from that binary's own help text; a
launcher that cannot read it gets the long-standing flags. The same help text
says whether the build has `--fit`: a launcher may then set `auto_fit` so an
unplaced launch (no `-ngl`, no MoE offload, no tensor override) omits `-ngl` and
lets llama.cpp place the model, optionally with a `--fit-target` margin. It
also says whether the build has `--lazy-mode`: such a build reads a per-layer
embedding table larger than 4 GiB from disk on demand and keeps it mapped even
without mmap, which a caller needs to know to size a model's private memory.

llama.cpp ships its own memory fitter, `llama-fit-params`: given a model and
a launch shape it prints, in seconds and without loading tensor data, how many
layers and which MoE expert tensors fit in free VRAM. `fit` (core) builds its
arguments from the exact server argv — placement flags removed, server-only flags
dropped, a `--fit-target` margin added — and reads its answer; `fit` (runtime)
runs it. The fitter cannot see a vision projector or draft model, so callers
widen the margin by their size.

The tuner store keeps its document schema and measurement methodology as
separate compatibility axes. Schema-1 files remain readable, but consumers
only replay entries whose `tuner_version` matches the shared
`CURRENT_TUNER_VERSION` constant; superseded measurements remain on disk for
migration and diagnosis.

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

Part of the LocalX ecosystem.

## License

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm_Noncommercial_1.0.0-blue.svg)

LocalX-owned source is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a
separate license. See [LICENSING.md](LICENSING.md) for the commercial contact,
the 30 August 2026 licensing boundary, and third-party terms.
