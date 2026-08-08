# ONNX decision heads

The mesh's decision head maps mesh state to one of four control actions. By default it
is a small built-in network. With the `onnx` feature it can be any conforming ONNX
graph — a trained policy drops in without touching the equations, the routing, or the
trust dynamics.

```bash
cargo test -p hatcher-neural --features onnx
cargo run -p hatcher-ux --features onnx
```

## The contract

```text
input    float32   [1, spec.input_dim]     default [1, 8]
output   float32   [1, spec.output_dim]    default [1, 4]
```

**Input slots** — the canonical mesh feature layout. Mesh statistics come first so a
narrower model still sees the mesh's own state:

| Slot | Name | Range | Meaning |
|---|---|---|---|
| 0 | `omega` | 0–1 | Global intelligence `Ω`, normalized by the ceiling (16.0) |
| 1 | `emergence_ratio` | 0–1 | Share of mesh intelligence `A` coming from connection |
| 2 | `mean_trust` | 0–1 | Mean `T_ij` across the cohort |
| 3 | `mean_confidence` | 0–1 | Mean calibrated agent confidence |
| 4 | `priority` | 0–1 | `P`, squashed by a logistic |
| 5 | `uncertainty` | 0–1 | Task `U` |
| 6 | `budget` | 0–1 | Task `B` |
| 7 | `implementation` | 0–1 | Task `I` |

Raw task features follow slot 7 for models wide enough to consume them.

**Output slots** — one score per action, in this order:

| Slot | Action | Meaning |
|---|---|---|
| 0 | `observe` | Not enough signal; keep watching, do not act |
| 1 | `delegate` | Hand the work onward |
| 2 | `stabilize` | Accept and consolidate |
| 3 | `escalate` | Return to a human |

Either raw logits or an already-normalized distribution is accepted — the decoder
detects which, so a model ending in `Softmax` is not squashed a second time.

## Exporting a model

`scripts/export_policy_onnx.py` exports a conforming policy and prints a sanity check:

```bash
python scripts/export_policy_onnx.py --out models/fixtures/hamns-policy-8x4.onnx
```

The shipped weights are hand-set rather than trained, which makes the policy
interpretable and testable — it escalates high-priority work an unconfident mesh cannot
justify, stabilizes routine work a confident mesh owns, delegates when the cohort is
trusted but unsure, and observes otherwise. Replace `policy_weights()` with trained
parameters to ship a learned policy; the contract and the export path are unchanged.

`models/fixtures/hamns-policy-8x4.onnx` (about 1 KB) is checked in and exercised by the
tests in `hatcher_neural::onnx`.

## Loading

```rust
use hatcher_neural::NeuralMesh;

// Fail loudly if the policy will not load.
let mesh = NeuralMesh::default().try_with_onnx("models/policy.onnx")?;

// Or degrade to the native head, keeping the reason.
use hatcher_core::ModelSpec;
use hatcher_neural::BackendKind;

let mesh = NeuralMesh::default().with_backend(
    BackendKind::Onnx { model_path: "models/policy.onnx".into() },
    ModelSpec::native_default(),
);
if let Some(reason) = mesh.degraded() {
    eprintln!("running the built-in head: {reason}");
}
```

Loading binds the input shape, optimizes the graph, and then runs a probe pass with a
zero vector. Shape and operator problems otherwise surface on the first real request,
which is the worst possible moment to find out the policy was never viable.

Errors are specific rather than a generic failure: `ModelNotFound` for a missing path,
`InputShape` / `OutputShape` for a contract mismatch, `Runtime` for a graph tract cannot
load or execute, and `BackendUnavailable` when the crate was built without the feature.

## Runtime choice

Inference runs on [tract](https://github.com/sonos/tract), a pure-Rust engine.

**Why not ONNX Runtime.** The `ort` crate binds the official Microsoft runtime and has
broader operator coverage plus GPU execution providers. It also requires a native
`onnxruntime` shared library that must be downloaded at build time or shipped alongside
the binary and kept version-matched with the host, and as of this writing it is still
pre-release (`2.0.0-rc.x`). For a Rust-native mesh whose selling point is a single
self-contained artifact, tract is the better default: `cargo build --features onnx`
produces a binary that runs ONNX models with nothing else to install.

**The tradeoff.** tract implements a large but not exhaustive subset of ONNX, so an
exotic graph may fail to load. That failure is reported at load time with the operator
named. If you need full operator coverage or GPU inference, add an `ort`-backed
`InferenceBackend` alongside `OnnxBackend` — the trait is the only integration point,
and `BackendKind` is where a new variant goes.

## Model files and version control

Model weights are not committed. `.gitignore` excludes `*.onnx` and the `models/`
directory, with an exception for `models/fixtures/` so small test models travel with the
repository. Put production policies in `models/` and distribute them out of band.
