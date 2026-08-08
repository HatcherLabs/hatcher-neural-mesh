# HatcherLabs Agent Mesh Neural System (HAMNS)

A Rust-native agentic mesh: an adaptive, decentralized multi-agent intelligence
framework where agent capability is multiplicative, collective intelligence is emergent
rather than aggregate, trust is learned from outcomes, and the whole mesh carries one
global intelligence state that rises and falls with what it actually accomplishes.

Ten equations, five layers, one execution loop, and a decision head you can swap for a
trained ONNX policy.

## The thesis

Most agent frameworks treat a fleet as a list. HAMNS treats it as a graph with dynamics:

* **Weaknesses matter.** `A_i = I·S·P·C·M` is a product, so a brilliant model with no
  memory scores as weak — and the mesh will tell you which factor is capping each agent.
* **Connection is worth something measurable.** `A = Σ A_i + γ Σ A_i A_j W_ij` separates
  raw compute from collective intelligence, and the second term is zero until agents
  actually work together.
* **The graph organizes itself.** Trust rises with successful handoffs and falls with
  failures; connection strength follows trust but pays for latency. Unreliable agents
  drift into isolation, reliable ones become hubs, unused links wither.
* **Scheduling is derived, not configured.** Priority balances uncertainty, budget,
  implementation cost, urgency, and the mesh's confidence *in that specific domain*.
* **Every run changes the mesh.** The mesh that ran your last task is not the one that
  will run the next.

## The ten equations

```text
1.  Global intelligence   Ω(t+1) = Ω(t) + α(L + E + C) − β(F + D)
2.  Agent capability      A_i    = I_i · S_i · P_i · C_i · M_i
3.  Mesh intelligence     A      = Σ A_i + γ Σ_{i≠j} A_i A_j W_ij
4.  Trust evolution       T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij
5.  Priority              P      = (U · B · I · urgency) / (C + τ)
6.  Memory evolution      M_i(t+1) = M_i(t) + η K_i − δ R_i
7.  Confidence            C_i(t+1) = C_i(t) + σ·success − ρ·error
8.  Specialization        S_i(t+1) = S_i(t) + κ·experience − ω·obsolescence
9.  Network plasticity    W_ij(t+1) = W_ij(t) + φ T_ij (1 − W_ij) − ψ·latency
10. Resource ratio        R_i    = P_i / (Energy_i + Latency_i)
```

[`docs/hamns.md`](docs/hamns.md) is the full specification: what every variable is
measured from in software, where each equation lives in the code, and the two places
this implementation deliberately refines the stated rules (and why).

## The pipeline

```text
Incoming Task → Task Parser → Priority Engine → Planner
                                                  ├── Research
                                                  └── Coding
                                                       ↓
                                                    Critic → Verifier
                                                       ↓
                              Memory Update → Trust Update → Ω Update
```

Ten stations. Five are staffed by agents chosen by the router; five are mesh
bookkeeping. Each pass returns a sealed `PipelineTrace` — the priority score, every
assignment, every stage record, the measured `Ω` terms, the decision, and a digest.

Outcomes are drawn from a hash of `(task id, agent id, stage, sequence)`, never a clock
or an RNG, so a run replays exactly. That is what makes rehearsal meaningful and lets a
trace digest be an attestation rather than a souvenir.

## Quick start

```bash
cargo test                      # 207 tests across the workspace
cargo run -p hatcher-ux         # interactive console
cargo run -p hatcher-terminal   # the live 3D observatory
```

### The observatory

`hatcher-terminal` links every crate in the mesh and renders a *running* mesh as
animated 3D ANSI — the frames are real observations taken after real pipeline work,
not a canned animation.

It also installs as a standalone binary, with no ONNX runtime or terminal backend
to match against the host:

```bash
cargo install hatcher-terminal
hatcher-terminal
```

From a checkout:

```bash
cargo run -p hatcher-terminal                              # full dashboard
cargo run -p hatcher-terminal -- --view tornado            # the vortex, full screen
cargo run -p hatcher-terminal -- --scenario frontier       # watch Ω erode
cargo run -p hatcher-terminal -- --once --plain            # one frame to stdout
```

| View | What it answers |
|---|---|
| `tornado` | Is the mesh live, and which agents are carrying it? |
| `graph` | Who is connected to whom, and how strongly? |
| `equations` | What are all ten update rules doing right now? |
| `dashboard` | All of the above, plus the roster and trust matrix |

The tornado is the headline, and it is driven by mesh state rather than decoration:
radius is inverse capability, so strong agents are drawn to the axis; height is trust
standing; angular speed is live activation; and the funnel's amplitude tracks `Ω`. A
mesh that is genuinely working spins up a tall, fast, bright vortex — one that is
stalling flattens toward a slow, dim ring.

In the console:

```text
run        submit one task through the full pipeline
scenario   run a 24-task batch (rehearsal | stress | frontier)
battle     compare coefficient tunings over identical work
overview   agent roster, capability bottlenecks, hubs, isolated agents
trust      the trust matrix and strongest collaborations
omega      the Ω ledger with its L, E, C, F, D terms
logs       recent pipeline traces
api        start the HTTP API for the Hatcher frontend
```

As a library:

```rust
use hatcher_core::TaskSpec;
use hatcher_neural::{pipeline, NeuralMesh};

let mut mesh = NeuralMesh::default();
let task = TaskSpec::new("task-1", "ship the router", "rust")
    .with_features(vec![0.3, 0.7, 0.2, 0.9])
    .parsed_from_features();

let trace = pipeline::run(&mut mesh, &task);
println!("{} → Ω {:.3}", trace.decision.action, trace.omega_after);
```

Run it a few dozen times and the interesting part shows up: hubs form, unreliable agents
stop receiving work, and `Ω` compounds or erodes depending on whether the cohort is
actually good enough for the work it is being given.

## Frontend integration

The mesh runs as an intelligence sidecar to
[`hatcher-host-frontend`](https://github.com/HatcherLabs/hatcher-host-frontend). That app
is a Next.js dashboard that talks to its backend only through `lib/api.ts` and renders an
agent across thirteen tabs, so the endpoints map onto the tabs that are about
intelligence:

| Frontend tab | Endpoint |
|---|---|
| Overview | `GET /api/mesh/overview` |
| Config | `GET` / `PUT /api/mesh/config` |
| Analytics | `GET /api/mesh/analytics` |
| Logs | `GET /api/mesh/logs` |
| Workflows / Chat | `POST /api/tasks` |
| Versions / audit | `GET /api/tasks/{id}` (full sealed trace) |
| — | `GET /api/mesh/graph`, `/api/mesh/trust`, `/api/mesh/agents`, `/api/mesh/memory` |

```bash
HATCHER_MESH_PORT=3030 cargo run -p hatcher-ux -- serve
```

The frontend's own backend stays at `:3001`; this listens on `HATCHER_MESH_PORT`
(default `3030`). Set `HATCHER_MESH_ALLOWED_ORIGIN` to lock CORS down before exposing it
past localhost. Coefficient writes are validated, so a bad tuning gets a `400` instead of
quietly making the mesh diverge.

## ONNX decision heads

The equations govern the graph; the decision head governs which of four control actions
(`observe`, `delegate`, `stabilize`, `escalate`) goes back to Hatcher. It ships as a
small built-in network and can be replaced with a trained ONNX policy:

```bash
python scripts/export_policy_onnx.py          # export a conforming model
cargo test -p hatcher-neural --features onnx  # exercise it
HATCHER_MESH_MODEL=models/policy.onnx cargo run -p hatcher-ux --features onnx
```

Inference runs on [tract](https://github.com/sonos/tract), which is pure Rust — enabling
the feature adds no native ONNX Runtime library to ship or version-match. Loading binds
the input shape, optimizes the graph, and runs a probe pass, so a broken policy fails at
startup rather than on the first real request. A failed load degrades to the built-in
head and says why, instead of leaving the mesh unable to decide anything.

The head proposes; the mesh disposes: a model is never allowed to claim `stabilize` on
work that failed verification, and high-priority unverified work always escalates to a
human. See [`docs/onnx.md`](docs/onnx.md) for the model contract.

## Workspace

| Crate | Role |
|---|---|
| `hatcher-core` | Typed contracts, digest-backed envelopes, frontend read models |
| `hatcher-neural` | The ten equations, trust graph, message passing, router, pipeline, inference |
| `hatcher-playground` | Scenarios, cohorts, coefficient battles |
| `hatcher-ux` | Terminal console and the HTTP API |
| `hatcher-terminal` | The observatory: 3D ANSI node graphs, live equations, and the active-node tornado |

## Docs

* [`docs/hamns.md`](docs/hamns.md) — the full specification
* [`docs/architecture.md`](docs/architecture.md) — how the layers fit together
* [`docs/onnx.md`](docs/onnx.md) — decision-head model contract

## License

MIT
