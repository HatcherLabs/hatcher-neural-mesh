# Hatcher Neural Mesh

> Originally created by [Meta-Oracle](https://github.com/Meta-Oracle) and continued by
> [HatcherLabs](https://github.com/HatcherLabs) under the MIT license. See
> [`NOTICE.md`](./NOTICE.md) for attribution and project history.

## Project status

This repository is the open-source intelligence engine for Hatcher's adaptive agent
routing work. It is currently a **technical preview**: the typed routing, outcome,
replay, trust, and ONNX contracts are implemented, while Hatcher production routing
remains in shadow evaluation. The included policy is not a trained reinforcement-learning
model, and the digest layer is not a zero-knowledge proof system.

A Rust-native agentic mesh: an adaptive, decentralized multi-agent intelligence
framework where agent capability is multiplicative, collective intelligence is emergent
rather than aggregate, trust is learned from outcomes, and the whole mesh carries one
global intelligence state that rises and falls with what it actually accomplishes.

Ten equations, five layers, one execution loop, a decision head you can swap for a trained
ONNX policy, and a four-verb integration contract for driving it from a real agent runtime.

**Foundation in 1.0.0:** the mesh no longer has to invent its own outcomes. Register your agents,
submit a task, receive a routing plan, run it for real, and report back what actually
happened — success, verifier score, latency, cost, and a classified error. See
[`docs/adapter.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/adapter.md).

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

[`docs/hamns.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/hamns.md) is the full specification: what every variable is
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

## The integration contract

The mesh decides *which* agent should run a stage. It does not execute agents, and it must
not invent what they did. So a run is a session:

```text
  1. register   AgentRegistration   →  RegistrationAck
  2. plan       TaskEnvelope        →  RoutingPlan      ─┐
                 …you execute for real…                  │  one run,
  3. report     StageOutcomeReport  →  RunStatus         │  held open
  4. finalize   run_id              →  MeshReceipt      ─┘
```

`plan` moves nothing — a mesh that learned from work it merely *scheduled* would be
learning from its own intentions. Trust, memory, capability, and `Ω` move at `finalize`,
when the reports are in.

```rust
use hatcher_core::{StageOutcomeReport, TaskEnvelope};
use hatcher_neural::MeshAdapter;

let mut adapter = MeshAdapter::new();
let plan = adapter.plan(&TaskEnvelope::new("harden the settlement path")
    .with_domain("rust")
    .with_features(vec![0.35, 0.72, 0.28, 0.64]))?;

for planned in &plan.stages {
    // …dispatch real work to planned.agent_id, then:
    adapter.report(&plan.run_id,
        StageOutcomeReport::success(planned.stage, &planned.agent_id, 0.88)
            .with_latency_ms(18_400.0)
            .with_cost(0.21))?;
}

let receipt = adapter.finalize(&plan.run_id)?;
assert!(receipt.provenance.is_real());   // this was real work
```

Three things this contract is careful about, because they are the ones that quietly ruin a
learning system:

* **A failed stage is not automatically the agent's fault.** Every failure carries an
  `ErrorClass`, and a rate limit or a provider outage is damped to 35% blame and kept out
  of the trust ledger entirely. Otherwise the mesh learns to distrust a perfectly good
  agent because someone else's service was down.
* **`quality` and `confidence` are different numbers.** One is the reviewer's score, the
  other is the producer's. Keeping them apart is what lets the mesh notice an agent that is
  confidently wrong.
* **Simulated and reported runs can never be confused.** Every trace carries its
  `OutcomeProvenance`, and it is *inside* the digest — so a rehearsal and a production run
  over identical outcomes hash differently, and the first can never be presented as the
  second.

Only reported outcomes teach the mesh what an agent costs. A simulated latency is derived
from the agent's own profile, so folding it back would be a closed loop that drifts with
every rehearsal and calls the drift evidence.

[`docs/adapter.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/adapter.md) is the full contract, including the error-class table,
the calibration model, and an explicit list of what it does *not* do.

## Does the routing pay for itself?

```bash
cargo run -p hatcher-playground --example benchmark
```

On routine work, against a fixed one-agent-per-role assignment — what most agent frameworks
ship — over 24 identical tasks:

| Policy | Quality | Verified | Cost/verified | p50 latency |
|---|---|---|---|---|
| **mesh** | 0.627 | 42% | **3.920** | 89 408 ms |
| `static-role` (baseline) | 0.617 | 42% | 6.510 | 154 481 ms |
| `cheapest` | 0.407 | 4% | 7.350 | 20 160 ms |

Same verified rate, **40% less money, 42% less wall-clock time**. `cheapest` is the
cautionary arm: the lowest cost per task and the *worst* cost per verified task, because
routing on price alone does not save money, it moves the bill.

The harness also reports where the mesh **loses** — on `frontier`, a domain nobody has
mastered, it under-provisions and verifies nothing while a fixed assignment manages 4%. That
result is pinned by a test, so fixing the router forces the documentation to be updated
rather than letting a stale claim survive. See [`docs/benchmark.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/benchmark.md).

Point the replay harness at real recorded runs and it scores the mesh's routing against the
choices that were actually made, without ever seeing your prompts or outputs.

## Quick start

```bash
cargo test                      # 322 tests across the workspace
cargo test --features hatcher-neural/onnx   # 334, adding the ONNX decision head
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

| Tab | What it answers |
|---|---|
| `dashboard` | All of the below, plus the roster and trust matrix |
| `tornado` | Is the mesh live, and which agents are carrying it? |
| `graph` | Who is connected to whom, and how strongly? |
| `equations` | What are all ten update rules doing right now? |
| `contract` | What is driving the mesh, and is any of it real? |
| `benchmark` | Does mesh routing beat a fixed assignment? |
| `economics` | What does this cohort cost, and how fast is it? |

The last three exist because once something outside the mesh starts driving it, "the
tornado is spinning" stops being the interesting question. A mesh running entirely on
declared numbers looks identical to one running on measured ones — so the `contract` tab
states which it is and what share of the cohort has ever been measured, and `economics`
marks every unmeasured figure as `declared` rather than letting a guess read as evidence.

```bash
cargo run -p hatcher-terminal -- --view contract --outcomes reported
cargo run -p hatcher-terminal -- --view benchmark --benchmark
```

`--outcomes reported` drives the full integration contract instead of the built-in
simulator: the observatory plans a run, reports synthesized stage outcomes back, and
finalizes it. The outcomes are still made up — what is real is the code path.

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
contract   drive one task through the integration contract, verb by verb
bench      mesh routing vs the baselines on quality, cost, latency, reliability
replay     replay a recording and score the mesh's routing against it
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

The integration contract is served alongside it:

| Verb | Endpoint |
|---|---|
| descriptor | `GET /api/contract` |
| register | `POST /api/mesh/agents` |
| plan | `POST /api/runs` |
| report | `POST /api/runs/{id}/outcomes` |
| finalize | `POST /api/runs/{id}/finalize` |
| inspect / abandon | `GET /api/runs`, `GET /api/runs/{id}`, `DELETE /api/runs/{id}` |
| evidence | `GET /api/benchmark`, `POST /api/replay` |
| tenant-local shadow rank | `POST /api/shadow/route` |

```bash
HATCHER_MESH_PORT=3030 cargo run -p hatcher-ux -- serve
```

The frontend's own backend stays at `:3001`; this listens on `HATCHER_MESH_PORT`
(default `3030`). Set `HATCHER_MESH_ALLOWED_ORIGIN` to lock CORS down before exposing it
past localhost. Coefficient writes are validated, so a bad tuning gets a `400` instead of
quietly making the mesh diverge.

Hatcher's shadow adapter uses the stateless `/api/shadow/route` surface. Every request
supplies one owner's complete candidate cohort, builds an empty request-local mesh, and
returns a recommendation without changing shared state. Keep this sidecar private to the
Hatcher API host. The endpoint rejects cohorts larger than 128 agents and caps request
bodies at 256 KiB.

### Production shadow sidecar

Production must use the restricted surface and fail-closed token policy:

```dotenv
HATCHER_MESH_PORT=3030
HATCHER_MESH_BIND_ADDR=0.0.0.0
HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND=true
HATCHER_MESH_INTERNAL_TOKEN=<at-least-32-random-characters>
HATCHER_MESH_REQUIRE_INTERNAL_TOKEN=true
HATCHER_MESH_SHADOW_ONLY=true
```

Keep those values in `/etc/hatcher/services/hatcher-neural-mesh.env`, owned by root with
mode `0600`. `scripts/deploy-production.sh` builds the pinned Docker image, installs the
systemd unit, binds the process only to `127.0.0.1`, enforces resource and privilege
limits, waits for `/health`, and restores the previous image if the health gate fails.

```bash
sudo install -d -o root -g root -m 0750 /etc/hatcher/services
sudo install -o root -g root -m 0600 .env.production.mesh \
  /etc/hatcher/services/hatcher-neural-mesh.env
./scripts/deploy-production.sh
```

The Hatcher API receives the same secret through `HATCHER_NEURAL_MESH_TOKEN`. Do not put
the sidecar behind the public reverse proxy. In shadow-only mode, every demo/stateful
endpoint is removed; only `/health`, `/api/contract`, and `/api/shadow/route` remain.

## ONNX decision heads

The equations govern the graph; the decision head governs which of four control actions
(`observe`, `delegate`, `stabilize`, `escalate`) goes back to Hatcher. It ships as a
small built-in network and can be replaced with a trained ONNX policy:

```bash
python scripts/export_policy_onnx.py          # export a conforming model
cargo test -p hatcher-neural --features onnx  # 164 tests, 12 of them ONNX
HATCHER_MESH_MODEL=models/policy.onnx cargo run -p hatcher-ux --features onnx
cargo run -p hatcher-terminal --features onnx -- --model models/policy.onnx --view contract
```

Inference runs on [tract](https://github.com/sonos/tract), which is pure Rust — enabling
the feature adds no native ONNX Runtime library to ship or version-match. Loading binds
the input shape, optimizes the graph, and runs a probe pass, so a broken policy fails at
startup rather than on the first real request.

The head proposes; the mesh disposes: a model is never allowed to claim `stabilize` on
work that failed verification, and high-priority unverified work always escalates to a
human. See [`docs/onnx.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/onnx.md) for the model contract.

**The head is part of the contract, not an implementation detail.** Every `MeshReceipt`
carries a `decision_head` — name, version, dimensions, and a reason if the mesh is not
running the head it was asked for — and `GET /api/contract` reports it before you send any
work. **The mesh digest commits to it**, so swapping the policy changes the attestation:
two meshes with identical nodes and identical trust but different heads return different
answers, and a commitment that ignored that would attest to state while saying nothing
about what the state was used to decide.

Where a bad model path *degrades* to the built-in head it always says why — on the receipt,
in `mesh.degraded()`, and in red on the observatory's `contract` tab. Where you named the
policy explicitly (`try_with_onnx`, `MeshAdapter::try_with_policy`, `hatcher-terminal
--model`) it fails instead. Silent fallback is never on offer: it would mean acting on
decisions from a model you think you replaced.

## Workspace

| Crate | Role |
|---|---|
| `hatcher-core` | Typed contracts, the integration contract, digest-backed envelopes, read models |
| `hatcher-neural` | The ten equations, trust graph, message passing, router, pipeline, inference, the adapter |
| `hatcher-playground` | Scenarios, cohorts, coefficient battles, the routing benchmark and replay harness |
| `hatcher-ux` | Terminal console and the HTTP API |
| `hatcher-terminal` | The observatory: 3D ANSI node graphs, live equations, and the active-node tornado |

A client that only needs the wire types can depend on `hatcher-core` alone.

## Docs

* [`docs/adapter.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/adapter.md) — **the integration contract**: the four verbs, the
  error-class table, calibration, and what it deliberately does not do
* [`docs/benchmark.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/benchmark.md) — mesh routing vs. the baselines, with results
  and the scenario where the mesh loses
* [`docs/zk.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/zk.md) — the canonical digest as shipped, and a precise statement of
  what a ZK proof would prove and when a signature would do instead
* [`docs/hamns.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/hamns.md) — the full specification
* [`docs/architecture.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/architecture.md) — how the layers fit together
* [`docs/onnx.md`](https://github.com/HatcherLabs/hatcher-neural-mesh/blob/main/docs/onnx.md) — decision-head model contract

## License

MIT
