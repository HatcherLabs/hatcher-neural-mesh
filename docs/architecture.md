# Architecture

How the pieces fit. For the mathematics and the measurement definitions, see
[`hamns.md`](./hamns.md).

## System posture

Two planes, deliberately separate:

* **The Hatcher control plane** owns identity, permissions, secrets, integrations, and
  execution. It is the existing platform and this project does not replace any of it.
* **The Rust intelligence layer** owns the mesh: capability, trust, routing, memory, and
  the global intelligence state. It is a first-class runtime rather than a library
  bolted onto a request handler.

The intelligence layer never executes side effects. It decides, records, and returns a
control action; Hatcher acts. That boundary is what makes `ExecutionMode` meaningful —
the mesh can rehearse in a sandbox at reduced plasticity without anything happening in
the world.

## Crate layout

```text
hatcher-core        contracts, state, digests, read models   (no logic)
       ▲
hatcher-neural      equations, trust, routing, pipeline, inference
       ▲
hatcher-playground  scenarios, cohorts, coefficient battles
       ▲
hatcher-ux          terminal console + HTTP API
```

Dependencies point one way. `hatcher-core` holds no dynamics at all: it defines the state
the equations operate on and seals that state into digest-backed envelopes. Everything
that *moves* lives in `hatcher-neural`.

## The five layers, in code

| Layer | Type | Lives in |
|---|---|---|
| Global intelligence `Ω` | `GlobalState`, `OmegaDelta`, `OmegaLedger` | `hatcher_core::global` |
| Agent capability `A_i` | `CapabilityVector`, `AgentNode` | `hatcher_core::agent` |
| Mesh intelligence `A` | `MeshIntelligence` | `hatcher_core::graph` |
| Trust `T`, plasticity `W` | `TrustGraph` | `hatcher_neural::trust` |
| Priority & resources | `PriorityScore`, `ResourceProfile`, `router` | `hatcher_core::task`, `hatcher_neural::router` |

## Components

### 1. The Hatcher bridge

`HatcherRequest` / `HatcherResponse` carry intent, features, execution mode, and policy
context across the boundary. A request parses into a `TaskSpec` — deriving `U`, `B`, and
`I` from the feature vector — and comes back as a `NeuralSignal`: one of `observe`,
`delegate`, `stabilize`, `escalate`, with a rationale that names the numbers behind it.

`accepted` is exactly the inverse of `escalate`, because an escalation is the mesh
declining to own the outcome.

### 2. The mesh

`NeuralMesh` holds the cohort, the trust graph, the memory graph, `Ω`, the ledger, and
the decision head. It exposes three kinds of operation:

* **Mutating** — `pipeline::run` executes real work and moves everything.
* **Rehearsal** — `simulate` and `rehearse` clone the mesh, so nothing observable changes.
* **Read models** — `overview`, `analytics`, `graph_view`, `trust_view`, `config_view`
  are shaped for the frontend and computed on demand.

Message passing is a two-round GNN layer over the trust-gated weight matrix. It is what
makes the `C` factor of the capability vector measurable: context awareness tracks how
much mesh signal actually reaches a node.

### 3. The pipeline

Ten stations, five staffed. The router assigns each staffed stage using capability,
trust, domain mastery, and the resource ratio, weighted by the task's priority band.
Outcomes are hash-derived, so a run is reproducible. Credit then flows backwards: memory,
confidence, specialization, and performance move per agent; trust and plasticity settle
across the handoff chain; `Ω` absorbs the measured `L, E, C, F, D`.

### 4. The decision head

An `InferenceBackend` maps the canonical eight-slot mesh feature vector to four action
logits. Two implementations: a deterministic built-in MLP, and an ONNX policy loaded
through tract. The mesh applies hard policy overrides *after* the model, so a learned
policy can be swapped in without widening what the mesh is allowed to do.

### 5. The playground

Scenarios generate comparable task batches at controlled difficulty. `AgentBattle` runs
identical cohorts on identical work under different coefficient tunings — the honest
version of an agent battle, where what wins is a tuning rather than a lucky agent.

### 6. The UX shell

A terminal console for local inspection and a warp HTTP API for the frontend. Both drive
the same live mesh through a single lock; a panicked request recovers the poisoned lock
rather than taking the mesh offline.

## Runtime flow

```text
Hatcher workflow or operator
        │  HatcherRequest / POST /api/tasks
        ▼
   Task Parser ──► Priority Engine ──► Router
        │                                │
        │                                ▼
        │                    Plan → Research → Code → Critique → Verify
        │                                │
        ▼                                ▼
   Decision head ◄──────── Memory / Trust / Ω updates
        │
        ▼
   NeuralSignal + sealed PipelineTrace  ──►  Hatcher executes or escalates
```

## Design decisions worth knowing

**State is sealed, not just serialized.** Any observation can be committed to a SHA-256
digest over canonical key-sorted JSON, so a mesh state or a trace can be attested to
rather than merely logged.

**Determinism is a feature, not an accident.** No clocks, no RNG. Rehearsals are
comparable, traces are replayable, and tests assert behaviour rather than tolerate noise.

**Degradation is visible.** When a requested ONNX policy cannot be loaded, the mesh falls
back to the built-in head and records why in `degraded()`. Silent fallback would mean
running a different model than the operator believes they deployed.

**The equations are bounded.** Every update clamps, `Ω` has a floor and a numerical
ceiling, and `MeshCoefficients::validate` rejects tunings that would diverge. A
misconfigured mesh should degrade, not explode.
