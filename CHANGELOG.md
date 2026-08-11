# Changelog

All crates are versioned in lockstep: a HAMNS release is a release of the whole
mesh. This project follows [Semantic Versioning](https://semver.org).

## [1.0.0] — 2026-08-10

The mesh stops having to invent its own outcomes.

Through `0.23.0` every stage result came from a deterministic competence model. That is the
right behaviour for rehearsal — it replays exactly, which is what makes a trace digest an
attestation — but it is the wrong behaviour for a mesh that is supposed to learn from
agents that really ran. `1.0.0` makes the outcome the *input* it always should have been,
and adds the contract for supplying it.

**This release is breaking.** Stage records, the pipeline entry points, and the learning
API all changed shape. `SCHEMA_VERSION` moves to `3`, so digests taken under `0.23.0` do
not compare to digests taken under `1.0.0`.

### Added

- **The integration contract** (`hatcher_core::contract`, contract version `1.0`) — four
  verbs: register an agent and its capabilities, submit a task and its measurable features,
  report the real outcome of each stage, receive the selected agent, action, confidence,
  rationale, and trace digest. Documented in [`docs/adapter.md`](docs/adapter.md).
- **`MeshAdapter`** (`hatcher_neural::adapter`) — the stateful facade implementing those
  verbs. `plan` chooses agents without moving trust, memory, capability, or `Ω`; the mesh
  only moves at `finalize`, when the reports are in. Between the two the run is held open
  with its assignments frozen, so a mesh that changes underneath a caller cannot silently
  re-route work that has already been dispatched.
- **`OutcomeSource`** (`hatcher_neural::outcomes`) — the seam that makes outcomes swappable.
  `SimulatedOutcomes` is the old competence model, moved intact; `ReportedOutcomes` takes
  results supplied from outside. Everything downstream is identical either way.
- **`ErrorClass` and blame attribution** — a failed stage is no longer automatically
  evidence that the agent is weak. Rate limits and infrastructure failures are attributed to
  the environment, damped to 35% blame, and kept out of the trust ledger entirely; caller
  cancellations are discarded. Without this a mesh learns to distrust a perfectly good agent
  because someone else's service was down.
- **`OutcomeProvenance`, inside the digest** — `Simulated`, `Reported`, or `Mixed`. A
  rehearsal and a production run over identical outcomes now produce *different* trace
  digests, so a simulation can never be presented as evidence of work.
- **Measured cost and latency** — `ResourceProfile` gained `observed_latency_ms`,
  `observed_cost`, and `observations`. Reported outcomes fold into an EWMA and move the
  normalized factors with them, so reported latency reaches `R_i` instead of being
  decorative. Only *reported* outcomes do this: a simulated latency is derived from the
  profile, and folding it back would be a closed loop.
- **`TaskConstraints`** — `max_latency_ms` and `max_cost` on a task. The router honours them
  everywhere, narrowing the candidate pool but never emptying it; a plan reports any stage
  whose constraint it could not satisfy rather than refusing to route.
- **The routing benchmark** (`hatcher_playground::bench`) — six routing policies over
  identical work, scored on quality, cost, latency, and reliability, with signed deltas
  against a fixed one-agent-per-role baseline. Results and the scenario where the mesh loses
  are in [`docs/benchmark.md`](docs/benchmark.md).
- **The replay harness** — `Replay` / `RecordedRun`, JSON Lines in and out, scoring the
  mesh's routing against recorded production choices. Reports agreement, the success rates
  where it agreed and disagreed, and a projected cost counterfactual.
- **`contested_cohort()`** — three differentiated candidates per role. `Benchmark::new()`
  uses it because the default cohort has one agent per role, which forces every policy into
  the same pick and produces six identical scorecards that look like a finding.
- **Three observatory tabs** — `contract`, `benchmark`, and `economics`, plus a tab strip
  across every view. A mesh running on declared numbers looks identical to one running on
  measured ones, so `contract` states which it is and `economics` marks every unmeasured
  figure as `declared`.
- **`hatcher-terminal --outcomes reported`** — drives the real contract loop instead of the
  simulator, so the integration path is exercised rather than described.
- **HTTP**: `GET /api/contract`, `POST /api/mesh/agents`, `POST /api/runs`,
  `POST /api/runs/{id}/outcomes`, `POST /api/runs/{id}/finalize`, `GET`/`DELETE
  /api/runs/{id}`, `GET /api/benchmark`, `POST /api/replay`. Console: `contract`, `bench`,
  `replay`.
- **The decision head is part of the contract.** `DecisionHead` (name, model, version,
  dimensions, degradation reason) rides on every `MeshReceipt` and on `GET /api/contract`,
  so a caller acting on a `stabilize` can tell whether it came from the built-in network or
  a trained policy — and whether the mesh is running the head it was *asked* to run.
- **The mesh digest now commits to the decision head.** Two meshes with identical nodes and
  identical trust but different policies return different answers; a commitment that ignored
  the head would attest to state while saying nothing about what that state was used to
  decide. Only the head's *identity* is committed, never the degradation reason — that
  contains a local filesystem path and would make the same mesh hash differently on two
  machines. This is what makes statement S2 in [`docs/zk.md`](docs/zk.md) checkable.
- **`MeshAdapter::try_with_policy`** and **`hatcher-terminal --model <path>`** — load a
  trained policy, failing loudly rather than degrading. An operator who names a policy
  explicitly should not be quietly shown decisions from a different model.
- **`Benchmark::with_head`** — run every arm against a specific decision head, with
  `PolicyScorecard::head` recording what actually ran so two reports are never compared
  blind. The head is held constant *across* arms on purpose: that benchmark varies routing,
  and an arm that also changed the head would not say which of the two moved the numbers.
- **[`docs/zk.md`](docs/zk.md)** — the canonical digest as shipped, and four candidate
  statements a real ZK proof could prove, each with an honest assessment of whether a
  signature would do instead. The recommendation is to build signing first and treat ZK as
  the private-aggregate-performance feature.

### Changed

- **`StageRecord`** gained `quality`, `latency_ms`, `error`, and `provenance`. `quality` is
  the reviewer's score and `confidence` is the producer's — keeping them apart is what lets
  the mesh detect an agent that is confidently wrong.
- **`Ω`'s drift term and the stabilize guardrail now key on the verifier's *score*** rather
  than the verifier's self-confidence. `Ω` is supposed to track whether the mesh produces
  good work, and the producer's opinion of its own output is exactly the thing that has to
  be checked rather than believed.
- **`pipeline::run_bound`** is the new general entry point, taking `RunInputs` (config,
  calibration, outcome source, optional pre-bound assignments). `run` and `run_with` are
  unchanged wrappers over it with simulated outcomes.
- **`learning::apply_outcome`** takes a `RuntimeCalibration` and consults blame attribution.
  `calibrate_confidence` takes a blame weight. `StageOutcome` carries quality, error,
  latency, cost, and provenance.
- **Memory salience scales with quality** — a stage that scraped a pass consolidates less
  than a clean one, so a mesh stops writing mediocrity into `M_i` just because it
  technically did not fail.
- **`PipelineStage` derives `Ord`** in execution order, so a keyed collection of stages
  iterates the way the pipeline does.
- `hatcher-ux` drives a `MeshAdapter` rather than an `AgentArena`; its routes are combined
  in boxed groups because a single `.or()` chain of that length nests deeply enough to be a
  problem for the compiler.
- The `onnx` feature is verified against this release: `cargo test --features
  hatcher-neural/onnx` runs 334 tests, twelve of them exercising the real
  `models/fixtures/hamns-policy-8x4.onnx` through tract. `OnnxBackend` itself is unchanged —
  what changed is that the head it loads is now visible on the contract and inside the
  digest.

### Fixed

- The routing benchmark reported six identical scorecards on the default cohort, because one
  agent per role leaves nothing to route between. `Benchmark::new()` now uses a contested
  cohort, and two tests pin the failure so it cannot return silently.

### Known limitations

Stated in [`docs/adapter.md`](docs/adapter.md) rather than discovered at integration time:
no authentication, no persistence across restarts, requests serialize behind one mutex, open
runs never time out, and the contract has no vocabulary for a task that is not shaped like
the five-station pipeline.

## [0.23.0] — 2026-08-08

Ships `hatcher-terminal`, the observatory — the mesh becomes something you watch rather
than something you read logs about.

### Added

- **`hatcher-terminal`**, a fifth crate that links the whole mesh (`hatcher-core`,
  `hatcher-neural`, `hatcher-playground`) and renders a live one as animated 3D ANSI.
  The binary runs real work through the ten-stage pipeline every frame, so what is on
  screen is measured mesh state, not an animation loop.
- **Active node tornado** — the cohort spun up into a 3D vortex whose geometry is read
  from mesh state: radius is inverse capability (strong agents pull to the axis), height
  is trust standing, angular speed is live activation, and funnel amplitude tracks `Ω`.
  An idle mesh decays to a slow dim ring; only an executing mesh spins up.
- **Node graph view** — agents placed on a Fibonacci sphere with trust-weighted edges,
  under an orbiting camera.
- **Trust matrix, equations, pipeline, and roster panels** — all ten update rules shown
  live as labelled meters beside the stage-by-stage trace of the last task.
- **A depth-buffered ANSI canvas** with clip regions, truecolor output, and a plain-text
  mode, so the entire view layer is unit-testable without a terminal.
- `--view`, `--scenario`, `--size`, `--frames`, `--fps`, `--spin`, `--once`, and
  `--plain` on the `hatcher-terminal` binary.

### Changed

- The workspace moves to `0.23.0`; all five crates are pinned in lockstep as before.
- `README` documents the observatory and the workspace gains a fifth crate row.

## [0.2.0] — 2026-08-08

The full articulation of the agent mesh. `0.1.0` sketched five of the equations against
placeholder state; `0.2.0` implements all ten against measured observations, adds the
execution pipeline, the self-organizing trust graph, and a swappable ONNX decision head.

**This release is a breaking rewrite of the public API.** Pin `0.1.0` if you depend on
the old surface.

### Added

- **All ten HAMNS equations** as pure, bounded, unit-tested functions in
  `hatcher_neural::equations` — global intelligence `Ω`, agent capability, mesh
  intelligence, trust, priority, memory, confidence, specialization, plasticity, and the
  resource ratio.
- **`TrustGraph`** — ndarray-backed `T` and `W` matrices with settlement, hub and
  isolation detection, and partner ranking. Unreliable agents drift out of the routing
  graph; reliable ones become hubs.
- **Message passing** — a two-round GNN layer, `m_i = Σ W_ij·T_ij·A_j·h_j`, gated by
  trust and scaled by sender capability.
- **The ten-station pipeline** (`hatcher_neural::pipeline`) — parse, prioritize, plan,
  research, code, critique, verify, memory, trust, `Ω` — returning a sealed
  `PipelineTrace`. Outcomes are hash-derived, so runs replay exactly.
- **`hatcher_neural::router`** — stage assignment from capability, trust, domain mastery,
  and cost, with the cost weighting set by the task's priority band.
- **`hatcher_neural::learning`** — credit assignment from stage outcomes into the
  capability vector.
- **Inference backends** — an `InferenceBackend` trait, a deterministic built-in MLP, and
  an **ONNX backend** behind the `onnx` feature, running on pure-Rust tract. Includes an
  exporter (`scripts/export_policy_onnx.py`) and a working model fixture.
- **`MemoryGraph`** with salience, verification-weighted retention, and decay that
  reports per-agent loss — the measured `K_i` and `R_i`.
- **`MeshCoefficients`** — the full Greek-letter knob surface in one validated struct,
  with `conservative()` and `exploratory()` presets.
- **Frontend read models** — `MeshOverview`, `MeshAnalytics`, `MeshGraphView`,
  `TrustMatrixView`, `MeshConfigView`, mapped to `hatcher-host-frontend` tabs.
- **HTTP API** in `hatcher-ux` (`serve`), with CORS, validated config writes, and a
  bounded trace log.
- **Playground scenarios** — `rehearsal`, `stress`, `frontier` — and `AgentBattle`, which
  runs identical cohorts on identical work under different tunings.
- `docs/hamns.md` (full specification), `docs/onnx.md` (model contract), `CHANGELOG.md`,
  and a `.gitignore`.
- Declared, compilation-verified MSRVs: `1.81` for the libraries, `1.86` for
  `hatcher-ux`, `1.91` for `hatcher-neural` with `--features onnx`.

### Changed — breaking

- `AgenticNodeData` is **removed**. The mesh node is now `AgentNode`, carrying a
  `CapabilityVector` (`I, S, P, C, M`), a `ResourceProfile`, per-domain expertise, and
  telemetry. The old flat seventeen-scalar record had no relationship to the equations.
- `SerializedNodeEnvelope` is replaced by the generic `Envelope<T>`, which can seal any
  observation and can verify its own digest. Schema version is now `2`.
- `MeshEdge` gains `trust`, `latency`, `successes`, and `failures`; `weight` now means
  connection strength `W_ij` specifically, distinct from trust.
- `MeshState` gains `global`; `MeshStepResult` and `MeshSimulation` report the
  intelligence decomposition and `Ω` instead of a single aggregate scalar.
- `NeuralMesh` is stateful. `evaluate` and `execute` take `&mut self` and advance the
  mesh; `simulate` and `rehearse` remain immutable.
- `HatcherResponse` gains `omega`, `intelligence`, and `priority`. `accepted` now means
  "the mesh will own this outcome" — exactly the inverse of an `escalate` decision.
- `MeshConfig` is now a deprecated alias for `MeshCoefficients`, which carries all
  fourteen coefficients rather than six.
- `AgentRole` gains `Planner`, `Researcher`, `Coder`, and `Verifier`.
- `ModelSpec::native_default()` is now `8 → 12 → 4`: four mesh statistics plus four task
  statistics in, one logit per control action out.
- `repository` and `homepage` metadata now point at the actual repository. The `0.1.0`
  crates advertised a URL that does not resolve.

### Deviations from the source specification

Five, each documented in `docs/hamns.md` with its reasoning:

1. **Priority urgency** — `τ` stays as the denominator floor; urgency multiplies the
   numerator. Taken literally, an urgency bonus in the denominator *lowers* priority.
2. **Plasticity saturation** — `W += φT(1−W) − ψ·latency`. The linear form is unbounded,
   so every link that ever carried traffic pinned at `1.0` and `W` stopped
   distinguishing partners.
3. **Idle links get no plasticity gain**, only the latency cost — otherwise the graph
   converges on fully-connected instead of organizing itself.
4. **Stage competence multiplies capability** rather than adding it, so capability is a
   ceiling. As a sum, mastery and trust floored an incapable agent near 45%.
5. **Degraded-regime escalation** — a mesh at `Ω ≈ 0` escalates everything it cannot
   verify, regardless of priority.

### Notes on tuning

`λ` and `μ` are effectively one knob: the linear trust rule is a threshold classifier at
`μ / (λ + μ)`. The `0.1.0` values put break-even at 29%, meaning an agent failing two
jobs in three still accumulated full trust. The default is now ≈71%.

## [0.1.0]

Initial release: mesh contracts, a digest-backed node envelope, a five-equation
simulation loop, a rehearsal playground, and a terminal demo.
