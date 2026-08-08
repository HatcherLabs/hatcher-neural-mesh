# Changelog

All crates are versioned in lockstep: a HAMNS release is a release of the whole
mesh. This project follows [Semantic Versioning](https://semver.org); while the major
version is `0`, breaking changes bump the minor version.

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
