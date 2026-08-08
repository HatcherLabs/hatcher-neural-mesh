# HAMNS — HatcherLabs Agent Mesh Neural System

An adaptive, decentralized multi-agent intelligence framework. Five layers, ten
equations, one execution loop.

* Individual agent capability is **multiplicative** over intelligence, specialization,
  performance, context, and memory — so weaknesses matter.
* Collective intelligence is **emergent**, not aggregate: connected agents are worth
  more than the sum of disconnected ones.
* Trust **evolves from outcomes**, so the mesh learns who should talk to whom.
* Scheduling balances **uncertainty, urgency, and cost**.
* The whole mesh carries one **global intelligence state** that rises and falls with
  what it actually accomplishes.

This document is the contract between the mathematics and the code. Every equation
below names the module that implements it and — the part that usually goes missing —
the concrete observable each variable is measured from.

---

## 1. The five layers

| Layer | Question | Crate module |
|---|---|---|
| 1. Global intelligence | Is the mesh getting smarter? | `hatcher_core::global` |
| 2. Agent capability | What is one agent worth? | `hatcher_core::agent` |
| 3. Mesh intelligence | What is the cohort worth, connected? | `hatcher_core::graph` |
| 4. Dynamic trust | Who should talk to whom? | `hatcher_neural::trust` |
| 5. Priority & resources | What runs next, and where? | `hatcher_core::task`, `hatcher_neural::router` |

All ten update rules live as pure, bounded, side-effect-free functions in
`hatcher_neural::equations`. The stateful engine decides *when* they fire and *what*
is measured; the equations decide only *how* the numbers move.

---

## 2. The graph

**Nodes** are `AgentNode`s. **Edges** are two `n × n` matrices held in `TrustGraph`:

* `T` — trust: the belief that `j` will do good work for `i`.
* `W` — connection strength: what the mesh actually spends on that belief.

There is no separate adjacency list. An edge exists exactly to the degree that
`W_ij > 0`, which is why an untrusted agent is not removed from the mesh so much as
forgotten by it.

**Message passing** is a graph neural network layer, run for `MESSAGE_ROUNDS = 2`:

```text
m_i = Σ_{j≠i} W_ij · T_ij · A_j · h_j
h_i ← squash(h_i + m_i)
```

Three deliberate choices: messages are gated by trust as well as weight; they are
scaled by the sender's capability `A_j`, so a confident but incapable agent cannot
shout down the mesh; and the update is residual, so an isolated node keeps its own
signal instead of decaying to zero. Two rounds let information travel two hops — a
planner feels a verifier through a coder — without the over-smoothing that makes every
node in a deep GNN look alike.

---

## 3. The ten equations

### 3.1 Global intelligence state

```text
Ω(t+1) = Ω(t) + α(L + E + C) − β(F + D)
```

`hatcher_neural::equations::global_intelligence`. Bounded to `[0, 16]` — the floor
because a mesh cannot have negative intelligence, the ceiling as a numerical guard so a
misconfigured `α` cannot produce `inf` and poison every downstream ratio.

Each pipeline run produces exactly one `OmegaDelta`, measured as follows:

| Term | Meaning | Measured from |
|---|---|---|
| `L` | Learning gain | Salience actually committed to the memory graph this run, summed over participating agents, divided by cohort size |
| `E` | Emergent behaviors | `MeshIntelligence::emergence_ratio()` — the share of `A` produced by the pairwise term rather than by headcount |
| `C` | Collaboration efficiency | Trust-weighted successful handoffs ÷ total handoffs, from `TrustSettlement` |
| `F` | Failure accumulation | Failed staffed stages ÷ executed staffed stages |
| `D` | Drift from objectives | `0.5·(1 − verifier confidence) + 0.5·(obsolescence applied ÷ cohort size)` |

`D` deserves a note: drift is half "we are not sure this was right" and half "our
expertise is aging". Both are real ways a mesh stops matching its objectives.

Every update is appended to `OmegaLedger`, which is the series the Analytics view plots
and the source of `omega_slope`.

`Ω` is **cumulative, not normalized**. A healthy mesh accrues it run after run and will
eventually reach the ceiling; a struggling one grinds to the floor. The absolute value is
therefore a poor health metric on its own — `OmegaRegime` and `omega_slope` are the
signals worth alerting on. What the level is good for is comparison: identical cohorts on
identical work under different tunings, which is exactly what `AgentBattle` does.

### Calibration matters more than it looks

`P` is measured from outcomes and feeds straight back into `A_i`, so the `Ω` loop has
real feedback in it. A cohort tuned to coin-flip on easy work does not stay at a coin
flip: failures depress `P`, which depresses capability, which produces more failures, and
`Ω` spirals to the floor. The default cohort is deliberately placed on the right side of
that loop (fitness ≈ 0.75, about a 75% stage success rate on routine work). Observed
behaviour of the three shipped scenarios:

| Scenario | Difficulty | Stage success | Verified | `Ω` after 24 tasks |
|---|---|---|---|---|
| `rehearsal` | 0.25 | ~72% | ~46% | 1.00 → 4.82 |
| `stress` | 0.85 | ~32% | ~8% | 1.00 → 0.00, 22 escalations |
| `frontier` | 0.60 | ~43% | ~4% | 1.00 → 0.00 |

That spread is the system working: the mesh compounds on work it is suited to, collapses
on work it is not, and says so.

### 3.2 Agent capability

```text
A_i = I_i · S_i · P_i · C_i · M_i
```

`CapabilityVector::capability()`. Multiplicative on purpose: a brilliant model with no
memory is weak, and a specialist with no context is weak.
`CapabilityVector::bottleneck()` names the factor currently capping an agent, which is
surfaced per-agent in the Overview roster.

| Factor | Meaning | Measured from |
|---|---|---|
| `I` | Intelligence | Configured per agent — a property of the underlying model, not something the mesh can grant |
| `S` | Specialization | Grown by equation 3.8 from domain experience |
| `P` | Performance | Laplace-smoothed lifetime success rate, `NodeTelemetry::observed_performance` |
| `C` | Context awareness | Tracks how much mesh signal actually reaches the node during message passing |
| `M` | Memory quality | Moved by equation 3.6 from real memory-graph activity |

Note that `I` and `C` are the only factors the pipeline does not lift. That is what
makes a persistent weak agent possible at all: an agent handicapped only in `S`, `P`, or
`M` trains its way out of the handicap within a couple of dozen runs.

### 3.3 Mesh intelligence

```text
A = Σ_i A_i + γ Σ_{i≠j} A_i A_j W_ij
```

`equations::mesh_intelligence`, returning the decomposition rather than a scalar:

* `raw = Σ A_i` — what the cohort is worth with the wires cut.
* `emergent = γ Σ A_i A_j W_ij` — collective intelligence, which is zero while `W` is
  zero no matter how good the agents are.
* `total = raw + emergent`, and `emergence_ratio = emergent / total`.

`equations::top_emergent_pairs` names the specific collaborations producing the most
collective value, so emergence is legible rather than a number that went up.

### 3.4 Dynamic trust

```text
T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij
```

`equations::trust_update`, applied by `TrustGraph::settle`. Held in `[0, 1]` so a long
success streak cannot make an agent unfalsifiably trusted.

Evidence accumulates during a run and is applied as one coherent update at the end.
`S_ij` and `E_ij` are counts of successful and failed handoffs from `i` to `j` in that
interval, scaled by the execution mode's plasticity gate — a sandbox run teaches the
mesh a quarter of what a production run does.

Credit flows along the handoff chain: each agent judges the one it handed to. The
verifier's judgment additionally feeds back to the planner, which is how a planner that
keeps producing unworkable plans loses trust despite never writing a line.

**λ and μ are one knob, not two.** Because the rule is linear and clamped to `[0, 1]`,
trust is a *threshold classifier*, not a reliability estimate: a partner whose success
rate exceeds `μ / (λ + μ)` drifts to full trust, and one below it drifts to zero. The
break-even rate is what actually decides who the mesh routes to, so it is what the
presets are tuned against — roughly 71% by default, 83% under `conservative()`, and 33%
under `exploratory()`. Getting this wrong is subtle and expensive: with a break-even of
29%, an agent failing two jobs in three still ends up fully trusted.

**Hubs and isolation.** A hub has inbound trust at least `1.25×` the cohort mean.
Isolation requires *both* thin inbound weight *and* low inbound trust: thin weight alone
only means "idle", which is the normal state of a fresh agent nobody has tried. Since
the router skips isolated candidates, conflating the two would starve every new agent.

### 3.5 Priority

```text
P = (U · B · I · urgency_gain) / (C + τ)
```

`equations::priority`. High `P` means "uncertain, expensive, and the mesh is not
confident — schedule it now on the strongest agents". Low `P` means "the mesh already
knows how to do this — defer it or hand it to a cheaper agent". Bands:
`P ≥ 0.60` immediate, `P < 0.20` deferred, otherwise standard.

| Term | Measured from |
|---|---|
| `U` | Feature-vector dispersion — a spread-out request is an ambiguous one |
| `B` | Feature mass, i.e. the budget the request justifies |
| `I` | How many dimensions the request touches |
| `C` | `NeuralMesh::domain_confidence` — agent confidence weighted by mastery of *this* domain |
| `τ` | Coefficient; the denominator floor |

**One deviation from the source specification, stated plainly.** The spec gives
`P = (U · B · I) / (C + τ)` with `τ` described as an urgency bonus. Taken literally,
urgency sits in the denominator and *lowers* priority, inverting its intent. This
implementation keeps the literal shape — `τ` is the floor that keeps `P` finite when the
mesh is fully confident — and routes urgency through `urgency_gain = 1 + urgency` in the
numerator, where it belongs. With `urgency_gain = 1.0` the formula reduces exactly to
the spec.

### 3.6 Memory evolution

```text
M_i(t+1) = M_i(t) + η K_i − δ R_i
```

`equations::memory_update`, fed by the `MemoryGraph`. Neither term is a free parameter:

* `K_i` is the salience actually admitted when the agent's work was written to memory.
  Salience scales with novelty, and novelty is `U · (1 − mastery)` — an agent already
  fluent in a domain learns little from doing it again. That is what keeps `L` from
  growing forever on repetitive work, and it is why a mesh grinding familiar tasks
  plateaus instead of compounding. Unverified work is retained at 35% salience: a
  failure still teaches something.
* `R_i` is what per-run decay removed, reported per agent by `MemoryGraph::decay`.

### 3.7 Confidence calibration

```text
C_i(t+1) = C_i(t) + σ·success − ρ·error
```

`equations::confidence_update`. Evidence is graded rather than binary: an agent that
succeeded while claiming low confidence gains less than one that succeeded confidently,
and an agent that failed while claiming high confidence is penalized more. That is what
calibration means — the update is about the *claim*, not just the outcome.

### 3.8 Specialization evolution

```text
S_i(t+1) = S_i(t) + κ·experience − ω·obsolescence
```

`equations::specialization_update`, applied to both the global `S` factor and the
per-domain mastery map, so an agent that repeatedly solves Rust problems becomes a Rust
specialist specifically rather than a vaguely better agent. Only successful work counts
as full experience — repeating a mistake is not practice. Every domain the agent did
*not* exercise this run ages, and the total aging applied is what feeds the `D` term.

### 3.9 Network plasticity

```text
W_ij(t+1) = W_ij(t) + φ T_ij (1 − W_ij) − ψ·latency
```

`equations::plasticity_update`. Connections strengthen through trust and weaken under
communication cost, so the mesh learns not just who is good but who is worth the round
trip.

**Two deviations, both load-bearing.**

1. *The `(1 − W_ij)` factor.* Written literally as `W += φT − ψ·latency`, the update is
   linear and unbounded: any link whose trust clears `ψ·latency / φ` grows without limit
   until it hits the `[0, 1]` clamp. In practice every link that ever carried traffic
   pins at `1.0`, and `W` stops distinguishing a strong partner from a tolerable one.
   Making the bound smooth instead of a wall gives the rule an interior fixed point,

   ```text
   W* = 1 − (ψ·latency) / (φ·T)
   ```

   so a link settles at a strength reflecting its trust-to-cost ratio, and a link whose
   trust falls below `ψ·latency / φ` dies off on its own.

2. *Idle links get no gain.* A link that carried traffic this pass gets the full rule; a
   link that carried nothing pays only the latency cost. Without this, baseline trust on
   an unproven pair is high enough that `φT` outruns `ψ·latency`, so every pair would
   thicken whether or not it was ever used and the mesh would converge on
   fully-connected — the opposite of self-organizing.

### 3.10 Resource ratio

```text
R_i = P_i / (Energy_i + Latency_i)
```

`equations::resource_efficiency` / `AgentNode::resource_efficiency`. Performance per unit
of cost, the tiebreaker when several agents can do the job. The denominator is floored so
a zero-cost agent cannot divide by zero.

---

## 4. Routing

`hatcher_neural::router` is where four equations meet. A candidate scores

```text
score = A_i^(1/5) · (0.20 + inbound_trust) · (0.20 + mastery) · (R_i / (1 + R_i))^cost_exponent
```

with the cost exponent set by the task's priority band:

| Band | Cost exponent | Behaviour |
|---|---|---|
| Immediate | 0.10 | Cost nearly ignored — get it right, not cheap |
| Standard | 0.50 | Cost is a tiebreaker |
| Deferred | 1.20 | Cost dominates — cheapest adequate agent wins |

That is the priority equation's intent made operational: urgent work goes to the
strongest agent, deferred work to the cheapest one that can still do it.

Capability enters as `A_i^(1/5)`, the geometric mean of the five factors, not as `A_i`.
`A_i` is a product of five sub-unit numbers, so raw capability differences are quintic —
a cohort-leading agent would outscore a merely-good one by so much that trust, mastery,
and cost could never matter. The geometric mean puts capability on the same scale as the
other terms, and on the same scale the pipeline uses to resolve success, so routing and
execution agree about what "capable" means.

The floors (`0.20`) keep a single zeroed term from acting as a veto rather than a
preference. Ties break on agent id, so routing is reproducible.

---

## 5. The pipeline

```text
Incoming Task
      │
      ▼
 Task Parser        →  U, B, I measured from the request
      ▼
 Priority Engine    →  P = (U·B·I·urgency) / (C + τ)
      ▼
 Planner Agent      →  routed by capability × trust × mastery × cost
   ┌──┴───┐
   ▼      ▼
Research  Coding
   └──┬───┘
      ▼
 Critic Agent       →  does the work survive review?
      ▼
 Verifier           →  is it actually correct?
      ▼
 Memory Update      →  M_i(t+1) = M_i + ηK_i − δR_i
      ▼
 Trust Graph Update →  T_ij(t+1) = T_ij + λS_ij − μE_ij,  W_ij += φT_ij(1−W_ij) − ψL
      ▼
 Ω Update           →  Ω(t+1) = Ω + α(L+E+C) − β(F+D)
```

Each pass returns a sealed `PipelineTrace`: the priority score, every assignment, every
stage record, the measured `OmegaDelta`, the decision, and a digest.

**Stage resolution.** Whether an assigned agent succeeds is

```text
fitness    = A_i^(1/5) · (0.60 + 0.25·mastery + 0.15·inbound_trust)
threshold  = fitness^(0.5 + 1.5·difficulty) · carry
success    = hash(task_id, agent_id, stage, sequence) < threshold
```

Capability multiplies rather than adds for the same reason `A_i` is a product: as a
bracketed sum, mastery and trust alone would floor an incapable agent at roughly a 45%
success rate, and since specialization grows with every attempt, *any* agent would
eventually pass regardless of its intelligence or context.

Difficulty enters as an **exponent** rather than a multiplier, because difficulty
compounds against weakness. A linear penalty moves every agent by the same amount, so a
hopeless agent and an excellent one lose the same margin on a hard task. As an exponent,
a fitness-0.9 agent barely notices (`0.9² = 0.81`) while a fitness-0.2 agent collapses
(`0.2² = 0.04`) — which is what "hard" means.

`carry` is `1.0` normally and `0.55` when an upstream stage failed. A bad plan does not
stop the pipeline, it poisons it — downstream agents are working from broken input.

**Verification.** A result is verified only when the work passed review *and* both
reviewers did their jobs. A critic that could not form a judgment is not a pass. Stage
success is recorded against the reviewer's own performance, so a critic is never
punished for correctly reviewing bad code.

**Determinism.** Outcomes are drawn from a hash of `(task id, agent id, stage, mesh
sequence)`, never from a clock or an RNG. The same task against the same mesh state
replays exactly — which is what makes playground rehearsal meaningful, and what lets a
trace digest be an attestation rather than a souvenir.

---

## 6. The decision head

The equations govern the graph. The decision head governs the *decision*: given the mesh
feature vector, which of four control actions goes back to Hatcher —
`observe`, `delegate`, `stabilize`, `escalate`.

**Canonical model input** (`NeuralMesh::model_input`), mesh statistics first so a
narrower model still sees the mesh's own state:

| Slot | Meaning |
|---|---|
| 0 | `Ω` normalized by the ceiling (16.0) |
| 1 | emergence ratio |
| 2 | mean trust |
| 3 | mean calibrated confidence |
| 4 | priority `P`, squashed into `(0, 1)` |
| 5 | task uncertainty `U` |
| 6 | task budget `B` |
| 7 | task implementation cost `I` |

Raw task features follow slot 7 for models wide enough to use them.

Two backends implement the same contract:

* **`NativeBackend`** — a small dense network, `8 → tanh(12) → softmax(4)`, with weights
  derived deterministically from a seed string. No external dependency; a fresh mesh on
  a different machine decides identically.
* **`OnnxBackend`** — a real ONNX graph. See [`onnx.md`](./onnx.md).

**The head proposes; the mesh disposes.** Three overrides are applied *after* the model,
never by it, which is why the mesh can be trusted with a `Production` execution mode at
all:

1. A model may not claim `stabilize` on work that failed verification.
2. High-priority unverified work always escalates to a human.
3. A mesh in the `Degraded` regime escalates *everything* it could not verify, whatever
   the priority. Without this, a mesh that has failed its way to `Ω ≈ 0` keeps quietly
   returning `observe` on low-priority work and never asks for help — the exact failure
   mode `Ω` exists to detect would go unreported by the system detecting it.

---

## 7. Coefficients

The entire knob surface is one serializable struct, `MeshCoefficients`, so the Config tab
can read and write it as a single JSON document. `validate()` rejects negative rates,
rates above 1.0, a non-positive `τ`, and a `γ` large enough for the pairwise term to
dominate.

| Symbol | Field | Governs |
|---|---|---|
| α | `alpha` | `Ω` gain from `L + E + C` |
| β | `beta` | `Ω` penalty from `F + D` |
| γ | `gamma` | Emergence coefficient |
| λ | `lambda` | Trust growth per success |
| μ | `mu` | Trust decay per failure |
| η | `eta` | Memory learning rate |
| δ | `delta` | Forgetting rate |
| σ | `sigma` | Confidence gain |
| ρ | `rho` | Confidence loss |
| κ | `kappa` | Specialization gain |
| ω | `obsolescence` | Specialization decay (named in full to avoid colliding with `Ω`) |
| φ | `phi` | Plasticity gain |
| ψ | `psi` | Plasticity cost |
| τ | `tau` | Priority denominator floor |

Three presets ship: `default()`, `conservative()` (slow to trust, hard to destabilize),
and `exploratory()` (fast emergence, more churn). `AgentBattle` runs identical cohorts on
identical work under each and reports which tuning ends up with the highest `Ω`.

---

## 8. Execution modes

`ExecutionMode::plasticity_gate()` scales how strongly a run is allowed to move the graph:
`Sandbox` 0.25, `Controlled` 0.6, `Production` 1.0. A sandbox failure is genuinely less
informative than a production one, so both trust gains and trust losses are discounted.

---

## 9. Attestation

Any mesh observation can be sealed into a digest-backed `Envelope<T>`. The digest is
SHA-256 over canonical, key-sorted JSON, so it is reproducible across processes and
machines. `NeuralMesh::digest()` folds every node digest with the edge set and the global
state into one commitment; a `PipelineTrace` digest commits to the task, every stage, the
outcome, and the mesh that produced it.

---

## 10. Where things live

| Module | Role |
|---|---|
| `hatcher_core::agent` | `CapabilityVector`, `ResourceProfile`, `AgentNode`, telemetry |
| `hatcher_core::graph` | `MeshEdge`, `MeshIntelligence`, `MeshState`, step and simulation results |
| `hatcher_core::global` | `Ω`, `OmegaDelta`, `OmegaLedger`, regimes |
| `hatcher_core::task` | `TaskSpec`, `PriorityScore`, `PipelineStage`, `PipelineTrace` |
| `hatcher_core::memory` | `MemoryGraph`, salience, decay |
| `hatcher_core::coefficients` | The Greek-letter knob surface and its validation |
| `hatcher_core::envelope` | Canonical digests and sealed envelopes |
| `hatcher_core::api` | Bridge contracts and the frontend read models |
| `hatcher_neural::equations` | The ten rules, pure and bounded |
| `hatcher_neural::trust` | `T` and `W`, settlement, hubs, isolation, partner ranking |
| `hatcher_neural::mesh` | Graph, message passing, `Ω` bookkeeping, read models |
| `hatcher_neural::router` | Stage assignment |
| `hatcher_neural::learning` | Credit assignment into capability movement |
| `hatcher_neural::pipeline` | The ten-station loop |
| `hatcher_neural::inference` | Decision head contract, native MLP |
| `hatcher_neural::onnx` | ONNX policy loading (feature `onnx`) |
| `hatcher_playground` | Scenarios, cohorts, coefficient battles |
| `hatcher_ux` | Terminal console and the HTTP API |
