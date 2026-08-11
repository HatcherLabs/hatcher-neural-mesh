# The Hatcher integration contract

**Contract version `1.0`.** Shipped in HAMNS `1.0.0`. Defined in
[`hatcher_core::contract`](../crates/hatcher-core/src/contract.rs), implemented by
[`hatcher_neural::adapter::MeshAdapter`](../crates/hatcher-neural/src/adapter.rs), and
exposed over HTTP by `hatcher-ux`.

This is the whole surface an external agent runtime binds to. It is deliberately four
verbs wide. Everything else in the workspace is either the engine behind it or a way to
look at it.

```text
  1. register   AgentRegistration   →  RegistrationAck
  2. plan       TaskEnvelope        →  RoutingPlan      ─┐
                 …you execute for real…                  │  one run,
  3. report     StageOutcomeReport  →  RunStatus         │  held open
  4. finalize   run_id              →  MeshReceipt      ─┘
```

---

## Why plan and report are separate calls

The mesh does not execute agents. It decides *which* agent should run a stage, and it
learns from what that agent then did.

In simulation those two halves collapse into one call, because the outcome is derived
from a hash and is available immediately. Against a real runtime they cannot collapse:
minutes of real work happen in between, and **the mesh must not invent the result**.

So a run is a session:

* `plan` is the mesh's half. It names an agent per stage, prices each one, and is issued
  against a committed mesh digest. **It does not move trust, memory, capability, or `Ω`** —
  nothing has happened yet, and a mesh that learned from work it merely *scheduled* would
  be learning from its own intentions.
* `report` is your half: success, a verifier score, latency, cost, and a classified error.
* `finalize` is where the mesh actually moves.

Between `plan` and `finalize` the run is held open with its assignments **frozen**. If the
mesh moves in the meantime — another run finalizes, an agent registers — the plan is still
honoured as issued. You have already dispatched real work to those agents; re-routing under
you would make the reported outcomes describe a run that never happened.

---

## 1. Register an agent

```jsonc
POST /api/mesh/agents
{
  "id": "claude-coder-01",          // stable across restarts
  "label": "Claude coder",
  "role": "Coder",                  // Planner | Researcher | Coder | Critic | Verifier
                                    // (also Orchestrator | Executor | Explorer | Guardian)
  "capability": {                   // the I·S·P·C·M prior — see below
    "intelligence": 0.90,
    "specialization": 0.70,
    "performance": 0.75,
    "context": 0.85,
    "memory": 0.60
  },
  "resources": { "energy": 0.35, "latency": 0.20 },
  "confidence": 0.6,
  "expertise": { "rust": 0.85, "general": 0.65 }
}
```

```jsonc
→ 200
{
  "contract_version": "1.0",
  "agent_id": "claude-coder-01",
  "created": true,
  "cohort_size": 6,
  "capability": 0.241,              // A_i = I·S·P·C·M
  "bottleneck": "memory",           // the factor currently capping this agent
  "mesh_digest": "e7a23c52…"
}
```

**The declared capability vector is a prior, not an assertion.** The moment the agent
starts reporting outcomes, `P` is measured from its success history, `M` from what it
commits to memory, `S` from what it repeatedly does well, and the declaration stops
mattering. Declaring `1.0` across the board buys about three tasks of optimism.

`bottleneck` is on the ack for a reason: `A_i` is a *product*, so one weak factor collapses
it. An agent declared at `0.9` everywhere except memory at `0.05` scores below one declared
at `0.6` everywhere, and the ack tells you that immediately rather than leaving you to
wonder why it never gets routed to.

**Re-registering a known id updates the declaration but keeps the measured history** —
telemetry, observed cost and latency, and everything the learning equations have moved.
Only `intelligence` and `context` are overwritten, because those are the two factors the
equations never lift, so a re-declaration is the only way they can change. A restarting
worker re-announcing itself must not be able to wipe its own track record.

Out-of-range values are **rejected, not clamped** (`400`). Clamping at a contract boundary
turns your bug into a silently different mesh and you never find out.

---

## 2. Submit a task and get a plan

```jsonc
POST /api/runs
{
  "task_id": "PR-4417",             // optional; the mesh assigns one if absent
  "description": "harden the trust settlement path",
  "domain": "rust",                 // drives specialization routing
  "features": [0.35, 0.72, 0.28, 0.64],
  "urgency": 0.6,
  "execution_mode": "Controlled",   // Sandbox | Controlled | Production
  "constraints": { "max_latency_ms": 30000, "max_cost": 0.50 },
  "metadata": { "pr": "4417" }      // echoed back on the receipt, never interpreted
}
```

```jsonc
→ 200
{
  "contract_version": "1.0",
  "run_id": "run-9f2c1a0b7d4e3c85",
  "task_id": "PR-4417",
  "priority": { "value": 0.31, "band": "Standard", … },
  "stages": [
    {
      "stage": "Plan",
      "agent_id": "planner-expert",
      "score": 0.4461,
      "capability": 0.514,
      "inbound_trust": 0.50,
      "mastery": 0.94,
      "expected_latency_ms": 33000.0,
      "expected_cost": 0.62,
      "rationale": "A=0.514 trust=0.50 mastery=0.94 R=0.74 → score 0.4461"
    },
    … four more
  ],
  "expected_latency_ms": 128000.0,
  "expected_cost": 2.31,
  "mesh_digest": "e7a23c52…",
  "omega": 1.2088,
  "constraint_violations": ["Plan"]
}
```

### `features` is the honest part of this struct

The mesh derives uncertainty `U`, budget `B`, and implementation cost `I` from feature
dispersion, mass, and width. **Supply real measurements** — diff size, file count, test
coverage, retrieval hit rate, token estimate — and you get real routing. Supply noise and
you get noise. Slot `k` must mean the same measurement across every task you submit.

If you already measure `U`, `B`, or `I` directly, set them explicitly and skip the
derivation.

### Constraints narrow the pool but never empty it

If nothing fits your limits, the mesh **still routes** and lists the stage in
`constraint_violations`. Refusing to route is worse than routing over budget, and you are
told which happened so you can decline instead. An agent is priced from its measured
history once it has any, and from its declared prior until then.

---

## 3. Report what actually happened

```jsonc
POST /api/runs/run-9f2c1a0b7d4e3c85/outcomes
[
  {
    "stage": "Code",
    "agent_id": "coder-expert",     // must match the plan
    "success": true,
    "quality": 0.88,                // the reviewer's score, not the producer's
    "confidence": 0.75,             // the producer's own confidence
    "latency_ms": 18400,
    "cost": 0.21,                   // your units — dollars, tokens, credits
    "error": "None",
    "note": "3 files, 2 tests added"
  }
]
```

```jsonc
→ 200  { "run_id": "…", "reported": ["Plan","Code"], "missing": ["Research","Critique","Verify"], "complete": false }
```

Send one at a time or a batch. A batch is validated in full before any of it is stored, so
one bad entry leaves the run untouched rather than half-updated.

### `quality` versus `confidence`

`quality` is what the **reviewer** thought of the output. `confidence` is what the
**producer** thought of it. Keeping them apart is what lets the mesh detect an agent that
is confidently wrong — the confidence equation penalizes a high-confidence failure harder
than a humble one, and that only works if the two numbers come from different places.

If you have no grader, report `1.0` on success and `0.0` on failure. The mesh degrades to a
binary signal cleanly.

### The error class is the most important field on this struct

A failed stage is **not** automatically evidence that the agent is weak. A provider outage,
a rate limit, and a caller-side cancellation all produce a failed stage, and a mesh that
treats them as capability signals will quietly learn to distrust a perfectly good agent
because someone else's service was down.

| `error` | Attribution | Blame | Reaches the trust ledger? |
|---|---|---|---|
| `None` | — | — | yes, as a success |
| `Quality` | agent | 1.00 | yes |
| `Refusal` | agent | 1.00 | yes |
| `Schema` | agent | 1.00 | yes |
| `Tool` | agent | 1.00 | yes |
| `Timeout` | agent | 1.00 | yes |
| `RateLimit` | environment | 0.35 | **no** |
| `Infrastructure` | environment | 0.35 | **no** |
| `Cancelled` | caller | 0.00 | no — discarded entirely |
| `Unknown` | agent | 1.00 | yes |

Two calls worth stating plainly:

* **`Timeout` is the agent's.** Taking too long *is* a performance property of an agent,
  and the resource ratio `R_i = P_i / (Energy_i + Latency_i)` already exists to price it.
* **`RateLimit` is not.** Being throttled is a property of the account, not the reasoning.

What blame protects is `C_i` — a belief about the agent's *reasoning* — and the trust
ledger, a peer judgement about its *work*. Neither was tested by an outage. What blame
does **not** protect is `P` (completed work over attempted work; the outage really did cost
a completion) or the `F` term in `Ω` (the mesh failed to deliver, whoever's fault it was).

`Unknown` is attributed to the agent on purpose: letting an unclassified failure launder
itself into a free pass would make `Unknown` the cheapest thing to send.

A report that says `"success": true, "error": "Timeout"` is rejected as a `400`. A
contradiction must not enter a digest that is supposed to be an attestation.

---

## 4. Receive the decision

```jsonc
POST /api/runs/run-9f2c1a0b7d4e3c85/finalize
→ 200
{
  "contract_version": "1.0",
  "run_id": "run-9f2c1a0b7d4e3c85",
  "task_id": "PR-4417",
  "accepted": true,
  "selected_agent": "coder-expert",
  "decision": {
    "action": "stabilize",          // observe | delegate | stabilize | escalate
    "confidence": 0.81,
    "rationale": "verified | omega=1.284 A=2.01 P=0.310 (standard) verified=true | head proposed stabilize",
    "agent": "PR-4417",
    "intent": "harden the trust settlement path"
  },
  "verified": true,
  "trace_digest": "40ce9edd1ccce9c2…",
  "mesh_digest": "e7a23c52f03a5a03…",
  "provenance": "Reported",
  "omega_before": 1.2088, "omega_after": 1.2863,
  "total_latency_ms": 64800, "total_cost": 1.663, "mean_quality": 0.88,
  "stages": [ … per-stage receipts … ],
  "metadata": { "pr": "4417" }
}
```

`accepted` is exactly the inverse of `escalate` — an escalation is the mesh declining to
own the outcome.

`selected_agent` is the agent the mesh holds accountable for the output: the coding station
when one ran, otherwise the highest-scoring assignment.

The mesh's guardrails outrank the decision head. A model is never allowed to claim
`stabilize` on work that failed verification, and high-priority unverified work always
escalates to a human. A mesh that has failed its way into the `Degraded` regime escalates
*everything* it could not verify, regardless of priority.

### `provenance` — read this before believing anything else

| Value | Meaning |
|---|---|
| `Reported` | every staffed stage was reported by a real runtime. **Only this attests to real work.** |
| `Simulated` | every outcome came from the deterministic simulator. A rehearsal. |
| `Mixed` | some reported, some simulated. Useful for debugging; never evidence. |

Provenance is **inside** the trace digest, not beside it. A digest over a rehearsal and a
digest over production work are different values, so the first can never be presented as
the second.

---

## Calibration: what a millisecond is worth

The equations work in `[0, 1]`; `R_i = P_i / (Energy_i + Latency_i)` is only meaningful if
energy and latency share a scale. Your reports arrive in milliseconds and dollars, so
something has to define what `1.0` means.

```jsonc
GET /api/contract
{ "calibration": { "latency_ceiling_ms": 60000.0, "cost_ceiling": 1.0, "observation_rate": 0.25 } }
```

* `latency_ceiling_ms` — the latency that normalizes to `1.0`, i.e. "the slowest agent we
  would tolerate". Default one minute.
* `cost_ceiling` — the cost that normalizes to `1.0`, in your units. Default `1.0`.
* `observation_rate` — EWMA weight given to each new observation. Low keeps a profile
  stable; high lets it chase the last run.

This is explicit rather than inferred, because an inferred ceiling would move every agent's
score the moment one slow outlier appeared.

**Only reported outcomes update an agent's cost and latency profile.** A simulated latency
is *derived from* that profile, so folding it back would be a closed loop that drifts with
every rehearsal and calls the drift evidence. The mesh learns what an agent costs from the
world, or it does not learn it at all.

The first real measurement **replaces** the declared prior outright rather than averaging
with it — a declaration is a guess, and one measurement is strictly better evidence than a
guess nobody checked.

---

## Errors

| Status | Variants | When |
|---|---|---|
| `400` | `Invalid`, `Mismatch` | a field is missing, out of range, self-contradictory, or names an agent the plan did not assign |
| `404` | `UnknownRun`, `UnknownAgent` | no such run or agent |
| `409` | `DuplicateReport`, `Incomplete` | finalize called before every staffed stage reported |

`finalize` on an incomplete run is a `409`, not a silent partial. A run finalized with holes
would let the mesh learn from a story with pieces missing. If you genuinely want the holes
filled by the simulator, that is `MeshAdapter::finalize_partial`, and the trace comes back
marked `Mixed`.

---

## The shortcuts

Both go through exactly the same pipeline as the long form.

| Call | Outcomes from | Provenance | Use |
|---|---|---|---|
| `MeshAdapter::execute(envelope)` | simulator | `Simulated` | rehearsal, tests |
| `MeshAdapter::submit_reported(envelope, reports)` | your reports | `Reported` | replay, or a runtime that batches |

`POST /api/tasks` is the legacy one-call form and remains simulated.

---

## In Rust

```rust
use hatcher_core::{AgentRegistration, AgentRole, StageOutcomeReport, TaskEnvelope};
use hatcher_neural::MeshAdapter;

let mut adapter = MeshAdapter::empty();
adapter.register_agent(
    AgentRegistration::new("claude-coder-01", "Claude coder", AgentRole::Coder)
        .with_expertise("rust", 0.85),
)?;

let plan = adapter.plan(&TaskEnvelope::new("harden the settlement path")
    .with_domain("rust")
    .with_features(vec![0.35, 0.72, 0.28, 0.64]))?;

for planned in &plan.stages {
    // …dispatch real work to planned.agent_id, then:
    adapter.report(
        &plan.run_id,
        StageOutcomeReport::success(planned.stage, &planned.agent_id, 0.88)
            .with_latency_ms(18_400.0)
            .with_cost(0.21),
    )?;
}

let receipt = adapter.finalize(&plan.run_id)?;
assert!(receipt.provenance.is_real());   // this was real work
```

Watch it end to end without writing a client:

```bash
cargo run -p hatcher-ux            # then type: contract
cargo run -p hatcher-terminal -- --view contract --outcomes reported
```

---

## What this contract does not do

Stated plainly so nobody discovers it at integration time.

* **No authentication, authorization, or tenancy.** The sidecar listens on localhost and
  trusts every caller. Put it behind something.
* **No persistence.** The mesh lives in memory and starts fresh on restart. Registrations,
  learned trust, and observed cost profiles are lost. Digests are reproducible, so state can
  be rebuilt by replaying a recording, but there is no store.
* **No concurrency control beyond a single mutex.** Requests serialize.
* **Open runs are unbounded and never time out.** A caller that plans and disappears leaks
  the run until `cancel`.
* **Stage-level only.** The contract models the five-station pipeline. It has no vocabulary
  for a task that is not shaped like plan → research → code → critique → verify.

None of these are hard to add. They are absent because guessing at your requirements would
be worse than leaving them for the review.
