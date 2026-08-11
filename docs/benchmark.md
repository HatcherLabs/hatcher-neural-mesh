# The routing benchmark

Does the mesh's routing actually beat something simpler?

That is the whole reason HAMNS has a router, and it is not settled by the architecture
being interesting. [`hatcher_playground::bench`](../crates/hatcher-playground/src/bench.rs)
answers it the only way it can be answered: run identical work through identical cohorts
under different routing policies and compare what came out.

```bash
cargo run -p hatcher-playground --example benchmark
cargo run -p hatcher-ux                      # then: bench [rehearsal|stress|frontier]
cargo run -p hatcher-terminal -- --view benchmark --benchmark
curl localhost:3030/api/benchmark
```

---

## The policies

| Policy | Strategy |
|---|---|
| `mesh` | capability × trust × domain mastery × cost, weighted by priority band |
| `static-role` | the same agent for a role, every time — **the baseline** |
| `round-robin` | spread the work evenly across eligible agents |
| `cheapest` | always the cheapest eligible agent |
| `strongest` | always the most capable eligible agent |
| `arbitrary` | a deterministic arbitrary pick — the control arm |

`static-role` is the baseline rather than `arbitrary` because it is what a reasonable team
actually ships: name a coder, name a reviewer, send every task to the same one forever.
Beating a random pick proves nothing. Beating a sensible fixed assignment is the claim
worth testing.

Every policy picks from the same candidate pool the mesh router builds — role eligibility,
exclusions, and isolation are structural facts, not routing opinions. The policies differ
only in which candidate they take.

## What makes the comparison fair

Every arm gets its own mesh built from the same cohort, the same task batch in the same
order, and the same outcome model. Stage outcomes are drawn from a hash of
`(task, agent, stage, sequence)`, so choosing a different agent genuinely changes the
result — but nothing is random, and running the benchmark twice gives identical numbers.
The only variable is which agent each policy picked.

### The cohort has to offer a choice

`Benchmark::new()` uses `contested_cohort()`, not `default_cohort()`. This is not a
preference. The default cohort has **exactly one agent per role**, so every policy is
forced into the same pick and the benchmark reports six identical scorecards that look like
a finding. A routing benchmark needs something to route between. `the_default_cohort_is_too_thin_to_benchmark_and_that_is_visible`
pins this so it cannot quietly come back.

`contested_cohort()` is three candidates per role:

| Tier | Capability | Domain mastery | Cost & latency |
|---|---|---|---|
| `expert` | high | high | high |
| `generalist` | middling | middling, broad | low |
| `novice` | low | low | very low |

The tiers are deliberately not rank-ordered on every axis at once. If the expert were
simply better at everything there would be one right answer and the benchmark would measure
nothing; the point is that the expert is worth its cost on hard work and wastes it on easy
work, which is exactly the judgement the priority band exists to make.

## The four axes

Quality, cost, latency, and reliability, kept separate on purpose. A policy that routes
everything to the strongest agent wins quality and loses cost; one that routes everything
to the cheapest wins cost and loses reliability. Collapsing them into a single score would
hide exactly the trade-off the mesh exists to make.

`cost_per_verified_task` is the number that decides whether a policy is affordable. A cheap
policy that never produces working output has infinite cost per unit of work, reported as
`n/a` — and a delta against it scores `-1.0`, because failing cheaply is not a saving.

Deltas are **signed so positive always means better**, including for cost and latency where
the raw numbers move the other way. A reader should not have to remember which columns are
inverted.

---

## Results

Measured on HAMNS `1.0.0`, `contested_cohort()`, 24 tasks per scenario. Reproduce with
`cargo run -p hatcher-playground --example benchmark`.

### `rehearsal` — routine work the cohort can handle (difficulty 0.25)

| Policy | Quality | Verified | Stages | Cost/task | Cost/verified | p50 latency | Escalations | Ω |
|---|---|---|---|---|---|---|---|---|
| **mesh** | 0.627 | 42% | 73% | 1.633 | **3.920** | 89 408 ms | 0% | 2.883 |
| `static-role` | 0.617 | 42% | 73% | 2.712 | 6.510 | 154 481 ms | 0% | 3.137 |
| `round-robin` | 0.512 | 21% | 55% | 1.323 | 6.352 | 77 910 ms | 46% | 0.678 |
| `cheapest` | 0.407 | 4% | 32% | 0.306 | 7.350 | 20 160 ms | 88% | 0.000 |
| `strongest` | 0.638 | 54% | 78% | 2.042 | 3.769 | 105 525 ms | 0% | 4.194 |
| `arbitrary` | 0.504 | 25% | 53% | 1.313 | 5.250 | 73 500 ms | 21% | 0.430 |

**The mesh delivers the same verified rate as the fixed assignment for 40% less money and
42% less wall-clock time.** That is the claim, and it is the one pinned by
`on_routine_work_the_mesh_matches_a_fixed_assignment_for_much_less_money`.

`strongest` edges it on cost-per-verified (3.77 vs 3.92) by buying a higher verified rate,
but costs 25% more per task and runs 18% slower. Which of those you want is a business
decision; the benchmark's job is to show you both rather than pick for you.

`cheapest` is the cautionary arm: 88% escalation, `Ω` collapsed to the floor, and the
*worst* cost per verified task despite the lowest cost per task. Routing on price alone
does not save money, it just moves the bill.

### `stress` — work at the edge of the cohort (difficulty 0.85, urgency 0.9)

| Policy | Quality | Verified | Cost/verified | Escalations |
|---|---|---|---|---|
| `mesh` | 0.499 | 25% | 14.492 | 75% |
| `static-role` | 0.499 | 25% | 14.570 | 75% |
| `strongest` | 0.529 | 29% | **11.884** | 71% |
| `round-robin` / `cheapest` / `arbitrary` | ≤0.411 | 0% | n/a | 100% |

Under stress the priority band goes `Immediate`, the cost exponent drops to 0.10, and the
mesh converges on the expert tier — so it and `static-role` become the same policy. That is
correct behaviour: on hard urgent work you should use your best agent. The mesh gets there
by reasoning rather than by configuration, which is worth something, but on this scenario
it is not worth *measurably* something.

### `frontier` — a domain nobody has mastered (difficulty 0.6, new domain)

| Policy | Quality | Verified | Cost/task | Cost/verified |
|---|---|---|---|---|
| `mesh` | 0.444 | **0%** | 2.028 | **n/a** |
| `static-role` | 0.460 | 4% | 3.255 | 78.120 |
| `strongest` | 0.524 | 4% | 2.170 | 52.080 |
| `arbitrary` | 0.423 | 4% | 1.599 | 38.377 |

**The mesh loses here, and the harness says so.**

In an unmastered domain every candidate's `mastery` is near the floor, so the mastery term
carries no signal, the cost term dominates, and the mesh under-provisions work that needed
the expert. It spends 38% less than `static-role` and produces nothing verified at all.

This is a real weakness, not a quirk of the scenario. It is pinned by
`the_benchmark_is_willing_to_report_that_the_mesh_lost`, which fails if the mesh ever
matches `strongest` here — so fixing the router forces this document to be updated rather
than letting a stale claim survive.

The likely fix is that mastery should not be allowed to collapse the score when *no*
candidate has any: a domain nobody knows is a reason to spend more, not less. That is a
router change, and it belongs in a release where it can be measured rather than bolted onto
this one.

---

## What this cannot tell you

**Whether the simulator is right.** A benchmark against a competence model measures whether
the router exploits *that model*, not whether real agents behave like it. Every number above
is a statement about HAMNS's internal physics.

That is what the replay harness is for.

---

## Replay: scoring the router against what really happened

Point it at recorded production runs and it scores the mesh's routing against the choices
that were actually made.

```jsonc
// JSON Lines — one run per line, so a runtime can append without rewriting the file
{"envelope":{"description":"…","domain":"rust","features":[…]},
 "outcomes":[{"stage":"Code","agent_id":"coder-expert","success":true,"quality":0.9,
              "latency_ms":18400,"cost":0.21,"error":"None","confidence":0.8}, …]}
```

```bash
curl -X POST localhost:3030/api/replay --data @recording.jsonl   # or: hatcher-ux → replay
```

The mesh learns from the recording as it goes, exactly as it would have in production, so
agreement is measured by a mesh that has seen the history rather than one guessing cold.

### Reading a replay report

```text
replay 40 runs | agreement 58% | agreed 100% vs disagreed 67% success
              | recorded cost 123.845 vs projected 89.761 (+28%) | Ω 7.829
```

**The two success rates are the load-bearing pair.** Agreement on its own says nothing — a
router that always picks what production picked has learned to imitate, not to route.

* `agreed_success_rate` — how often the stages went well where the mesh would have made the
  same choice.
* `disagreed_success_rate` — how often they went well where it would have chosen differently.

If the second is *lower* than the first, the mesh was disagreeing precisely where production
was struggling; `router_added_value()` reports exactly that. In the synthetic recording
above — every stage routed to the expert tier, the way a team does when it decides the safe
thing is to send everything to its best model — the mesh disagreed on 42% of stages, and
those were the stages that succeeded only 67% of the time.

`projected_cost_saving` is an **estimate, not a measurement**: it prices the mesh's picks at
their observed averages, which is not the same as having run them. Treat it as a reason to
run an experiment, not as a result.

### Generating example data

`synthetic_recording(runs, tier)` produces a recording that routes every stage to one tier
of `contested_cohort()`. It is deliberately patterned rather than noisy so the harness has
a real counterfactual to price. **Real recordings will be considerably messier**, and the
first thing worth checking against real data is whether `routing_agreement` is suspiciously
high — that usually means the mesh has been handed a cohort with no alternatives, not that
it agrees with you.
