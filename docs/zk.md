# Attestation and zero knowledge

Split into two milestones, deliberately.

| Milestone | Status | What it is |
|---|---|---|
| **1 — canonical digest and envelope** | **shipped in 1.0.0** | a reproducible commitment over any mesh artifact |
| **2 — a real ZK proof** | not started, specified below | proving a statement about a run without revealing the run |

Milestone 1 is done and useful on its own. Milestone 2 is specified here rather than built,
because the honest answer to "should this be a ZK proof?" depends on facts about the
deployment that are not yet settled — and this document says which facts.

---

# Milestone 1 — the canonical digest

Shipped. Defined in [`hatcher_core::envelope`](../crates/hatcher-core/src/envelope.rs).

## The commitment

```text
digest(v) = SHA-256( "zk-canonical-v1" ‖ ":" ‖ canonical_json(v) )
```

* `canonical_json` is `serde_json` over a `serde_json::Value`, whose maps are `BTreeMap`s,
  so **object keys are sorted** and the encoding does not depend on struct field order.
* The codec tag is inside the hash, not beside it. A future `zk-canonical-v2` therefore
  produces different digests for the same payload by construction, rather than by
  convention.

Folding many digests into one, order-sensitively:

```text
fold(d₁…dₙ) = SHA-256( "zk-fold-v1:" ‖ d₁ ‖ … ‖ dₙ )
```

An `Envelope<T>` is `{ codec, schema_version, payload, digest }` and `verify()` recomputes
the digest and compares. `SCHEMA_VERSION` is `3` as of 1.0.0 — it was bumped when stage
records gained quality, latency, error class, and provenance, so a digest taken under
schema 2 does not compare to one taken under 3. That is the entire point of the field.

## The commitment tree

```text
mesh_digest   = fold( digest(node₁) … digest(nodeₙ),
                      digest(edges), digest(global), digest(head) )

head          = { name, model, version, input_dim, output_dim, degraded: bool }
                // identity only — never the degradation *reason*, which contains a
                // local filesystem path and would make the digest machine-dependent

trace_digest  = digest({
                  task_id,
                  stages,        // every StageRecord: agent, success, quality,
                                 // latency_ms, cost, error, provenance, note
                  omega_after,
                  verified,
                  provenance,    // Simulated | Reported | Mixed
                  mesh_digest,   // the mesh that produced it
                })
```

Three properties follow, and they are what make the digest an attestation rather than a
souvenir:

1. **A trace commits to the mesh that produced it, including the decision head.** You
   cannot present a trace from a well-trained mesh as having come from a fresh one, and you
   cannot swap the policy that proposed the action without the commitment changing. That
   last part is what makes statement **S2** below checkable at all.
2. **A trace commits to its provenance.** A rehearsal and a production run over identical
   outcomes produce *different* digests — pinned by
   `a_rehearsal_and_a_real_run_never_share_a_digest`. Provenance is inside the commitment
   specifically so a simulation can never be presented as evidence of work.
3. **Runs are replayable.** Simulated outcomes come from a hash of
   `(task, agent, stage, sequence)`, never a clock or an RNG, so the same task against the
   same mesh state reproduces the same digest on any machine.

## What milestone 1 already gives you

A `MeshReceipt` carries `trace_digest` and `mesh_digest`. A caller that keeps nothing but
receipts can still demonstrate, later, exactly what the mesh said and what state it said it
from — provided both parties have the underlying data to recompute against.

That last clause is the limitation, and it is the whole reason milestone 2 exists.

## What milestone 1 does **not** give you

* **No signature.** A digest proves integrity, not origin. Anyone can compute one. There is
  currently nothing binding a receipt to *this* mesh instance rather than a mesh someone
  ran on their laptop with a favourable cohort.
* **No privacy.** Verifying a digest requires the payload. Handing someone a trace to check
  means handing them the task ids, the agent ids, the per-stage costs, and the quality
  scores.
* **No inclusion proofs.** `fold_digests` is a flat concatenation hash, not a Merkle tree,
  so proving "node *k* was in this mesh" requires all *n* nodes. A Merkle root would fix
  this and is cheap; it is not in 1.0.0 because nothing needs it yet.
* **Not a domain-separated hash-to-field.** SHA-256 over JSON is the wrong primitive to
  prove *inside* a circuit — see below.

---

# Milestone 2 — what a ZK proof would prove, and why

## The setting

Once Hatcher runs the mesh against real agents, there are two parties who want incompatible
things:

* **The operator** holds the interesting data: which tasks were run, which agents ran them,
  what each cost, what the verifier scored. This is commercially sensitive — it is a
  detailed map of a company's engineering workload and its vendors' comparative performance.
* **A verifier** — a customer, an auditor, a counterparty, a governance process — wants to
  believe a claim about that data without being handed it.

ZK is only the right tool when both halves are true: **the inputs must stay private, and
the verifier must not be willing to take the operator's word for it.** If either fails,
something cheaper is correct, and this document says so explicitly rather than assuming the
answer.

## The candidate statements

Four, in increasing order of how much they are worth.

### S1 — Routing integrity

> "Given a mesh committed to by `mesh_digest` and a task committed to by `task_digest`, the
> agent named in the receipt is the arg-max of the published scoring function over the
> eligible candidate set."

*Prevents:* an operator hand-picking which agent gets credit for a run, or claiming a
routing decision was principled when it was manual.

*Honest assessment:* **weak on its own.** The operator controls the mesh state that is
committed to, so it can produce any routing outcome it wants by choosing the cohort. This
statement is only meaningful in combination with S3.

### S2 — Policy compliance

> "The `action` in the receipt is the one the published guardrail policy produces from the
> committed stage outcomes."

Concretely: no run marked `stabilize` had `verified = false`; every unverified run with
`priority ≥ escalation_priority` was escalated; every unverified run in the `Degraded`
regime was escalated.

*Prevents:* the failure mode the guardrails exist for — a mesh quietly marking unverified
work as accepted, at scale, in `Production` execution mode.

*Honest assessment:* **the most defensible statement in the list.** It is a pure function of
data already inside the commitment, the policy is small and public, and the property is one
an auditor would actually ask about. If only one statement gets built, build this one.

Note that this statement is about the *guardrails*, not the model. The guardrails are a
handful of comparisons and are cheap to arithmetize; the decision head is a neural network
and proving its execution is a different and much larger problem. That split is deliberate
and it is the reason S2 is tractable: the mesh's safety property — no unverified work marked
`stabilize`, high-priority unverified work always escalated — holds *regardless* of what the
head proposed, so proving it never requires proving the model ran correctly. The head's
identity is in the commitment, so a verifier knows which model was in play without anyone
having to prove its arithmetic.

### S3 — Aggregate performance over private runs

> "Over *N* runs committed to by a Merkle root *R*, the verified rate was ≥ *x*, the mean
> verifier score ≥ *q*, and the mean cost per verified task ≤ *c* — revealing nothing about
> any individual run, task, or agent."

*Prevents:* nothing, exactly. It **enables** something: making a checkable performance claim
without publishing the workload.

*Honest assessment:* **the only statement with a commercial reason to exist.** It is the
one where privacy is not a nice-to-have but the entire product: "our mesh achieves a 94%
verified rate at $0.31 per verified task" is a claim a customer wants verified and an
operator cannot substantiate without exposing its book of work. Everything else on this
list has a cheaper non-ZK answer.

### S4 — Learning integrity

> "`Ω_after = f(Ω_before, L, E, C, F, D)` for the committed coefficients, and each term was
> computed from the committed outcomes."

*Prevents:* an operator inflating the headline intelligence number.

*Honest assessment:* **only worth proving if `Ω` becomes externally consequential** — priced,
reported, or used to gate something. Today it is a diagnostic. Proving a diagnostic is
theatre.

## Why not just sign it?

This is the question that decides the milestone, so it should be answered before any circuit
is written.

| Mechanism | Gives you | Costs | Fails when |
|---|---|---|---|
| **Signed receipt** (Ed25519 over `trace_digest`) | origin + integrity | ~nothing | the verifier does not trust the signer's *honesty*, only its identity |
| **TEE attestation** | the above, plus "the published binary computed it" | moderate; a hardware trust assumption | you distrust the silicon vendor, or need the guarantee to outlive the enclave |
| **ZK proof** | the above, plus **the inputs never leave** | high — circuit, prover, a new primitive stack | the statement is about data the verifier is entitled to see anyway |

**For S1, S2, and S4, a signed receipt is very likely sufficient**, and it is roughly a day
of work rather than a quarter. The verifier in those cases is a customer or auditor who
already trusts the operator enough to be doing business with it; what they want is a
tamper-evident record, and a signature provides that.

**S3 is the one that genuinely needs ZK**, because there the verifier is being asked to
believe a summary statistic over data it will never see, and no signature can make an
unverifiable aggregate verifiable.

So the recommendation is: **build signing first, and treat ZK as the S3 feature.** Doing
this in the other order would produce an expensive proof of something a signature already
covered.

## What would have to change in the code

The current digest is not provable as-is. SHA-256 over canonical JSON is excellent for
reproducibility and terrible inside a circuit — JSON serialization is not arithmetizable,
and SHA-256 is expensive in every proof system that is not specifically built around it.

A milestone-2 implementation needs:

1. **A field-friendly parallel commitment.** Keep the JSON digest as the interoperability
   format; add a Poseidon (or Rescue) commitment over a fixed-width field encoding of the
   same data. Both go in the envelope. The JSON digest stays authoritative for humans and
   HTTP; the algebraic one is what the circuit consumes.
2. **A fixed-width canonical encoding.** Every provable field pinned to a known number of
   field elements: `success` as a bit, `quality`/`cost`/`latency` as fixed-point integers
   with a declared scale, `error` as a small enum index, ids as hashes rather than strings.
   Variable-length strings cannot be in the provable part.
3. **A Merkle tree, not a fold.** `fold_digests` cannot produce an inclusion proof.
   S3 needs one per run.
4. **The guardrail policy as an arithmetic circuit.** For S2 this is genuinely small — a
   handful of comparisons and boolean gates over values already in the commitment. This is
   the cheapest statement to build and the best place to start.
5. **A pinned proof system and trust-setup story**, with the ceremony question answered
   before anything is deployed.

## Open questions for the review

These are decisions for HatcherLabs, not for this repository. They determine whether
milestone 2 is worth starting at all.

1. **Who is the verifier?** A customer reading a dashboard, an auditor with a contract, a
   regulator, or a smart contract? Only the last two make a signature clearly insufficient.
2. **Is the data actually private?** If the verifier is entitled to the trace anyway, the
   whole premise collapses and S2-with-a-signature is the right answer.
3. **Is `Ω` ever going to be externally consequential?** If not, drop S4.
4. **On-chain or off-chain verification?** On-chain forces a proof system choice and makes
   proof size a hard constraint; off-chain leaves both open.
5. **What is the claim you actually want to sell?** If it is a performance number over
   private workloads, that is S3 and ZK is justified. If it is "we did not mark broken work
   as done", that is S2 and a signature will do.

Until questions 1 and 2 have answers, building a circuit would be committing significant
effort to a guess about what needs proving.
