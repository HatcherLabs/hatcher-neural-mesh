//! The Hatcher integration contract: the typed surface a real agent runtime binds to.
//!
//! Everything else in this crate describes what the mesh *is*. This module describes
//! how something outside the mesh talks to it, and it is deliberately the narrowest
//! surface that supports the whole loop:
//!
//! ```text
//!  1. register   AgentRegistration   →  RegistrationAck
//!  2. submit     TaskEnvelope        →  RoutingPlan      (who should do what)
//!  3. report     StageOutcomeReport  →  ()               (what actually happened)
//!  4. receive    finalize            →  MeshReceipt      (decision + trace digest)
//! ```
//!
//! ## Why steps 2 and 3 are separate
//!
//! The mesh does not execute agents. It decides *which* agent should run a stage, and
//! it learns from what that agent then did. In simulation those two halves collapse
//! into one call, because the outcome is derived from a hash and is available
//! immediately. Against a real runtime they cannot collapse: minutes of real work
//! happen in between, and the mesh must not invent the result.
//!
//! So a run is a session. [`RoutingPlan`] is the mesh's half — it names an agent per
//! stage and is issued against a committed mesh digest. [`StageOutcomeReport`] is the
//! caller's half — success, a verifier score, latency, cost, and a classified error.
//! Only when the reports are in does the mesh move trust, memory, capability, and `Ω`.
//!
//! ## Why the error class matters
//!
//! A failed stage is not automatically evidence that the agent is weak. A provider
//! outage, a rate limit, and a caller-side cancellation all produce a failed stage, and
//! a mesh that treats them as capability signals will quietly learn to distrust a
//! perfectly good agent because someone else's service was down. Every reported failure
//! therefore carries an [`ErrorClass`], and every class carries an [`Attribution`] that
//! decides how much of the failure the agent actually wears. See
//! [`ErrorClass::blame_weight`].
//!
//! ## Simulated versus reported
//!
//! Both kinds of run produce the same [`PipelineTrace`](crate::task::PipelineTrace) and
//! the same digest machinery, which is exactly why the trace records its
//! [`OutcomeProvenance`]. A digest over simulated outcomes attests to a rehearsal; a
//! digest over reported outcomes attests to work that was really done. They must never
//! be confusable, so provenance is inside the commitment, not beside it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::agent::{AgentRole, CapabilityVector, ExecutionMode, ResourceProfile};
use crate::api::NeuralSignal;
use crate::graph::MeshIntelligence;
use crate::task::{PipelineStage, PriorityBand, PriorityScore, TaskConstraints, TaskSpec};

/// Version of this integration contract, independent of the crate version.
///
/// A caller pins this. It changes only when the wire shape changes in a way that would
/// break a client that was compiled against the previous value.
pub const CONTRACT_VERSION: &str = "1.0";

// ---------------------------------------------------------------------------
// Failure classification
// ---------------------------------------------------------------------------

/// Who a failure belongs to.
///
/// This is the single most important judgement in the contract, because it decides
/// what the mesh is allowed to learn from a failure.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Attribution {
    /// The agent did the work and the work was not good enough. Full learning signal.
    Agent,
    /// Something under the agent broke — provider, network, quota, host. The agent is
    /// partially accountable (it chose to depend on that thing, and the run still cost
    /// real time) but it is not evidence about its reasoning.
    Environment,
    /// The caller withdrew the work. Not evidence about anything; discarded.
    Caller,
}

/// Why a stage failed.
///
/// `None` is the success case. Every other variant is a distinct thing a real agent
/// runtime can report, chosen so that a caller never has to squeeze an outage into a
/// bucket that means "the agent was wrong".
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorClass {
    /// The stage succeeded.
    #[default]
    None,
    /// The output was produced but failed review or verification.
    Quality,
    /// The agent declined the work.
    Refusal,
    /// The output did not conform to the required schema or contract.
    Schema,
    /// A tool, subprocess, or downstream call the agent made failed.
    Tool,
    /// The agent did not finish inside its time budget.
    Timeout,
    /// Throttled or out of quota upstream.
    RateLimit,
    /// Provider, network, or host failure below the agent.
    Infrastructure,
    /// The caller cancelled before the stage completed.
    Cancelled,
    /// Reported as failed without a class. Treated as the agent's, because assuming
    /// otherwise would let an unclassified failure launder itself into a free pass.
    Unknown,
}

impl ErrorClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorClass::None => "none",
            ErrorClass::Quality => "quality",
            ErrorClass::Refusal => "refusal",
            ErrorClass::Schema => "schema",
            ErrorClass::Tool => "tool",
            ErrorClass::Timeout => "timeout",
            ErrorClass::RateLimit => "rate_limit",
            ErrorClass::Infrastructure => "infrastructure",
            ErrorClass::Cancelled => "cancelled",
            ErrorClass::Unknown => "unknown",
        }
    }

    /// Who wears this failure.
    ///
    /// `Timeout` is the interesting one: it is attributed to the agent, not the
    /// environment, because taking too long *is* a performance property of an agent and
    /// the resource ratio `R_i` already exists to price it. `RateLimit` is not, because
    /// being throttled is a property of the account, not the reasoning.
    pub fn attribution(&self) -> Attribution {
        match self {
            ErrorClass::None | ErrorClass::Quality | ErrorClass::Refusal | ErrorClass::Schema => Attribution::Agent,
            ErrorClass::Timeout | ErrorClass::Tool | ErrorClass::Unknown => Attribution::Agent,
            ErrorClass::RateLimit | ErrorClass::Infrastructure => Attribution::Environment,
            ErrorClass::Cancelled => Attribution::Caller,
        }
    }

    /// How much of this failure moves the agent's own state, in `[0, 1]`.
    ///
    /// Environment failures are damped rather than ignored: a run still burned time and
    /// the mesh still did not deliver, so the cohort should mildly prefer an agent whose
    /// dependencies stay up — but one bad afternoon at a provider must not undo months
    /// of measured capability.
    pub fn blame_weight(&self) -> f64 {
        match self.attribution() {
            Attribution::Agent => 1.0,
            Attribution::Environment => 0.35,
            Attribution::Caller => 0.0,
        }
    }

    /// Whether this outcome should be written to the trust ledger at all.
    ///
    /// Trust is a claim about whether one agent can rely on another's *work*. An outage
    /// is not a betrayal, so it is excluded rather than damped — recording it at reduced
    /// weight is not possible in a ledger whose entries are booleans, and recording it
    /// at full weight would be a lie.
    pub fn is_trust_evidence(&self) -> bool {
        matches!(self.attribution(), Attribution::Agent)
    }

    /// Whether the run should be counted as an attempt in the agent's telemetry.
    pub fn is_countable(&self) -> bool {
        !matches!(self.attribution(), Attribution::Caller)
    }
}

// ---------------------------------------------------------------------------
// The decision head
// ---------------------------------------------------------------------------

/// Which model proposed the action on a receipt.
///
/// The equations govern the graph; the decision head governs which of the four control
/// actions goes back to Hatcher. Those are separate systems and they can be swapped
/// independently, so a caller acting on a `stabilize` is entitled to know whether it came
/// from the built-in head or from a trained policy — and whether the mesh is running the
/// policy it was *asked* to run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionHead {
    /// `native`, or `onnx:<file stem>`.
    pub name: String,
    /// The shape contract the head was bound against.
    pub model: String,
    pub version: String,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Why the mesh is *not* running the head it was asked for.
    ///
    /// A failed ONNX load degrades to the built-in head rather than leaving the mesh
    /// unable to decide anything — but silently degrading would mean an operator acting on
    /// decisions from a model they think they replaced. `Some` here says they did not.
    pub degraded: Option<String>,
}

impl DecisionHead {
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        version: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        Self {
            name: name.into(),
            model: model.into(),
            version: version.into(),
            input_dim,
            output_dim,
            degraded: None,
        }
    }

    pub fn with_degraded(mut self, reason: Option<String>) -> Self {
        self.degraded = reason;
        self
    }

    /// True when the mesh is running the head that was requested.
    pub fn is_intact(&self) -> bool {
        self.degraded.is_none()
    }

    /// Whether this is a loaded policy rather than the built-in network.
    pub fn is_policy(&self) -> bool {
        self.name != "native"
    }

    /// What a digest commits to.
    ///
    /// Identity and dimensions, plus *whether* the head is degraded — never the reason.
    /// The reason contains a local filesystem path, and committing to it would make the
    /// same mesh running the same model hash differently on two machines.
    pub fn commitment(&self) -> HeadCommitment<'_> {
        HeadCommitment {
            name: &self.name,
            model: &self.model,
            version: &self.version,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            degraded: self.degraded.is_some(),
        }
    }
}

/// The machine-independent part of a [`DecisionHead`], for the mesh commitment.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HeadCommitment<'a> {
    pub name: &'a str,
    pub model: &'a str,
    pub version: &'a str,
    pub input_dim: usize,
    pub output_dim: usize,
    pub degraded: bool,
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// Where the outcomes in a trace came from.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum OutcomeProvenance {
    /// Every outcome was derived from the deterministic simulator. A rehearsal.
    #[default]
    Simulated,
    /// Every staffed stage was reported by a real runtime.
    Reported,
    /// Some stages were reported and some were simulated. A trace in this state is
    /// useful for debugging and must not be treated as an attestation of real work.
    Mixed,
}

impl OutcomeProvenance {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutcomeProvenance::Simulated => "simulated",
            OutcomeProvenance::Reported => "reported",
            OutcomeProvenance::Mixed => "mixed",
        }
    }

    /// True only when every outcome came from the outside world.
    pub fn is_real(&self) -> bool {
        matches!(self, OutcomeProvenance::Reported)
    }

    /// Combine per-stage provenance into the trace-level verdict.
    pub fn merge(self, other: OutcomeProvenance) -> OutcomeProvenance {
        if self == other {
            self
        } else {
            OutcomeProvenance::Mixed
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Registration
// ---------------------------------------------------------------------------

/// Register an agent and declare what it can do.
///
/// The declared capability vector is a *prior*, not an assertion. The moment the agent
/// starts reporting outcomes, `P` is measured from its success history, `M` from what it
/// commits to memory, `S` from what it repeatedly does well, and the declaration stops
/// mattering. Declaring `1.0` across the board buys about three tasks of optimism.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentRegistration {
    /// Stable identifier, unique in the mesh. Reused across restarts.
    pub id: String,
    /// Human-readable name for dashboards.
    pub label: String,
    pub role: AgentRole,
    /// Declared `I, S, P, C, M` prior.
    #[serde(default)]
    pub capability: CapabilityVector,
    /// Declared cost and latency prior. Superseded by observation once reports arrive.
    #[serde(default)]
    pub resources: ResourceProfile,
    /// Starting calibration `C_i`, in `[0, 1]`.
    #[serde(default = "half")]
    pub confidence: f64,
    /// Declared per-domain mastery, `domain -> [0, 1]`.
    #[serde(default)]
    pub expertise: BTreeMap<String, f64>,
    /// Free-form tags carried through to the roster. Not interpreted by the mesh.
    #[serde(default)]
    pub tags: Vec<String>,
}

fn half() -> f64 {
    0.5
}

impl AgentRegistration {
    pub fn new(id: impl Into<String>, label: impl Into<String>, role: AgentRole) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            role,
            capability: CapabilityVector::default(),
            resources: ResourceProfile::default(),
            confidence: 0.5,
            expertise: BTreeMap::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_capability(mut self, capability: CapabilityVector) -> Self {
        self.capability = capability;
        self
    }

    pub fn with_resources(mut self, resources: ResourceProfile) -> Self {
        self.resources = resources;
        self
    }

    pub fn with_expertise(mut self, domain: impl Into<String>, mastery: f64) -> Self {
        self.expertise.insert(domain.into(), mastery);
        self
    }

    /// Reject anything the mesh cannot represent.
    ///
    /// Out-of-range factors are rejected rather than clamped. Clamping at a contract
    /// boundary turns a caller's bug into a silently different mesh, and the caller
    /// never finds out; a `400` is more useful than a mesh that quietly disagrees with
    /// the client about what it registered.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.id.trim().is_empty() {
            return Err(ContractError::invalid("id", "must not be empty"));
        }
        for (name, value) in [
            ("capability.intelligence", self.capability.intelligence),
            ("capability.specialization", self.capability.specialization),
            ("capability.performance", self.capability.performance),
            ("capability.context", self.capability.context),
            ("capability.memory", self.capability.memory),
            ("confidence", self.confidence),
        ] {
            unit_range(name, value)?;
        }
        for (name, value) in [
            ("resources.energy", self.resources.energy),
            ("resources.latency", self.resources.latency),
        ] {
            unit_range(name, value)?;
        }
        for (domain, mastery) in &self.expertise {
            unit_range(&format!("expertise.{domain}"), *mastery)?;
        }
        Ok(())
    }
}

fn unit_range(name: &str, value: f64) -> Result<(), ContractError> {
    if !value.is_finite() {
        return Err(ContractError::invalid(name, "must be a finite number"));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ContractError::invalid(name, "must be within [0, 1]"));
    }
    Ok(())
}

/// What the mesh says back after a registration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegistrationAck {
    pub contract_version: String,
    pub agent_id: String,
    /// False when the id already existed and the record was updated in place.
    pub created: bool,
    pub cohort_size: usize,
    /// `A_i` the mesh computed from the declared vector.
    pub capability: f64,
    /// The factor currently capping this agent, so a caller can see immediately why a
    /// generous declaration did not buy what it expected.
    pub bottleneck: String,
    /// Mesh commitment after the registration was applied.
    pub mesh_digest: String,
}

// ---------------------------------------------------------------------------
// 2. Task submission
// ---------------------------------------------------------------------------

/// A unit of work handed to the mesh, with its measurable features.
///
/// `features` is the honest part of this struct. The mesh derives uncertainty, budget,
/// and implementation cost from feature dispersion, mass, and width — so a caller that
/// supplies real measurements (diff size, file count, test coverage, retrieval hit rate,
/// token estimate) gets real routing, and a caller that supplies noise gets noise. The
/// explicit `uncertainty` / `budget` / `implementation_cost` overrides exist for callers
/// that already measure those directly and would rather not round-trip through features.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskEnvelope {
    /// Caller-supplied id. When absent the mesh assigns a sequence-stable one.
    #[serde(default)]
    pub task_id: Option<String>,
    pub description: String,
    /// Domain tag used for specialization routing, e.g. `"rust"`, `"infra"`.
    #[serde(default)]
    pub domain: Option<String>,
    /// Measurable features of the request. Order is caller-defined but must be stable:
    /// the mesh treats slot `k` as the same measurement across tasks.
    #[serde(default)]
    pub features: Vec<f64>,
    /// `[0, 1]`. Defaults to `0.5`.
    #[serde(default)]
    pub urgency: Option<f64>,
    #[serde(default)]
    pub execution_mode: Option<ExecutionMode>,
    /// Direct `U` override, skipping feature derivation.
    #[serde(default)]
    pub uncertainty: Option<f64>,
    /// Direct `B` override.
    #[serde(default)]
    pub budget: Option<f64>,
    /// Direct `I` override.
    #[serde(default)]
    pub implementation_cost: Option<f64>,
    /// Hard limits the router must respect when it can.
    #[serde(default)]
    pub constraints: TaskConstraints,
    /// Opaque caller metadata, echoed back on the receipt. Never interpreted.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl TaskEnvelope {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            task_id: None,
            description: description.into(),
            domain: None,
            features: Vec::new(),
            urgency: None,
            execution_mode: None,
            uncertainty: None,
            budget: None,
            implementation_cost: None,
            constraints: TaskConstraints::default(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }

    pub fn with_constraints(mut self, constraints: TaskConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = Some(urgency);
        self
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.description.trim().is_empty() {
            return Err(ContractError::invalid("description", "must not be empty"));
        }
        if self.features.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::invalid("features", "must all be finite numbers"));
        }
        for (name, value) in [
            ("urgency", self.urgency),
            ("uncertainty", self.uncertainty),
            ("budget", self.budget),
            ("implementation_cost", self.implementation_cost),
        ] {
            if let Some(value) = value {
                unit_range(name, value)?;
            }
        }
        self.constraints.validate()
    }

    /// Turn the envelope into a task, deriving `U`, `B`, `I` from features and then
    /// applying any explicit overrides on top.
    pub fn to_task(&self, sequence: u64) -> TaskSpec {
        let id = self
            .task_id
            .clone()
            .unwrap_or_else(|| format!("task-{sequence:06}"));

        let mut task = TaskSpec::new(
            id,
            self.description.clone(),
            self.domain.clone().unwrap_or_else(|| "general".to_string()),
        )
        .with_features(self.features.clone())
        .with_execution_mode(self.execution_mode.unwrap_or(ExecutionMode::Controlled))
        .parsed_from_features()
        .with_constraints(self.constraints);

        task.urgency = self.urgency.unwrap_or(0.5).clamp(0.0, 1.0);
        if let Some(uncertainty) = self.uncertainty {
            task.uncertainty = uncertainty.clamp(0.0, 1.0);
        }
        if let Some(budget) = self.budget {
            task.budget = budget.clamp(0.0, 1.0);
        }
        if let Some(cost) = self.implementation_cost {
            task.implementation_cost = cost.clamp(0.0, 1.0);
        }
        task
    }
}

/// One stage of a routing plan: the agent the mesh chose, and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannedStage {
    pub stage: PipelineStage,
    pub agent_id: String,
    pub role: AgentRole,
    /// Composite routing score that won the assignment.
    pub score: f64,
    /// `A_i` at planning time.
    pub capability: f64,
    /// Mean trust flowing into this agent from the cohort.
    pub inbound_trust: f64,
    /// Domain mastery at planning time.
    pub mastery: f64,
    /// What the mesh expects this stage to take, from the agent's observed history.
    pub expected_latency_ms: f64,
    /// What the mesh expects this stage to cost, in the caller's cost units.
    pub expected_cost: f64,
    /// Why this agent and not another, in one line.
    pub rationale: String,
}

/// The mesh's answer to "who should do this".
///
/// A plan is issued against a specific mesh state, named by `mesh_digest`. If the mesh
/// moves before the plan is finalized — because another run completed — the plan is
/// still honoured as written. That is deliberate: the caller has already dispatched real
/// work to these agents, and silently re-routing under it would make the reported
/// outcomes describe a run that never happened.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingPlan {
    pub contract_version: String,
    /// Handle for reporting outcomes and finalizing.
    pub run_id: String,
    pub task_id: String,
    pub domain: String,
    pub execution_mode: ExecutionMode,
    pub priority: PriorityScore,
    pub band: PriorityBand,
    /// One entry per staffed stage, in pipeline order.
    pub stages: Vec<PlannedStage>,
    /// Sum of `expected_latency_ms` across the plan.
    pub expected_latency_ms: f64,
    /// Sum of `expected_cost` across the plan.
    pub expected_cost: f64,
    /// Mesh commitment the plan was computed against.
    pub mesh_digest: String,
    /// `Ω` at planning time.
    pub omega: f64,
    /// Stages whose constraint could not be satisfied by any candidate. The plan still
    /// names an agent for them — refusing to route is worse than routing over budget —
    /// but the caller is told, so it can decline instead.
    #[serde(default)]
    pub constraint_violations: Vec<String>,
}

impl RoutingPlan {
    /// The agent the mesh holds accountable for the run's output.
    ///
    /// The coding station when one is staffed, because that is where the artifact is
    /// produced; otherwise the highest-scoring assignment.
    pub fn primary_agent(&self) -> Option<&str> {
        self.stages
            .iter()
            .find(|planned| planned.stage == PipelineStage::Code)
            .or_else(|| {
                self.stages
                    .iter()
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|planned| planned.agent_id.as_str())
    }

    pub fn agent_for(&self, stage: PipelineStage) -> Option<&str> {
        self.stages
            .iter()
            .find(|planned| planned.stage == stage)
            .map(|planned| planned.agent_id.as_str())
    }
}

// ---------------------------------------------------------------------------
// 3. Outcome reporting
// ---------------------------------------------------------------------------

/// What actually happened when a real agent ran a stage.
///
/// This is the struct that turns HAMNS from a simulator into a system that learns.
/// Every field here is something a real runtime already knows and would otherwise
/// throw away.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageOutcomeReport {
    pub stage: PipelineStage,
    /// The agent that ran it. Must match the plan; a mismatch is rejected, because a
    /// mesh that credits outcomes to the wrong agent learns the exact opposite of the
    /// truth.
    pub agent_id: String,
    pub success: bool,
    /// Verifier or grader score in `[0, 1]`.
    ///
    /// Distinct from `confidence`: this is what the *reviewer* thought of the output,
    /// not what the producer thought of it. When a caller has no grader, report `1.0`
    /// on success and `0.0` on failure and the mesh degrades to a binary signal.
    pub quality: f64,
    /// Wall-clock time the stage took, in milliseconds.
    pub latency_ms: f64,
    /// What the stage cost, in the caller's own units — dollars, tokens, credits. The
    /// mesh only ever compares costs to each other, so the unit is the caller's choice
    /// as long as it stays the same one.
    pub cost: f64,
    /// The agent's own confidence in the output it produced, in `[0, 1]`.
    #[serde(default = "half")]
    pub confidence: f64,
    /// Why it failed. `None` on success.
    #[serde(default)]
    pub error: ErrorClass,
    /// Free-form detail for the trace. Not interpreted.
    #[serde(default)]
    pub note: String,
}

impl StageOutcomeReport {
    /// A successful stage.
    pub fn success(stage: PipelineStage, agent_id: impl Into<String>, quality: f64) -> Self {
        Self {
            stage,
            agent_id: agent_id.into(),
            success: true,
            quality,
            latency_ms: 0.0,
            cost: 0.0,
            confidence: quality,
            error: ErrorClass::None,
            note: String::new(),
        }
    }

    /// A failed stage with a classified cause.
    pub fn failure(stage: PipelineStage, agent_id: impl Into<String>, error: ErrorClass) -> Self {
        Self {
            stage,
            agent_id: agent_id.into(),
            success: false,
            quality: 0.0,
            latency_ms: 0.0,
            cost: 0.0,
            confidence: 0.5,
            error,
            note: String::new(),
        }
    }

    pub fn with_latency_ms(mut self, latency_ms: f64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = note.into();
        self
    }

    /// Reject reports the mesh cannot learn from.
    ///
    /// The success/error cross-check is the load-bearing one: a report that says
    /// `success: true, error: Timeout` is a client bug, and accepting it would put a
    /// contradiction inside a digest that is supposed to be an attestation.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.agent_id.trim().is_empty() {
            return Err(ContractError::invalid("agent_id", "must not be empty"));
        }
        unit_range("quality", self.quality)?;
        unit_range("confidence", self.confidence)?;
        if !self.latency_ms.is_finite() || self.latency_ms < 0.0 {
            return Err(ContractError::invalid("latency_ms", "must be finite and non-negative"));
        }
        if !self.cost.is_finite() || self.cost < 0.0 {
            return Err(ContractError::invalid("cost", "must be finite and non-negative"));
        }
        if self.success && self.error != ErrorClass::None {
            return Err(ContractError::invalid(
                "error",
                "a successful stage must report ErrorClass::None",
            ));
        }
        if !self.success && self.error == ErrorClass::None {
            return Err(ContractError::invalid(
                "error",
                "a failed stage must report why it failed",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 4. The receipt
// ---------------------------------------------------------------------------

/// What one stage did, as it appears on a receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageReceipt {
    pub stage: PipelineStage,
    pub agent_id: Option<String>,
    pub success: bool,
    pub quality: f64,
    pub latency_ms: f64,
    pub cost: f64,
    pub error: ErrorClass,
    pub provenance: OutcomeProvenance,
}

/// The mesh's final answer for a run: what it decided, and a digest that pins it.
///
/// This is the whole of step 4. A caller that keeps nothing but the receipt can still
/// prove later what the mesh said, because `trace_digest` commits to the stages, the
/// outcomes, the provenance, and the mesh that produced them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshReceipt {
    pub contract_version: String,
    pub run_id: String,
    pub task_id: String,
    /// True unless the mesh escalated. An escalation is the mesh declining to own the
    /// outcome, so it is the exact inverse.
    pub accepted: bool,
    /// The agent the mesh holds accountable for the output.
    pub selected_agent: Option<String>,
    /// `observe` | `delegate` | `stabilize` | `escalate`, plus confidence and rationale.
    pub decision: NeuralSignal,
    /// Which model proposed that action, and whether it is the one that was asked for.
    pub decision_head: DecisionHead,
    /// Whether the work passed review and verification.
    pub verified: bool,
    /// Commitment over this run.
    pub trace_digest: String,
    /// Commitment over the mesh after the run was applied.
    pub mesh_digest: String,
    /// Whether these outcomes were real or simulated. Read this before believing
    /// anything else on the receipt.
    pub provenance: OutcomeProvenance,
    pub omega_before: f64,
    pub omega_after: f64,
    pub intelligence: MeshIntelligence,
    pub priority: PriorityScore,
    pub stages: Vec<StageReceipt>,
    /// Sum of reported (or simulated) stage latency.
    pub total_latency_ms: f64,
    /// Sum of reported (or simulated) stage cost.
    pub total_cost: f64,
    /// Mean verifier score across staffed stages.
    pub mean_quality: f64,
    /// Echoed back from the submitted envelope.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// How reported absolutes become the normalized quantities the equations use.
///
/// The equation set works in `[0, 1]`: `R_i = P_i / (Energy_i + Latency_i)` is only
/// meaningful if energy and latency are on a common scale. Real reports arrive in
/// milliseconds and dollars, so something has to define what "1.0" means. That is this
/// struct, and it is explicit rather than inferred because an inferred ceiling would
/// move every agent's score whenever one slow outlier appeared.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RuntimeCalibration {
    /// Latency that normalizes to `1.0` — "the slowest agent we would tolerate".
    pub latency_ceiling_ms: f64,
    /// Cost that normalizes to `1.0`, in the caller's cost units.
    pub cost_ceiling: f64,
    /// EWMA rate for folding a new observation into an agent's profile, in `[0, 1]`.
    /// Low values keep a profile stable; high values let it chase the last run.
    pub observation_rate: f64,
}

impl Default for RuntimeCalibration {
    fn default() -> Self {
        Self {
            latency_ceiling_ms: 60_000.0,
            cost_ceiling: 1.0,
            observation_rate: 0.25,
        }
    }
}

impl RuntimeCalibration {
    pub fn new(latency_ceiling_ms: f64, cost_ceiling: f64) -> Self {
        Self {
            latency_ceiling_ms,
            cost_ceiling,
            ..Self::default()
        }
    }

    /// Milliseconds into the normalized latency the equations consume.
    pub fn normalize_latency(&self, latency_ms: f64) -> f64 {
        if self.latency_ceiling_ms <= 0.0 {
            return 0.0;
        }
        (latency_ms / self.latency_ceiling_ms).clamp(0.0, 1.0)
    }

    /// Cost units into normalized energy.
    pub fn normalize_cost(&self, cost: f64) -> f64 {
        if self.cost_ceiling <= 0.0 {
            return 0.0;
        }
        (cost / self.cost_ceiling).clamp(0.0, 1.0)
    }

    /// Normalized latency back into milliseconds, for expectations on a plan.
    pub fn denormalize_latency(&self, normalized: f64) -> f64 {
        normalized.clamp(0.0, 1.0) * self.latency_ceiling_ms
    }

    /// Normalized energy back into cost units.
    pub fn denormalize_cost(&self, normalized: f64) -> f64 {
        normalized.clamp(0.0, 1.0) * self.cost_ceiling
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if !(self.latency_ceiling_ms.is_finite() && self.latency_ceiling_ms > 0.0) {
            return Err(ContractError::invalid("latency_ceiling_ms", "must be finite and positive"));
        }
        if !(self.cost_ceiling.is_finite() && self.cost_ceiling > 0.0) {
            return Err(ContractError::invalid("cost_ceiling", "must be finite and positive"));
        }
        unit_range("observation_rate", self.observation_rate)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong at the contract boundary.
///
/// Each variant maps to exactly one HTTP status, so the transport layer never has to
/// guess: `Invalid` and `Mismatch` are `400`, `UnknownRun` and `UnknownAgent` are `404`,
/// `DuplicateReport` and `Incomplete` are `409`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContractError {
    /// A field was missing, out of range, or self-contradictory.
    Invalid { field: String, reason: String },
    /// No open run with that id.
    UnknownRun { run_id: String },
    /// No agent with that id in the cohort.
    UnknownAgent { agent_id: String },
    /// A report named a stage that is not in the plan, or an agent the plan did not
    /// assign to that stage.
    Mismatch { expected: String, found: String },
    /// The same stage was reported twice for one run.
    DuplicateReport { stage: String },
    /// Finalize was called before every staffed stage had a report.
    Incomplete { missing: Vec<String> },
}

impl ContractError {
    pub fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        ContractError::Invalid {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// The HTTP status a transport should answer with.
    pub fn status(&self) -> u16 {
        match self {
            ContractError::Invalid { .. } | ContractError::Mismatch { .. } => 400,
            ContractError::UnknownRun { .. } | ContractError::UnknownAgent { .. } => 404,
            ContractError::DuplicateReport { .. } | ContractError::Incomplete { .. } => 409,
        }
    }
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractError::Invalid { field, reason } => write!(formatter, "invalid `{field}`: {reason}"),
            ContractError::UnknownRun { run_id } => write!(formatter, "no open run `{run_id}`"),
            ContractError::UnknownAgent { agent_id } => write!(formatter, "no agent `{agent_id}`"),
            ContractError::Mismatch { expected, found } => {
                write!(formatter, "expected {expected}, found {found}")
            }
            ContractError::DuplicateReport { stage } => {
                write!(formatter, "stage `{stage}` was already reported for this run")
            }
            ContractError::Incomplete { missing } => {
                write!(formatter, "cannot finalize, missing outcomes for: {}", missing.join(", "))
            }
        }
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_outage_is_not_evidence_that_an_agent_is_weak() {
        assert_eq!(ErrorClass::Infrastructure.attribution(), Attribution::Environment);
        assert!(ErrorClass::Infrastructure.blame_weight() < ErrorClass::Quality.blame_weight());
        assert!(
            !ErrorClass::Infrastructure.is_trust_evidence(),
            "an outage must never reach the trust ledger"
        );
        assert!(ErrorClass::Infrastructure.is_countable(), "but the run still happened");
    }

    #[test]
    fn a_cancelled_run_teaches_the_mesh_nothing() {
        assert_eq!(ErrorClass::Cancelled.attribution(), Attribution::Caller);
        assert_eq!(ErrorClass::Cancelled.blame_weight(), 0.0);
        assert!(!ErrorClass::Cancelled.is_countable());
    }

    #[test]
    fn slowness_is_the_agents_problem_but_throttling_is_not() {
        assert_eq!(ErrorClass::Timeout.attribution(), Attribution::Agent);
        assert_eq!(ErrorClass::RateLimit.attribution(), Attribution::Environment);
    }

    #[test]
    fn an_unclassified_failure_does_not_get_a_free_pass() {
        assert_eq!(ErrorClass::Unknown.attribution(), Attribution::Agent);
        assert_eq!(ErrorClass::Unknown.blame_weight(), 1.0);
    }

    #[test]
    fn registration_rejects_impossible_declarations_instead_of_clamping() {
        let mut registration = AgentRegistration::new("a", "A", AgentRole::Coder);
        registration.capability.intelligence = 1.4;
        assert!(matches!(
            registration.validate(),
            Err(ContractError::Invalid { ref field, .. }) if field == "capability.intelligence"
        ));

        registration.capability.intelligence = f64::NAN;
        assert!(registration.validate().is_err());

        registration.capability.intelligence = 0.9;
        assert!(registration.validate().is_ok());
    }

    #[test]
    fn registration_rejects_an_empty_id() {
        assert!(AgentRegistration::new("  ", "A", AgentRole::Coder).validate().is_err());
    }

    #[test]
    fn a_report_cannot_claim_success_and_an_error_at_once() {
        let mut report = StageOutcomeReport::success(PipelineStage::Code, "coder-01", 0.9);
        report.error = ErrorClass::Timeout;
        assert!(report.validate().is_err(), "a contradiction must not enter a digest");

        let mut silent_failure = StageOutcomeReport::failure(PipelineStage::Code, "coder-01", ErrorClass::Quality);
        silent_failure.error = ErrorClass::None;
        assert!(silent_failure.validate().is_err(), "a failure must say why");
    }

    #[test]
    fn a_report_rejects_negative_latency_and_cost() {
        let report = StageOutcomeReport::success(PipelineStage::Code, "coder-01", 0.9).with_latency_ms(-1.0);
        assert!(report.validate().is_err());

        let report = StageOutcomeReport::success(PipelineStage::Code, "coder-01", 0.9).with_cost(-0.5);
        assert!(report.validate().is_err());
    }

    #[test]
    fn envelope_derives_a_task_and_honours_overrides() {
        let envelope = TaskEnvelope::new("ship the adapter")
            .with_domain("rust")
            .with_features(vec![0.1, 0.9, 0.2, 0.8])
            .with_urgency(0.9);

        let derived = envelope.to_task(7);
        assert_eq!(derived.id, "task-000007");
        assert_eq!(derived.domain, "rust");
        assert_eq!(derived.urgency, 0.9);
        assert!(derived.uncertainty > 0.3, "dispersed features mean an ambiguous request");

        let mut pinned = envelope.clone();
        pinned.uncertainty = Some(0.05);
        assert_eq!(pinned.to_task(7).uncertainty, 0.05, "an explicit override wins");
    }

    #[test]
    fn envelope_keeps_a_caller_supplied_id() {
        let mut envelope = TaskEnvelope::new("do a thing");
        envelope.task_id = Some("PR-4417".into());
        assert_eq!(envelope.to_task(1).id, "PR-4417");
    }

    #[test]
    fn envelope_validation_catches_junk_features() {
        let envelope = TaskEnvelope::new("x").with_features(vec![0.5, f64::INFINITY]);
        assert!(envelope.validate().is_err());
        assert!(TaskEnvelope::new("   ").validate().is_err());
    }

    #[test]
    fn calibration_round_trips_absolutes_through_the_normalized_scale() {
        let calibration = RuntimeCalibration::new(30_000.0, 2.0);
        assert_eq!(calibration.normalize_latency(15_000.0), 0.5);
        assert_eq!(calibration.denormalize_latency(0.5), 15_000.0);
        assert_eq!(calibration.normalize_cost(1.0), 0.5);
        assert_eq!(calibration.normalize_latency(1e9), 1.0, "the ceiling is a ceiling");
        assert!(calibration.validate().is_ok());
        assert!(RuntimeCalibration::new(0.0, 1.0).validate().is_err());
    }

    #[test]
    fn provenance_only_survives_as_real_when_nothing_was_simulated() {
        assert!(OutcomeProvenance::Reported.is_real());
        assert!(!OutcomeProvenance::Mixed.is_real());
        assert_eq!(
            OutcomeProvenance::Reported.merge(OutcomeProvenance::Simulated),
            OutcomeProvenance::Mixed
        );
        assert_eq!(
            OutcomeProvenance::Reported.merge(OutcomeProvenance::Reported),
            OutcomeProvenance::Reported
        );
    }

    #[test]
    fn a_head_commitment_is_machine_independent() {
        let clean = DecisionHead::new("onnx:policy", "hamns-native", "2.0.0", 8, 4);
        let here = clean
            .clone()
            .with_degraded(Some("onnx model not found at `/home/a/policy.onnx`".into()));
        let there = clean
            .clone()
            .with_degraded(Some("onnx model not found at `D:\\models\\policy.onnx`".into()));

        assert_eq!(
            here.commitment(),
            there.commitment(),
            "the same failure on two machines must not produce two different digests"
        );
        assert_ne!(
            clean.commitment(),
            here.commitment(),
            "but running the wrong head is exactly what a digest should record"
        );
    }

    #[test]
    fn a_head_reports_whether_it_is_the_one_that_was_asked_for() {
        let native = DecisionHead::new("native", "hamns-native", "2.0.0", 8, 4);
        assert!(native.is_intact());
        assert!(!native.is_policy());

        let fallen_back = native.clone().with_degraded(Some("model not found".into()));
        assert!(!fallen_back.is_intact());

        assert!(DecisionHead::new("onnx:trained-v3", "hamns-native", "2.0.0", 8, 4).is_policy());
    }

    #[test]
    fn contract_errors_map_onto_distinct_http_statuses() {
        assert_eq!(ContractError::invalid("x", "y").status(), 400);
        assert_eq!(ContractError::UnknownRun { run_id: "r".into() }.status(), 404);
        assert_eq!(ContractError::DuplicateReport { stage: "code".into() }.status(), 409);
        assert!(ContractError::invalid("x", "y").to_string().contains("invalid `x`"));
    }
}
