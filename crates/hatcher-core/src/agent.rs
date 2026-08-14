//! Agent-level contracts: the capability vector, resource profile, and node record.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::envelope::Envelope;

/// The role an agent plays in the mesh pipeline.
///
/// The first five variants are the original Hatcher control-plane roles; the
/// remainder name the specialized stations of the HAMNS pipeline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentRole {
    Orchestrator,
    Executor,
    Critic,
    Explorer,
    Guardian,
    Planner,
    Researcher,
    Coder,
    Verifier,
}

impl AgentRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Orchestrator => "orchestrator",
            AgentRole::Executor => "executor",
            AgentRole::Critic => "critic",
            AgentRole::Explorer => "explorer",
            AgentRole::Guardian => "guardian",
            AgentRole::Planner => "planner",
            AgentRole::Researcher => "researcher",
            AgentRole::Coder => "coder",
            AgentRole::Verifier => "verifier",
        }
    }
}

/// How much authority the mesh has over side effects for a given request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionMode {
    Sandbox,
    Controlled,
    Production,
}

impl ExecutionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionMode::Sandbox => "sandbox",
            ExecutionMode::Controlled => "controlled",
            ExecutionMode::Production => "production",
        }
    }

    /// Gate multiplier applied to how strongly a run is allowed to move the mesh.
    pub fn plasticity_gate(&self) -> f64 {
        match self {
            ExecutionMode::Sandbox => 0.25,
            ExecutionMode::Controlled => 0.6,
            ExecutionMode::Production => 1.0,
        }
    }
}

/// The five multiplicative factors of agent capability.
///
/// `A_i = I_i · S_i · P_i · C_i · M_i`
///
/// Multiplication is the point: a single weak factor collapses the product, so a
/// brilliant model with no memory or no context is correctly scored as weak.
/// Every factor is held in `[0, 1]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CapabilityVector {
    /// `I` — raw reasoning strength of the underlying model.
    pub intelligence: f64,
    /// `S` — depth of domain specialization, grown by repeated experience in a domain.
    pub specialization: f64,
    /// `P` — observed performance: completed work over attempted work.
    pub performance: f64,
    /// `C` — context awareness: fraction of relevant context the agent actually holds.
    pub context: f64,
    /// `M` — memory quality: retained, retrievable knowledge.
    pub memory: f64,
}

impl CapabilityVector {
    pub fn new(
        intelligence: f64,
        specialization: f64,
        performance: f64,
        context: f64,
        memory: f64,
    ) -> Self {
        Self {
            intelligence,
            specialization,
            performance,
            context,
            memory,
        }
    }

    /// Uniform vector, useful as a cohort baseline.
    pub fn uniform(value: f64) -> Self {
        Self::new(value, value, value, value, value)
    }

    /// `A_i` — the multiplicative capability scalar.
    pub fn capability(&self) -> f64 {
        (self.intelligence * self.specialization * self.performance * self.context * self.memory)
            .max(0.0)
    }

    /// The weakest factor, i.e. the one that is actually capping this agent.
    pub fn bottleneck(&self) -> (&'static str, f64) {
        let factors = [
            ("intelligence", self.intelligence),
            ("specialization", self.specialization),
            ("performance", self.performance),
            ("context", self.context),
            ("memory", self.memory),
        ];
        factors
            .into_iter()
            .fold(("intelligence", f64::MAX), |acc, item| {
                if item.1 < acc.1 {
                    item
                } else {
                    acc
                }
            })
    }

    /// Clamp every factor into `[0, 1]`.
    pub fn clamped(mut self) -> Self {
        self.intelligence = self.intelligence.clamp(0.0, 1.0);
        self.specialization = self.specialization.clamp(0.0, 1.0);
        self.performance = self.performance.clamp(0.0, 1.0);
        self.context = self.context.clamp(0.0, 1.0);
        self.memory = self.memory.clamp(0.0, 1.0);
        self
    }
}

impl Default for CapabilityVector {
    fn default() -> Self {
        Self::new(0.7, 0.5, 0.6, 0.6, 0.5)
    }
}

/// What it costs to run an agent.
///
/// Feeds the resource ratio `R_i = P_i / (Energy_i + Latency_i)`.
///
/// The first two fields are normalized into `[0, 1]` because that is the scale the
/// equations work on. The last two are the absolutes a real runtime actually reports —
/// milliseconds and cost units — kept alongside so the mesh can answer "how long will
/// this take" in units a caller recognizes, and so a declared prior can be told apart
/// from a measured fact. Both are `0.0` until something is observed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ResourceProfile {
    /// Normalized compute/token cost per task.
    pub energy: f64,
    /// Normalized round-trip latency per task (`1.0` == the slowest tolerated agent).
    pub latency: f64,
    /// Exponentially-weighted mean of reported stage latency, in milliseconds.
    /// `0.0` means nothing has been observed yet.
    #[serde(default)]
    pub observed_latency_ms: f64,
    /// Exponentially-weighted mean of reported stage cost, in the caller's cost units.
    /// `0.0` means nothing has been observed yet.
    #[serde(default)]
    pub observed_cost: f64,
    /// How many reports have folded into the observed means.
    #[serde(default)]
    pub observations: u64,
}

impl ResourceProfile {
    pub fn new(energy: f64, latency: f64) -> Self {
        Self {
            energy,
            latency,
            observed_latency_ms: 0.0,
            observed_cost: 0.0,
            observations: 0,
        }
    }

    /// Total cost denominator, floored so the ratio never divides by zero.
    pub fn cost(&self) -> f64 {
        (self.energy.max(0.0) + self.latency.max(0.0)).max(1e-6)
    }

    /// Whether any real report has landed on this profile.
    pub fn is_observed(&self) -> bool {
        self.observations > 0
    }

    /// Expected stage latency in milliseconds.
    ///
    /// Measured history when there is any, otherwise the declared prior projected onto
    /// the given ceiling. A caller asking "will this agent fit in my 5-second budget"
    /// deserves an answer either way; it just gets a better one once reports arrive.
    pub fn expected_latency_ms(&self, latency_ceiling_ms: f64) -> f64 {
        if self.is_observed() {
            self.observed_latency_ms
        } else {
            self.latency.clamp(0.0, 1.0) * latency_ceiling_ms
        }
    }

    /// Expected stage cost in the caller's cost units.
    pub fn expected_cost(&self, cost_ceiling: f64) -> f64 {
        if self.is_observed() {
            self.observed_cost
        } else {
            self.energy.clamp(0.0, 1.0) * cost_ceiling
        }
    }

    /// Fold one real observation into the profile.
    ///
    /// The normalized factors move with the absolutes, because they are two views of
    /// the same fact: an agent that is measurably slower must become measurably worse on
    /// `R_i`, or reporting latency would be decorative. `rate` is the EWMA weight given
    /// to the new observation.
    pub fn observe(
        &mut self,
        latency_ms: f64,
        cost: f64,
        latency_ceiling_ms: f64,
        cost_ceiling: f64,
        rate: f64,
    ) {
        let rate = rate.clamp(0.0, 1.0);
        let latency_ms = latency_ms.max(0.0);
        let cost = cost.max(0.0);

        // The first observation replaces the prior outright rather than averaging with
        // it: a declared profile is a guess, and one real measurement is strictly better
        // evidence than a guess it was never checked against.
        if self.observations == 0 {
            self.observed_latency_ms = latency_ms;
            self.observed_cost = cost;
        } else {
            self.observed_latency_ms = self.observed_latency_ms * (1.0 - rate) + latency_ms * rate;
            self.observed_cost = self.observed_cost * (1.0 - rate) + cost * rate;
        }
        self.observations += 1;

        if latency_ceiling_ms > 0.0 {
            self.latency = (self.observed_latency_ms / latency_ceiling_ms).clamp(0.0, 1.0);
        }
        if cost_ceiling > 0.0 {
            self.energy = (self.observed_cost / cost_ceiling).clamp(0.0, 1.0);
        }
    }
}

impl Default for ResourceProfile {
    fn default() -> Self {
        Self::new(0.4, 0.3)
    }
}

/// Rolling counters that make the capability factors measurable rather than guessed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NodeTelemetry {
    /// Tasks this agent was assigned.
    pub attempts: u64,
    /// Assignments that passed the verifier.
    pub successes: u64,
    /// Assignments that failed the critic or the verifier.
    pub failures: u64,
    /// Distinct collaborations recorded on the trust graph.
    pub collaborations: u64,
    /// Cumulative `K_i` — knowledge admitted into memory.
    pub knowledge: f64,
    /// Cumulative `R_i` — knowledge lost to decay.
    pub decay: f64,
    /// Cumulative obsolescence pressure applied to specialization.
    pub obsolescence: f64,
    /// Last action this agent emitted, for the Logs view.
    pub last_action: Option<String>,
}

impl NodeTelemetry {
    /// Observed performance `P` = successes / attempts, with an optimistic prior
    /// so an unproven agent is not permanently locked out by a zero product.
    pub fn observed_performance(&self, prior: f64) -> f64 {
        if self.attempts == 0 {
            return prior.clamp(0.0, 1.0);
        }
        let successes = self.successes as f64;
        let attempts = self.attempts as f64;
        // Laplace-smoothed rate keeps a single early failure from zeroing the agent.
        ((successes + prior) / (attempts + 1.0)).clamp(0.0, 1.0)
    }
}

/// A single agent in the mesh: one node of the HAMNS graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentNode {
    pub id: String,
    pub label: String,
    pub role: AgentRole,
    /// `I, S, P, C, M`
    pub capability: CapabilityVector,
    /// `C_i` in the confidence-calibration equation. Distinct from
    /// `capability.context`, which is the `C` of the capability product.
    pub confidence: f64,
    pub resources: ResourceProfile,
    /// Per-domain specialization, `domain -> mastery in [0, 1]`.
    pub expertise: BTreeMap<String, f64>,
    pub telemetry: NodeTelemetry,
}

impl AgentNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>, role: AgentRole) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            role,
            capability: CapabilityVector::default(),
            confidence: 0.5,
            resources: ResourceProfile::default(),
            expertise: BTreeMap::new(),
            telemetry: NodeTelemetry::default(),
        }
    }

    pub fn with_capability(mut self, capability: CapabilityVector) -> Self {
        self.capability = capability.clamped();
        self
    }

    pub fn with_resources(mut self, resources: ResourceProfile) -> Self {
        self.resources = resources;
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_expertise(mut self, domain: impl Into<String>, mastery: f64) -> Self {
        self.expertise
            .insert(domain.into(), mastery.clamp(0.0, 1.0));
        self
    }

    /// `A_i` — capability scalar for this node.
    pub fn influence(&self) -> f64 {
        self.capability.capability()
    }

    /// Mastery in a domain.
    ///
    /// An unseen domain falls back to the agent's `general` expertise if it has one, and
    /// only then to half its specialization factor. The order matters: a broadly capable
    /// agent meeting an unfamiliar domain is better described by its general competence
    /// than by an arbitrary fraction of a number that measures how *narrow* it is.
    pub fn mastery(&self, domain: &str) -> f64 {
        if let Some(mastery) = self.expertise.get(domain) {
            return *mastery;
        }
        if let Some(general) = self.expertise.get("general") {
            return *general;
        }
        self.capability.specialization * 0.5
    }

    /// `R_i = P_i / (Energy_i + Latency_i)` — performance per unit of cost.
    pub fn resource_efficiency(&self) -> f64 {
        self.capability.performance / self.resources.cost()
    }

    /// Seal this node into a digest-backed envelope.
    pub fn seal(&self) -> Result<Envelope<AgentNode>, serde_json::Error> {
        Envelope::seal(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_is_multiplicative_so_weaknesses_dominate() {
        let balanced = CapabilityVector::uniform(0.6);
        let brilliant_but_amnesiac = CapabilityVector::new(1.0, 0.9, 0.9, 0.9, 0.02);
        assert!(
            brilliant_but_amnesiac.capability() < balanced.capability(),
            "a near-zero memory factor must collapse the product"
        );
        assert_eq!(brilliant_but_amnesiac.bottleneck().0, "memory");
    }

    #[test]
    fn resource_efficiency_prefers_cheap_fast_agents() {
        let cheap = AgentNode::new("a", "cheap", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.6))
            .with_resources(ResourceProfile::new(0.1, 0.1));
        let costly = AgentNode::new("b", "costly", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.6))
            .with_resources(ResourceProfile::new(0.9, 0.8));
        assert!(cheap.resource_efficiency() > costly.resource_efficiency());
    }

    #[test]
    fn observed_performance_smooths_early_history() {
        let mut telemetry = NodeTelemetry::default();
        assert_eq!(
            telemetry.observed_performance(0.6),
            0.6,
            "unproven agents keep their prior"
        );
        telemetry.attempts = 1;
        telemetry.failures = 1;
        assert!(
            telemetry.observed_performance(0.6) > 0.0,
            "one failure must not zero the agent"
        );
    }

    #[test]
    fn an_unobserved_profile_still_answers_the_latency_question() {
        let profile = ResourceProfile::new(0.5, 0.25);
        assert!(!profile.is_observed());
        assert_eq!(
            profile.expected_latency_ms(60_000.0),
            15_000.0,
            "projected from the prior"
        );
        assert_eq!(profile.expected_cost(2.0), 1.0);
    }

    #[test]
    fn the_first_real_measurement_replaces_the_guess() {
        let mut profile = ResourceProfile::new(0.9, 0.9);
        profile.observe(1_000.0, 0.10, 60_000.0, 1.0, 0.25);

        assert_eq!(profile.observations, 1);
        assert_eq!(
            profile.observed_latency_ms, 1_000.0,
            "one measurement beats an unchecked prior"
        );
        assert!(
            profile.latency < 0.9,
            "and it must move the normalized factor too"
        );
        assert!((profile.energy - 0.10).abs() < 1e-9);
    }

    #[test]
    fn later_measurements_are_smoothed_rather_than_chased() {
        let mut profile = ResourceProfile::new(0.5, 0.5);
        profile.observe(1_000.0, 0.1, 60_000.0, 1.0, 0.25);
        profile.observe(5_000.0, 0.1, 60_000.0, 1.0, 0.25);

        assert_eq!(profile.observations, 2);
        assert!(
            profile.observed_latency_ms > 1_000.0 && profile.observed_latency_ms < 5_000.0,
            "an outlier moves the mean without becoming it, got {}",
            profile.observed_latency_ms
        );
    }

    #[test]
    fn a_measurably_slower_agent_becomes_measurably_less_efficient() {
        let mut quick = AgentNode::new("quick", "quick", AgentRole::Coder);
        let mut slow = AgentNode::new("slow", "slow", AgentRole::Coder);
        quick.resources.observe(500.0, 0.05, 60_000.0, 1.0, 0.5);
        slow.resources.observe(45_000.0, 0.05, 60_000.0, 1.0, 0.5);

        assert!(
            quick.resource_efficiency() > slow.resource_efficiency(),
            "reported latency has to reach R_i or reporting it is decorative"
        );
    }

    #[test]
    fn node_seals_into_a_verifiable_envelope() {
        let node = AgentNode::new("node-1", "guardian", AgentRole::Guardian);
        let envelope = node.seal().unwrap();
        assert!(envelope.verify().unwrap());
        assert_eq!(envelope.payload.id, "node-1");
    }
}
