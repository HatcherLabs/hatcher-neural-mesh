//! Layer 5: incoming work, its priority, and the trace of the pipeline that ran it.

use serde::{Deserialize, Serialize};

use crate::agent::{AgentRole, ExecutionMode};
use crate::api::NeuralSignal;
use crate::global::OmegaDelta;
use crate::graph::MeshIntelligence;

/// A unit of work submitted to the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: String,
    pub description: String,
    /// Domain tag used for specialization routing, e.g. `"rust"`, `"research"`, `"infra"`.
    pub domain: String,
    /// `U` — uncertainty: how unknown the solution path is, in `[0, 1]`.
    pub uncertainty: f64,
    /// `B` — estimated compute budget the task justifies, in `[0, 1]`.
    pub budget: f64,
    /// `I` — implementation cost: how much work the change represents, in `[0, 1]`.
    pub implementation_cost: f64,
    /// Urgency, in `[0, 1]`. See [`PriorityScore`] for exactly how this enters `P`.
    pub urgency: f64,
    /// Raw feature vector from the Hatcher bridge, used to seed node activations.
    pub features: Vec<f64>,
    pub execution_mode: ExecutionMode,
}

impl TaskSpec {
    pub fn new(id: impl Into<String>, description: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            domain: domain.into(),
            uncertainty: 0.5,
            budget: 0.5,
            implementation_cost: 0.5,
            urgency: 0.5,
            features: Vec::new(),
            execution_mode: ExecutionMode::Controlled,
        }
    }

    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }

    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Derive `U`, `B`, and `I` from a raw feature vector.
    ///
    /// This is the Task Parser's job in software terms: uncertainty is feature
    /// dispersion (a spread-out feature vector means the request is ambiguous),
    /// budget is feature mass, and implementation cost grows with how many
    /// dimensions the request touches.
    pub fn parsed_from_features(mut self) -> Self {
        if self.features.is_empty() {
            return self;
        }
        let count = self.features.len() as f64;
        let mean = self.features.iter().sum::<f64>() / count;
        let variance = self.features.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count;

        self.uncertainty = variance.sqrt().clamp(0.05, 1.0);
        self.budget = mean.clamp(0.05, 1.0);
        self.implementation_cost = (count / 8.0).clamp(0.05, 1.0);
        self
    }
}

/// Where a task lands in the schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PriorityBand {
    /// Schedule now, on the strongest available agents.
    Immediate,
    /// Normal queue.
    Standard,
    /// Delay, or hand to the cheapest agent that can plausibly do it.
    Deferred,
}

impl PriorityBand {
    pub fn as_str(&self) -> &'static str {
        match self {
            PriorityBand::Immediate => "immediate",
            PriorityBand::Standard => "standard",
            PriorityBand::Deferred => "deferred",
        }
    }
}

/// The scheduler's output for one task.
///
/// `P = (U · B · I · urgency_gain) / (C + τ)`
///
/// The spec's form is `P = (U · B · I) / (C + τ)` with `τ` as an urgency bonus.
/// Taken literally, urgency sits in the denominator and *lowers* priority, which
/// inverts its intent. This implementation keeps the literal shape — `τ` is the
/// denominator floor that keeps `P` finite when a mesh is fully confident — and
/// routes urgency through `urgency_gain` in the numerator, where it belongs.
/// With `urgency_gain = 1.0` the formula reduces exactly to the spec.
///
/// High `P` means "expensive, uncertain, and the mesh is not confident about it —
/// schedule it now, on the best agents". Low `P` means "the mesh already knows how
/// to do this — defer it or give it to a cheap agent".
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PriorityScore {
    /// `P`
    pub value: f64,
    /// `U`
    pub uncertainty: f64,
    /// `B`
    pub budget: f64,
    /// `I`
    pub implementation_cost: f64,
    /// `C` — mesh confidence about work of this kind.
    pub confidence: f64,
    /// `τ`
    pub tau: f64,
    /// Numerator multiplier carrying task urgency, `>= 1.0`.
    pub urgency_gain: f64,
    pub band: PriorityBand,
}

/// `P >= IMMEDIATE_THRESHOLD` schedules right away.
pub const IMMEDIATE_THRESHOLD: f64 = 0.60;
/// `P < DEFERRED_THRESHOLD` is safe to delay or downgrade.
pub const DEFERRED_THRESHOLD: f64 = 0.20;

impl PriorityScore {
    pub fn band_for(value: f64) -> PriorityBand {
        if value >= IMMEDIATE_THRESHOLD {
            PriorityBand::Immediate
        } else if value < DEFERRED_THRESHOLD {
            PriorityBand::Deferred
        } else {
            PriorityBand::Standard
        }
    }
}

/// The stations of the mesh pipeline, in execution order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PipelineStage {
    Parse,
    Prioritize,
    Plan,
    Research,
    Code,
    Critique,
    Verify,
    MemoryUpdate,
    TrustUpdate,
    OmegaUpdate,
}

impl PipelineStage {
    /// Every stage, in pipeline order.
    pub const ALL: [PipelineStage; 10] = [
        PipelineStage::Parse,
        PipelineStage::Prioritize,
        PipelineStage::Plan,
        PipelineStage::Research,
        PipelineStage::Code,
        PipelineStage::Critique,
        PipelineStage::Verify,
        PipelineStage::MemoryUpdate,
        PipelineStage::TrustUpdate,
        PipelineStage::OmegaUpdate,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            PipelineStage::Parse => "parse",
            PipelineStage::Prioritize => "prioritize",
            PipelineStage::Plan => "plan",
            PipelineStage::Research => "research",
            PipelineStage::Code => "code",
            PipelineStage::Critique => "critique",
            PipelineStage::Verify => "verify",
            PipelineStage::MemoryUpdate => "memory_update",
            PipelineStage::TrustUpdate => "trust_update",
            PipelineStage::OmegaUpdate => "omega_update",
        }
    }

    /// The role the mesh prefers to staff this stage with, if any.
    pub fn preferred_role(&self) -> Option<AgentRole> {
        match self {
            PipelineStage::Plan => Some(AgentRole::Planner),
            PipelineStage::Research => Some(AgentRole::Researcher),
            PipelineStage::Code => Some(AgentRole::Coder),
            PipelineStage::Critique => Some(AgentRole::Critic),
            PipelineStage::Verify => Some(AgentRole::Verifier),
            _ => None,
        }
    }

    /// Whether this stage is executed by an agent (as opposed to being mesh bookkeeping).
    pub fn is_staffed(&self) -> bool {
        self.preferred_role().is_some()
    }
}

/// One agent's assignment to one stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Assignment {
    pub stage: PipelineStage,
    pub agent_id: String,
    pub role: AgentRole,
    /// Composite routing score that won this agent the assignment.
    pub score: f64,
    /// `A_i` at assignment time.
    pub capability: f64,
    /// Mean trust from the rest of the cohort into this agent.
    pub inbound_trust: f64,
    /// `R_i` at assignment time.
    pub efficiency: f64,
    /// Domain mastery at assignment time.
    pub mastery: f64,
}

/// What happened at one stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageRecord {
    pub stage: PipelineStage,
    pub agent: Option<String>,
    pub success: bool,
    /// The executing agent's calibrated confidence in its own output.
    pub confidence: f64,
    /// Resource cost consumed by this stage.
    pub cost: f64,
    pub note: String,
}

/// The complete, sealed record of one pass through the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTrace {
    pub task_id: String,
    pub domain: String,
    pub execution_mode: ExecutionMode,
    pub priority: PriorityScore,
    pub assignments: Vec<Assignment>,
    pub stages: Vec<StageRecord>,
    pub omega_before: f64,
    pub omega_after: f64,
    pub delta: OmegaDelta,
    pub intelligence: MeshIntelligence,
    pub decision: NeuralSignal,
    /// Whether the verifier accepted the work.
    pub verified: bool,
    pub digest: String,
}

impl PipelineTrace {
    /// Stages that failed, in order.
    pub fn failures(&self) -> Vec<&StageRecord> {
        self.stages.iter().filter(|record| !record.success).collect()
    }

    /// Total resource cost of the run.
    pub fn total_cost(&self) -> f64 {
        self.stages.iter().map(|record| record.cost).sum()
    }

    /// Did this run make the mesh smarter?
    pub fn omega_gain(&self) -> f64 {
        self.omega_after - self.omega_before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_derives_priority_inputs_from_features() {
        let task = TaskSpec::new("t-1", "ship the router", "rust")
            .with_features(vec![0.1, 0.9, 0.2, 0.8])
            .parsed_from_features();

        assert!(task.uncertainty > 0.3, "a dispersed feature vector is an ambiguous request");
        assert!((task.budget - 0.5).abs() < 1e-9);
        assert!((task.implementation_cost - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parser_leaves_defaults_when_there_are_no_features() {
        let task = TaskSpec::new("t-2", "no features", "infra").parsed_from_features();
        assert_eq!(task.uncertainty, 0.5);
    }

    #[test]
    fn bands_partition_the_priority_range() {
        assert_eq!(PriorityScore::band_for(0.9), PriorityBand::Immediate);
        assert_eq!(PriorityScore::band_for(0.4), PriorityBand::Standard);
        assert_eq!(PriorityScore::band_for(0.05), PriorityBand::Deferred);
    }

    #[test]
    fn staffed_stages_map_to_roles() {
        assert_eq!(PipelineStage::Code.preferred_role(), Some(AgentRole::Coder));
        assert!(!PipelineStage::OmegaUpdate.is_staffed());
        assert_eq!(PipelineStage::ALL.iter().filter(|stage| stage.is_staffed()).count(), 5);
    }
}
