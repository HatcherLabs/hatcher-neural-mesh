//! Layer 5: incoming work, its priority, and the trace of the pipeline that ran it.

use serde::{Deserialize, Serialize};

use crate::agent::{AgentRole, ExecutionMode};
use crate::api::NeuralSignal;
use crate::contract::{ContractError, ErrorClass, OutcomeProvenance};
use crate::global::OmegaDelta;
use crate::graph::MeshIntelligence;

/// Hard limits a caller places on how a task may be executed.
///
/// These are constraints on *routing*, not on the equations. An unset limit is not a
/// limit of infinity so much as an absence of opinion, which is why both fields are
/// optional rather than defaulted to something large: a caller that says nothing gets
/// the mesh's own judgement, and a caller that says `max_cost` means it.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct TaskConstraints {
    /// Reject candidates whose observed per-stage latency exceeds this, in milliseconds.
    #[serde(default)]
    pub max_latency_ms: Option<f64>,
    /// Reject candidates whose observed per-stage cost exceeds this, in the caller's
    /// cost units.
    #[serde(default)]
    pub max_cost: Option<f64>,
}

impl TaskConstraints {
    pub fn latency(max_latency_ms: f64) -> Self {
        Self {
            max_latency_ms: Some(max_latency_ms),
            max_cost: None,
        }
    }

    pub fn cost(max_cost: f64) -> Self {
        Self {
            max_latency_ms: None,
            max_cost: Some(max_cost),
        }
    }

    pub fn with_max_latency_ms(mut self, max_latency_ms: f64) -> Self {
        self.max_latency_ms = Some(max_latency_ms);
        self
    }

    pub fn with_max_cost(mut self, max_cost: f64) -> Self {
        self.max_cost = Some(max_cost);
        self
    }

    /// True when nothing is constrained.
    pub fn is_open(&self) -> bool {
        self.max_latency_ms.is_none() && self.max_cost.is_none()
    }

    /// Whether an agent's expected latency and cost fit inside these limits.
    pub fn admits(&self, expected_latency_ms: f64, expected_cost: f64) -> bool {
        self.max_latency_ms.map_or(true, |limit| expected_latency_ms <= limit)
            && self.max_cost.map_or(true, |limit| expected_cost <= limit)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, value) in [
            ("constraints.max_latency_ms", self.max_latency_ms),
            ("constraints.max_cost", self.max_cost),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(ContractError::invalid(name, "must be finite and non-negative"));
                }
            }
        }
        Ok(())
    }
}

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
    /// Caller-imposed routing limits. Empty by default.
    #[serde(default)]
    pub constraints: TaskConstraints,
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
            constraints: TaskConstraints::default(),
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

    pub fn with_constraints(mut self, constraints: TaskConstraints) -> Self {
        self.constraints = constraints;
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
///
/// The derived ordering is that execution order — the variants are declared in the
/// sequence they run, so sorting a set of stages sorts them into pipeline order and a
/// keyed collection iterates the way the pipeline does.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
///
/// The three fields after `cost` are what make a trace comparable to a real run rather
/// than only to another simulation. `quality` is the reviewer's score and `confidence`
/// is the producer's — keeping them apart is what lets the mesh detect an agent that is
/// confidently wrong. `error` carries the classified cause so the learning layer can
/// tell an incapable agent from an unlucky one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageRecord {
    pub stage: PipelineStage,
    pub agent: Option<String>,
    pub success: bool,
    /// The executing agent's calibrated confidence in its own output.
    pub confidence: f64,
    /// Resource cost consumed by this stage, in the caller's cost units.
    pub cost: f64,
    /// Verifier or grader score for this stage's output, in `[0, 1]`.
    #[serde(default)]
    pub quality: f64,
    /// Wall-clock time this stage took, in milliseconds.
    #[serde(default)]
    pub latency_ms: f64,
    /// Why it failed. [`ErrorClass::None`] on success.
    #[serde(default)]
    pub error: ErrorClass,
    /// Whether this stage's outcome was simulated or reported by a real runtime.
    ///
    /// `None` on the five bookkeeping stations, which have no external outcome to
    /// source — the mesh measured those itself. Only the `Some` values decide a trace's
    /// provenance, so a fully simulated run does not read as `Mixed` just because its
    /// trust update was real.
    #[serde(default)]
    pub provenance: Option<OutcomeProvenance>,
    pub note: String,
}

impl StageRecord {
    /// A bookkeeping station: no agent, no cost, nothing external to grade.
    ///
    /// The five unstaffed stations all record the same shape, and spelling it out at
    /// each call site is how a field silently drifts out of sync between them.
    pub fn bookkeeping(stage: PipelineStage, confidence: f64, note: String) -> Self {
        Self {
            stage,
            agent: None,
            success: true,
            confidence,
            cost: 0.0,
            quality: confidence,
            latency_ms: 0.0,
            error: ErrorClass::None,
            provenance: None,
            note,
        }
    }

    /// A staffed station the router could not fill.
    pub fn unstaffed(stage: PipelineStage) -> Self {
        Self {
            stage,
            agent: None,
            success: false,
            confidence: 0.0,
            cost: 0.0,
            quality: 0.0,
            latency_ms: 0.0,
            error: ErrorClass::Infrastructure,
            provenance: None,
            note: "no agent available for this stage".to_string(),
        }
    }
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
    /// Whether the outcomes behind this trace were simulated or really happened.
    ///
    /// Inside the digest on purpose: a commitment over a rehearsal and a commitment over
    /// real work must never be confusable, and putting provenance beside the digest
    /// rather than inside it would let one be presented as the other.
    #[serde(default)]
    pub provenance: OutcomeProvenance,
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

    /// Total wall-clock time across every stage.
    ///
    /// A sum rather than a critical path: the pipeline is sequential, so the sum *is*
    /// the elapsed time. If a caller ever runs research and coding concurrently, this
    /// becomes an upper bound and should be read as total agent time, not latency.
    pub fn total_latency_ms(&self) -> f64 {
        self.stages.iter().map(|record| record.latency_ms).sum()
    }

    /// Mean verifier score across the staffed stations.
    ///
    /// Unstaffed stations are excluded rather than counted as zero: a mesh with no
    /// verifier should look under-staffed, not low-quality.
    pub fn mean_quality(&self) -> f64 {
        let staffed: Vec<&StageRecord> = self
            .stages
            .iter()
            .filter(|record| record.stage.is_staffed() && record.agent.is_some())
            .collect();
        if staffed.is_empty() {
            return 0.0;
        }
        staffed.iter().map(|record| record.quality).sum::<f64>() / staffed.len() as f64
    }

    /// Fraction of staffed stations that succeeded.
    pub fn stage_success_rate(&self) -> f64 {
        let staffed: Vec<&StageRecord> = self.stages.iter().filter(|record| record.stage.is_staffed()).collect();
        if staffed.is_empty() {
            return 0.0;
        }
        staffed.iter().filter(|record| record.success).count() as f64 / staffed.len() as f64
    }

    /// The verifier's score for this run, or `0.0` if the station never ran.
    pub fn verify_quality(&self) -> f64 {
        self.stages
            .iter()
            .find(|record| record.stage == PipelineStage::Verify)
            .map(|record| record.quality)
            .unwrap_or(0.0)
    }

    /// Every distinct error class this run produced, in pipeline order.
    pub fn error_classes(&self) -> Vec<ErrorClass> {
        let mut seen: Vec<ErrorClass> = Vec::new();
        for record in &self.stages {
            if record.error != ErrorClass::None && !seen.contains(&record.error) {
                seen.push(record.error);
            }
        }
        seen
    }

    /// Did this run make the mesh smarter?
    pub fn omega_gain(&self) -> f64 {
        self.omega_after - self.omega_before
    }

    /// The agent the mesh holds accountable for the output: the coding station when one
    /// ran, otherwise the first agent that did.
    pub fn primary_agent(&self) -> Option<&str> {
        self.assignments
            .iter()
            .find(|assignment| assignment.stage == PipelineStage::Code)
            .or_else(|| self.assignments.first())
            .map(|assignment| assignment.agent_id.as_str())
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

    #[test]
    fn an_absent_constraint_is_an_absence_of_opinion() {
        let open = TaskConstraints::default();
        assert!(open.is_open());
        assert!(open.admits(1e9, 1e9), "saying nothing must not exclude anyone");
    }

    #[test]
    fn constraints_exclude_only_what_they_name() {
        let latency_only = TaskConstraints::latency(5_000.0);
        assert!(latency_only.admits(4_000.0, 99.0), "cost is unconstrained here");
        assert!(!latency_only.admits(6_000.0, 0.0));

        let both = TaskConstraints::latency(5_000.0).with_max_cost(0.5);
        assert!(both.admits(5_000.0, 0.5), "the limit itself is admissible");
        assert!(!both.admits(5_000.0, 0.51));
    }

    #[test]
    fn constraints_reject_nonsense_limits() {
        assert!(TaskConstraints::latency(-1.0).validate().is_err());
        assert!(TaskConstraints::cost(f64::NAN).validate().is_err());
        assert!(TaskConstraints::latency(10.0).validate().is_ok());
    }

    #[test]
    fn bookkeeping_stations_have_no_outcome_to_source() {
        let record = StageRecord::bookkeeping(PipelineStage::TrustUpdate, 0.5, "settled".into());
        assert!(record.success);
        assert_eq!(record.error, ErrorClass::None);
        assert!(
            record.provenance.is_none(),
            "the mesh measured this itself; it is neither simulated nor reported"
        );
    }

    #[test]
    fn an_unstaffed_station_is_a_failure_not_a_silent_pass() {
        let record = StageRecord::unstaffed(PipelineStage::Verify);
        assert!(!record.success);
        assert_eq!(record.error, ErrorClass::Infrastructure);
        assert_eq!(record.quality, 0.0);
    }

    fn trace_with(stages: Vec<StageRecord>, assignments: Vec<Assignment>) -> PipelineTrace {
        PipelineTrace {
            task_id: "t".into(),
            domain: "rust".into(),
            execution_mode: ExecutionMode::Controlled,
            priority: PriorityScore {
                value: 0.4,
                uncertainty: 0.5,
                budget: 0.5,
                implementation_cost: 0.5,
                confidence: 0.5,
                tau: 0.1,
                urgency_gain: 1.0,
                band: PriorityBand::Standard,
            },
            assignments,
            stages,
            omega_before: 1.0,
            omega_after: 1.2,
            delta: OmegaDelta::default(),
            intelligence: MeshIntelligence::default(),
            decision: NeuralSignal {
                agent: "t".into(),
                intent: "x".into(),
                confidence: 0.5,
                action: "observe".into(),
                rationale: String::new(),
            },
            verified: true,
            provenance: OutcomeProvenance::Simulated,
            digest: String::new(),
        }
    }

    fn staffed_record(stage: PipelineStage, agent: &str, quality: f64, latency_ms: f64, cost: f64) -> StageRecord {
        StageRecord {
            stage,
            agent: Some(agent.to_string()),
            success: quality > 0.0,
            confidence: quality,
            cost,
            quality,
            latency_ms,
            error: if quality > 0.0 { ErrorClass::None } else { ErrorClass::Quality },
            provenance: Some(OutcomeProvenance::Reported),
            note: String::new(),
        }
    }

    #[test]
    fn trace_metrics_sum_the_stages_that_actually_ran() {
        let trace = trace_with(
            vec![
                StageRecord::bookkeeping(PipelineStage::Parse, 1.0, String::new()),
                staffed_record(PipelineStage::Code, "coder-01", 0.8, 1_200.0, 0.30),
                staffed_record(PipelineStage::Verify, "verifier-01", 0.6, 400.0, 0.05),
            ],
            Vec::new(),
        );

        assert!((trace.total_latency_ms() - 1_600.0).abs() < 1e-9);
        assert!((trace.total_cost() - 0.35).abs() < 1e-9);
        assert!((trace.mean_quality() - 0.7).abs() < 1e-9, "bookkeeping must not dilute quality");
        assert!((trace.verify_quality() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn an_unstaffed_mesh_reads_as_under_staffed_not_low_quality() {
        let trace = trace_with(vec![StageRecord::unstaffed(PipelineStage::Code)], Vec::new());
        assert_eq!(trace.mean_quality(), 0.0);
        assert_eq!(trace.stage_success_rate(), 0.0);
    }

    #[test]
    fn error_classes_are_deduplicated_in_pipeline_order() {
        let mut first = staffed_record(PipelineStage::Code, "a", 0.0, 0.0, 0.0);
        first.error = ErrorClass::Timeout;
        let mut second = staffed_record(PipelineStage::Critique, "b", 0.0, 0.0, 0.0);
        second.error = ErrorClass::Timeout;
        let mut third = staffed_record(PipelineStage::Verify, "c", 0.0, 0.0, 0.0);
        third.error = ErrorClass::RateLimit;

        let trace = trace_with(vec![first, second, third], Vec::new());
        assert_eq!(trace.error_classes(), vec![ErrorClass::Timeout, ErrorClass::RateLimit]);
    }

    #[test]
    fn the_coding_station_owns_the_output() {
        let assignment = |stage, agent: &str| Assignment {
            stage,
            agent_id: agent.to_string(),
            role: AgentRole::Coder,
            score: 1.0,
            capability: 0.5,
            inbound_trust: 0.5,
            efficiency: 1.0,
            mastery: 0.5,
        };
        let trace = trace_with(
            Vec::new(),
            vec![
                assignment(PipelineStage::Plan, "planner-01"),
                assignment(PipelineStage::Code, "coder-01"),
            ],
        );
        assert_eq!(trace.primary_agent(), Some("coder-01"));

        let planning_only = trace_with(Vec::new(), vec![assignment(PipelineStage::Plan, "planner-01")]);
        assert_eq!(planning_only.primary_agent(), Some("planner-01"));
    }
}
