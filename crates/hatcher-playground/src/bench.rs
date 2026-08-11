//! Does the mesh's routing actually beat something simpler?
//!
//! That question is the whole reason HAMNS has a router at all, and it is not settled by
//! the architecture being interesting. This module answers it the only way it can be
//! answered: run identical work through identical cohorts under different routing
//! policies, and compare what came out.
//!
//! ## What makes the comparison fair
//!
//! Every arm gets its own mesh built from the same cohort, the same task batch in the
//! same order, and the same outcome model. Stage outcomes are drawn from a hash of
//! `(task, agent, stage, sequence)`, so choosing a different agent genuinely changes the
//! result — but nothing is random, and running the benchmark twice gives the same
//! numbers. The only variable is which agent each policy picked.
//!
//! The one thing this cannot tell you is whether the *simulator* is right. A benchmark
//! against a competence model measures whether the router exploits that model, not
//! whether real agents behave like it. That is what [`Replay`] is for: point it at real
//! recorded runs and it scores the mesh against what actually happened.
//!
//! ## The four axes
//!
//! Quality, cost, latency, and reliability, kept separate on purpose. A policy that
//! routes everything to the strongest agent wins quality and loses cost; one that routes
//! everything to the cheapest wins cost and loses reliability. Collapsing them into a
//! single score would hide exactly the trade-off the mesh exists to make.

use std::collections::BTreeMap;

use hatcher_core::{
    Assignment, ErrorClass, PipelineStage, PipelineTrace, PriorityBand, RuntimeCalibration, StageOutcomeReport,
    TaskEnvelope, TaskSpec,
};
use hatcher_neural::{pipeline, router, BackendKind, NeuralMesh, RunInputs, SimulatedOutcomes};
use serde::{Deserialize, Serialize};

use crate::Scenario;

/// The five staffed stations, in order.
const STAFFED: [PipelineStage; 5] = [
    PipelineStage::Plan,
    PipelineStage::Research,
    PipelineStage::Code,
    PipelineStage::Critique,
    PipelineStage::Verify,
];

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

/// How a policy chooses an agent for a stage.
///
/// Each variant is a strategy a real system might plausibly ship, which is what makes
/// them useful baselines. `StaticRole` in particular is what most agent frameworks do:
/// name a coder, name a reviewer, and send every task to the same one forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingPolicy {
    /// The mesh router: capability × trust × domain mastery × cost, weighted by band.
    Mesh,
    /// Always the same agent per role. No learning, no trust, no cost awareness.
    StaticRole,
    /// Cycle through eligible agents, spreading load evenly.
    RoundRobin,
    /// Whichever eligible agent is expected to cost least.
    Cheapest,
    /// Whichever eligible agent has the highest `A_i`, cost be damned.
    Strongest,
    /// A deterministic arbitrary pick — the control arm. Any policy that cannot beat
    /// this is not routing, it is shuffling.
    Arbitrary,
}

impl RoutingPolicy {
    pub const ALL: [RoutingPolicy; 6] = [
        RoutingPolicy::Mesh,
        RoutingPolicy::StaticRole,
        RoutingPolicy::RoundRobin,
        RoutingPolicy::Cheapest,
        RoutingPolicy::Strongest,
        RoutingPolicy::Arbitrary,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            RoutingPolicy::Mesh => "mesh",
            RoutingPolicy::StaticRole => "static-role",
            RoutingPolicy::RoundRobin => "round-robin",
            RoutingPolicy::Cheapest => "cheapest",
            RoutingPolicy::Strongest => "strongest",
            RoutingPolicy::Arbitrary => "arbitrary",
        }
    }

    /// One line on what this policy is trying to do.
    pub fn description(&self) -> &'static str {
        match self {
            RoutingPolicy::Mesh => "capability × trust × mastery × cost, weighted by priority band",
            RoutingPolicy::StaticRole => "the same agent for a role, every time",
            RoutingPolicy::RoundRobin => "spread the work evenly across eligible agents",
            RoutingPolicy::Cheapest => "always the cheapest eligible agent",
            RoutingPolicy::Strongest => "always the most capable eligible agent",
            RoutingPolicy::Arbitrary => "a deterministic arbitrary pick — the control arm",
        }
    }

    /// Choose an agent for one stage.
    ///
    /// Every policy picks from the same candidate pool the mesh router builds — role
    /// eligibility, exclusions, and isolation are structural facts, not routing opinions.
    /// The policies differ only in which candidate they take.
    fn choose(
        &self,
        mesh: &NeuralMesh,
        stage: PipelineStage,
        task: &TaskSpec,
        band: PriorityBand,
        taken: &[String],
        calibration: &RuntimeCalibration,
        turn: usize,
    ) -> Option<Assignment> {
        let ranked = router::rank_with(mesh, stage, task, band, taken, calibration);
        if ranked.is_empty() {
            return None;
        }

        let picked = match self {
            // `rank` already sorts by the mesh score.
            RoutingPolicy::Mesh => 0,
            // Ties in `rank` break on agent id, so the last entry of an id-sorted view is
            // stable across runs regardless of how capability has drifted.
            RoutingPolicy::StaticRole => {
                let mut by_id: Vec<usize> = (0..ranked.len()).collect();
                by_id.sort_by(|a, b| ranked[*a].agent_id.cmp(&ranked[*b].agent_id));
                by_id[0]
            }
            RoutingPolicy::RoundRobin => turn % ranked.len(),
            RoutingPolicy::Cheapest => index_of_best(&ranked, |assignment| {
                -expected_cost(mesh, &assignment.agent_id, calibration)
            }),
            RoutingPolicy::Strongest => index_of_best(&ranked, |assignment| assignment.capability),
            RoutingPolicy::Arbitrary => {
                let seed: usize = task
                    .id
                    .bytes()
                    .map(|byte| byte as usize)
                    .sum::<usize>()
                    .wrapping_add(stage as usize * 31);
                seed % ranked.len()
            }
        };

        ranked.get(picked).cloned()
    }
}

/// Index of the highest-scoring entry, ties broken on agent id so the pick is stable.
fn index_of_best(ranked: &[Assignment], score: impl Fn(&Assignment) -> f64) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (index, assignment) in ranked.iter().enumerate() {
        let value = score(assignment);
        let better = value > best_score
            || (value == best_score && assignment.agent_id < ranked[best].agent_id);
        if better {
            best = index;
            best_score = value;
        }
    }
    best
}

fn expected_cost(mesh: &NeuralMesh, agent_id: &str, calibration: &RuntimeCalibration) -> f64 {
    mesh.node(agent_id)
        .map(|node| router::expected_profile(node, calibration).1)
        .unwrap_or(f64::MAX)
}

/// Build a full five-stage plan under one policy.
fn plan_under(
    policy: RoutingPolicy,
    mesh: &NeuralMesh,
    task: &TaskSpec,
    calibration: &RuntimeCalibration,
    turn: usize,
) -> Vec<Assignment> {
    let band = mesh.priority_for(task).band;
    let mut taken: Vec<String> = Vec::new();
    let mut planned = Vec::new();

    for (offset, stage) in STAFFED.into_iter().enumerate() {
        if let Some(assignment) = policy.choose(mesh, stage, task, band, &taken, calibration, turn + offset) {
            taken.push(assignment.agent_id.clone());
            planned.push(assignment);
        }
    }
    planned
}

// ---------------------------------------------------------------------------
// Scorecards
// ---------------------------------------------------------------------------

/// What one policy achieved, on the four axes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyScorecard {
    pub policy: String,
    pub tasks: usize,

    // --- quality
    /// Mean verifier score across every staffed stage that ran.
    pub mean_quality: f64,
    /// Fraction of tasks that passed review and verification.
    pub verified_rate: f64,

    // --- reliability
    /// Fraction of staffed stages that succeeded.
    pub stage_success_rate: f64,
    /// Fraction of tasks handed back to a human.
    pub escalation_rate: f64,
    /// The most common failure class, when there was one.
    pub dominant_error: Option<String>,

    // --- cost
    pub total_cost: f64,
    pub cost_per_task: f64,
    /// Cost divided by *verified* tasks — the number that decides whether a policy is
    /// affordable. A cheap policy that never produces working output has infinite cost
    /// per unit of work, which this reports as `None`.
    pub cost_per_verified_task: Option<f64>,

    // --- latency
    pub total_latency_ms: f64,
    pub mean_latency_ms: f64,
    /// Median task latency. Reported alongside the mean because a policy that is usually
    /// fast and occasionally catastrophic should not look the same as a steady one.
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,

    // --- what it did to the mesh
    pub omega_end: f64,
    pub mean_trust: f64,
    pub emergence_ratio: f64,
    /// How many distinct agents the policy actually used.
    pub agents_used: usize,
    /// The decision head that actually ran, so two reports are never compared blind.
    pub head: String,
}

impl PolicyScorecard {
    fn from_traces(policy: RoutingPolicy, mesh: &NeuralMesh, traces: &[PipelineTrace]) -> Self {
        let tasks = traces.len().max(1) as f64;
        let verified = traces.iter().filter(|trace| trace.verified).count();

        let staffed: Vec<&hatcher_core::StageRecord> = traces
            .iter()
            .flat_map(|trace| trace.stages.iter())
            .filter(|record| record.stage.is_staffed() && record.agent.is_some())
            .collect();

        let mut errors: BTreeMap<&'static str, usize> = BTreeMap::new();
        for record in &staffed {
            if record.error != ErrorClass::None {
                *errors.entry(record.error.as_str()).or_insert(0) += 1;
            }
        }

        let mut latencies: Vec<f64> = traces.iter().map(|trace| trace.total_latency_ms()).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let total_cost: f64 = traces.iter().map(|trace| trace.total_cost()).sum();
        let total_latency_ms: f64 = latencies.iter().sum();

        let mut agents: Vec<&str> = traces
            .iter()
            .flat_map(|trace| trace.assignments.iter())
            .map(|assignment| assignment.agent_id.as_str())
            .collect();
        agents.sort_unstable();
        agents.dedup();

        let intelligence = mesh.intelligence();

        Self {
            policy: policy.as_str().to_string(),
            tasks: traces.len(),
            mean_quality: if staffed.is_empty() {
                0.0
            } else {
                staffed.iter().map(|record| record.quality).sum::<f64>() / staffed.len() as f64
            },
            verified_rate: verified as f64 / tasks,
            stage_success_rate: if staffed.is_empty() {
                0.0
            } else {
                staffed.iter().filter(|record| record.success).count() as f64 / staffed.len() as f64
            },
            escalation_rate: traces
                .iter()
                .filter(|trace| trace.decision.action == "escalate")
                .count() as f64
                / tasks,
            dominant_error: errors
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class.to_string()),
            total_cost,
            cost_per_task: total_cost / tasks,
            cost_per_verified_task: (verified > 0).then(|| total_cost / verified as f64),
            total_latency_ms,
            mean_latency_ms: total_latency_ms / tasks,
            p50_latency_ms: percentile(&latencies, 0.50),
            p95_latency_ms: percentile(&latencies, 0.95),
            omega_end: mesh.global.omega,
            mean_trust: mesh.trust.mean_trust(),
            emergence_ratio: intelligence.emergence_ratio(),
            agents_used: agents.len(),
            head: mesh.head().name,
        }
    }

    /// One line for a terminal.
    pub fn headline(&self) -> String {
        format!(
            "{:<12} quality {:.3} | verified {:>3.0}% | stages {:>3.0}% | cost/task {:.3} | cost/verified {} | p50 {:>6.0}ms | esc {:>3.0}% | Ω {:.3}",
            self.policy,
            self.mean_quality,
            self.verified_rate * 100.0,
            self.stage_success_rate * 100.0,
            self.cost_per_task,
            self.cost_per_verified_task
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "  n/a".to_string()),
            self.p50_latency_ms,
            self.escalation_rate * 100.0,
            self.omega_end
        )
    }
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// How one policy compares to the baseline, per axis.
///
/// Signed so that positive always means "better than the baseline", including for cost
/// and latency, where the raw numbers move the other way. A reader should not have to
/// remember which columns are inverted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyDelta {
    pub policy: String,
    pub baseline: String,
    /// Percentage-point change in mean verifier score.
    pub quality_gain: f64,
    /// Percentage-point change in verified rate.
    pub reliability_gain: f64,
    /// Fraction of baseline cost per verified task saved. `0.20` means 20% cheaper.
    pub cost_saving: f64,
    /// Fraction of baseline median latency saved.
    pub latency_saving: f64,
    pub omega_gain: f64,
}

impl PolicyDelta {
    fn between(candidate: &PolicyScorecard, baseline: &PolicyScorecard) -> Self {
        let cost_saving = match (candidate.cost_per_verified_task, baseline.cost_per_verified_task) {
            (Some(candidate_cost), Some(baseline_cost)) if baseline_cost > 0.0 => {
                (baseline_cost - candidate_cost) / baseline_cost
            }
            // A policy that verified nothing has no meaningful cost per unit of work, and
            // reporting it as a saving would reward failing cheaply.
            (None, Some(_)) => -1.0,
            (Some(_), None) => 1.0,
            _ => 0.0,
        };

        Self {
            policy: candidate.policy.clone(),
            baseline: baseline.policy.clone(),
            quality_gain: candidate.mean_quality - baseline.mean_quality,
            reliability_gain: candidate.verified_rate - baseline.verified_rate,
            cost_saving,
            latency_saving: if baseline.p50_latency_ms > 0.0 {
                (baseline.p50_latency_ms - candidate.p50_latency_ms) / baseline.p50_latency_ms
            } else {
                0.0
            },
            omega_gain: candidate.omega_end - baseline.omega_end,
        }
    }

    /// Whether this policy beat the baseline on every axis at once.
    pub fn dominates(&self) -> bool {
        self.quality_gain > 0.0 && self.reliability_gain > 0.0 && self.cost_saving > 0.0 && self.latency_saving >= 0.0
    }
}

// ---------------------------------------------------------------------------
// The benchmark
// ---------------------------------------------------------------------------

/// A head-to-head run of several routing policies over identical work.
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub scenario: Scenario,
    pub policies: Vec<RoutingPolicy>,
    /// The arm everything else is measured against.
    pub baseline: RoutingPolicy,
    pub calibration: RuntimeCalibration,
    /// Cohort every arm starts from. Cloned per arm, so no arm inherits another's
    /// learning.
    pub cohort: Vec<hatcher_core::AgentNode>,
    /// The decision head every arm runs.
    ///
    /// Held constant across arms on purpose: this benchmark varies *routing*, and an arm
    /// that also changed the head would not tell you which of the two moved the numbers.
    /// To compare heads, run the whole benchmark twice with different values here.
    pub head: BackendKind,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmark {
    /// Every policy over the standard rehearsal scenario, against `StaticRole`.
    ///
    /// `StaticRole` is the baseline rather than `Arbitrary` because it is what a
    /// reasonable team actually ships: one named agent per job. Beating a random pick
    /// proves nothing; beating a sensible fixed assignment is the claim worth testing.
    ///
    /// The cohort is [`contested_cohort`], not [`default_cohort`]. That is not a
    /// preference — the default cohort has exactly one agent per role, so every policy
    /// is forced into the same pick and the benchmark reports six identical scorecards.
    /// A routing benchmark needs something to route between.
    pub fn new() -> Self {
        Self {
            scenario: Scenario::rehearsal(),
            policies: RoutingPolicy::ALL.to_vec(),
            baseline: RoutingPolicy::StaticRole,
            calibration: RuntimeCalibration::default(),
            cohort: contested_cohort(),
            head: BackendKind::Native,
        }
    }

    pub fn with_scenario(mut self, scenario: Scenario) -> Self {
        self.scenario = scenario;
        self
    }

    pub fn with_cohort(mut self, cohort: Vec<hatcher_core::AgentNode>) -> Self {
        self.cohort = cohort;
        self
    }

    pub fn with_policies(mut self, policies: Vec<RoutingPolicy>) -> Self {
        self.policies = policies;
        self
    }

    pub fn with_baseline(mut self, baseline: RoutingPolicy) -> Self {
        self.baseline = baseline;
        self
    }

    pub fn with_calibration(mut self, calibration: RuntimeCalibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Run every arm against a specific decision head.
    ///
    /// The head does not change routing, outcomes, cost, or `Ω` — it only decides which
    /// control action comes back. What it moves is `escalation_rate`, which is exactly the
    /// question worth asking of a trained policy: does it hand less work to a human
    /// without handing over work it should have escalated?
    pub fn with_head(mut self, head: BackendKind) -> Self {
        self.head = head;
        self
    }

    /// Run every policy over the same batch and score them.
    pub fn run(&self) -> BenchmarkReport {
        let tasks = self.scenario.task_batch();
        let mut scorecards = Vec::with_capacity(self.policies.len());

        for policy in &self.policies {
            let (scorecard, _) = self.run_policy(*policy, &tasks);
            scorecards.push(scorecard);
        }

        let baseline_name = self.baseline.as_str().to_string();
        let baseline = scorecards
            .iter()
            .find(|scorecard| scorecard.policy == baseline_name)
            .cloned()
            // A benchmark whose baseline was not among the policies still reports the
            // arms; it just cannot express deltas, so it names the first arm instead of
            // silently comparing against nothing.
            .or_else(|| scorecards.first().cloned());

        let deltas = baseline
            .as_ref()
            .map(|baseline| {
                scorecards
                    .iter()
                    .filter(|scorecard| scorecard.policy != baseline.policy)
                    .map(|scorecard| PolicyDelta::between(scorecard, baseline))
                    .collect()
            })
            .unwrap_or_default();

        let winner = scorecards
            .iter()
            .max_by(|a, b| {
                verified_throughput_per_cost(a)
                    .partial_cmp(&verified_throughput_per_cost(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|scorecard| scorecard.policy.clone())
            .unwrap_or_default();

        BenchmarkReport {
            scenario: self.scenario.name.clone(),
            tasks: tasks.len(),
            baseline: baseline.as_ref().map(|scorecard| scorecard.policy.clone()),
            winner,
            scorecards,
            deltas,
        }
    }

    /// Run a single policy and return its scorecard plus every trace.
    pub fn run_policy(&self, policy: RoutingPolicy, tasks: &[TaskSpec]) -> (PolicyScorecard, Vec<PipelineTrace>) {
        // A head that will not load degrades to the built-in one rather than aborting the
        // benchmark; `PolicyScorecard::head` records which one actually ran, so a reader
        // is never comparing arms that silently used different models.
        let mut mesh = NeuralMesh::with_cohort(self.cohort.clone())
            .with_backend(self.head.clone(), hatcher_core::ModelSpec::native_default());
        let source = SimulatedOutcomes;
        let mut traces = Vec::with_capacity(tasks.len());

        for (turn, task) in tasks.iter().enumerate() {
            let planned = plan_under(policy, &mesh, task, &self.calibration, turn);
            let trace = pipeline::run_bound(
                &mut mesh,
                task,
                &RunInputs::new(&source)
                    .with_calibration(self.calibration)
                    .with_assignments(&planned),
            );
            traces.push(trace);
        }

        (PolicyScorecard::from_traces(policy, &mesh, &traces), traces)
    }
}

/// Verified tasks per unit of cost — the single number used only to name a winner.
///
/// Deliberately not part of [`PolicyScorecard`]. The four axes are reported separately
/// because the trade-off between them is the interesting part; this exists so the report
/// can put one name at the top, and it is defined as "working output per dollar" because
/// that is the least arbitrary way to rank policies that differ on every axis.
fn verified_throughput_per_cost(scorecard: &PolicyScorecard) -> f64 {
    match scorecard.cost_per_verified_task {
        Some(cost) if cost > 0.0 => 1.0 / cost,
        _ => 0.0,
    }
}

/// Every policy's results, plus how each compares to the baseline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkReport {
    pub scenario: String,
    pub tasks: usize,
    pub baseline: Option<String>,
    /// Highest verified output per unit of cost.
    pub winner: String,
    pub scorecards: Vec<PolicyScorecard>,
    pub deltas: Vec<PolicyDelta>,
}

impl BenchmarkReport {
    pub fn scorecard(&self, policy: RoutingPolicy) -> Option<&PolicyScorecard> {
        self.scorecards
            .iter()
            .find(|scorecard| scorecard.policy == policy.as_str())
    }

    pub fn delta(&self, policy: RoutingPolicy) -> Option<&PolicyDelta> {
        self.deltas.iter().find(|delta| delta.policy == policy.as_str())
    }

    /// A table for a terminal.
    pub fn table(&self) -> String {
        let mut lines = vec![format!(
            "scenario `{}` — {} tasks, baseline `{}`",
            self.scenario,
            self.tasks,
            self.baseline.as_deref().unwrap_or("none")
        )];
        for scorecard in &self.scorecards {
            lines.push(scorecard.headline());
        }
        lines.push(format!("winner (verified output per unit cost): {}", self.winner));
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// One recorded run: what was asked, and what really happened.
///
/// This is the serialization format for a captured production trace. A caller records
/// these as JSON lines from its own runtime and hands the file back; the mesh replays it
/// without ever having to see the prompts, the outputs, or the agents themselves.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordedRun {
    pub envelope: TaskEnvelope,
    /// One report per stage that ran, naming the agent that actually ran it.
    pub outcomes: Vec<StageOutcomeReport>,
}

impl RecordedRun {
    /// The agent that really ran a given stage.
    pub fn agent_for(&self, stage: PipelineStage) -> Option<&str> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.stage == stage)
            .map(|outcome| outcome.agent_id.as_str())
    }

    pub fn total_cost(&self) -> f64 {
        self.outcomes.iter().map(|outcome| outcome.cost).sum()
    }

    pub fn total_latency_ms(&self) -> f64 {
        self.outcomes.iter().map(|outcome| outcome.latency_ms).sum()
    }

    pub fn mean_quality(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        self.outcomes.iter().map(|outcome| outcome.quality).sum::<f64>() / self.outcomes.len() as f64
    }

    pub fn verified(&self) -> bool {
        !self.outcomes.is_empty() && self.outcomes.iter().all(|outcome| outcome.success)
    }
}

/// A set of recorded runs, replayable against a mesh.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Replay {
    pub runs: Vec<RecordedRun>,
}

impl Replay {
    pub fn new(runs: Vec<RecordedRun>) -> Self {
        Self { runs }
    }

    /// Parse JSON Lines — one [`RecordedRun`] per line, blank lines ignored.
    ///
    /// JSONL rather than a JSON array so a runtime can append a run at a time without
    /// rewriting the file, which is how these recordings are actually produced.
    pub fn from_jsonl(source: &str) -> Result<Self, serde_json::Error> {
        let mut runs = Vec::new();
        for line in source.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            runs.push(serde_json::from_str(line)?);
        }
        Ok(Self { runs })
    }

    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut lines = Vec::with_capacity(self.runs.len());
        for run in &self.runs {
            lines.push(serde_json::to_string(run)?);
        }
        Ok(lines.join("\n"))
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Replay every run and score the mesh's routing against what really happened.
    ///
    /// The mesh learns from the recording as it goes, exactly as it would have in
    /// production, so agreement measured at the end reflects a mesh that has seen the
    /// history — not one guessing cold.
    pub fn evaluate(&self, cohort: Vec<hatcher_core::AgentNode>, calibration: RuntimeCalibration) -> ReplayReport {
        let mut mesh = NeuralMesh::with_cohort(cohort);
        let mut sequence = 0u64;

        let mut agreements = 0usize;
        let mut comparable = 0usize;
        let mut agreed_success = 0usize;
        let mut disagreed_success = 0usize;
        let mut counterfactual_cost = 0.0;
        let mut counterfactual_latency = 0.0;
        let mut per_stage: BTreeMap<&'static str, StageAgreement> = BTreeMap::new();

        for run in &self.runs {
            sequence += 1;
            let task = run.envelope.to_task(sequence);
            let band = mesh.priority_for(&task).band;

            // What the mesh would have chosen, given everything it has learned so far.
            let mut taken: Vec<String> = Vec::new();
            for stage in STAFFED {
                let Some(recorded_agent) = run.agent_for(stage) else {
                    continue;
                };
                let Some(choice) = router::select_with(&mesh, stage, &task, band, &taken, &calibration) else {
                    continue;
                };
                taken.push(recorded_agent.to_string());

                let outcome = run.outcomes.iter().find(|outcome| outcome.stage == stage);
                let agreed = choice.agent_id == recorded_agent;
                comparable += 1;
                let entry = per_stage.entry(stage.as_str()).or_default();
                entry.compared += 1;

                if agreed {
                    agreements += 1;
                    entry.agreed += 1;
                    if outcome.map(|outcome| outcome.success).unwrap_or(false) {
                        agreed_success += 1;
                    }
                } else if outcome.map(|outcome| outcome.success).unwrap_or(false) {
                    disagreed_success += 1;
                }

                if let Some(node) = mesh.node(&choice.agent_id) {
                    let (latency_ms, cost) = router::expected_profile(node, &calibration);
                    counterfactual_cost += cost;
                    counterfactual_latency += latency_ms;
                }
            }

            // Now let the mesh actually learn from what happened, bound to the agents
            // that really ran — not to the ones it would have picked.
            let planned: Vec<Assignment> = STAFFED
                .into_iter()
                .filter_map(|stage| {
                    let agent = run.agent_for(stage)?;
                    let node = mesh.node(agent)?;
                    Some(Assignment {
                        stage,
                        agent_id: node.id.clone(),
                        role: node.role,
                        score: 0.0,
                        capability: node.influence(),
                        inbound_trust: mesh.trust.inbound_trust(&node.id),
                        efficiency: node.resource_efficiency(),
                        mastery: node.mastery(&task.domain),
                    })
                })
                .collect();

            let source: hatcher_neural::ReportedOutcomes = run.outcomes.iter().cloned().collect();
            pipeline::run_bound(
                &mut mesh,
                &task,
                &RunInputs::new(&source)
                    .with_calibration(calibration)
                    .with_assignments(&planned),
            );
        }

        let recorded_cost: f64 = self.runs.iter().map(RecordedRun::total_cost).sum();
        let recorded_latency: f64 = self.runs.iter().map(RecordedRun::total_latency_ms).sum();
        let runs = self.runs.len().max(1) as f64;

        ReplayReport {
            runs: self.runs.len(),
            stages_compared: comparable,
            routing_agreement: if comparable == 0 {
                0.0
            } else {
                agreements as f64 / comparable as f64
            },
            agreed_success_rate: rate(agreed_success, agreements),
            disagreed_success_rate: rate(disagreed_success, comparable.saturating_sub(agreements)),
            recorded_mean_quality: self.runs.iter().map(RecordedRun::mean_quality).sum::<f64>() / runs,
            recorded_verified_rate: self.runs.iter().filter(|run| run.verified()).count() as f64 / runs,
            recorded_cost,
            recorded_latency_ms: recorded_latency,
            counterfactual_cost,
            counterfactual_latency_ms: counterfactual_latency,
            omega_end: mesh.global.omega,
            mean_trust: mesh.trust.mean_trust(),
            per_stage_agreement: per_stage
                .into_iter()
                .map(|(stage, agreement)| (stage.to_string(), agreement))
                .collect(),
        }
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Agreement between the mesh and a recording, for one stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct StageAgreement {
    pub compared: usize,
    pub agreed: usize,
}

impl StageAgreement {
    pub fn rate(&self) -> f64 {
        rate(self.agreed, self.compared)
    }
}

/// What a replay revealed.
///
/// The two success rates are the load-bearing pair. If the mesh's picks succeeded more
/// often than the recording's picks where they *disagreed*, the router was adding value;
/// if they succeeded less often, it was not. Agreement on its own says nothing — a router
/// that always picks what production picked has learned to imitate, not to route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayReport {
    pub runs: usize,
    pub stages_compared: usize,
    /// Fraction of stages where the mesh would have picked the agent that really ran.
    pub routing_agreement: f64,
    /// Success rate of the stages where the mesh agreed with production.
    pub agreed_success_rate: f64,
    /// Success rate of the stages where it did not.
    pub disagreed_success_rate: f64,
    pub recorded_mean_quality: f64,
    pub recorded_verified_rate: f64,
    pub recorded_cost: f64,
    pub recorded_latency_ms: f64,
    /// What the mesh's own picks were expected to cost, over the same stages.
    pub counterfactual_cost: f64,
    pub counterfactual_latency_ms: f64,
    pub omega_end: f64,
    pub mean_trust: f64,
    pub per_stage_agreement: BTreeMap<String, StageAgreement>,
}

impl ReplayReport {
    /// Fraction of recorded cost the mesh's own routing would have saved.
    ///
    /// An estimate, not a measurement: it prices the mesh's picks at their observed
    /// averages, which is not the same as having actually run them.
    pub fn projected_cost_saving(&self) -> f64 {
        if self.recorded_cost <= 0.0 {
            return 0.0;
        }
        (self.recorded_cost - self.counterfactual_cost) / self.recorded_cost
    }

    /// Whether the mesh's disagreements with production looked better than production.
    pub fn router_added_value(&self) -> bool {
        self.routing_agreement < 1.0 && self.disagreed_success_rate < self.agreed_success_rate
    }

    pub fn headline(&self) -> String {
        // With no disagreements there is no `disagreed_success_rate` to report, and
        // printing the `0.0` placeholder would read as "the mesh's alternatives all
        // failed" — the opposite of what an empty set means.
        let verdict = if self.routing_agreement >= 1.0 {
            "no disagreements".to_string()
        } else {
            format!(
                "agreed {:.0}% vs disagreed {:.0}% success",
                self.agreed_success_rate * 100.0,
                self.disagreed_success_rate * 100.0
            )
        };

        format!(
            "replay {} runs | agreement {:.0}% | {} | recorded cost {:.3} vs projected {:.3} ({:+.0}%) | Ω {:.3}",
            self.runs,
            self.routing_agreement * 100.0,
            verdict,
            self.recorded_cost,
            self.counterfactual_cost,
            self.projected_cost_saving() * 100.0,
            self.omega_end
        )
    }
}

// ---------------------------------------------------------------------------
// Cohorts
// ---------------------------------------------------------------------------

/// A cohort with a real choice at every station.
///
/// Three candidates per role, differentiated on the axes a router is supposed to trade
/// off against each other:
///
/// | Tier | Capability | Domain mastery | Cost & latency |
/// |---|---|---|---|
/// | expert | high | high | high |
/// | generalist | middling | middling, broad | low |
/// | novice | low | low | very low |
///
/// The tiers are deliberately not rank-ordered on every axis at once. If the expert were
/// simply better at everything the benchmark would have one right answer and would be
/// measuring nothing; the point is that the expert is worth its cost on hard work and
/// wastes it on easy work, which is exactly the judgement the priority band exists to
/// make.
pub fn contested_cohort() -> Vec<hatcher_core::AgentNode> {
    use hatcher_core::{AgentNode, AgentRole, CapabilityVector, ResourceProfile};

    let roles = [
        (AgentRole::Planner, "planning"),
        (AgentRole::Researcher, "research"),
        (AgentRole::Coder, "rust"),
        (AgentRole::Critic, "review"),
        (AgentRole::Verifier, "verification"),
    ];

    let mut cohort = Vec::with_capacity(roles.len() * 3);
    for (role, speciality) in roles {
        let name = role.as_str();

        cohort.push(
            AgentNode::new(format!("{name}-expert"), format!("{name} expert"), role)
                .with_capability(CapabilityVector::new(0.92, 0.88, 0.86, 0.90, 0.82))
                .with_resources(ResourceProfile::new(0.62, 0.55))
                .with_confidence(0.72)
                .with_expertise("rust", 0.92)
                .with_expertise(speciality, 0.94)
                .with_expertise("general", 0.70),
        );
        cohort.push(
            AgentNode::new(format!("{name}-generalist"), format!("{name} generalist"), role)
                .with_capability(CapabilityVector::new(0.78, 0.68, 0.76, 0.80, 0.70))
                .with_resources(ResourceProfile::new(0.22, 0.18))
                .with_confidence(0.60)
                .with_expertise("rust", 0.62)
                .with_expertise(speciality, 0.66)
                .with_expertise("general", 0.74),
        );
        cohort.push(
            AgentNode::new(format!("{name}-novice"), format!("{name} novice"), role)
                .with_capability(CapabilityVector::new(0.58, 0.42, 0.55, 0.60, 0.45))
                .with_resources(ResourceProfile::new(0.07, 0.06))
                .with_confidence(0.45)
                .with_expertise("rust", 0.30)
                .with_expertise(speciality, 0.34)
                .with_expertise("general", 0.40),
        );
    }
    cohort
}

// ---------------------------------------------------------------------------
// Synthetic recordings
// ---------------------------------------------------------------------------

/// Generate an example recording, so the replay harness runs out of the box.
///
/// The recording routes every stage to one tier of [`contested_cohort`] — by default the
/// `expert` tier, which is what a team ships when it decides the safest thing is to send
/// everything to its best model. That is a deliberate pattern rather than uniform noise:
/// it is expensive in a way a router can see and a fixed assignment never will, so the
/// replay has an actual counterfactual to price. Anyone wiring up a real recording should
/// expect their data to be considerably messier than this.
pub fn synthetic_recording(runs: usize, tier: &str) -> Replay {
    let cohort = contested_cohort();
    let mut recorded = Vec::with_capacity(runs);

    for index in 0..runs {
        let phase = index as f64 / runs.max(1) as f64;
        let difficulty = 0.25 + 0.5 * phase;

        let envelope = TaskEnvelope::new(format!("recorded task {index}"))
            .with_domain("rust")
            .with_features(vec![phase, 1.0 - phase, difficulty, 0.5]);

        let outcomes = STAFFED
            .into_iter()
            .enumerate()
            .filter_map(|(offset, stage)| {
                let role = stage.preferred_role()?;
                let wanted = format!("{}-{tier}", role.as_str());
                let node = cohort.iter().find(|node| node.id == wanted)?;

                // Deterministic and legible: every third run fails at the review stations.
                let success = !(index % 3 == 0 && offset >= 3);
                let quality = if success { 0.70 + 0.25 * (1.0 - difficulty) } else { 0.15 };

                let mut report = if success {
                    StageOutcomeReport::success(stage, &node.id, quality)
                } else {
                    StageOutcomeReport::failure(stage, &node.id, ErrorClass::Quality)
                };
                report.confidence = quality;
                Some(
                    report
                        .with_latency_ms(node.resources.latency * 60_000.0 * (0.8 + 0.4 * difficulty))
                        .with_cost(node.resources.energy * (0.9 + 0.2 * difficulty)),
                )
            })
            .collect();

        recorded.push(RecordedRun { envelope, outcomes });
    }

    Replay::new(recorded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_neural::default_cohort;

    fn quick_scenario() -> Scenario {
        Scenario::new("bench", "rust", 12, 0.35)
    }

    // --- policies ----------------------------------------------------------

    #[test]
    fn every_policy_staffs_every_station() {
        let mesh = NeuralMesh::default();
        let task = quick_scenario().task_batch().remove(0);
        let calibration = RuntimeCalibration::default();

        for policy in RoutingPolicy::ALL {
            let planned = plan_under(policy, &mesh, &task, &calibration, 0);
            assert_eq!(planned.len(), 5, "{} left a station empty", policy.as_str());

            let mut agents: Vec<&str> = planned.iter().map(|a| a.agent_id.as_str()).collect();
            agents.sort_unstable();
            let before = agents.len();
            agents.dedup();
            assert_eq!(agents.len(), before, "{} double-booked an agent", policy.as_str());
        }
    }

    #[test]
    fn the_cheapest_policy_picks_the_cheapest_agent() {
        let mut mesh = NeuralMesh::default();
        let calibration = RuntimeCalibration::default();
        // Give the coder a measured cost, and add a cheaper alternative.
        mesh.add_agent(
            hatcher_core::AgentNode::new("coder-02", "budget coder", hatcher_core::AgentRole::Coder)
                .with_resources(hatcher_core::ResourceProfile::new(0.02, 0.02)),
        );

        let task = quick_scenario().task_batch().remove(0);
        let planned = plan_under(RoutingPolicy::Cheapest, &mesh, &task, &calibration, 0);
        let coding = planned.iter().find(|a| a.stage == PipelineStage::Code).unwrap();
        assert_eq!(coding.agent_id, "coder-02");
    }

    #[test]
    fn round_robin_spreads_work_that_a_static_policy_concentrates() {
        let mut cohort = default_cohort();
        for index in 0..3 {
            cohort.push(
                hatcher_core::AgentNode::new(
                    format!("coder-1{index}"),
                    format!("coder {index}"),
                    hatcher_core::AgentRole::Coder,
                )
                .with_capability(hatcher_core::CapabilityVector::uniform(0.7)),
            );
        }

        let benchmark = Benchmark::new().with_scenario(quick_scenario()).with_cohort(cohort);
        let tasks = benchmark.scenario.task_batch();

        let (round_robin, _) = benchmark.run_policy(RoutingPolicy::RoundRobin, &tasks);
        let (statically, _) = benchmark.run_policy(RoutingPolicy::StaticRole, &tasks);
        assert!(
            round_robin.agents_used > statically.agents_used,
            "round-robin {} should touch more agents than static {}",
            round_robin.agents_used,
            statically.agents_used
        );
    }

    // --- the benchmark -----------------------------------------------------

    #[test]
    fn a_benchmark_scores_every_policy_on_all_four_axes() {
        let report = Benchmark::new().with_scenario(quick_scenario()).run();

        assert_eq!(report.scorecards.len(), RoutingPolicy::ALL.len());
        assert_eq!(report.tasks, 12);
        assert_eq!(report.baseline.as_deref(), Some("static-role"));

        for scorecard in &report.scorecards {
            assert_eq!(scorecard.tasks, 12);
            assert!((0.0..=1.0).contains(&scorecard.mean_quality));
            assert!((0.0..=1.0).contains(&scorecard.verified_rate));
            assert!(scorecard.total_cost > 0.0, "{} reported no cost", scorecard.policy);
            assert!(scorecard.total_latency_ms > 0.0, "{} reported no latency", scorecard.policy);
            assert!(scorecard.p95_latency_ms >= scorecard.p50_latency_ms);
        }
    }

    #[test]
    fn benchmarks_are_reproducible() {
        let benchmark = Benchmark::new().with_scenario(quick_scenario());
        assert_eq!(benchmark.run(), benchmark.run());
    }

    #[test]
    fn the_benchmark_actually_discriminates_between_policies() {
        // The failure this guards against is silent and total: with one agent per role
        // there is nothing to route between, every policy is forced into the same pick,
        // and the report is six identical rows that look like a finding.
        let report = Benchmark::new().with_scenario(quick_scenario()).run();

        let mut costs: Vec<String> = report
            .scorecards
            .iter()
            .map(|scorecard| format!("{:.6}", scorecard.total_cost))
            .collect();
        costs.sort();
        costs.dedup();

        assert!(
            costs.len() > 1,
            "every policy spent the same amount — the cohort offers no choices, so this \
             benchmark is measuring nothing: {costs:?}"
        );
    }

    #[test]
    fn the_default_cohort_is_too_thin_to_benchmark_and_that_is_visible() {
        let report = Benchmark::new()
            .with_scenario(quick_scenario())
            .with_cohort(default_cohort())
            .run();

        let distinct: std::collections::BTreeSet<String> = report
            .scorecards
            .iter()
            .map(|scorecard| format!("{:.6}", scorecard.total_cost))
            .collect();

        assert_eq!(
            distinct.len(),
            1,
            "one agent per role means one possible plan; this is why `Benchmark::new` \
             does not use the default cohort"
        );
    }

    #[test]
    fn the_contested_cohort_offers_a_real_choice_at_every_station() {
        let cohort = contested_cohort();
        assert_eq!(cohort.len(), 15, "three candidates for each of five roles");

        for stage in STAFFED {
            let role = stage.preferred_role().unwrap();
            let candidates = cohort.iter().filter(|node| node.role == role).count();
            assert_eq!(candidates, 3, "{} has nothing to choose between", stage.as_str());
        }

        // The tiers must actually differ on cost, or "cheapest" is the same policy as
        // "strongest" wearing a different name.
        let expert = cohort.iter().find(|node| node.id == "coder-expert").unwrap();
        let novice = cohort.iter().find(|node| node.id == "coder-novice").unwrap();
        assert!(expert.influence() > novice.influence());
        assert!(expert.resources.cost() > novice.resources.cost());
        assert!(expert.mastery("rust") > novice.mastery("rust"));
    }

    #[test]
    fn cost_and_capability_policies_genuinely_pick_different_agents() {
        let benchmark = Benchmark::new().with_scenario(quick_scenario());
        let mesh = NeuralMesh::with_cohort(benchmark.cohort.clone());
        let task = benchmark.scenario.task_batch().remove(0);

        let cheapest = plan_under(RoutingPolicy::Cheapest, &mesh, &task, &benchmark.calibration, 0);
        let strongest = plan_under(RoutingPolicy::Strongest, &mesh, &task, &benchmark.calibration, 0);

        assert_ne!(
            cheapest.iter().map(|a| a.agent_id.as_str()).collect::<Vec<_>>(),
            strongest.iter().map(|a| a.agent_id.as_str()).collect::<Vec<_>>(),
            "if these agree, the cohort has no cost/capability trade-off in it"
        );
    }

    #[test]
    fn deltas_are_signed_so_positive_always_means_better() {
        let report = Benchmark::new().with_scenario(quick_scenario()).run();
        let baseline = report.scorecard(RoutingPolicy::StaticRole).unwrap();
        let strongest = report.scorecard(RoutingPolicy::Strongest).unwrap();
        let delta = report.delta(RoutingPolicy::Strongest).unwrap();

        assert!((delta.quality_gain - (strongest.mean_quality - baseline.mean_quality)).abs() < 1e-9);
        if let (Some(candidate), Some(base)) =
            (strongest.cost_per_verified_task, baseline.cost_per_verified_task)
        {
            let cheaper = candidate < base;
            assert_eq!(cheaper, delta.cost_saving > 0.0, "a saving must mean it cost less");
        }
    }

    #[test]
    fn a_policy_that_verifies_nothing_is_not_rewarded_for_being_cheap() {
        let broke = PolicyScorecard {
            cost_per_verified_task: None,
            ..Benchmark::new()
                .with_scenario(quick_scenario())
                .run()
                .scorecards
                .remove(0)
        };
        let working = Benchmark::new().with_scenario(quick_scenario()).run().scorecards.remove(0);

        let delta = PolicyDelta::between(&broke, &working);
        assert!(delta.cost_saving < 0.0, "failing cheaply is not a saving");
        assert_eq!(verified_throughput_per_cost(&broke), 0.0);
    }

    #[test]
    fn on_routine_work_the_mesh_matches_a_fixed_assignment_for_much_less_money() {
        // This is the actual claim the router makes, and the only one worth pinning in a
        // test: on work the cohort can handle, spending capability where it is needed and
        // saving it where it is not gets the same output for less.
        let report = Benchmark::new().with_scenario(Scenario::rehearsal()).run();
        let delta = report.delta(RoutingPolicy::Mesh).unwrap();

        assert!(
            delta.reliability_gain >= 0.0,
            "the saving must not come out of how often the work actually lands: {:+.3}",
            delta.reliability_gain
        );
        assert!(
            delta.cost_saving > 0.20,
            "expected a material saving against a fixed assignment, got {:+.1}%",
            delta.cost_saving * 100.0
        );
        assert!(
            delta.latency_saving > 0.0,
            "and it should not buy the saving with wall-clock time: {:+.1}%",
            delta.latency_saving * 100.0
        );
    }

    #[test]
    fn the_mesh_router_beats_an_arbitrary_pick_on_output_per_cost() {
        let report = Benchmark::new().with_scenario(Scenario::rehearsal()).run();

        let mesh = report.scorecard(RoutingPolicy::Mesh).unwrap();
        let arbitrary = report.scorecard(RoutingPolicy::Arbitrary).unwrap();

        assert!(
            verified_throughput_per_cost(mesh) > verified_throughput_per_cost(arbitrary),
            "a router that cannot beat a coin flip is shuffling, not routing: mesh={:?} arbitrary={:?}",
            mesh.cost_per_verified_task,
            arbitrary.cost_per_verified_task
        );
    }

    #[test]
    fn the_benchmark_is_willing_to_report_that_the_mesh_lost() {
        // On `frontier` — a domain nobody has mastered — mastery carries no signal, the
        // cost term dominates, and the mesh under-provisions work that needed the expert.
        // This test exists so that stops being true loudly rather than quietly: if the
        // router is fixed, this fails and the finding in `docs/benchmark.md` gets updated.
        let report = Benchmark::new().with_scenario(Scenario::frontier()).run();

        let mesh = report.scorecard(RoutingPolicy::Mesh).unwrap();
        let strongest = report.scorecard(RoutingPolicy::Strongest).unwrap();

        assert!(
            mesh.verified_rate <= strongest.verified_rate,
            "the mesh now matches `strongest` in an unmastered domain — update the \
             documented finding: mesh={:.3} strongest={:.3}",
            mesh.verified_rate,
            strongest.verified_rate
        );
    }

    #[test]
    fn every_arm_records_the_head_it_actually_ran() {
        let report = Benchmark::new().with_scenario(quick_scenario()).run();
        assert!(
            report.scorecards.iter().all(|scorecard| scorecard.head == "native"),
            "arms that silently used different models must never be compared as if they had not"
        );
    }

    #[test]
    fn an_unloadable_head_degrades_rather_than_aborting_the_benchmark() {
        // Reported, not hidden: the scorecards still say `native`, so a reader comparing
        // this run against one with a real policy can see the two are not comparable.
        let report = Benchmark::new()
            .with_scenario(quick_scenario())
            .with_head(BackendKind::Onnx {
                model_path: "models/fixtures/definitely-not-here.onnx".into(),
            })
            .run();

        assert_eq!(report.scorecards.len(), RoutingPolicy::ALL.len());
        assert!(report.scorecards.iter().all(|scorecard| scorecard.head == "native"));
    }

    #[test]
    fn the_report_renders_a_table_naming_a_winner() {
        let report = Benchmark::new().with_scenario(quick_scenario()).run();
        let table = report.table();

        assert!(table.contains("baseline `static-role`"));
        assert!(table.contains(&report.winner));
        assert!(table.lines().count() >= RoutingPolicy::ALL.len() + 2);
    }

    // --- replay ------------------------------------------------------------

    #[test]
    fn a_recording_round_trips_through_jsonl() {
        let replay = synthetic_recording(6, "expert");
        let encoded = replay.to_jsonl().unwrap();

        assert_eq!(encoded.lines().count(), 6, "one run per line, appendable");
        assert_eq!(Replay::from_jsonl(&encoded).unwrap(), replay);
        assert!(Replay::from_jsonl("\n\n").unwrap().is_empty(), "blank lines are ignored");
    }

    #[test]
    fn replaying_a_recording_scores_routing_against_what_really_happened() {
        let replay = synthetic_recording(20, "expert");
        let report = replay.evaluate(default_cohort(), RuntimeCalibration::default());

        assert_eq!(report.runs, 20);
        assert_eq!(report.stages_compared, 100, "five stations over twenty runs");
        assert!((0.0..=1.0).contains(&report.routing_agreement));
        assert!(report.recorded_cost > 0.0);
        assert!(report.per_stage_agreement.len() == 5);
        assert!(report.headline().contains("replay 20 runs"));
    }

    #[test]
    fn a_replay_teaches_the_mesh_what_the_expensive_agent_really_costs() {
        let replay = synthetic_recording(24, "expert");
        let report = replay.evaluate(default_cohort(), RuntimeCalibration::default());

        assert!(report.mean_trust > 0.0);
        assert!(
            report.counterfactual_cost > 0.0,
            "the mesh must be able to price its own alternative"
        );
        assert!(
            report.projected_cost_saving().abs() <= 1.0,
            "a saving is a fraction of what was spent"
        );
    }

    #[test]
    fn replaying_an_empty_recording_reports_nothing_rather_than_panicking() {
        let report = Replay::default().evaluate(default_cohort(), RuntimeCalibration::default());
        assert_eq!(report.runs, 0);
        assert_eq!(report.routing_agreement, 0.0);
        assert_eq!(report.projected_cost_saving(), 0.0);
        assert!(!report.router_added_value());
    }

    #[test]
    fn a_replay_against_an_unknown_cohort_does_not_credit_anyone() {
        let replay = synthetic_recording(4, "expert");
        let report = replay.evaluate(
            vec![hatcher_core::AgentNode::new(
                "stranger",
                "stranger",
                hatcher_core::AgentRole::Executor,
            )],
            RuntimeCalibration::default(),
        );

        assert_eq!(report.routing_agreement, 0.0, "none of the recorded agents exist here");
        assert!(
            !report.router_added_value(),
            "a mesh that knows none of the agents has not proven anything"
        );
    }

    #[test]
    fn percentiles_handle_the_degenerate_cases() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[7.0], 0.95), 7.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.95), 4.0);
    }
}
