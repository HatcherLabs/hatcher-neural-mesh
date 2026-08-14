//! The neural mesh pipeline: one pass of work through the mesh.
//!
//! ```text
//! Incoming Task
//!       │
//!       ▼
//!  Task Parser        →  U, B, I measured from the request
//!       ▼
//!  Priority Engine    →  P = (U·B·I·urgency) / (C + τ)
//!       ▼
//!  Planner Agent      →  routed by capability × trust × mastery × cost
//!    ┌──┴───┐
//!    ▼      ▼
//! Research  Coding
//!    └──┬───┘
//!       ▼
//!  Critic Agent       →  does the work survive review?
//!       ▼
//!  Verifier           →  is it actually correct?
//!       ▼
//!  Memory Update      →  M_i(t+1) = M_i + ηK_i − δR_i
//!       ▼
//!  Trust Graph Update →  T_ij(t+1) = T_ij + λS_ij − μE_ij,  W_ij += φT_ij(1−W_ij) − ψL
//!       ▼
//!  Ω Update           →  Ω(t+1) = Ω + α(L+E+C) − β(F+D)
//! ```
//!
//! Every iteration improves — or degrades — trust, routing, memory, specialization,
//! and collaboration, which is the point: the mesh that ran the task is not the mesh
//! that will run the next one.
//!
//! ## Where outcomes come from
//!
//! The pipeline decides *who* runs each stage. It does not decide what happened — that
//! comes from an [`OutcomeSource`], which is either the deterministic competence model
//! ([`SimulatedOutcomes`]) or a set of results reported by a real runtime
//! ([`ReportedOutcomes`]). Everything downstream of the staffed stations — memory,
//! trust, capability, `Ω`, the decision — is identical either way, which is the whole
//! point: a mesh that learned from a rehearsal and a mesh that learned from production
//! ran the same code.
//!
//! Simulated outcomes are drawn from a hash of `(task id, agent id, stage, sequence)`,
//! never from a clock or an RNG. The same task against the same mesh state always
//! replays exactly, which is what makes rehearsal meaningful and what lets a trace
//! digest be an attestation rather than a souvenir. Every trace records its
//! [`OutcomeProvenance`] inside the digest, so a rehearsal can never be presented as
//! evidence of work that really happened.

use hatcher_core::{
    canonical_digest, Assignment, ExecutionMode, MeshAction, NeuralSignal, OmegaDelta, OmegaRegime,
    OutcomeProvenance, PipelineStage, PipelineTrace, PriorityScore, RuntimeCalibration,
    StageRecord, TaskSpec,
};

use crate::inference::Decision;
use crate::learning::{apply_outcome, StageOutcome};
use crate::mesh::NeuralMesh;
use crate::outcomes::{OutcomeSource, SimulatedOutcomes, StageContext};
use crate::router;

/// Thresholds and rates that govern one pipeline pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineConfig {
    /// Per-run memory decay rate applied before new knowledge is written.
    pub memory_decay_rate: f64,
    /// Verifier score below which the mesh will not claim a result is stabilized.
    pub stabilize_threshold: f64,
    /// Priority above which an unverified result must be escalated to a human.
    pub escalation_priority: f64,
    /// How strongly task difficulty suppresses an agent's success chance. Applied as an
    /// exponent on fitness, so difficulty punishes weak agents far harder than strong
    /// ones — see [`SimulatedOutcomes`].
    pub difficulty_weight: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            memory_decay_rate: 0.03,
            stabilize_threshold: 0.55,
            escalation_priority: 0.60,
            difficulty_weight: 1.50,
        }
    }
}

/// The five staffed stations, in order.
const STAFFED: [PipelineStage; 5] = [
    PipelineStage::Plan,
    PipelineStage::Research,
    PipelineStage::Code,
    PipelineStage::Critique,
    PipelineStage::Verify,
];

/// Everything a run needs that is not the mesh or the task.
///
/// Bundled into one struct rather than threaded as five parameters because three of the
/// four are usually defaults, and a call site that has to spell out
/// `run(mesh, task, config, calibration, source, None)` invites getting the order wrong.
pub struct RunInputs<'a> {
    pub config: PipelineConfig,
    pub calibration: RuntimeCalibration,
    /// Resolves what happened at each staffed stage.
    pub outcomes: &'a dyn OutcomeSource,
    /// Pre-bound assignments, in place of asking the router.
    ///
    /// Set by the adapter when a caller has already dispatched real work to specific
    /// agents. Re-routing under a caller that has already acted would make the reported
    /// outcomes describe a run that never happened.
    pub assignments: Option<&'a [Assignment]>,
}

impl<'a> RunInputs<'a> {
    pub fn new(outcomes: &'a dyn OutcomeSource) -> Self {
        Self {
            config: PipelineConfig::default(),
            calibration: RuntimeCalibration::default(),
            outcomes,
            assignments: None,
        }
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_calibration(mut self, calibration: RuntimeCalibration) -> Self {
        self.calibration = calibration;
        self
    }

    pub fn with_assignments(mut self, assignments: &'a [Assignment]) -> Self {
        self.assignments = Some(assignments);
        self
    }
}

/// Run a task through the mesh with the default configuration and simulated outcomes.
pub fn run(mesh: &mut NeuralMesh, task: &TaskSpec) -> PipelineTrace {
    run_with(mesh, task, &PipelineConfig::default())
}

/// Run a task with an explicit configuration and simulated outcomes.
pub fn run_with(mesh: &mut NeuralMesh, task: &TaskSpec, config: &PipelineConfig) -> PipelineTrace {
    let source = SimulatedOutcomes;
    run_bound(mesh, task, &RunInputs::new(&source).with_config(*config))
}

/// Run a task against outcomes supplied from outside.
///
/// This is the production path: the mesh routes, something else executes, and the
/// results come back in. The resulting trace is marked
/// [`OutcomeProvenance::Reported`] and is the only kind that attests to real work.
pub fn run_with_outcomes(
    mesh: &mut NeuralMesh,
    task: &TaskSpec,
    outcomes: &dyn OutcomeSource,
    calibration: &RuntimeCalibration,
) -> PipelineTrace {
    run_bound(
        mesh,
        task,
        &RunInputs::new(outcomes).with_calibration(*calibration),
    )
}

/// Run a task through the mesh, mutating trust, memory, capability, and `Ω`.
pub fn run_bound(mesh: &mut NeuralMesh, task: &TaskSpec, inputs: &RunInputs<'_>) -> PipelineTrace {
    let config = &inputs.config;
    let coefficients = mesh.coefficients;
    let gate = task.execution_mode.plasticity_gate();
    let omega_before = mesh.global.omega;
    mesh.sequence += 1;

    let mut stages: Vec<StageRecord> = Vec::with_capacity(PipelineStage::ALL.len());
    let mut assignments: Vec<Assignment> = Vec::new();
    let mut outcomes: Vec<StageOutcome> = Vec::new();

    // --- 1. Task Parser -----------------------------------------------------
    stages.push(StageRecord::bookkeeping(
        PipelineStage::Parse,
        1.0,
        format!(
            "domain={} U={:.2} B={:.2} I={:.2} urgency={:.2} mode={}",
            task.domain,
            task.uncertainty,
            task.budget,
            task.implementation_cost,
            task.urgency,
            task.execution_mode.as_str()
        ),
    ));

    // --- 2. Priority Engine -------------------------------------------------
    let priority = mesh.priority_for(task);
    stages.push(StageRecord::bookkeeping(
        PipelineStage::Prioritize,
        priority.confidence,
        format!(
            "P={:.3} band={} C={:.2} tau={:.2}",
            priority.value,
            priority.band.as_str(),
            priority.confidence,
            priority.tau
        ),
    ));

    // --- 3-7. Staffed stations ---------------------------------------------
    let mut taken: Vec<String> = Vec::new();
    let mut plan_ok = true;
    let mut research_ok = true;
    let mut code_ok = true;
    let mut critique_ok = true;
    let mut verify_ok = true;
    let mut provenance: Option<OutcomeProvenance> = None;

    for stage in STAFFED {
        let carry = match stage {
            PipelineStage::Plan => 1.0,
            PipelineStage::Research => carry_for(plan_ok),
            PipelineStage::Code => carry_for(plan_ok && research_ok),
            PipelineStage::Critique => carry_for(code_ok),
            PipelineStage::Verify => carry_for(code_ok && critique_ok),
            _ => 1.0,
        };

        let selected = match inputs.assignments {
            Some(bound) => bound
                .iter()
                .find(|assignment| assignment.stage == stage)
                .cloned(),
            None => router::select(mesh, stage, task, priority.band, &taken),
        };

        let Some(assignment) = selected else {
            // No candidate at all means an empty mesh. An unstaffed station is a failed
            // station — it must not read as a silent pass to everything downstream.
            match stage {
                PipelineStage::Plan => plan_ok = false,
                PipelineStage::Research => research_ok = false,
                PipelineStage::Code => code_ok = false,
                PipelineStage::Critique => critique_ok = false,
                PipelineStage::Verify => verify_ok = false,
                _ => {}
            }
            stages.push(StageRecord::unstaffed(stage));
            continue;
        };
        taken.push(assignment.agent_id.clone());

        let resolved = inputs.outcomes.resolve(&StageContext {
            mesh,
            task,
            assignment: &assignment,
            carry,
            config,
            calibration: &inputs.calibration,
        });

        match stage {
            PipelineStage::Plan => plan_ok = resolved.success,
            PipelineStage::Research => research_ok = resolved.success,
            PipelineStage::Code => code_ok = resolved.success,
            PipelineStage::Critique => critique_ok = resolved.success,
            PipelineStage::Verify => verify_ok = resolved.success,
            _ => {}
        }

        provenance = Some(match provenance {
            Some(seen) => seen.merge(resolved.provenance),
            None => resolved.provenance,
        });

        stages.push(StageRecord {
            stage,
            agent: Some(assignment.agent_id.clone()),
            success: resolved.success,
            confidence: resolved.confidence,
            cost: resolved.cost,
            quality: resolved.quality,
            latency_ms: resolved.latency_ms,
            error: resolved.error,
            provenance: Some(resolved.provenance),
            note: resolved.note.clone(),
        });

        outcomes.push(StageOutcome {
            agent_id: assignment.agent_id.clone(),
            stage,
            domain: task.domain.clone(),
            success: resolved.success,
            confidence: resolved.confidence,
            quality: resolved.quality,
            // Diminishing returns: an agent already fluent in this domain learns
            // little from doing it again, which is what keeps L from growing forever
            // on repetitive work.
            novelty: (task.uncertainty * (1.0 - assignment.mastery.clamp(0.0, 1.0)))
                .clamp(0.0, 1.0),
            error: resolved.error,
            latency_ms: resolved.latency_ms,
            cost: resolved.cost,
            provenance: resolved.provenance,
        });

        assignments.push(assignment);
    }

    // A result is verified only when the work passed review *and* both reviewers
    // actually did their jobs. A critic that could not form a judgment is not a pass.
    let verified = code_ok && critique_ok && verify_ok;

    // --- 8. Memory Update ---------------------------------------------------
    let decayed = mesh.memory.decay(config.memory_decay_rate);
    let mut credited: Vec<(StageOutcome, f64, f64)> = Vec::with_capacity(outcomes.len());
    for outcome in &outcomes {
        let knowledge = mesh.memory.remember(outcome.to_memory_record(&task.id));
        let decay = decayed.get(&outcome.agent_id).copied().unwrap_or(0.0);
        credited.push((outcome.clone(), knowledge, decay));
    }

    let mut learning_total = 0.0;
    let mut staleness_total = 0.0;
    for (outcome, knowledge, decay) in &credited {
        if let Some(node) = mesh.node_mut(&outcome.agent_id) {
            staleness_total += apply_outcome(
                node,
                outcome,
                *knowledge,
                *decay,
                &coefficients,
                &inputs.calibration,
            );
            learning_total += *knowledge;
        }
    }
    let cohort = mesh.nodes.len().max(1) as f64;

    stages.push(StageRecord::bookkeeping(
        PipelineStage::MemoryUpdate,
        mesh.mean_memory(),
        format!(
            "K={:.3} decayed={:.3} records={} obsolescence={:.3}",
            learning_total,
            decayed.values().sum::<f64>(),
            mesh.memory.records.len(),
            staleness_total
        ),
    ));

    // --- 9. Trust Graph Update ---------------------------------------------
    // Credit flows along the handoff chain: each agent judges the one it handed to.
    let mut handoffs = 0usize;
    let mut excused = 0usize;
    for window in assignments.windows(2) {
        let (from, to) = (&window[0], &window[1]);
        let downstream = stages.iter().find(|record| record.stage == to.stage);
        let downstream_ok = downstream.map(|record| record.success).unwrap_or(false);

        // A failure that was not the agent's doing never reaches the ledger. Trust is a
        // claim about whether one agent can rely on another's *work*; an outage is not a
        // betrayal, and a boolean ledger has no way to record it at reduced weight.
        let admissible = downstream_ok
            || downstream
                .map(|record| record.error.is_trust_evidence())
                .unwrap_or(true);
        if !admissible {
            excused += 1;
            continue;
        }
        if mesh
            .trust
            .record(&from.agent_id, &to.agent_id, downstream_ok)
        {
            handoffs += 1;
        }
    }
    // The verifier's judgment also feeds back to the planner: whoever planned this owns
    // the outcome, which is how a planner that keeps producing unworkable plans loses
    // trust even though it never wrote a line.
    if let (Some(first), Some(last)) = (assignments.first(), assignments.last()) {
        if first.agent_id != last.agent_id
            && mesh.trust.record(&last.agent_id, &first.agent_id, verified)
        {
            handoffs += 1;
        }
    }

    let settlement = mesh.trust.settle(&coefficients, gate);
    for node in &mut mesh.nodes {
        if taken.contains(&node.id) {
            node.telemetry.collaborations += 1;
        }
    }

    stages.push(StageRecord::bookkeeping(
        PipelineStage::TrustUpdate,
        settlement.collaboration_efficiency,
        format!(
            "handoffs={} excused={} pairs_moved={} C={:.3} gate={:.2}",
            handoffs, excused, settlement.updated_pairs, settlement.collaboration_efficiency, gate
        ),
    ));

    // --- 10. Ω Update -------------------------------------------------------
    let intelligence = mesh.intelligence();
    let staffed: Vec<&StageRecord> = stages
        .iter()
        .filter(|record| record.stage.is_staffed())
        .collect();
    let executed = staffed.len().max(1) as f64;
    let failed = staffed.iter().filter(|record| !record.success).count() as f64;

    // The verifier's *score*, not its self-confidence. `Ω` is supposed to track whether
    // the mesh is producing good work, and the producer's opinion of its own output is
    // exactly the thing that has to be checked rather than believed.
    let verify_quality = stages
        .iter()
        .find(|record| record.stage == PipelineStage::Verify)
        .map(|record| record.quality)
        .unwrap_or(0.0);

    let delta = OmegaDelta {
        learning: (learning_total / cohort).clamp(0.0, 1.0),
        emergence: intelligence.emergence_ratio(),
        collaboration: settlement.collaboration_efficiency,
        // Every failed station counts, whoever caused it. Blame decides what an *agent*
        // learns; `F` measures whether the *mesh* delivered, and an outage means it did not.
        failure: (failed / executed).clamp(0.0, 1.0),
        // Drift is half "we are not sure this was right" and half "our expertise is aging".
        drift: (0.5 * (1.0 - verify_quality) + 0.5 * (staleness_total / cohort)).clamp(0.0, 1.0),
    };

    mesh.apply_omega(delta, intelligence);

    stages.push(StageRecord::bookkeeping(
        PipelineStage::OmegaUpdate,
        verify_quality,
        format!(
            "omega {:.4} -> {:.4} | L={:.3} E={:.3} C={:.3} F={:.3} D={:.3}",
            omega_before,
            mesh.global.omega,
            delta.learning,
            delta.emergence,
            delta.collaboration,
            delta.failure,
            delta.drift
        ),
    ));

    // --- Decision -----------------------------------------------------------
    let decision = decide(mesh, task, &priority, verified, verify_quality, config);

    // A run with nothing staffed has no external outcome to source, so it defaults to a
    // rehearsal. It is emphatically not an attestation of real work.
    let provenance = provenance.unwrap_or(OutcomeProvenance::Simulated);

    let mut trace = PipelineTrace {
        task_id: task.id.clone(),
        domain: task.domain.clone(),
        execution_mode: task.execution_mode,
        priority,
        assignments,
        stages,
        omega_before,
        omega_after: mesh.global.omega,
        delta,
        intelligence,
        decision,
        verified,
        provenance,
        digest: String::new(),
    };
    trace.digest = canonical_digest(&TraceCommitment {
        task_id: &trace.task_id,
        stages: &trace.stages,
        omega_after: trace.omega_after,
        verified: trace.verified,
        provenance: trace.provenance,
        mesh_digest: mesh.digest(),
    })
    .unwrap_or_default();

    trace
}

/// What the trace digest commits to: the work, the outcome, and the mesh that produced it.
///
/// `provenance` is inside the commitment rather than beside it. A digest over a
/// rehearsal and a digest over production work must be different values, or the first
/// could be presented as the second.
#[derive(serde::Serialize)]
struct TraceCommitment<'a> {
    task_id: &'a str,
    stages: &'a [StageRecord],
    omega_after: f64,
    verified: bool,
    provenance: OutcomeProvenance,
    mesh_digest: String,
}

/// A prerequisite failure does not stop the pipeline, it poisons it — downstream
/// agents are working from a bad plan or broken code.
fn carry_for(upstream_ok: bool) -> f64 {
    if upstream_ok {
        1.0
    } else {
        0.55
    }
}

/// Turn the model's proposal plus the run's outcome into a control action.
///
/// The decision head proposes; the mesh's guardrails dispose. A model is not allowed
/// to claim `stabilize` on work that failed verification, and high-priority unverified
/// work always goes back to a human — that override is the whole reason the mesh can be
/// trusted with a `Production` execution mode.
fn decide(
    mesh: &NeuralMesh,
    task: &TaskSpec,
    priority: &PriorityScore,
    verified: bool,
    verify_quality: f64,
    config: &PipelineConfig,
) -> NeuralSignal {
    let proposed = mesh
        .decide(task, priority)
        .unwrap_or_else(|_| Decision::from_distribution(&[1.0, 0.0, 0.0, 0.0]));

    // A degraded mesh escalates everything it could not verify, regardless of priority.
    // Without this, a mesh that has failed its way to Ω ≈ 0 keeps quietly returning
    // `observe` on low-priority work and never asks for help — the failure mode `Ω` is
    // there to detect would go unreported by the very system detecting it.
    let degraded = mesh.global.regime() == OmegaRegime::Degraded;

    let action = if !verified && (degraded || priority.value >= config.escalation_priority) {
        MeshAction::Escalate
    } else if verified && verify_quality >= config.stabilize_threshold {
        MeshAction::Stabilize
    } else if !verified {
        // Unverified but low stakes: keep it in the mesh rather than waking anyone.
        match proposed.action {
            MeshAction::Stabilize => MeshAction::Observe,
            other => other,
        }
    } else {
        proposed.action
    };

    let confidence = if verified {
        (0.5 * verify_quality + 0.5 * proposed.confidence).clamp(0.0, 1.0)
    } else {
        (0.5 * verify_quality * 0.5 + 0.5 * proposed.confidence).clamp(0.0, 1.0)
    };

    let overridden = action != proposed.action;
    NeuralSignal {
        agent: task.id.clone(),
        intent: task.description.clone(),
        confidence,
        action: action.as_str().to_string(),
        rationale: format!(
            "{} | omega={:.3} A={:.3} P={:.3} ({}) verified={} | head proposed {}{}",
            if verified {
                "verified"
            } else if degraded {
                "unverified, mesh degraded"
            } else {
                "unverified"
            },
            mesh.global.omega,
            mesh.intelligence().total,
            priority.value,
            priority.band.as_str(),
            verified,
            proposed.action.as_str(),
            if overridden {
                ", overridden by mesh policy"
            } else {
                ""
            }
        ),
    }
}

/// Run several tasks in sequence, returning every trace.
///
/// This is how the mesh is actually meant to be used: the interesting behaviour —
/// hubs forming, weak agents going quiet, `Ω` compounding — only appears over many runs.
pub fn run_batch(mesh: &mut NeuralMesh, tasks: &[TaskSpec]) -> Vec<PipelineTrace> {
    tasks.iter().map(|task| run(mesh, task)).collect()
}

/// Convenience: which execution mode a trace ran under.
pub fn trace_mode(trace: &PipelineTrace) -> ExecutionMode {
    trace.execution_mode
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{
        AgentNode, AgentRole, CapabilityVector, ErrorClass, OmegaRegime, PriorityBand,
        StageOutcomeReport,
    };

    use crate::outcomes::ReportedOutcomes;

    fn task(id: &str) -> TaskSpec {
        TaskSpec::new(id, "wire the mesh router", "rust")
            .with_features(vec![0.4, 0.6, 0.3, 0.7])
            .parsed_from_features()
    }

    #[test]
    fn a_run_visits_every_pipeline_stage_in_order() {
        let mut mesh = NeuralMesh::default();
        let trace = run(&mut mesh, &task("task-1"));

        let visited: Vec<PipelineStage> = trace.stages.iter().map(|record| record.stage).collect();
        assert_eq!(visited, PipelineStage::ALL.to_vec());
        assert_eq!(trace.assignments.len(), 5);
        assert!(!trace.digest.is_empty());
    }

    #[test]
    fn every_staffed_stage_gets_a_distinct_agent() {
        let mut mesh = NeuralMesh::default();
        let trace = run(&mut mesh, &task("task-2"));

        let mut agents: Vec<&str> = trace
            .assignments
            .iter()
            .map(|a| a.agent_id.as_str())
            .collect();
        agents.sort_unstable();
        let before = agents.len();
        agents.dedup();
        assert_eq!(agents.len(), before, "one agent must not hold two stations");
    }

    #[test]
    fn runs_are_deterministic_and_replayable() {
        let mut left = NeuralMesh::default();
        let mut right = NeuralMesh::default();
        let left_trace = run(&mut left, &task("task-3"));
        let right_trace = run(&mut right, &task("task-3"));

        assert_eq!(left_trace.digest, right_trace.digest);
        assert_eq!(left.global.omega, right.global.omega);
        assert_eq!(left_trace.verified, right_trace.verified);
    }

    #[test]
    fn different_tasks_take_different_paths() {
        let mut mesh = NeuralMesh::default();
        let first = run(&mut mesh, &task("task-a"));
        let second = run(&mut mesh, &task("task-b"));
        assert_ne!(first.digest, second.digest);
    }

    #[test]
    fn the_same_task_run_twice_is_not_the_same_event() {
        let mut mesh = NeuralMesh::default();
        let first = run(&mut mesh, &task("task-repeat"));
        let second = run(&mut mesh, &task("task-repeat"));
        assert_ne!(
            first.digest, second.digest,
            "the mesh has changed, so the second run is a different event"
        );
        assert_eq!(mesh.global.epoch, 2);
    }

    #[test]
    fn a_run_moves_trust_memory_and_omega() {
        let mut mesh = NeuralMesh::default();
        let trust_before = mesh.trust.mean_trust();
        let memory_before = mesh.mean_memory();
        let omega_before = mesh.global.omega;

        let trace = run(&mut mesh, &task("task-4"));

        assert_ne!(mesh.trust.mean_trust(), trust_before, "trust must move");
        assert_ne!(mesh.mean_memory(), memory_before, "memory must move");
        assert_ne!(mesh.global.omega, omega_before, "omega must move");
        assert_eq!(trace.omega_before, omega_before);
        assert_eq!(trace.omega_after, mesh.global.omega);
        assert!(!mesh.memory.records.is_empty());
    }

    #[test]
    fn agents_accumulate_telemetry_for_the_stages_they_ran() {
        let mut mesh = NeuralMesh::default();
        run(&mut mesh, &task("task-5"));

        for node in &mesh.nodes {
            assert_eq!(
                node.telemetry.attempts, 1,
                "{} should have one attempt",
                node.id
            );
            assert_eq!(node.telemetry.collaborations, 1);
            assert!(node.telemetry.last_action.is_some());
        }
    }

    #[test]
    fn omega_terms_are_all_measured_and_bounded() {
        let mut mesh = NeuralMesh::default();
        let trace = run(&mut mesh, &task("task-6"));
        let delta = trace.delta;

        for (name, value) in [
            ("L", delta.learning),
            ("E", delta.emergence),
            ("C", delta.collaboration),
            ("F", delta.failure),
            ("D", delta.drift),
        ] {
            assert!((0.0..=1.0).contains(&value), "{name} out of range: {value}");
        }
        assert!(
            delta.learning > 0.0,
            "a run that wrote memories must show learning"
        );
    }

    #[test]
    fn a_capable_mesh_compounds_omega_over_many_runs() {
        let mut mesh = NeuralMesh::with_cohort(strong_cohort());
        let start = mesh.global.omega;
        for index in 0..40 {
            let mut task = task(&format!("task-{index}"));
            task.uncertainty = 0.2;
            task.implementation_cost = 0.2;
            run(&mut mesh, &task);
        }
        assert!(
            mesh.global.omega > start,
            "a competent mesh doing easy work should get smarter, got {} from {}",
            mesh.global.omega,
            start
        );
        assert!(mesh.trust.mean_trust() > 0.5);
    }

    #[test]
    fn a_weak_mesh_erodes_and_stops_claiming_success() {
        let mut mesh = NeuralMesh::with_cohort(weak_cohort());
        for index in 0..30 {
            let mut task = task(&format!("hard-{index}"));
            task.uncertainty = 0.95;
            task.implementation_cost = 0.95;
            run(&mut mesh, &task);
        }
        assert!(
            mesh.global.omega < 1.0,
            "failure must cost the mesh, got {}",
            mesh.global.omega
        );
        assert!(
            mesh.mean_confidence() < 0.5,
            "confidence must decay with failure"
        );
    }

    #[test]
    fn the_mesh_learns_who_to_route_to() {
        // One deliberately incompetent critic among an otherwise capable cohort.
        let cohort: Vec<AgentNode> = crate::mesh::default_cohort()
            .into_iter()
            .map(|node| {
                if node.role == AgentRole::Critic {
                    node.with_capability(CapabilityVector::uniform(0.05))
                        .with_expertise("rust", 0.02)
                } else {
                    node.with_capability(CapabilityVector::uniform(0.9))
                        .with_expertise("rust", 0.9)
                }
            })
            .collect();

        let mut mesh = NeuralMesh::with_cohort(cohort);
        for index in 0..25 {
            run(&mut mesh, &task(&format!("route-{index}")));
        }

        assert!(
            mesh.trust.inbound_trust("critic-01") < mesh.trust.inbound_trust("coder-01"),
            "the mesh must learn that the critic cannot be relied on: critic={:.3} coder={:.3}",
            mesh.trust.inbound_trust("critic-01"),
            mesh.trust.inbound_trust("coder-01")
        );
        assert!(
            mesh.trust.inbound_weight("critic-01") < mesh.trust.inbound_weight("coder-01"),
            "and must thin the connection accordingly"
        );
    }

    #[test]
    fn unverified_high_priority_work_is_escalated_to_a_human() {
        let mut mesh = NeuralMesh::with_cohort(weak_cohort());
        let mut hard = task("crisis");
        hard.uncertainty = 1.0;
        hard.budget = 1.0;
        hard.implementation_cost = 1.0;
        hard.urgency = 1.0;

        let trace = run(&mut mesh, &hard);
        assert_eq!(trace.priority.band, PriorityBand::Immediate);
        assert!(!trace.verified);
        assert_eq!(trace.decision.action, "escalate");
        assert!(trace.decision.rationale.contains("unverified"));
    }

    #[test]
    fn a_degraded_mesh_escalates_even_trivial_work_it_cannot_verify() {
        let mut mesh = NeuralMesh::with_cohort(weak_cohort());
        for index in 0..20 {
            let mut grind = task(&format!("grind-{index}"));
            grind.uncertainty = 0.9;
            grind.implementation_cost = 0.9;
            run(&mut mesh, &grind);
        }
        assert_eq!(mesh.global.regime(), OmegaRegime::Degraded);

        let mut trivial = task("trivial");
        trivial.uncertainty = 0.05;
        trivial.budget = 0.05;
        trivial.implementation_cost = 0.05;
        trivial.urgency = 0.0;

        let trace = run(&mut mesh, &trivial);
        assert_eq!(
            trace.priority.band,
            PriorityBand::Deferred,
            "this is low-priority work"
        );
        assert!(!trace.verified);
        assert_eq!(
            trace.decision.action, "escalate",
            "a collapsed mesh must ask for help even about small things"
        );
        assert!(trace.decision.rationale.contains("degraded"));
    }

    #[test]
    fn verified_work_is_never_escalated() {
        let mut mesh = NeuralMesh::with_cohort(strong_cohort());
        let mut verified_runs = 0;

        for index in 0..20 {
            let mut easy = task(&format!("routine-{index}"));
            easy.uncertainty = 0.1;
            easy.implementation_cost = 0.1;
            easy.budget = 0.1;

            let trace = run(&mut mesh, &easy);
            if trace.verified {
                verified_runs += 1;
                assert_ne!(
                    trace.decision.action, "escalate",
                    "verified work must not be handed back to a human"
                );
            }
        }

        assert!(
            verified_runs > 0,
            "a strong cohort on easy work should verify sometimes"
        );
    }

    #[test]
    fn a_prerequisite_failure_poisons_rather_than_stops_the_pipeline() {
        assert_eq!(carry_for(true), 1.0);
        assert!(
            carry_for(false) < 1.0,
            "downstream agents work from a bad plan, not from nothing"
        );
    }

    #[test]
    fn sandbox_runs_teach_the_mesh_less_than_production_runs() {
        let mut sandbox = NeuralMesh::default();
        let mut production = NeuralMesh::default();

        for index in 0..10 {
            let mut sandbox_task = task(&format!("t-{index}"));
            sandbox_task.execution_mode = ExecutionMode::Sandbox;
            run(&mut sandbox, &sandbox_task);

            let mut production_task = task(&format!("t-{index}"));
            production_task.execution_mode = ExecutionMode::Production;
            run(&mut production, &production_task);
        }

        assert!(
            (sandbox.trust.mean_trust() - 0.5).abs() < (production.trust.mean_trust() - 0.5).abs(),
            "sandbox trust should move less than production trust"
        );
    }

    #[test]
    fn an_empty_mesh_records_the_gap_instead_of_panicking() {
        let mut mesh = NeuralMesh::with_cohort(Vec::new());
        let trace = run(&mut mesh, &task("orphan"));

        assert!(trace.assignments.is_empty());
        assert!(!trace.verified);
        let staffed_failures = trace
            .stages
            .iter()
            .filter(|record| record.stage.is_staffed() && !record.success)
            .count();
        assert_eq!(staffed_failures, 5);
        assert!(trace
            .stages
            .iter()
            .any(|record| record.note.contains("no agent available")));
    }

    #[test]
    fn a_single_agent_mesh_doubles_up_rather_than_stalling() {
        let mut mesh =
            NeuralMesh::with_cohort(vec![AgentNode::new("solo", "solo", AgentRole::Executor)]);
        let trace = run(&mut mesh, &task("solo-run"));
        assert_eq!(trace.assignments.len(), 5);
        assert!(trace.assignments.iter().all(|a| a.agent_id == "solo"));
        assert_eq!(mesh.nodes[0].telemetry.attempts, 5);
    }

    #[test]
    fn batch_runs_return_one_trace_per_task() {
        let mut mesh = NeuralMesh::default();
        let tasks: Vec<TaskSpec> = (0..4)
            .map(|index| task(&format!("batch-{index}")))
            .collect();
        let traces = run_batch(&mut mesh, &tasks);

        assert_eq!(traces.len(), 4);
        assert_eq!(mesh.global.epoch, 4);
        assert_eq!(mesh.ledger.len(), 4);
        assert_eq!(trace_mode(&traces[0]), ExecutionMode::Controlled);
    }

    #[test]
    fn traces_report_cost_and_failures() {
        let mut mesh = NeuralMesh::default();
        let trace = run(&mut mesh, &task("cost"));
        assert!(trace.total_cost() > 0.0, "staffed stages consume resources");
        assert_eq!(
            trace.failures().len(),
            trace.stages.iter().filter(|record| !record.success).count()
        );
    }

    // -----------------------------------------------------------------------
    // Externally supplied outcomes
    // -----------------------------------------------------------------------

    /// Plan a run the way the adapter does, so reports can be bound to real assignments.
    fn plan_for(mesh: &NeuralMesh, task: &TaskSpec) -> Vec<Assignment> {
        let band = mesh.priority_for(task).band;
        let mut taken: Vec<String> = Vec::new();
        let mut planned = Vec::new();
        for stage in STAFFED {
            if let Some(assignment) = router::select(mesh, stage, task, band, &taken) {
                taken.push(assignment.agent_id.clone());
                planned.push(assignment);
            }
        }
        planned
    }

    fn all_reported(planned: &[Assignment], success: bool, quality: f64) -> ReportedOutcomes {
        planned
            .iter()
            .map(|assignment| {
                if success {
                    StageOutcomeReport::success(assignment.stage, &assignment.agent_id, quality)
                        .with_latency_ms(1_500.0)
                        .with_cost(0.08)
                } else {
                    StageOutcomeReport::failure(
                        assignment.stage,
                        &assignment.agent_id,
                        ErrorClass::Quality,
                    )
                    .with_latency_ms(1_500.0)
                    .with_cost(0.08)
                }
            })
            .collect()
    }

    #[test]
    fn a_reported_run_is_marked_as_real_work() {
        let mut mesh = NeuralMesh::default();
        let task = task("reported-1");
        let planned = plan_for(&mesh, &task);
        let source = all_reported(&planned, true, 0.88);
        let calibration = RuntimeCalibration::default();

        let trace = run_bound(
            &mut mesh,
            &task,
            &RunInputs::new(&source)
                .with_calibration(calibration)
                .with_assignments(&planned),
        );

        assert_eq!(trace.provenance, OutcomeProvenance::Reported);
        assert!(trace.provenance.is_real());
        assert!(trace.verified);
        assert!((trace.mean_quality() - 0.88).abs() < 1e-9);
        assert!(
            (trace.total_latency_ms() - 7_500.0).abs() < 1e-9,
            "five stages at 1.5s each"
        );
    }

    #[test]
    fn a_rehearsal_and_a_real_run_never_share_a_digest() {
        let task = task("provenance");

        let mut simulated_mesh = NeuralMesh::default();
        let planned = plan_for(&simulated_mesh, &task);
        let simulated = run(&mut simulated_mesh, &task);

        // Report exactly what the simulator produced, stage for stage.
        let reports: Vec<StageOutcomeReport> = simulated
            .stages
            .iter()
            .filter(|record| record.stage.is_staffed())
            .filter_map(|record| {
                let agent = record.agent.clone()?;
                let mut report = if record.success {
                    StageOutcomeReport::success(record.stage, agent, record.quality)
                } else {
                    StageOutcomeReport::failure(record.stage, agent, record.error)
                };
                report.confidence = record.confidence;
                report.quality = record.quality;
                Some(
                    report
                        .with_latency_ms(record.latency_ms)
                        .with_cost(record.cost),
                )
            })
            .collect();

        let mut reported_mesh = NeuralMesh::default();
        let source = ReportedOutcomes::new(reports);
        let reported = run_bound(
            &mut reported_mesh,
            &task,
            &RunInputs::new(&source).with_assignments(&planned),
        );

        assert_eq!(
            reported.verified, simulated.verified,
            "identical outcomes, identical verdict"
        );
        assert_ne!(
            reported.digest, simulated.digest,
            "provenance is inside the commitment, so a rehearsal cannot be passed off as real work"
        );
    }

    #[test]
    fn reported_outcomes_teach_the_mesh_what_its_agents_actually_cost() {
        let mut mesh = NeuralMesh::default();
        let task = task("costed");
        let planned = plan_for(&mesh, &task);

        let source: ReportedOutcomes = planned
            .iter()
            .map(|assignment| {
                StageOutcomeReport::success(assignment.stage, &assignment.agent_id, 0.9)
                    .with_latency_ms(12_000.0)
                    .with_cost(0.4)
            })
            .collect();

        run_bound(
            &mut mesh,
            &task,
            &RunInputs::new(&source).with_assignments(&planned),
        );

        for assignment in &planned {
            let node = mesh.node(&assignment.agent_id).unwrap();
            assert!(
                node.resources.is_observed(),
                "{} learned nothing",
                assignment.agent_id
            );
            assert!((node.resources.observed_latency_ms - 12_000.0).abs() < 1e-6);
            assert!(
                (node.resources.latency - 0.2).abs() < 1e-6,
                "12s against a 60s ceiling"
            );
        }
    }

    #[test]
    fn an_outage_costs_the_mesh_omega_without_costing_the_agent_its_trust() {
        let task = task("outage");

        let mut blamed = NeuralMesh::default();
        let blamed_plan = plan_for(&blamed, &task);
        let blamed_source: ReportedOutcomes = blamed_plan
            .iter()
            .map(|assignment| {
                StageOutcomeReport::failure(
                    assignment.stage,
                    &assignment.agent_id,
                    ErrorClass::Quality,
                )
            })
            .collect();
        let blamed_trace = run_bound(
            &mut blamed,
            &task,
            &RunInputs::new(&blamed_source).with_assignments(&blamed_plan),
        );

        let mut unlucky = NeuralMesh::default();
        let unlucky_plan = plan_for(&unlucky, &task);
        let unlucky_source: ReportedOutcomes = unlucky_plan
            .iter()
            .map(|assignment| {
                StageOutcomeReport::failure(
                    assignment.stage,
                    &assignment.agent_id,
                    ErrorClass::Infrastructure,
                )
            })
            .collect();
        let unlucky_trace = run_bound(
            &mut unlucky,
            &task,
            &RunInputs::new(&unlucky_source).with_assignments(&unlucky_plan),
        );

        assert!(
            (blamed_trace.delta.failure - unlucky_trace.delta.failure).abs() < 1e-9,
            "the mesh failed to deliver either way, so F is the same"
        );
        assert!(
            unlucky.trust.mean_trust() > blamed.trust.mean_trust(),
            "an outage is not a betrayal: unlucky={:.4} blamed={:.4}",
            unlucky.trust.mean_trust(),
            blamed.trust.mean_trust()
        );
        assert!(
            unlucky.mean_confidence() > blamed.mean_confidence(),
            "and nothing about the agents' reasoning was tested"
        );
    }

    #[test]
    fn a_stage_nobody_reported_on_is_not_a_silent_pass() {
        let mut mesh = NeuralMesh::default();
        let task = task("silence");
        let planned = plan_for(&mesh, &task);

        // Everything reported except the verifier.
        let source: ReportedOutcomes = planned
            .iter()
            .filter(|assignment| assignment.stage != PipelineStage::Verify)
            .map(|assignment| {
                StageOutcomeReport::success(assignment.stage, &assignment.agent_id, 0.95)
            })
            .collect();

        let trace = run_bound(
            &mut mesh,
            &task,
            &RunInputs::new(&source).with_assignments(&planned),
        );

        assert!(!trace.verified, "silence must not read as success");
        assert!(trace.error_classes().contains(&ErrorClass::Unknown));
    }

    #[test]
    fn a_partial_recording_can_be_replayed_but_is_marked_mixed() {
        let mut mesh = NeuralMesh::default();
        let task = task("partial");
        let planned = plan_for(&mesh, &task);

        let source: ReportedOutcomes = planned
            .iter()
            .filter(|assignment| assignment.stage == PipelineStage::Code)
            .map(|assignment| {
                StageOutcomeReport::success(assignment.stage, &assignment.agent_id, 0.9)
            })
            .collect::<ReportedOutcomes>()
            .simulating_gaps();

        let trace = run_bound(
            &mut mesh,
            &task,
            &RunInputs::new(&source).with_assignments(&planned),
        );

        assert_eq!(trace.provenance, OutcomeProvenance::Mixed);
        assert!(
            !trace.provenance.is_real(),
            "a trace with invented stages must never read as evidence"
        );
    }

    #[test]
    fn bound_assignments_survive_a_mesh_that_moved_underneath_them() {
        let mut mesh = NeuralMesh::default();
        let bound = task("bound");
        let planned = plan_for(&mesh, &bound);

        // Something else runs in between, moving trust and capability.
        for index in 0..6 {
            run(&mut mesh, &task(&format!("interleaved-{index}")));
        }

        let source = all_reported(&planned, true, 0.8);
        let trace = run_bound(
            &mut mesh,
            &bound,
            &RunInputs::new(&source).with_assignments(&planned),
        );

        let used: Vec<&str> = trace
            .assignments
            .iter()
            .map(|a| a.agent_id.as_str())
            .collect();
        let expected: Vec<&str> = planned.iter().map(|a| a.agent_id.as_str()).collect();
        assert_eq!(
            used, expected,
            "the caller already dispatched to these agents"
        );
        assert_eq!(trace.provenance, OutcomeProvenance::Reported);
    }

    fn strong_cohort() -> Vec<AgentNode> {
        crate::mesh::default_cohort()
            .into_iter()
            .map(|node| {
                let domain = node.role.as_str().to_string();
                node.with_capability(CapabilityVector::uniform(0.95))
                    .with_confidence(0.85)
                    .with_expertise("rust", 0.95)
                    .with_expertise(domain, 0.95)
            })
            .collect()
    }

    fn weak_cohort() -> Vec<AgentNode> {
        crate::mesh::default_cohort()
            .into_iter()
            .map(|node| {
                node.with_capability(CapabilityVector::uniform(0.12))
                    .with_confidence(0.2)
                    .with_expertise("rust", 0.05)
            })
            .collect()
    }
}
