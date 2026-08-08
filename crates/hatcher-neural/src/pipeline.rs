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
//! ## Determinism
//!
//! Stage outcomes are drawn from a hash of `(task id, agent id, stage)`, never from a
//! clock or an RNG. The same task against the same mesh state always replays exactly,
//! which is what makes rehearsal in the playground meaningful and what lets a trace
//! digest be an attestation rather than a souvenir.

use hatcher_core::{
    canonical_digest, Assignment, ExecutionMode, MeshAction, NeuralSignal, OmegaDelta, OmegaRegime,
    PipelineStage, PipelineTrace, PriorityScore, StageRecord, TaskSpec,
};

use crate::equations::{deterministic_unit, seed_of};
use crate::inference::Decision;
use crate::learning::{apply_outcome, StageOutcome};
use crate::mesh::NeuralMesh;
use crate::router;

/// Thresholds and rates that govern one pipeline pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineConfig {
    /// Per-run memory decay rate applied before new knowledge is written.
    pub memory_decay_rate: f64,
    /// Confidence below which the mesh will not claim a result is stabilized.
    pub stabilize_threshold: f64,
    /// Priority above which an unverified result must be escalated to a human.
    pub escalation_priority: f64,
    /// How strongly task difficulty suppresses an agent's success chance. Applied as an
    /// exponent on fitness, so difficulty punishes weak agents far harder than strong
    /// ones — see [`execute_stage`].
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

/// Run a task through the mesh with the default configuration.
pub fn run(mesh: &mut NeuralMesh, task: &TaskSpec) -> PipelineTrace {
    run_with(mesh, task, &PipelineConfig::default())
}

/// Run a task through the mesh, mutating trust, memory, capability, and `Ω`.
pub fn run_with(mesh: &mut NeuralMesh, task: &TaskSpec, config: &PipelineConfig) -> PipelineTrace {
    let coefficients = mesh.coefficients;
    let gate = task.execution_mode.plasticity_gate();
    let omega_before = mesh.global.omega;
    mesh.sequence += 1;

    let mut stages: Vec<StageRecord> = Vec::with_capacity(PipelineStage::ALL.len());
    let mut assignments: Vec<Assignment> = Vec::new();
    let mut outcomes: Vec<StageOutcome> = Vec::new();

    // --- 1. Task Parser -----------------------------------------------------
    stages.push(StageRecord {
        stage: PipelineStage::Parse,
        agent: None,
        success: true,
        confidence: 1.0,
        cost: 0.0,
        note: format!(
            "domain={} U={:.2} B={:.2} I={:.2} urgency={:.2} mode={}",
            task.domain,
            task.uncertainty,
            task.budget,
            task.implementation_cost,
            task.urgency,
            task.execution_mode.as_str()
        ),
    });

    // --- 2. Priority Engine -------------------------------------------------
    let priority = mesh.priority_for(task);
    stages.push(StageRecord {
        stage: PipelineStage::Prioritize,
        agent: None,
        success: true,
        confidence: priority.confidence,
        cost: 0.0,
        note: format!(
            "P={:.3} band={} C={:.2} tau={:.2}",
            priority.value,
            priority.band.as_str(),
            priority.confidence,
            priority.tau
        ),
    });

    // --- 3-7. Staffed stations ---------------------------------------------
    let mut taken: Vec<String> = Vec::new();
    let mut plan_ok = true;
    let mut research_ok = true;
    let mut code_ok = true;
    let mut critique_ok = true;
    let mut verify_ok = true;

    for stage in STAFFED {
        let carry = match stage {
            PipelineStage::Plan => 1.0,
            PipelineStage::Research => carry_for(plan_ok),
            PipelineStage::Code => carry_for(plan_ok && research_ok),
            PipelineStage::Critique => carry_for(code_ok),
            PipelineStage::Verify => carry_for(code_ok && critique_ok),
            _ => 1.0,
        };

        let Some(assignment) = router::select(mesh, stage, task, priority.band, &taken) else {
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
            stages.push(StageRecord {
                stage,
                agent: None,
                success: false,
                confidence: 0.0,
                cost: 0.0,
                note: "no agent available for this stage".to_string(),
            });
            continue;
        };
        taken.push(assignment.agent_id.clone());

        let attempt = execute_stage(mesh, &assignment, task, carry, config);

        match stage {
            PipelineStage::Plan => plan_ok = attempt.success,
            PipelineStage::Research => research_ok = attempt.success,
            PipelineStage::Code => code_ok = attempt.success,
            PipelineStage::Critique => critique_ok = attempt.success,
            PipelineStage::Verify => verify_ok = attempt.success,
            _ => {}
        }

        stages.push(StageRecord {
            stage,
            agent: Some(assignment.agent_id.clone()),
            success: attempt.success,
            confidence: attempt.confidence,
            cost: attempt.cost,
            note: attempt.note.clone(),
        });

        outcomes.push(StageOutcome {
            agent_id: assignment.agent_id.clone(),
            stage,
            domain: task.domain.clone(),
            success: attempt.success,
            confidence: attempt.confidence,
            // Diminishing returns: an agent already fluent in this domain learns
            // little from doing it again, which is what keeps L from growing forever
            // on repetitive work.
            novelty: (task.uncertainty * (1.0 - assignment.mastery.clamp(0.0, 1.0))).clamp(0.0, 1.0),
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
            staleness_total += apply_outcome(node, outcome, *knowledge, *decay, &coefficients);
            learning_total += *knowledge;
        }
    }
    let cohort = mesh.nodes.len().max(1) as f64;

    stages.push(StageRecord {
        stage: PipelineStage::MemoryUpdate,
        agent: None,
        success: true,
        confidence: mesh.mean_memory(),
        cost: 0.0,
        note: format!(
            "K={:.3} decayed={:.3} records={} obsolescence={:.3}",
            learning_total,
            decayed.values().sum::<f64>(),
            mesh.memory.records.len(),
            staleness_total
        ),
    });

    // --- 9. Trust Graph Update ---------------------------------------------
    // Credit flows along the handoff chain: each agent judges the one it handed to.
    let mut handoffs = 0usize;
    for window in assignments.windows(2) {
        let (from, to) = (&window[0], &window[1]);
        let downstream_ok = stages
            .iter()
            .find(|record| record.stage == to.stage)
            .map(|record| record.success)
            .unwrap_or(false);
        if mesh.trust.record(&from.agent_id, &to.agent_id, downstream_ok) {
            handoffs += 1;
        }
    }
    // The verifier's judgment also feeds back to the planner: whoever planned this owns
    // the outcome, which is how a planner that keeps producing unworkable plans loses
    // trust even though it never wrote a line.
    if let (Some(first), Some(last)) = (assignments.first(), assignments.last()) {
        if first.agent_id != last.agent_id && mesh.trust.record(&last.agent_id, &first.agent_id, verified) {
            handoffs += 1;
        }
    }

    let settlement = mesh.trust.settle(&coefficients, gate);
    for node in &mut mesh.nodes {
        if taken.contains(&node.id) {
            node.telemetry.collaborations += 1;
        }
    }

    stages.push(StageRecord {
        stage: PipelineStage::TrustUpdate,
        agent: None,
        success: true,
        confidence: settlement.collaboration_efficiency,
        cost: 0.0,
        note: format!(
            "handoffs={} pairs_moved={} C={:.3} gate={:.2}",
            handoffs, settlement.updated_pairs, settlement.collaboration_efficiency, gate
        ),
    });

    // --- 10. Ω Update -------------------------------------------------------
    let intelligence = mesh.intelligence();
    let staffed: Vec<&StageRecord> = stages.iter().filter(|record| record.stage.is_staffed()).collect();
    let executed = staffed.len().max(1) as f64;
    let failed = staffed.iter().filter(|record| !record.success).count() as f64;

    let verify_confidence = stages
        .iter()
        .find(|record| record.stage == PipelineStage::Verify)
        .map(|record| record.confidence)
        .unwrap_or(0.0);

    let delta = OmegaDelta {
        learning: (learning_total / cohort).clamp(0.0, 1.0),
        emergence: intelligence.emergence_ratio(),
        collaboration: settlement.collaboration_efficiency,
        failure: (failed / executed).clamp(0.0, 1.0),
        // Drift is half "we are not sure this was right" and half "our expertise is aging".
        drift: (0.5 * (1.0 - verify_confidence) + 0.5 * (staleness_total / cohort)).clamp(0.0, 1.0),
    };

    mesh.apply_omega(delta, intelligence);

    stages.push(StageRecord {
        stage: PipelineStage::OmegaUpdate,
        agent: None,
        success: true,
        confidence: verify_confidence,
        cost: 0.0,
        note: format!(
            "omega {:.4} -> {:.4} | L={:.3} E={:.3} C={:.3} F={:.3} D={:.3}",
            omega_before,
            mesh.global.omega,
            delta.learning,
            delta.emergence,
            delta.collaboration,
            delta.failure,
            delta.drift
        ),
    });

    // --- Decision -----------------------------------------------------------
    let decision = decide(mesh, task, &priority, verified, verify_confidence, config);

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
        digest: String::new(),
    };
    trace.digest = canonical_digest(&TraceCommitment {
        task_id: &trace.task_id,
        stages: &trace.stages,
        omega_after: trace.omega_after,
        verified: trace.verified,
        mesh_digest: mesh.digest(),
    })
    .unwrap_or_default();

    trace
}

/// What the trace digest commits to: the work, the outcome, and the mesh that produced it.
#[derive(serde::Serialize)]
struct TraceCommitment<'a> {
    task_id: &'a str,
    stages: &'a [StageRecord],
    omega_after: f64,
    verified: bool,
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

/// What one agent did at one stage.
#[derive(Debug, Clone)]
struct Attempt {
    success: bool,
    confidence: f64,
    cost: f64,
    note: String,
}

/// Resolve whether an assigned agent succeeds.
///
/// Competence is
///
/// ```text
/// competence = A_i^(1/5) · (0.60 + 0.25·mastery + 0.15·inbound_trust)
/// ```
///
/// `A_i^(1/5)` is the geometric mean of the five capability factors — `A_i` put back on
/// the same scale as its parts — and it *multiplies* rather than adds. That matters: as
/// a bracketed sum, mastery and trust alone would floor an incapable agent at a ~45%
/// success rate, and since specialization grows with every attempt, any agent would
/// eventually pass no matter how weak its intelligence or context. Multiplying makes
/// capability a ceiling, which is the same reason `A_i` is a product in the first place:
/// weaknesses have to matter.
///
/// Difficulty then enters as an *exponent* rather than a multiplier:
///
/// ```text
/// threshold = fitness^(0.5 + difficulty_weight · difficulty) · carry
/// ```
///
/// Exponentiation is the right shape because difficulty compounds against weakness. A
/// linear penalty moves every agent by the same amount, so a hopeless agent and an
/// excellent one lose the same margin on a hard task. As an exponent, a fitness-0.9
/// agent barely notices a hard task (0.9² = 0.81) while a fitness-0.2 agent collapses
/// (0.2² = 0.04) — which is what "hard" actually means.
///
/// The `carry` factor then applies the upstream penalty, and the draw itself is a hash,
/// not a random number.
fn execute_stage(
    mesh: &NeuralMesh,
    assignment: &Assignment,
    task: &TaskSpec,
    carry: f64,
    config: &PipelineConfig,
) -> Attempt {
    let node_confidence = mesh
        .node(&assignment.agent_id)
        .map(|node| node.confidence)
        .unwrap_or(0.5);
    let cost = mesh
        .node(&assignment.agent_id)
        .map(|node| node.resources.cost())
        .unwrap_or(0.0);

    let geometric_competence = assignment.capability.max(0.0).powf(0.2);
    let modulation = 0.60 + 0.25 * assignment.mastery.clamp(0.0, 1.0) + 0.15 * assignment.inbound_trust.clamp(0.0, 1.0);
    let competence = (geometric_competence * modulation).clamp(0.0, 1.0);

    let difficulty = 0.5 * task.uncertainty + 0.5 * task.implementation_cost;
    let exponent = 0.5 + config.difficulty_weight * difficulty.clamp(0.0, 1.0);
    let threshold = (competence.powf(exponent) * carry).clamp(0.02, 0.98);

    let draw = deterministic_unit(seed_of(&format!(
        "{}:{}:{}:{}",
        task.id,
        assignment.agent_id,
        assignment.stage.as_str(),
        mesh.sequence
    )));
    let success = draw < threshold;

    // Reported confidence blends what the agent believes about itself with how
    // comfortably it cleared (or missed) the bar on this particular task.
    let margin = if success {
        (threshold - draw) / threshold.max(1e-6)
    } else {
        -((draw - threshold) / (1.0 - threshold).max(1e-6))
    };
    let confidence = (0.6 * node_confidence + 0.4 * (0.5 + 0.5 * margin)).clamp(0.0, 1.0);

    Attempt {
        success,
        confidence,
        cost,
        note: format!(
            "{} by {} | fitness={:.3} threshold={:.3} draw={:.3} carry={:.2}",
            assignment.stage.as_str(),
            assignment.agent_id,
            competence,
            threshold,
            draw,
            carry
        ),
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
    verify_confidence: f64,
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
    } else if verified && verify_confidence >= config.stabilize_threshold {
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
        (0.5 * verify_confidence + 0.5 * proposed.confidence).clamp(0.0, 1.0)
    } else {
        (0.5 * verify_confidence * 0.5 + 0.5 * proposed.confidence).clamp(0.0, 1.0)
    };

    let overridden = action != proposed.action;
    NeuralSignal {
        agent: task.id.clone(),
        intent: task.description.clone(),
        confidence,
        action: action.as_str().to_string(),
        rationale: format!(
            "{} | omega={:.3} A={:.3} P={:.3} ({}) verified={} | head proposed {}{}",
            if verified { "verified" } else if degraded { "unverified, mesh degraded" } else { "unverified" },
            mesh.global.omega,
            mesh.intelligence().total,
            priority.value,
            priority.band.as_str(),
            verified,
            proposed.action.as_str(),
            if overridden { ", overridden by mesh policy" } else { "" }
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
    use hatcher_core::{AgentNode, AgentRole, CapabilityVector, OmegaRegime, PriorityBand};

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

        let mut agents: Vec<&str> = trace.assignments.iter().map(|a| a.agent_id.as_str()).collect();
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
            assert_eq!(node.telemetry.attempts, 1, "{} should have one attempt", node.id);
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
        assert!(delta.learning > 0.0, "a run that wrote memories must show learning");
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
        assert!(mesh.global.omega < 1.0, "failure must cost the mesh, got {}", mesh.global.omega);
        assert!(mesh.mean_confidence() < 0.5, "confidence must decay with failure");
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
        assert_eq!(trace.priority.band, PriorityBand::Deferred, "this is low-priority work");
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

        assert!(verified_runs > 0, "a strong cohort on easy work should verify sometimes");
    }

    #[test]
    fn a_poisoned_plan_drags_the_rest_of_the_pipeline_down() {
        let config = PipelineConfig::default();
        let mesh = NeuralMesh::default();
        let assignment = router::select(
            &mesh,
            PipelineStage::Code,
            &task("carry"),
            PriorityBand::Standard,
            &[],
        )
        .unwrap();

        let clean = execute_stage(&mesh, &assignment, &task("carry"), 1.0, &config);
        let poisoned = execute_stage(&mesh, &assignment, &task("carry"), carry_for(false), &config);
        assert!(poisoned.note.contains("carry=0.55"));
        assert!(!poisoned.success || clean.success, "a poisoned upstream must never help");
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
        assert!(trace.stages.iter().any(|record| record.note.contains("no agent available")));
    }

    #[test]
    fn a_single_agent_mesh_doubles_up_rather_than_stalling() {
        let mut mesh = NeuralMesh::with_cohort(vec![AgentNode::new("solo", "solo", AgentRole::Executor)]);
        let trace = run(&mut mesh, &task("solo-run"));
        assert_eq!(trace.assignments.len(), 5);
        assert!(trace.assignments.iter().all(|a| a.agent_id == "solo"));
        assert_eq!(mesh.nodes[0].telemetry.attempts, 5);
    }

    #[test]
    fn batch_runs_return_one_trace_per_task() {
        let mut mesh = NeuralMesh::default();
        let tasks: Vec<TaskSpec> = (0..4).map(|index| task(&format!("batch-{index}"))).collect();
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
