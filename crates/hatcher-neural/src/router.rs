//! Assignment: which agent should execute a stage.
//!
//! This is where four of the equations meet. A candidate is scored on its capability
//! `A_i`, the trust the cohort has learned to place in it, its mastery of the task's
//! domain, and its resource ratio `R_i` — and the *weight* given to that last term
//! depends on the task's priority band, which is exactly the behaviour the priority
//! equation is supposed to produce: urgent work goes to the strongest agent, deferred
//! work goes to the cheapest one that can still do it.

use hatcher_core::{AgentNode, Assignment, PipelineStage, PriorityBand, RuntimeCalibration, TaskSpec};

use crate::mesh::NeuralMesh;

/// Floors that keep a single zeroed term from erasing an otherwise viable candidate.
/// Without them the product behaves like a veto rather than a preference.
const TRUST_FLOOR: f64 = 0.20;
const MASTERY_FLOOR: f64 = 0.20;

/// How much the cost term counts, per priority band.
///
/// * `Immediate` — cost is nearly ignored; get it right, not cheap.
/// * `Standard` — cost is a tiebreaker.
/// * `Deferred` — cost dominates; hand it to whoever is cheapest and adequate.
fn cost_exponent(band: PriorityBand) -> f64 {
    match band {
        PriorityBand::Immediate => 0.10,
        PriorityBand::Standard => 0.50,
        PriorityBand::Deferred => 1.20,
    }
}

/// Score one candidate for one stage.
///
/// The score is a product, so a candidate must be at least adequate on every axis;
/// the floors keep "adequate" from meaning "nonzero".
///
/// Capability enters as `A_i^(1/5)`, the geometric mean of the five capability
/// factors, not as `A_i` itself. `A_i` is a product of five sub-unit numbers, so raw
/// capability differences are quintic: a cohort-leading agent would outscore a
/// merely-good one by so much that trust, mastery, and cost could never matter. The
/// geometric mean puts capability back on the same scale as the other three terms,
/// which is also the scale [`crate::pipeline`] uses to resolve whether a stage
/// succeeds — so routing and execution agree about what "capable" means.
pub fn score_candidate(mesh: &NeuralMesh, node: &AgentNode, task: &TaskSpec, band: PriorityBand) -> f64 {
    let capability = node.influence().max(0.0).powf(0.2);
    let trust = TRUST_FLOOR + mesh.trust.inbound_trust(&node.id);
    let mastery = MASTERY_FLOOR + node.mastery(&task.domain);
    // Normalize the resource ratio into a comparable multiplier. Efficiency is
    // unbounded above (cheap, fast agents can score arbitrarily high), so squash it.
    let efficiency = node.resource_efficiency();
    let cost_term = (efficiency / (1.0 + efficiency)).max(1e-6).powf(cost_exponent(band));

    capability * trust * mastery * cost_term
}

/// What the mesh expects a stage to take and cost if this agent runs it.
///
/// Measured history when the agent has any; otherwise its declared prior projected onto
/// the calibration ceiling. The ceiling only matters for an unobserved agent — once real
/// reports land, the absolutes are used directly and the projection stops mattering.
pub fn expected_profile(node: &AgentNode, calibration: &RuntimeCalibration) -> (f64, f64) {
    (
        node.resources.expected_latency_ms(calibration.latency_ceiling_ms),
        node.resources.expected_cost(calibration.cost_ceiling),
    )
}

/// Whether an agent fits inside a task's declared limits.
pub fn admits(node: &AgentNode, task: &TaskSpec, calibration: &RuntimeCalibration) -> bool {
    if task.constraints.is_open() {
        return true;
    }
    let (latency_ms, cost) = expected_profile(node, calibration);
    task.constraints.admits(latency_ms, cost)
}

/// Rank every eligible candidate for a stage, best first.
///
/// Eligibility prefers agents holding the stage's role. If none exist, the whole
/// cohort is considered — a mesh missing a verifier should still verify, just worse.
/// Isolated agents are excluded unless they are all that is left, and the same applies
/// to a caller's latency and cost limits: they narrow the pool, but they never empty it.
/// Refusing to route is worse than routing over budget, and the caller is told which
/// happened either way — see [`crate::adapter::MeshAdapter::plan`].
pub fn rank(
    mesh: &NeuralMesh,
    stage: PipelineStage,
    task: &TaskSpec,
    band: PriorityBand,
    exclude: &[String],
) -> Vec<Assignment> {
    rank_with(mesh, stage, task, band, exclude, &RuntimeCalibration::default())
}

/// Rank candidates against an explicit calibration.
pub fn rank_with(
    mesh: &NeuralMesh,
    stage: PipelineStage,
    task: &TaskSpec,
    band: PriorityBand,
    exclude: &[String],
    calibration: &RuntimeCalibration,
) -> Vec<Assignment> {
    let isolated = mesh.trust.isolated();

    let by_role: Vec<&AgentNode> = match stage.preferred_role() {
        Some(role) => mesh.nodes.iter().filter(|node| node.role == role).collect(),
        None => mesh.nodes.iter().collect(),
    };
    let pool: Vec<&AgentNode> = if by_role.is_empty() {
        mesh.nodes.iter().collect()
    } else {
        by_role
    };

    let available: Vec<&AgentNode> = pool
        .iter()
        .copied()
        .filter(|node| !exclude.contains(&node.id))
        .collect();
    let available = if available.is_empty() { pool } else { available };

    let healthy: Vec<&AgentNode> = available
        .iter()
        .copied()
        .filter(|node| !isolated.contains(&node.id))
        .collect();
    let candidates = if healthy.is_empty() { available } else { healthy };

    let affordable: Vec<&AgentNode> = candidates
        .iter()
        .copied()
        .filter(|node| admits(node, task, calibration))
        .collect();
    let candidates = if affordable.is_empty() { candidates } else { affordable };

    let mut assignments: Vec<Assignment> = candidates
        .into_iter()
        .map(|node| Assignment {
            stage,
            agent_id: node.id.clone(),
            role: node.role,
            score: score_candidate(mesh, node, task, band),
            capability: node.influence(),
            inbound_trust: mesh.trust.inbound_trust(&node.id),
            efficiency: node.resource_efficiency(),
            mastery: node.mastery(&task.domain),
        })
        .collect();

    assignments.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Ties break on id so routing is reproducible.
            .then_with(|| a.agent_id.cmp(&b.agent_id))
    });
    assignments
}

/// Pick the best agent for a stage.
pub fn select(
    mesh: &NeuralMesh,
    stage: PipelineStage,
    task: &TaskSpec,
    band: PriorityBand,
    exclude: &[String],
) -> Option<Assignment> {
    rank(mesh, stage, task, band, exclude).into_iter().next()
}

/// Pick the best agent for a stage against an explicit calibration.
pub fn select_with(
    mesh: &NeuralMesh,
    stage: PipelineStage,
    task: &TaskSpec,
    band: PriorityBand,
    exclude: &[String],
    calibration: &RuntimeCalibration,
) -> Option<Assignment> {
    rank_with(mesh, stage, task, band, exclude, calibration)
        .into_iter()
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, CapabilityVector, ResourceProfile, TaskConstraints};

    fn task() -> TaskSpec {
        TaskSpec::new("task-1", "write the engine", "rust")
    }

    #[test]
    fn stages_route_to_their_role() {
        let mesh = NeuralMesh::default();
        let assignment = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(assignment.role, AgentRole::Coder);
        assert_eq!(assignment.agent_id, "coder-01");
        assert_eq!(assignment.stage, PipelineStage::Code);
    }

    #[test]
    fn domain_mastery_wins_the_assignment() {
        let mut mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("generalist", "generalist", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.7))
                .with_expertise("rust", 0.1),
            AgentNode::new("rustacean", "rustacean", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.7))
                .with_expertise("rust", 0.95),
        ]);
        mesh.sync_link_latencies();

        let winner = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(winner.agent_id, "rustacean");
    }

    #[test]
    fn deferred_work_goes_to_the_cheaper_agent_and_urgent_work_does_not() {
        let mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("strong", "strong", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.80))
                .with_resources(ResourceProfile::new(0.90, 0.80))
                .with_expertise("rust", 0.7),
            AgentNode::new("cheap", "cheap", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.62))
                .with_resources(ResourceProfile::new(0.05, 0.05))
                .with_expertise("rust", 0.7),
        ]);

        let deferred = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Deferred, &[]).unwrap();
        assert_eq!(deferred.agent_id, "cheap", "deferred work is cost-driven");

        let urgent = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Immediate, &[]).unwrap();
        assert_eq!(urgent.agent_id, "strong", "urgent work is capability-driven");
    }

    #[test]
    fn trust_breaks_ties_between_equal_agents() {
        let mut mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("a", "a", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.7))
                .with_expertise("rust", 0.7),
            AgentNode::new("b", "b", AgentRole::Coder)
                .with_capability(CapabilityVector::uniform(0.7))
                .with_expertise("rust", 0.7),
            AgentNode::new("judge", "judge", AgentRole::Critic),
        ]);

        let coefficients = mesh.coefficients;
        for _ in 0..10 {
            mesh.trust.record("judge", "b", true);
            mesh.trust.record("judge", "a", false);
            mesh.trust.settle(&coefficients, 1.0);
        }

        let winner = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(winner.agent_id, "b");
    }

    #[test]
    fn isolated_agents_are_skipped_while_alternatives_exist() {
        let mut mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("burned", "burned", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.9)),
            AgentNode::new("steady", "steady", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.5)),
            AgentNode::new("judge", "judge", AgentRole::Critic),
        ]);

        let coefficients = mesh.coefficients;
        for _ in 0..40 {
            mesh.trust.record("judge", "burned", false);
            mesh.trust.record("steady", "burned", false);
            mesh.trust.settle(&coefficients, 1.0);
        }
        assert!(mesh.trust.isolated().contains(&"burned".to_string()));

        let winner = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(winner.agent_id, "steady", "the mesh routes around an isolated agent");
    }

    #[test]
    fn a_missing_role_falls_back_to_the_whole_cohort() {
        let mesh = NeuralMesh::with_cohort(vec![AgentNode::new("solo", "solo", AgentRole::Executor)]);
        let assignment = select(&mesh, PipelineStage::Verify, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(assignment.agent_id, "solo", "a mesh with no verifier still verifies");
    }

    #[test]
    fn exclusion_prevents_one_agent_holding_two_stages() {
        let mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("coder-01", "first", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.8)),
            AgentNode::new("coder-02", "second", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.6)),
        ]);

        let unrestricted = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(unrestricted.agent_id, "coder-01");

        let excluded = vec!["coder-01".to_string()];
        let assignment = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &excluded).unwrap();
        assert_eq!(assignment.agent_id, "coder-02", "an excluded agent yields to its peer");
    }

    #[test]
    fn exclusion_yields_rather_than_stalling_the_pipeline() {
        let mesh = NeuralMesh::with_cohort(vec![AgentNode::new("only", "only", AgentRole::Coder)]);
        let assignment =
            select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &["only".to_string()]).unwrap();
        assert_eq!(assignment.agent_id, "only", "a stage must be staffed even if it doubles up");
    }

    #[test]
    fn a_latency_limit_narrows_the_pool() {
        let mut fast = AgentNode::new("fast", "fast", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.5))
            .with_expertise("rust", 0.5);
        let mut slow = AgentNode::new("slow", "slow", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.95))
            .with_expertise("rust", 0.95);
        fast.resources.observe(2_000.0, 0.1, 60_000.0, 1.0, 1.0);
        slow.resources.observe(40_000.0, 0.1, 60_000.0, 1.0, 1.0);

        let mesh = NeuralMesh::with_cohort(vec![fast, slow]);

        let unconstrained = select(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]).unwrap();
        assert_eq!(unconstrained.agent_id, "slow", "left alone, the mesh takes the strong agent");

        let urgent = task().with_constraints(TaskConstraints::latency(5_000.0));
        let constrained = select(&mesh, PipelineStage::Code, &urgent, PriorityBand::Standard, &[]).unwrap();
        assert_eq!(constrained.agent_id, "fast", "a deadline excludes the agent that cannot meet it");
    }

    #[test]
    fn an_impossible_limit_still_routes_rather_than_stalling() {
        let mesh = NeuralMesh::default();
        let impossible = task().with_constraints(TaskConstraints::latency(1.0));
        let assignment = select(&mesh, PipelineStage::Code, &impossible, PriorityBand::Standard, &[]);
        assert!(
            assignment.is_some(),
            "refusing to route is worse than routing over budget; the plan reports the violation"
        );
    }

    #[test]
    fn a_cost_limit_prices_agents_on_what_they_were_measured_to_cost() {
        let mut cheap = AgentNode::new("cheap", "cheap", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.6))
            .with_expertise("rust", 0.6);
        let mut pricey = AgentNode::new("pricey", "pricey", AgentRole::Coder)
            .with_capability(CapabilityVector::uniform(0.95))
            .with_expertise("rust", 0.95);
        cheap.resources.observe(1_000.0, 0.02, 60_000.0, 1.0, 1.0);
        pricey.resources.observe(1_000.0, 0.80, 60_000.0, 1.0, 1.0);

        let mesh = NeuralMesh::with_cohort(vec![cheap, pricey]);
        let budgeted = task().with_constraints(TaskConstraints::cost(0.10));
        let assignment = select(&mesh, PipelineStage::Code, &budgeted, PriorityBand::Standard, &[]).unwrap();
        assert_eq!(assignment.agent_id, "cheap");
    }

    #[test]
    fn an_unobserved_agent_is_judged_on_its_declared_prior() {
        let node = AgentNode::new("declared", "declared", AgentRole::Coder)
            .with_resources(ResourceProfile::new(0.5, 0.5));
        let calibration = RuntimeCalibration::default();
        let (latency_ms, cost) = expected_profile(&node, &calibration);

        assert_eq!(latency_ms, 30_000.0, "half of a 60s ceiling");
        assert_eq!(cost, 0.5);
        assert!(!admits(&node, &task().with_constraints(TaskConstraints::latency(10_000.0)), &calibration));
    }

    #[test]
    fn ranking_is_reproducible_for_identical_candidates() {
        let mesh = NeuralMesh::with_cohort(vec![
            AgentNode::new("b-agent", "b", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.6)),
            AgentNode::new("a-agent", "a", AgentRole::Coder).with_capability(CapabilityVector::uniform(0.6)),
        ]);
        let first = rank(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]);
        let second = rank(&mesh, PipelineStage::Code, &task(), PriorityBand::Standard, &[]);
        assert_eq!(first, second);
        assert_eq!(first[0].agent_id, "a-agent", "ties break on id");
    }
}
