//! Credit assignment: turning what happened into movement in the capability vector.
//!
//! Equations 6, 7, and 8 all have the same shape — a gain term and a decay term — and
//! all three are only as good as what feeds them. This module is the measurement
//! layer: it converts stage outcomes and memory-graph activity into the `K_i`, `R_i`,
//! `success`, `error`, `experience`, and `obsolescence` those equations consume, then
//! applies them.
//!
//! Nothing here invents evidence. If a run produced no verified work, `K_i` is small;
//! if a domain was not exercised, it decays.

use hatcher_core::{AgentNode, MemoryRecord, MeshCoefficients, PipelineStage};

use crate::equations::{confidence_update, memory_update, specialization_update};

/// Performance prior for an agent with no history. Optimistic enough that a new agent
/// gets work, low enough that it does not outrank a proven one.
pub const PERFORMANCE_PRIOR: f64 = 0.60;

/// How much salience a verified stage commits to memory.
pub const BASE_SALIENCE: f64 = 0.45;

/// One agent's measured result at one stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageOutcome {
    pub agent_id: String,
    pub stage: PipelineStage,
    pub domain: String,
    pub success: bool,
    /// The agent's own confidence in the output it produced.
    pub confidence: f64,
    /// How novel the work was, in `[0, 1]`. Novel work is worth remembering; routine
    /// work is not, which is why a mesh grinding familiar tasks stops gaining `M_i`.
    pub novelty: f64,
}

impl StageOutcome {
    /// Salience this outcome writes to memory: novelty, scaled by how hard the task
    /// was to be confident about.
    pub fn salience(&self) -> f64 {
        (BASE_SALIENCE * (0.5 + 0.5 * self.novelty)).clamp(0.0, 1.0)
    }

    /// Turn the outcome into a memory record.
    pub fn to_memory_record(&self, task_id: &str) -> MemoryRecord {
        MemoryRecord {
            task_id: task_id.to_string(),
            agent_id: self.agent_id.clone(),
            domain: self.domain.clone(),
            salience: self.salience(),
            verified: self.success,
        }
    }
}

/// Count an attempt and refresh the measured performance factor `P`.
///
/// `P` is never asserted — it is the Laplace-smoothed success rate over this agent's
/// whole history, so it moves quickly at first and stabilizes as evidence accumulates.
pub fn record_attempt(node: &mut AgentNode, success: bool, stage: PipelineStage) {
    node.telemetry.attempts += 1;
    if success {
        node.telemetry.successes += 1;
    } else {
        node.telemetry.failures += 1;
    }
    node.telemetry.last_action = Some(format!(
        "{} {}",
        stage.as_str(),
        if success { "succeeded" } else { "failed" }
    ));
    node.capability.performance = node.telemetry.observed_performance(PERFORMANCE_PRIOR);
}

/// `C_i(t+1) = C_i(t) + σ·success − ρ·error`
///
/// Evidence is graded rather than binary: an agent that succeeded while claiming low
/// confidence gains less than one that succeeded confidently, and an agent that failed
/// while claiming high confidence is penalized more. That is what calibration means.
pub fn calibrate_confidence(node: &mut AgentNode, success: bool, claimed: f64, coefficients: &MeshCoefficients) {
    let claimed = claimed.clamp(0.0, 1.0);
    let (success_evidence, error_evidence) = if success {
        (0.5 + 0.5 * claimed, 0.0)
    } else {
        (0.0, 0.5 + 0.5 * claimed)
    };
    node.confidence = confidence_update(node.confidence, success_evidence, error_evidence, coefficients);
}

/// `M_i(t+1) = M_i(t) + η K_i − δ R_i`
pub fn consolidate_memory(node: &mut AgentNode, knowledge: f64, decay: f64, coefficients: &MeshCoefficients) {
    node.telemetry.knowledge += knowledge.max(0.0);
    node.telemetry.decay += decay.max(0.0);
    node.capability.memory = memory_update(node.capability.memory, knowledge, decay, coefficients);
}

/// `S_i(t+1) = S_i(t) + κ·experience − ω·obsolescence`
///
/// Applied to both the global specialization factor and the per-domain mastery, so an
/// agent that repeatedly solves Rust problems becomes a Rust specialist specifically,
/// not a vaguely better agent.
pub fn grow_specialization(
    node: &mut AgentNode,
    domain: &str,
    experience: f64,
    obsolescence: f64,
    coefficients: &MeshCoefficients,
) {
    node.telemetry.obsolescence += obsolescence.max(0.0);
    node.capability.specialization =
        specialization_update(node.capability.specialization, experience, obsolescence, coefficients);

    let current = node.mastery(domain);
    let updated = specialization_update(current, experience, 0.0, coefficients);
    node.expertise.insert(domain.to_string(), updated);
}

/// Age every domain this agent did *not* just work in.
///
/// Returns the total obsolescence pressure applied, which is part of the drift term
/// `D` in the `Ω` update: a mesh whose expertise is aging is drifting from what its
/// objectives now require.
pub fn obsolescence_sweep(node: &mut AgentNode, exercised_domain: &str, coefficients: &MeshCoefficients) -> f64 {
    let stale: Vec<String> = node
        .expertise
        .keys()
        .filter(|domain| domain.as_str() != exercised_domain)
        .cloned()
        .collect();

    let mut applied = 0.0;
    for domain in stale {
        let current = node.expertise.get(&domain).copied().unwrap_or(0.0);
        let decayed = specialization_update(current, 0.0, 1.0, coefficients);
        applied += (current - decayed).max(0.0);
        node.expertise.insert(domain, decayed);
    }
    applied
}

/// Apply everything one stage outcome implies for one agent.
///
/// Ordering matters: the attempt is counted first so the performance factor reflects
/// this run, then confidence is calibrated against the claim, then memory and
/// specialization move on the knowledge the run actually produced.
pub fn apply_outcome(
    node: &mut AgentNode,
    outcome: &StageOutcome,
    knowledge: f64,
    decay: f64,
    coefficients: &MeshCoefficients,
) -> f64 {
    record_attempt(node, outcome.success, outcome.stage);
    calibrate_confidence(node, outcome.success, outcome.confidence, coefficients);
    consolidate_memory(node, knowledge, decay, coefficients);

    // Only successful work counts as experience — repeating a mistake is not practice.
    let experience = if outcome.success { 1.0 } else { 0.25 };
    let staleness = obsolescence_sweep(node, &outcome.domain, coefficients);
    grow_specialization(node, &outcome.domain, experience, staleness, coefficients);
    staleness
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, CapabilityVector};

    fn agent() -> AgentNode {
        AgentNode::new("coder-01", "coder", AgentRole::Coder)
            .with_capability(CapabilityVector::new(0.8, 0.5, 0.6, 0.7, 0.5))
            .with_confidence(0.5)
            .with_expertise("rust", 0.6)
            .with_expertise("python", 0.6)
    }

    fn outcome(success: bool, confidence: f64) -> StageOutcome {
        StageOutcome {
            agent_id: "coder-01".into(),
            stage: PipelineStage::Code,
            domain: "rust".into(),
            success,
            confidence,
            novelty: 0.8,
        }
    }

    #[test]
    fn performance_is_measured_from_history_not_asserted() {
        let mut node = agent();
        for _ in 0..8 {
            record_attempt(&mut node, true, PipelineStage::Code);
        }
        assert!(node.capability.performance > 0.8, "a success streak raises P");
        assert_eq!(node.telemetry.attempts, 8);
        assert_eq!(node.telemetry.successes, 8);

        for _ in 0..20 {
            record_attempt(&mut node, false, PipelineStage::Code);
        }
        assert!(node.capability.performance < 0.4, "sustained failure lowers P");
        assert!(node.telemetry.last_action.as_deref().unwrap().contains("failed"));
    }

    #[test]
    fn confidence_calibration_punishes_confident_failure_hardest() {
        let coefficients = MeshCoefficients::default();

        let mut humble = agent();
        calibrate_confidence(&mut humble, false, 0.1, &coefficients);
        let mut arrogant = agent();
        calibrate_confidence(&mut arrogant, false, 1.0, &coefficients);
        assert!(arrogant.confidence < humble.confidence);

        let mut bold = agent();
        calibrate_confidence(&mut bold, true, 1.0, &coefficients);
        let mut timid = agent();
        calibrate_confidence(&mut timid, true, 0.0, &coefficients);
        assert!(bold.confidence > timid.confidence);
    }

    #[test]
    fn memory_tracks_both_sides_of_its_ledger() {
        let coefficients = MeshCoefficients::default();
        let mut node = agent();
        consolidate_memory(&mut node, 1.0, 0.0, &coefficients);
        let learned = node.capability.memory;
        assert!(learned > 0.5);
        assert!((node.telemetry.knowledge - 1.0).abs() < 1e-12);

        consolidate_memory(&mut node, 0.0, 4.0, &coefficients);
        assert!(node.capability.memory < learned);
        assert!(node.telemetry.decay > 0.0);
    }

    #[test]
    fn specialization_deepens_in_the_domain_that_was_worked() {
        let coefficients = MeshCoefficients::default();
        let mut node = agent();
        for _ in 0..5 {
            grow_specialization(&mut node, "rust", 1.0, 0.0, &coefficients);
        }
        assert!(node.mastery("rust") > 0.6);
        assert!((node.mastery("python") - 0.6).abs() < 1e-12, "untouched domains are untouched here");
        assert!(node.capability.specialization > 0.5);
    }

    #[test]
    fn unexercised_domains_go_obsolete() {
        let coefficients = MeshCoefficients::default();
        let mut node = agent();
        let applied = obsolescence_sweep(&mut node, "rust", &coefficients);
        assert!(applied > 0.0);
        assert!(node.mastery("python") < 0.6);
        assert!((node.mastery("rust") - 0.6).abs() < 1e-12, "the exercised domain is spared");
    }

    #[test]
    fn salience_scales_with_novelty() {
        let mut routine = outcome(true, 0.8);
        routine.novelty = 0.0;
        let novel = outcome(true, 0.8);
        assert!(novel.salience() > routine.salience());
        assert!(novel.salience() <= 1.0);
    }

    #[test]
    fn applying_a_successful_outcome_moves_every_learning_factor() {
        let coefficients = MeshCoefficients::default();
        let mut node = agent();
        let before = node.capability;

        apply_outcome(&mut node, &outcome(true, 0.9), 0.5, 0.0, &coefficients);

        assert!(node.capability.performance != before.performance);
        assert!(node.capability.memory > before.memory);
        assert!(node.capability.specialization > before.specialization);
        assert!(node.confidence > 0.5);
        assert!(node.mastery("rust") > 0.6);
        assert!(node.mastery("python") < 0.6, "working rust ages the python mastery");
    }

    #[test]
    fn a_failed_outcome_still_teaches_a_little() {
        let coefficients = MeshCoefficients::default();
        let mut node = agent();
        apply_outcome(&mut node, &outcome(false, 0.9), 0.1, 0.0, &coefficients);

        assert!(node.confidence < 0.5, "confidence drops");
        assert!(node.capability.specialization > 0.5, "but the attempt is still practice");
        assert_eq!(node.telemetry.failures, 1);
    }

    #[test]
    fn memory_records_carry_the_verification_flag() {
        let record = outcome(false, 0.5).to_memory_record("task-9");
        assert_eq!(record.task_id, "task-9");
        assert_eq!(record.agent_id, "coder-01");
        assert!(!record.verified);
    }
}
