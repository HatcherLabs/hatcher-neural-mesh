//! Where a stage outcome comes from.
//!
//! Until 1.0 the pipeline resolved every stage itself, from a hash of
//! `(task, agent, stage, sequence)`. That is the right behaviour for rehearsal — it
//! replays exactly, so a trace digest means something — but it is the wrong behaviour
//! for a mesh that is supposed to learn from agents that really ran.
//!
//! So the resolution step is a seam. [`OutcomeSource`] answers one question — "what
//! happened when this agent ran this stage?" — and the pipeline does not care whether
//! the answer came from arithmetic or from a production runtime reporting back.
//!
//! | Source | Answer comes from | Use |
//! |---|---|---|
//! | [`SimulatedOutcomes`] | the deterministic competence model | rehearsal, tests, benchmarks |
//! | [`ReportedOutcomes`] | [`StageOutcomeReport`]s supplied by the caller | production, replay |
//!
//! Both produce the same [`ResolvedOutcome`], and every resolved outcome carries its
//! [`OutcomeProvenance`], so nothing downstream can lose track of which kind of run it
//! is looking at.

use std::collections::BTreeMap;

use hatcher_core::{
    Assignment, ErrorClass, OutcomeProvenance, PipelineStage, RuntimeCalibration,
    StageOutcomeReport, TaskSpec,
};

use crate::equations::{deterministic_unit, seed_of};
use crate::mesh::NeuralMesh;
use crate::pipeline::PipelineConfig;

/// Everything a source needs to know to resolve one stage.
pub struct StageContext<'a> {
    pub mesh: &'a NeuralMesh,
    pub task: &'a TaskSpec,
    pub assignment: &'a Assignment,
    /// Upstream penalty in `(0, 1]`: below `1.0`, this agent is working from a bad plan
    /// or broken code.
    pub carry: f64,
    pub config: &'a PipelineConfig,
    pub calibration: &'a RuntimeCalibration,
}

impl StageContext<'_> {
    pub fn stage(&self) -> PipelineStage {
        self.assignment.stage
    }

    pub fn agent_id(&self) -> &str {
        &self.assignment.agent_id
    }

    /// How hard this task is, in `[0, 1]`.
    pub fn difficulty(&self) -> f64 {
        (0.5 * self.task.uncertainty + 0.5 * self.task.implementation_cost).clamp(0.0, 1.0)
    }
}

/// What happened at one stage, whatever produced the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOutcome {
    pub success: bool,
    /// The producing agent's confidence in its own output.
    pub confidence: f64,
    /// The reviewer's score for that output.
    pub quality: f64,
    pub latency_ms: f64,
    pub cost: f64,
    pub error: ErrorClass,
    pub provenance: OutcomeProvenance,
    pub note: String,
}

/// Resolves what happened when an agent ran a stage.
pub trait OutcomeSource {
    fn resolve(&self, context: &StageContext<'_>) -> ResolvedOutcome;
}

// ---------------------------------------------------------------------------
// Simulated
// ---------------------------------------------------------------------------

/// The deterministic competence model.
///
/// Outcomes are drawn from a hash of `(task id, agent id, stage, mesh sequence)`, never
/// from a clock or an RNG, so the same task against the same mesh state always replays
/// exactly. That is what makes rehearsal meaningful and what lets a trace digest be an
/// attestation rather than a souvenir.
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
/// threshold = competence^(0.5 + difficulty_weight · difficulty) · carry
/// ```
///
/// Exponentiation is the right shape because difficulty compounds against weakness. A
/// linear penalty moves every agent by the same amount, so a hopeless agent and an
/// excellent one lose the same margin on a hard task. As an exponent, a competence-0.9
/// agent barely notices a hard task (0.9² = 0.81) while a competence-0.2 agent collapses
/// (0.2² = 0.04) — which is what "hard" actually means.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimulatedOutcomes;

/// How much slower a stage runs at maximum difficulty than at zero.
const DIFFICULTY_LATENCY_SPAN: f64 = 0.5;
/// Multiplier applied to a stage that failed. Failures are not free — they burn a
/// timeout or a full generation before anyone finds out they were wrong.
const FAILURE_LATENCY_PENALTY: f64 = 1.35;
const _: () = assert!(
    FAILURE_LATENCY_PENALTY > 1.0,
    "finding out you were wrong is not free"
);

impl OutcomeSource for SimulatedOutcomes {
    fn resolve(&self, context: &StageContext<'_>) -> ResolvedOutcome {
        let assignment = context.assignment;
        let node = context.mesh.node(&assignment.agent_id);
        let node_confidence = node.map(|node| node.confidence).unwrap_or(0.5);

        let geometric_competence = assignment.capability.max(0.0).powf(0.2);
        let modulation = 0.60
            + 0.25 * assignment.mastery.clamp(0.0, 1.0)
            + 0.15 * assignment.inbound_trust.clamp(0.0, 1.0);
        let competence = (geometric_competence * modulation).clamp(0.0, 1.0);

        let difficulty = context.difficulty();
        let exponent = 0.5 + context.config.difficulty_weight * difficulty;
        let threshold = (competence.powf(exponent) * context.carry).clamp(0.02, 0.98);

        let draw = deterministic_unit(seed_of(&format!(
            "{}:{}:{}:{}",
            context.task.id,
            assignment.agent_id,
            assignment.stage.as_str(),
            context.mesh.sequence
        )));
        let success = draw < threshold;

        // How comfortably the agent cleared (or missed) the bar on this particular task.
        let margin = if success {
            (threshold - draw) / threshold.max(1e-6)
        } else {
            -((draw - threshold) / (1.0 - threshold).max(1e-6))
        };

        // Quality is the reviewer's view — the margin alone. Confidence blends that with
        // what the agent already believed about itself. Keeping them apart is what lets
        // the mesh notice an agent that is confidently wrong.
        let quality = (0.5 + 0.5 * margin).clamp(0.0, 1.0);
        let confidence = (0.6 * node_confidence + 0.4 * quality).clamp(0.0, 1.0);

        let expected_latency = node
            .map(|node| {
                node.resources
                    .expected_latency_ms(context.calibration.latency_ceiling_ms)
            })
            .unwrap_or(0.0);
        let expected_cost = node
            .map(|node| {
                node.resources
                    .expected_cost(context.calibration.cost_ceiling)
            })
            .unwrap_or(0.0);

        let difficulty_scale =
            1.0 - DIFFICULTY_LATENCY_SPAN / 2.0 + DIFFICULTY_LATENCY_SPAN * difficulty;
        let failure_scale = if success {
            1.0
        } else {
            FAILURE_LATENCY_PENALTY
        };
        let latency_ms = expected_latency * difficulty_scale * failure_scale;
        let cost = expected_cost * difficulty_scale;

        ResolvedOutcome {
            success,
            confidence,
            quality,
            latency_ms,
            cost,
            // A competence model can only ever produce competence failures. Labelling
            // them anything else would put a cause in the trace that nothing measured.
            error: if success {
                ErrorClass::None
            } else {
                ErrorClass::Quality
            },
            provenance: OutcomeProvenance::Simulated,
            note: format!(
                "{} by {} | fitness={:.3} threshold={:.3} draw={:.3} carry={:.2}",
                assignment.stage.as_str(),
                assignment.agent_id,
                competence,
                threshold,
                draw,
                context.carry
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Reported
// ---------------------------------------------------------------------------

/// What to do when a stage has no report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingOutcome {
    /// Fall back to the simulator and mark the trace `Mixed`.
    ///
    /// Useful for replaying a partial recording. Never appropriate for a trace that is
    /// going to be presented as evidence of real work.
    Simulate,
    /// Record the stage as failed with [`ErrorClass::Unknown`].
    ///
    /// The safe default: a stage nobody reported on is a stage nobody can vouch for, and
    /// inventing a pass for it would let silence read as success.
    Fail,
}

/// Outcomes supplied from outside — a live runtime, or a recorded log being replayed.
#[derive(Debug, Clone)]
pub struct ReportedOutcomes {
    reports: BTreeMap<PipelineStage, StageOutcomeReport>,
    missing: MissingOutcome,
}

impl ReportedOutcomes {
    /// Build from a set of reports, one per staffed stage.
    ///
    /// A later report for the same stage replaces an earlier one, which is what makes
    /// this usable for a runtime that retries a station.
    pub fn new(reports: impl IntoIterator<Item = StageOutcomeReport>) -> Self {
        Self {
            reports: reports
                .into_iter()
                .map(|report| (report.stage, report))
                .collect(),
            missing: MissingOutcome::Fail,
        }
    }

    /// Fall back to the simulator for stages nobody reported.
    pub fn simulating_gaps(mut self) -> Self {
        self.missing = MissingOutcome::Simulate;
        self
    }

    pub fn insert(&mut self, report: StageOutcomeReport) -> Option<StageOutcomeReport> {
        self.reports.insert(report.stage, report)
    }

    pub fn get(&self, stage: PipelineStage) -> Option<&StageOutcomeReport> {
        self.reports.get(&stage)
    }

    pub fn contains(&self, stage: PipelineStage) -> bool {
        self.reports.contains_key(&stage)
    }

    pub fn len(&self) -> usize {
        self.reports.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reports.is_empty()
    }

    /// The reported stages, in pipeline order.
    pub fn stages(&self) -> Vec<PipelineStage> {
        self.reports.keys().copied().collect()
    }

    /// Which of the given stages have no report yet.
    pub fn missing_from(&self, expected: &[PipelineStage]) -> Vec<PipelineStage> {
        expected
            .iter()
            .copied()
            .filter(|stage| !self.reports.contains_key(stage))
            .collect()
    }
}

impl FromIterator<StageOutcomeReport> for ReportedOutcomes {
    fn from_iter<I: IntoIterator<Item = StageOutcomeReport>>(iter: I) -> Self {
        Self::new(iter)
    }
}

impl OutcomeSource for ReportedOutcomes {
    fn resolve(&self, context: &StageContext<'_>) -> ResolvedOutcome {
        let Some(report) = self.reports.get(&context.stage()) else {
            return match self.missing {
                MissingOutcome::Simulate => SimulatedOutcomes.resolve(context),
                MissingOutcome::Fail => ResolvedOutcome {
                    success: false,
                    confidence: 0.0,
                    quality: 0.0,
                    latency_ms: 0.0,
                    cost: 0.0,
                    error: ErrorClass::Unknown,
                    provenance: OutcomeProvenance::Reported,
                    note: format!("no outcome reported for {}", context.stage().as_str()),
                },
            };
        };

        // A report that names a different agent than the one the mesh assigned is not a
        // near miss — crediting it would teach the mesh the opposite of the truth. The
        // adapter rejects these at the boundary; this is the defence for callers that
        // build a source by hand.
        if report.agent_id != context.agent_id() {
            return ResolvedOutcome {
                success: false,
                confidence: 0.0,
                quality: 0.0,
                latency_ms: report.latency_ms,
                cost: report.cost,
                error: ErrorClass::Unknown,
                provenance: OutcomeProvenance::Reported,
                note: format!(
                    "outcome for {} was reported against `{}` but the mesh assigned `{}`; discarded",
                    context.stage().as_str(),
                    report.agent_id,
                    context.agent_id()
                ),
            };
        }

        ResolvedOutcome {
            success: report.success,
            confidence: report.confidence.clamp(0.0, 1.0),
            quality: report.quality.clamp(0.0, 1.0),
            latency_ms: report.latency_ms.max(0.0),
            cost: report.cost.max(0.0),
            error: report.error,
            provenance: OutcomeProvenance::Reported,
            note: if report.note.is_empty() {
                format!(
                    "{} by {} | reported quality={:.3} latency={:.0}ms cost={:.4} error={}",
                    report.stage.as_str(),
                    report.agent_id,
                    report.quality,
                    report.latency_ms,
                    report.cost,
                    report.error.as_str()
                )
            } else {
                report.note.clone()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, PriorityBand};

    use crate::router;

    fn task() -> TaskSpec {
        TaskSpec::new("task-1", "wire the adapter", "rust")
            .with_features(vec![0.4, 0.6, 0.3, 0.7])
            .parsed_from_features()
    }

    fn context<'a>(
        mesh: &'a NeuralMesh,
        task: &'a TaskSpec,
        assignment: &'a Assignment,
        carry: f64,
        config: &'a PipelineConfig,
        calibration: &'a RuntimeCalibration,
    ) -> StageContext<'a> {
        StageContext {
            mesh,
            task,
            assignment,
            carry,
            config,
            calibration,
        }
    }

    fn coding_assignment(mesh: &NeuralMesh, task: &TaskSpec) -> Assignment {
        router::select(mesh, PipelineStage::Code, task, PriorityBand::Standard, &[]).unwrap()
    }

    #[test]
    fn simulated_outcomes_replay_exactly() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let first = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        let second = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert_eq!(first, second);
        assert_eq!(first.provenance, OutcomeProvenance::Simulated);
    }

    #[test]
    fn a_simulated_failure_is_always_a_competence_failure() {
        let mesh = NeuralMesh::with_cohort(vec![hatcher_core::AgentNode::new(
            "weak",
            "weak",
            AgentRole::Coder,
        )
        .with_capability(hatcher_core::CapabilityVector::uniform(0.02))]);
        let mut task = task();
        task.uncertainty = 1.0;
        task.implementation_cost = 1.0;

        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();
        let outcome = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));

        assert!(!outcome.success);
        assert_eq!(
            outcome.error,
            ErrorClass::Quality,
            "a competence model must not claim causes it never measured"
        );
    }

    #[test]
    fn a_poisoned_upstream_never_helps() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let clean = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        let poisoned = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            0.55,
            &config,
            &calibration,
        ));
        assert!(!poisoned.success || clean.success);
    }

    #[test]
    fn simulated_quality_and_confidence_are_different_measurements() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let outcome = SimulatedOutcomes.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert!(
            (outcome.quality - outcome.confidence).abs() > 1e-9,
            "a producer's confidence must not be its reviewer's score"
        );
    }

    #[test]
    fn reported_outcomes_are_taken_at_face_value() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let source = ReportedOutcomes::new([StageOutcomeReport::success(
            PipelineStage::Code,
            &assignment.agent_id,
            0.91,
        )
        .with_latency_ms(4_200.0)
        .with_cost(0.17)]);

        let outcome = source.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert!(outcome.success);
        assert_eq!(outcome.quality, 0.91);
        assert_eq!(outcome.latency_ms, 4_200.0);
        assert_eq!(outcome.cost, 0.17);
        assert_eq!(outcome.provenance, OutcomeProvenance::Reported);
    }

    #[test]
    fn a_report_for_the_wrong_agent_is_discarded_not_credited() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let source = ReportedOutcomes::new([StageOutcomeReport::success(
            PipelineStage::Code,
            "somebody-else",
            1.0,
        )]);

        let outcome = source.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert!(
            !outcome.success,
            "crediting the wrong agent teaches the mesh a lie"
        );
        assert_eq!(outcome.error, ErrorClass::Unknown);
        assert!(outcome.note.contains("discarded"));
    }

    #[test]
    fn silence_reads_as_failure_by_default() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let source = ReportedOutcomes::new([]);
        let outcome = source.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert!(!outcome.success);
        assert_eq!(outcome.error, ErrorClass::Unknown);
    }

    #[test]
    fn gaps_can_be_simulated_when_a_recording_is_partial() {
        let mesh = NeuralMesh::default();
        let task = task();
        let assignment = coding_assignment(&mesh, &task);
        let config = PipelineConfig::default();
        let calibration = RuntimeCalibration::default();

        let source = ReportedOutcomes::new([]).simulating_gaps();
        let outcome = source.resolve(&context(
            &mesh,
            &task,
            &assignment,
            1.0,
            &config,
            &calibration,
        ));
        assert_eq!(
            outcome.provenance,
            OutcomeProvenance::Simulated,
            "a simulated gap must announce itself so the trace can be marked Mixed"
        );
    }

    #[test]
    fn a_source_tracks_which_stages_are_still_outstanding() {
        let mut source = ReportedOutcomes::new([]);
        assert!(source.is_empty());
        source.insert(StageOutcomeReport::success(
            PipelineStage::Plan,
            "planner-01",
            0.8,
        ));

        let expected = [PipelineStage::Plan, PipelineStage::Code];
        assert_eq!(source.missing_from(&expected), vec![PipelineStage::Code]);
        assert!(source.contains(PipelineStage::Plan));
        assert_eq!(source.len(), 1);
    }

    #[test]
    fn a_retry_replaces_the_earlier_report_for_that_stage() {
        let mut source = ReportedOutcomes::new([StageOutcomeReport::failure(
            PipelineStage::Code,
            "coder-01",
            ErrorClass::Timeout,
        )]);
        let previous = source.insert(StageOutcomeReport::success(
            PipelineStage::Code,
            "coder-01",
            0.9,
        ));

        assert!(previous.is_some());
        assert_eq!(source.len(), 1);
        assert!(source.get(PipelineStage::Code).unwrap().success);
    }
}
