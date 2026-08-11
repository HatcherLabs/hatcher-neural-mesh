//! The integration adapter: a stateful facade over [`NeuralMesh`] implementing the
//! four verbs of [`hatcher_core::contract`].
//!
//! ```text
//!   register_agent(AgentRegistration)      → RegistrationAck
//!   plan(TaskEnvelope)                     → RoutingPlan     ─┐
//!        …the caller executes for real…                       │  one run,
//!   report(run_id, StageOutcomeReport)     → RunStatus        │  held open
//!   finalize(run_id)                       → MeshReceipt     ─┘
//! ```
//!
//! ## The open run
//!
//! [`plan`](MeshAdapter::plan) does not touch trust, memory, capability, or `Ω`. It
//! chooses agents and hands the choice back. The mesh only moves at
//! [`finalize`](MeshAdapter::finalize), when the reports are in — because until then
//! nothing has actually happened, and a mesh that learned from work it merely *scheduled*
//! would be learning from its own intentions.
//!
//! Between those two calls the run is held open with its assignments frozen. If the mesh
//! moves in the meantime — another run finalizes, an agent is registered — the plan is
//! still honoured as issued. The caller has already dispatched real work to those
//! agents; re-routing under it would make the reported outcomes describe a run that
//! never happened.
//!
//! ## The shortcuts
//!
//! [`execute`](MeshAdapter::execute) collapses all four calls using simulated outcomes,
//! for rehearsal and tests. [`submit_reported`](MeshAdapter::submit_reported) collapses
//! them using outcomes the caller already has, for replaying a recording or for a
//! runtime that batches. Both go through exactly the same pipeline as the long form.

use std::collections::{BTreeMap, VecDeque};

use hatcher_core::{
    AgentNode, AgentRegistration, AgentSummary, Assignment, ContractError, DecisionHead, MeshReceipt,
    OutcomeProvenance, PipelineStage, PipelineTrace, PlannedStage, RegistrationAck, RoutingPlan,
    RuntimeCalibration, StageOutcomeReport, StageReceipt, TaskEnvelope, TaskSpec, CONTRACT_VERSION,
};
use serde::{Deserialize, Serialize};

use crate::inference::InferenceError;
use crate::mesh::NeuralMesh;
use crate::outcomes::{ReportedOutcomes, SimulatedOutcomes};
use crate::pipeline::{self, PipelineConfig, RunInputs};
use crate::router;

/// The five staffed stations, in order. A plan names an agent for each.
const STAFFED: [PipelineStage; 5] = [
    PipelineStage::Plan,
    PipelineStage::Research,
    PipelineStage::Code,
    PipelineStage::Critique,
    PipelineStage::Verify,
];

/// How many finalized traces the adapter keeps for lookup by id.
const DEFAULT_HISTORY_LIMIT: usize = 256;

/// Where an open run has got to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunStatus {
    pub run_id: String,
    pub task_id: String,
    /// Stages that have a report, in pipeline order.
    pub reported: Vec<PipelineStage>,
    /// Stages still outstanding.
    pub missing: Vec<PipelineStage>,
    /// True when [`MeshAdapter::finalize`] will succeed.
    pub complete: bool,
}

/// A run the caller has been given a plan for but has not yet finished reporting on.
#[derive(Debug, Clone)]
struct OpenRun {
    plan: RoutingPlan,
    task: TaskSpec,
    assignments: Vec<Assignment>,
    reports: ReportedOutcomes,
    metadata: BTreeMap<String, String>,
}

impl OpenRun {
    fn staffed_stages(&self) -> Vec<PipelineStage> {
        self.assignments.iter().map(|assignment| assignment.stage).collect()
    }

    fn status(&self) -> RunStatus {
        let expected = self.staffed_stages();
        let missing = self.reports.missing_from(&expected);
        RunStatus {
            run_id: self.plan.run_id.clone(),
            task_id: self.plan.task_id.clone(),
            reported: expected
                .iter()
                .copied()
                .filter(|stage| self.reports.contains(*stage))
                .collect(),
            missing: missing.clone(),
            complete: missing.is_empty(),
        }
    }
}

/// A mesh plus everything needed to drive it from outside.
#[derive(Debug, Clone)]
pub struct MeshAdapter {
    pub mesh: NeuralMesh,
    pub config: PipelineConfig,
    pub calibration: RuntimeCalibration,
    open: BTreeMap<String, OpenRun>,
    history: VecDeque<PipelineTrace>,
    history_limit: usize,
    sequence: u64,
}

impl Default for MeshAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshAdapter {
    /// An adapter over the default five-station cohort.
    pub fn new() -> Self {
        Self::with_mesh(NeuralMesh::default())
    }

    /// An adapter over an existing mesh.
    pub fn with_mesh(mesh: NeuralMesh) -> Self {
        Self {
            mesh,
            config: PipelineConfig::default(),
            calibration: RuntimeCalibration::default(),
            open: BTreeMap::new(),
            history: VecDeque::new(),
            history_limit: DEFAULT_HISTORY_LIMIT,
            sequence: 0,
        }
    }

    /// An adapter over an empty cohort, for callers that register everything themselves.
    pub fn empty() -> Self {
        Self::with_mesh(NeuralMesh::with_cohort(Vec::new()))
    }

    pub fn with_calibration(mut self, calibration: RuntimeCalibration) -> Self {
        self.calibration = calibration;
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit.max(1);
        self
    }

    /// Bind a trained ONNX policy as the decision head.
    ///
    /// Fails loudly rather than degrading. An operator who deliberately swaps the policy
    /// on an adapter that is about to serve production traffic should find out here, not
    /// by noticing later that the receipts say `native`.
    pub fn try_with_policy(mut self, model_path: impl Into<String>) -> Result<Self, InferenceError> {
        self.mesh = self.mesh.try_with_onnx(model_path)?;
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // 1. Register
    // -----------------------------------------------------------------------

    /// Register an agent, or update one that already exists.
    ///
    /// Re-registering a known id updates the declaration but **keeps the agent's
    /// measured history** — telemetry, observed cost and latency, and everything the
    /// learning equations have moved. A restarting worker re-announcing itself must not
    /// be able to wipe its own track record by declaring a fresh capability vector.
    pub fn register_agent(&mut self, registration: AgentRegistration) -> Result<RegistrationAck, ContractError> {
        registration.validate()?;

        let created = self.mesh.node(&registration.id).is_none();
        if created {
            let mut node = AgentNode::new(&registration.id, &registration.label, registration.role)
                .with_capability(registration.capability)
                .with_resources(registration.resources)
                .with_confidence(registration.confidence);
            for (domain, mastery) in &registration.expertise {
                node.expertise.insert(domain.clone(), mastery.clamp(0.0, 1.0));
            }
            self.mesh.add_agent(node);
        } else if let Some(node) = self.mesh.node_mut(&registration.id) {
            node.label = registration.label.clone();
            node.role = registration.role;
            // `intelligence` and `context` are the two factors the learning equations
            // never lift, so a re-declaration is the only way they can change. The other
            // three are measured, and a declaration must not overwrite a measurement.
            node.capability.intelligence = registration.capability.intelligence;
            node.capability.context = registration.capability.context;
            for (domain, mastery) in &registration.expertise {
                node.expertise
                    .entry(domain.clone())
                    .or_insert_with(|| mastery.clamp(0.0, 1.0));
            }
            if !node.resources.is_observed() {
                node.resources = registration.resources;
            }
        }

        self.mesh.sync_link_latencies();

        let node = self
            .mesh
            .node(&registration.id)
            .ok_or_else(|| ContractError::UnknownAgent {
                agent_id: registration.id.clone(),
            })?;

        Ok(RegistrationAck {
            contract_version: CONTRACT_VERSION.to_string(),
            agent_id: registration.id.clone(),
            created,
            cohort_size: self.mesh.nodes.len(),
            capability: node.influence(),
            bottleneck: node.capability.bottleneck().0.to_string(),
            mesh_digest: self.mesh.digest(),
        })
    }

    /// The current roster, strongest first.
    pub fn roster(&self) -> Vec<AgentSummary> {
        self.mesh.agent_summaries()
    }

    /// Which model is deciding, and whether it is the one that was asked for.
    pub fn head(&self) -> DecisionHead {
        self.mesh.head()
    }

    // -----------------------------------------------------------------------
    // 2. Plan
    // -----------------------------------------------------------------------

    /// Choose an agent for every staffed stage, without moving the mesh.
    ///
    /// The returned plan is held open under its `run_id` until it is finalized or
    /// cancelled.
    pub fn plan(&mut self, envelope: &TaskEnvelope) -> Result<RoutingPlan, ContractError> {
        envelope.validate()?;

        self.sequence += 1;
        let task = envelope.to_task(self.sequence);
        let priority = self.mesh.priority_for(&task);

        let mut taken: Vec<String> = Vec::new();
        let mut assignments: Vec<Assignment> = Vec::new();
        let mut stages: Vec<PlannedStage> = Vec::new();
        let mut violations: Vec<String> = Vec::new();

        for stage in STAFFED {
            let Some(assignment) =
                router::select_with(&self.mesh, stage, &task, priority.band, &taken, &self.calibration)
            else {
                continue;
            };
            taken.push(assignment.agent_id.clone());

            let node = self.mesh.node(&assignment.agent_id);
            let (expected_latency_ms, expected_cost) = node
                .map(|node| router::expected_profile(node, &self.calibration))
                .unwrap_or((0.0, 0.0));

            if !task.constraints.admits(expected_latency_ms, expected_cost) {
                violations.push(stage.as_str().to_string());
            }

            stages.push(PlannedStage {
                stage,
                agent_id: assignment.agent_id.clone(),
                role: assignment.role,
                score: assignment.score,
                capability: assignment.capability,
                inbound_trust: assignment.inbound_trust,
                mastery: assignment.mastery,
                expected_latency_ms,
                expected_cost,
                rationale: format!(
                    "A={:.3} trust={:.2} mastery={:.2} R={:.2} → score {:.4}",
                    assignment.capability,
                    assignment.inbound_trust,
                    assignment.mastery,
                    assignment.efficiency,
                    assignment.score
                ),
            });
            assignments.push(assignment);
        }

        let run_id = self.mint_run_id(&task);
        let plan = RoutingPlan {
            contract_version: CONTRACT_VERSION.to_string(),
            run_id: run_id.clone(),
            task_id: task.id.clone(),
            domain: task.domain.clone(),
            execution_mode: task.execution_mode,
            priority,
            band: priority.band,
            expected_latency_ms: stages.iter().map(|planned| planned.expected_latency_ms).sum(),
            expected_cost: stages.iter().map(|planned| planned.expected_cost).sum(),
            stages,
            mesh_digest: self.mesh.digest(),
            omega: self.mesh.global.omega,
            constraint_violations: violations,
        };

        self.open.insert(
            run_id,
            OpenRun {
                plan: plan.clone(),
                task,
                assignments,
                reports: ReportedOutcomes::new([]),
                metadata: envelope.metadata.clone(),
            },
        );

        Ok(plan)
    }

    /// A run id that is stable for a given mesh state and task, so a replayed session
    /// produces the same handles.
    fn mint_run_id(&self, task: &TaskSpec) -> String {
        let digest = hatcher_core::canonical_digest(&(task.id.as_str(), self.sequence, self.mesh.global.epoch))
            .unwrap_or_default();
        format!("run-{}", &digest[..16.min(digest.len())])
    }

    // -----------------------------------------------------------------------
    // 3. Report
    // -----------------------------------------------------------------------

    /// Record what actually happened at one stage.
    ///
    /// The stage must be in the plan and the agent must be the one the plan named.
    /// Crediting an outcome to an agent that did not produce it would teach the mesh the
    /// exact opposite of the truth, so a mismatch is rejected rather than coerced.
    pub fn report(&mut self, run_id: &str, report: StageOutcomeReport) -> Result<RunStatus, ContractError> {
        report.validate()?;

        let run = self.open.get_mut(run_id).ok_or_else(|| ContractError::UnknownRun {
            run_id: run_id.to_string(),
        })?;

        let planned = run
            .assignments
            .iter()
            .find(|assignment| assignment.stage == report.stage)
            .ok_or_else(|| ContractError::Mismatch {
                expected: format!("one of {:?}", run.staffed_stages()),
                found: format!("stage `{}`", report.stage.as_str()),
            })?;

        if planned.agent_id != report.agent_id {
            return Err(ContractError::Mismatch {
                expected: format!("agent `{}` for stage `{}`", planned.agent_id, report.stage.as_str()),
                found: format!("agent `{}`", report.agent_id),
            });
        }

        run.reports.insert(report);
        Ok(run.status())
    }

    /// Record several stages at once. Rejected reports leave the run untouched.
    ///
    /// Every report is validated before any is stored, so a batch with one bad entry
    /// does not leave the run half-updated and the caller unsure which half landed.
    pub fn report_many(
        &mut self,
        run_id: &str,
        reports: impl IntoIterator<Item = StageOutcomeReport>,
    ) -> Result<RunStatus, ContractError> {
        let reports: Vec<StageOutcomeReport> = reports.into_iter().collect();

        {
            let run = self.open.get(run_id).ok_or_else(|| ContractError::UnknownRun {
                run_id: run_id.to_string(),
            })?;
            for report in &reports {
                report.validate()?;
                let planned = run
                    .assignments
                    .iter()
                    .find(|assignment| assignment.stage == report.stage)
                    .ok_or_else(|| ContractError::Mismatch {
                        expected: format!("one of {:?}", run.staffed_stages()),
                        found: format!("stage `{}`", report.stage.as_str()),
                    })?;
                if planned.agent_id != report.agent_id {
                    return Err(ContractError::Mismatch {
                        expected: format!("agent `{}` for stage `{}`", planned.agent_id, report.stage.as_str()),
                        found: format!("agent `{}`", report.agent_id),
                    });
                }
            }
        }

        let run = self.open.get_mut(run_id).expect("checked above");
        for report in reports {
            run.reports.insert(report);
        }
        Ok(run.status())
    }

    /// Where an open run has got to.
    pub fn status(&self, run_id: &str) -> Result<RunStatus, ContractError> {
        self.open
            .get(run_id)
            .map(OpenRun::status)
            .ok_or_else(|| ContractError::UnknownRun {
                run_id: run_id.to_string(),
            })
    }

    /// Every run currently waiting on outcomes.
    pub fn open_runs(&self) -> Vec<RunStatus> {
        self.open.values().map(OpenRun::status).collect()
    }

    /// Abandon a run without letting it move the mesh.
    pub fn cancel(&mut self, run_id: &str) -> Result<(), ContractError> {
        self.open
            .remove(run_id)
            .map(|_| ())
            .ok_or_else(|| ContractError::UnknownRun {
                run_id: run_id.to_string(),
            })
    }

    // -----------------------------------------------------------------------
    // 4. Finalize
    // -----------------------------------------------------------------------

    /// Apply a fully-reported run to the mesh and return the receipt.
    ///
    /// Fails if any staffed stage is still outstanding. That is deliberate: a partial
    /// run finalized silently would let the mesh learn from a story with holes in it.
    /// Callers that genuinely want the holes filled in can say so with
    /// [`finalize_partial`](Self::finalize_partial).
    pub fn finalize(&mut self, run_id: &str) -> Result<MeshReceipt, ContractError> {
        let status = self.status(run_id)?;
        if !status.complete {
            return Err(ContractError::Incomplete {
                missing: status
                    .missing
                    .iter()
                    .map(|stage| stage.as_str().to_string())
                    .collect(),
            });
        }
        self.close(run_id, false)
    }

    /// Finalize a run with unreported stages resolved by the simulator.
    ///
    /// The resulting trace is marked [`OutcomeProvenance::Mixed`], and a mixed trace is
    /// never evidence of real work — it is a debugging and replay convenience.
    pub fn finalize_partial(&mut self, run_id: &str) -> Result<MeshReceipt, ContractError> {
        self.close(run_id, true)
    }

    fn close(&mut self, run_id: &str, simulate_gaps: bool) -> Result<MeshReceipt, ContractError> {
        let run = self.open.remove(run_id).ok_or_else(|| ContractError::UnknownRun {
            run_id: run_id.to_string(),
        })?;

        let source = if simulate_gaps {
            run.reports.clone().simulating_gaps()
        } else {
            run.reports.clone()
        };

        let trace = pipeline::run_bound(
            &mut self.mesh,
            &run.task,
            &RunInputs::new(&source)
                .with_config(self.config)
                .with_calibration(self.calibration)
                .with_assignments(&run.assignments),
        );

        let receipt = self.receipt_for(run_id, &trace, run.metadata);
        self.remember(trace);
        Ok(receipt)
    }

    // -----------------------------------------------------------------------
    // Shortcuts
    // -----------------------------------------------------------------------

    /// Plan and run a task in one call, with simulated outcomes.
    ///
    /// The rehearsal path. The receipt is marked [`OutcomeProvenance::Simulated`].
    pub fn execute(&mut self, envelope: &TaskEnvelope) -> Result<MeshReceipt, ContractError> {
        envelope.validate()?;
        self.sequence += 1;

        let task = envelope.to_task(self.sequence);
        let run_id = self.mint_run_id(&task);
        let source = SimulatedOutcomes;

        let trace = pipeline::run_bound(
            &mut self.mesh,
            &task,
            &RunInputs::new(&source)
                .with_config(self.config)
                .with_calibration(self.calibration),
        );

        let receipt = self.receipt_for(&run_id, &trace, envelope.metadata.clone());
        self.remember(trace);
        Ok(receipt)
    }

    /// Plan, report, and finalize in one call, from outcomes the caller already has.
    ///
    /// The replay path, and the right shape for a runtime that executes a whole task
    /// before talking to the mesh at all.
    pub fn submit_reported(
        &mut self,
        envelope: &TaskEnvelope,
        reports: impl IntoIterator<Item = StageOutcomeReport>,
    ) -> Result<MeshReceipt, ContractError> {
        let plan = self.plan(envelope)?;
        let run_id = plan.run_id.clone();

        // A replayed recording names the agents it actually used. Rewriting them onto
        // this mesh's choices would silently re-attribute someone else's history, so a
        // mismatch is surfaced here — and the caller cleans it up before the mesh learns.
        match self.report_many(&run_id, reports) {
            Ok(_) => {}
            Err(error) => {
                let _ = self.cancel(&run_id);
                return Err(error);
            }
        }

        match self.finalize(&run_id) {
            Ok(receipt) => Ok(receipt),
            Err(error) => {
                let _ = self.cancel(&run_id);
                Err(error)
            }
        }
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    fn remember(&mut self, trace: PipelineTrace) {
        self.history.push_back(trace);
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }
    }

    /// Finalized traces, oldest first.
    pub fn history(&self) -> impl Iterator<Item = &PipelineTrace> {
        self.history.iter()
    }

    /// The most recent finalized trace.
    pub fn last_trace(&self) -> Option<&PipelineTrace> {
        self.history.back()
    }

    /// Look a finalized trace up by task id.
    pub fn trace(&self, task_id: &str) -> Option<&PipelineTrace> {
        self.history.iter().rev().find(|trace| trace.task_id == task_id)
    }

    /// How many runs have been finalized, including any dropped from history.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    fn receipt_for(
        &self,
        run_id: &str,
        trace: &PipelineTrace,
        metadata: BTreeMap<String, String>,
    ) -> MeshReceipt {
        MeshReceipt {
            contract_version: CONTRACT_VERSION.to_string(),
            run_id: run_id.to_string(),
            task_id: trace.task_id.clone(),
            accepted: trace.decision.action != "escalate",
            selected_agent: trace.primary_agent().map(str::to_string),
            decision: trace.decision.clone(),
            decision_head: self.mesh.head(),
            verified: trace.verified,
            trace_digest: trace.digest.clone(),
            mesh_digest: self.mesh.digest(),
            provenance: trace.provenance,
            omega_before: trace.omega_before,
            omega_after: trace.omega_after,
            intelligence: trace.intelligence,
            priority: trace.priority,
            stages: trace
                .stages
                .iter()
                .filter(|record| record.stage.is_staffed())
                .map(|record| StageReceipt {
                    stage: record.stage,
                    agent_id: record.agent.clone(),
                    success: record.success,
                    quality: record.quality,
                    latency_ms: record.latency_ms,
                    cost: record.cost,
                    error: record.error,
                    provenance: record.provenance.unwrap_or(OutcomeProvenance::Simulated),
                })
                .collect(),
            total_latency_ms: trace.total_latency_ms(),
            total_cost: trace.total_cost(),
            mean_quality: trace.mean_quality(),
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, CapabilityVector, ErrorClass, ResourceProfile, TaskConstraints};

    fn envelope() -> TaskEnvelope {
        TaskEnvelope::new("ship the integration adapter")
            .with_domain("rust")
            .with_features(vec![0.3, 0.7, 0.2, 0.8])
    }

    fn report_plan(plan: &RoutingPlan, quality: f64, latency_ms: f64, cost: f64) -> Vec<StageOutcomeReport> {
        plan.stages
            .iter()
            .map(|planned| {
                StageOutcomeReport::success(planned.stage, &planned.agent_id, quality)
                    .with_latency_ms(latency_ms)
                    .with_cost(cost)
            })
            .collect()
    }

    // --- registration ------------------------------------------------------

    #[test]
    fn registering_an_agent_reports_what_the_mesh_made_of_it() {
        let mut adapter = MeshAdapter::empty();
        let ack = adapter
            .register_agent(
                AgentRegistration::new("worker-01", "Worker", AgentRole::Coder)
                    .with_capability(CapabilityVector::new(0.9, 0.8, 0.8, 0.9, 0.05))
                    .with_expertise("rust", 0.9),
            )
            .unwrap();

        assert!(ack.created);
        assert_eq!(ack.cohort_size, 1);
        assert_eq!(ack.bottleneck, "memory", "the caller should see what is capping it");
        assert!(!ack.mesh_digest.is_empty());
        assert_eq!(ack.contract_version, CONTRACT_VERSION);
    }

    #[test]
    fn a_restarting_worker_cannot_wipe_its_own_track_record() {
        let mut adapter = MeshAdapter::empty();
        adapter
            .register_agent(AgentRegistration::new("worker-01", "Worker", AgentRole::Coder))
            .unwrap();

        // Give it a history.
        let node = adapter.mesh.node_mut("worker-01").unwrap();
        node.telemetry.attempts = 40;
        node.telemetry.successes = 4;
        node.capability.performance = 0.1;
        node.resources.observe(9_000.0, 0.5, 60_000.0, 1.0, 1.0);

        let ack = adapter
            .register_agent(
                AgentRegistration::new("worker-01", "Worker v2", AgentRole::Coder)
                    .with_capability(CapabilityVector::uniform(1.0))
                    .with_resources(ResourceProfile::new(0.01, 0.01)),
            )
            .unwrap();

        assert!(!ack.created);
        let node = adapter.mesh.node("worker-01").unwrap();
        assert_eq!(node.label, "Worker v2", "the declaration still updates what it can");
        assert_eq!(node.telemetry.attempts, 40, "history survives");
        assert!(node.capability.performance < 0.2, "a declaration cannot overwrite a measurement");
        assert!(
            node.resources.observed_latency_ms > 0.0,
            "nor can it discard observed cost"
        );
    }

    #[test]
    fn an_invalid_registration_is_rejected_at_the_boundary() {
        let mut adapter = MeshAdapter::empty();
        let mut registration = AgentRegistration::new("bad", "Bad", AgentRole::Coder);
        registration.confidence = 4.0;

        let error = adapter.register_agent(registration).unwrap_err();
        assert_eq!(error.status(), 400);
        assert!(adapter.mesh.nodes.is_empty(), "nothing partially applied");
    }

    // --- planning ----------------------------------------------------------

    #[test]
    fn a_plan_names_an_agent_per_stage_without_moving_the_mesh() {
        let mut adapter = MeshAdapter::new();
        let omega_before = adapter.mesh.global.omega;
        let epoch_before = adapter.mesh.global.epoch;

        let plan = adapter.plan(&envelope()).unwrap();

        assert_eq!(plan.stages.len(), 5);
        assert_eq!(plan.agent_for(PipelineStage::Code), Some("coder-01"));
        assert_eq!(plan.primary_agent(), Some("coder-01"));
        assert!(plan.expected_latency_ms > 0.0);
        assert_eq!(adapter.mesh.global.omega, omega_before, "planning is not learning");
        assert_eq!(adapter.mesh.global.epoch, epoch_before);
        assert_eq!(adapter.open_runs().len(), 1);
    }

    #[test]
    fn a_plan_reports_a_constraint_it_could_not_satisfy() {
        let mut adapter = MeshAdapter::new();
        let impossible = envelope().with_constraints(TaskConstraints::latency(1.0));

        let plan = adapter.plan(&impossible).unwrap();
        assert_eq!(plan.stages.len(), 5, "the mesh still routes");
        assert_eq!(
            plan.constraint_violations.len(),
            5,
            "but the caller is told, so it can decline instead"
        );
    }

    #[test]
    fn an_open_plan_survives_the_mesh_moving_underneath_it() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();
        let promised: Vec<&str> = plan.stages.iter().map(|s| s.agent_id.as_str()).collect();

        for _ in 0..8 {
            adapter.execute(&envelope()).unwrap();
        }

        adapter.report_many(&plan.run_id, report_plan(&plan, 0.9, 1_000.0, 0.1)).unwrap();
        let receipt = adapter.finalize(&plan.run_id).unwrap();

        let used: Vec<&str> = receipt.stages.iter().filter_map(|s| s.agent_id.as_deref()).collect();
        assert_eq!(used, promised, "the caller already dispatched to these agents");
    }

    // --- reporting ---------------------------------------------------------

    #[test]
    fn the_full_loop_produces_a_receipt_that_attests_to_real_work() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();

        for report in report_plan(&plan, 0.92, 2_000.0, 0.15) {
            adapter.report(&plan.run_id, report).unwrap();
        }
        let receipt = adapter.finalize(&plan.run_id).unwrap();

        assert_eq!(receipt.run_id, plan.run_id);
        assert!(receipt.provenance.is_real());
        assert!(receipt.verified);
        assert!(receipt.accepted);
        assert_eq!(receipt.selected_agent.as_deref(), Some("coder-01"));
        assert!(!receipt.trace_digest.is_empty());
        assert!((receipt.total_latency_ms - 10_000.0).abs() < 1e-9);
        assert!((receipt.total_cost - 0.75).abs() < 1e-9);
        assert!((receipt.mean_quality - 0.92).abs() < 1e-9);
        assert!(adapter.open_runs().is_empty(), "a finalized run is closed");
    }

    #[test]
    fn a_report_for_the_wrong_agent_is_rejected_rather_than_coerced() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();

        let error = adapter
            .report(
                &plan.run_id,
                StageOutcomeReport::success(PipelineStage::Code, "somebody-else", 1.0),
            )
            .unwrap_err();

        assert_eq!(error.status(), 400);
        assert!(matches!(error, ContractError::Mismatch { .. }));
        assert!(!adapter.status(&plan.run_id).unwrap().complete);
    }

    #[test]
    fn a_report_for_an_unplanned_stage_is_rejected() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();

        let error = adapter
            .report(
                &plan.run_id,
                StageOutcomeReport::success(PipelineStage::OmegaUpdate, "coder-01", 1.0),
            )
            .unwrap_err();
        assert!(matches!(error, ContractError::Mismatch { .. }));
    }

    #[test]
    fn a_batch_with_one_bad_entry_leaves_the_run_untouched() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();

        let mut batch = report_plan(&plan, 0.9, 100.0, 0.1);
        batch.push(StageOutcomeReport::success(PipelineStage::Code, "impostor", 1.0));

        assert!(adapter.report_many(&plan.run_id, batch).is_err());
        let status = adapter.status(&plan.run_id).unwrap();
        assert!(status.reported.is_empty(), "nothing landed, so the caller knows where it stands");
        assert_eq!(status.missing.len(), 5);
    }

    #[test]
    fn status_tracks_progress_through_the_run() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();

        let status = adapter.status(&plan.run_id).unwrap();
        assert_eq!(status.missing.len(), 5);
        assert!(!status.complete);

        let reports = report_plan(&plan, 0.8, 100.0, 0.1);
        let status = adapter.report(&plan.run_id, reports[0].clone()).unwrap();
        assert_eq!(status.reported, vec![PipelineStage::Plan]);
        assert_eq!(status.missing.len(), 4);

        adapter.report_many(&plan.run_id, reports[1..].to_vec()).unwrap();
        assert!(adapter.status(&plan.run_id).unwrap().complete);
    }

    #[test]
    fn an_unknown_run_is_a_404() {
        let mut adapter = MeshAdapter::new();
        assert_eq!(adapter.status("nope").unwrap_err().status(), 404);
        assert_eq!(adapter.finalize("nope").unwrap_err().status(), 404);
        assert_eq!(
            adapter
                .report("nope", StageOutcomeReport::success(PipelineStage::Code, "a", 1.0))
                .unwrap_err()
                .status(),
            404
        );
    }

    // --- finalizing --------------------------------------------------------

    #[test]
    fn an_incomplete_run_will_not_finalize_by_accident() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();
        let reports = report_plan(&plan, 0.9, 100.0, 0.1);
        adapter.report(&plan.run_id, reports[0].clone()).unwrap();

        let error = adapter.finalize(&plan.run_id).unwrap_err();
        assert_eq!(error.status(), 409);
        assert!(matches!(error, ContractError::Incomplete { .. }));
        assert!(
            adapter.status(&plan.run_id).is_ok(),
            "a rejected finalize must leave the run open"
        );
    }

    #[test]
    fn a_partial_run_can_be_forced_but_is_marked_mixed() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();
        let reports = report_plan(&plan, 0.9, 100.0, 0.1);
        adapter.report(&plan.run_id, reports[0].clone()).unwrap();

        let receipt = adapter.finalize_partial(&plan.run_id).unwrap();
        assert_eq!(receipt.provenance, OutcomeProvenance::Mixed);
        assert!(!receipt.provenance.is_real(), "a story with holes is not evidence");
    }

    #[test]
    fn cancelling_a_run_leaves_the_mesh_exactly_where_it_was() {
        let mut adapter = MeshAdapter::new();
        let omega = adapter.mesh.global.omega;
        let digest = adapter.mesh.digest();

        let plan = adapter.plan(&envelope()).unwrap();
        adapter.report_many(&plan.run_id, report_plan(&plan, 1.0, 100.0, 0.1)).unwrap();
        adapter.cancel(&plan.run_id).unwrap();

        assert_eq!(adapter.mesh.global.omega, omega);
        assert_eq!(adapter.mesh.digest(), digest);
        assert!(adapter.open_runs().is_empty());
        assert_eq!(adapter.cancel(&plan.run_id).unwrap_err().status(), 404);
    }

    // --- learning ----------------------------------------------------------

    #[test]
    fn reported_runs_move_the_mesh_and_rehearsals_are_marked_apart() {
        let mut adapter = MeshAdapter::new();

        let rehearsal = adapter.execute(&envelope()).unwrap();
        assert_eq!(rehearsal.provenance, OutcomeProvenance::Simulated);
        assert!(!rehearsal.provenance.is_real());

        let plan = adapter.plan(&envelope()).unwrap();
        adapter.report_many(&plan.run_id, report_plan(&plan, 0.95, 3_000.0, 0.2)).unwrap();
        let real = adapter.finalize(&plan.run_id).unwrap();

        assert!(real.provenance.is_real());
        assert_ne!(real.trace_digest, rehearsal.trace_digest);
        assert_eq!(adapter.history().count(), 2);
    }

    #[test]
    fn the_mesh_learns_an_agents_real_cost_from_reports() {
        let mut adapter = MeshAdapter::new();
        let before = adapter.mesh.node("coder-01").unwrap().resources;
        assert!(!before.is_observed());

        for _ in 0..3 {
            let plan = adapter.plan(&envelope()).unwrap();
            adapter.report_many(&plan.run_id, report_plan(&plan, 0.9, 6_000.0, 0.25)).unwrap();
            adapter.finalize(&plan.run_id).unwrap();
        }

        let after = adapter.mesh.node("coder-01").unwrap().resources;
        assert!(after.is_observed());
        assert!((after.observed_latency_ms - 6_000.0).abs() < 1.0);
        assert!((after.latency - 0.1).abs() < 1e-3, "6s against a 60s ceiling");
    }

    #[test]
    fn a_plan_prices_a_stage_from_measured_history_once_there_is_any() {
        let mut adapter = MeshAdapter::new();
        let first = adapter.plan(&envelope()).unwrap();
        let declared = first
            .stages
            .iter()
            .find(|planned| planned.stage == PipelineStage::Code)
            .unwrap()
            .expected_latency_ms;

        adapter.report_many(&first.run_id, report_plan(&first, 0.9, 1_500.0, 0.05)).unwrap();
        adapter.finalize(&first.run_id).unwrap();

        let second = adapter.plan(&envelope()).unwrap();
        let measured = second
            .stages
            .iter()
            .find(|planned| planned.stage == PipelineStage::Code)
            .unwrap()
            .expected_latency_ms;

        assert_ne!(declared, measured);
        assert!((measured - 1_500.0).abs() < 1.0, "the plan now quotes what was measured");
    }

    #[test]
    fn a_run_of_outages_does_not_teach_the_mesh_to_distrust_its_agents() {
        let mut blamed = MeshAdapter::new();
        let mut unlucky = MeshAdapter::new();

        for _ in 0..6 {
            for (adapter, error) in [
                (&mut blamed, ErrorClass::Quality),
                (&mut unlucky, ErrorClass::Infrastructure),
            ] {
                let plan = adapter.plan(&envelope()).unwrap();
                let reports: Vec<StageOutcomeReport> = plan
                    .stages
                    .iter()
                    .map(|planned| StageOutcomeReport::failure(planned.stage, &planned.agent_id, error))
                    .collect();
                adapter.report_many(&plan.run_id, reports).unwrap();
                adapter.finalize(&plan.run_id).unwrap();
            }
        }

        assert!(
            unlucky.mesh.trust.mean_trust() > blamed.mesh.trust.mean_trust(),
            "six outages must not read as six betrayals"
        );
        assert!(
            unlucky.mesh.trust.isolated().len() <= blamed.mesh.trust.isolated().len(),
            "and must not push good agents out of the routing graph"
        );
    }

    // --- shortcuts ---------------------------------------------------------

    #[test]
    fn submit_reported_collapses_the_loop_for_a_replayed_recording() {
        let mut adapter = MeshAdapter::new();
        let plan = adapter.plan(&envelope()).unwrap();
        let reports = report_plan(&plan, 0.85, 900.0, 0.03);
        adapter.cancel(&plan.run_id).unwrap();

        // The same agents, submitted in one call.
        let receipt = adapter.submit_reported(&envelope(), reports).unwrap();
        assert!(receipt.provenance.is_real());
        assert!((receipt.mean_quality - 0.85).abs() < 1e-9);
        assert!(adapter.open_runs().is_empty());
    }

    #[test]
    fn a_replay_that_names_agents_this_mesh_would_not_choose_is_surfaced_not_swallowed() {
        let mut adapter = MeshAdapter::new();
        let stale = vec![StageOutcomeReport::success(PipelineStage::Code, "retired-agent", 0.9)];

        let error = adapter.submit_reported(&envelope(), stale).unwrap_err();
        assert!(matches!(error, ContractError::Mismatch { .. }));
        assert!(
            adapter.open_runs().is_empty(),
            "a failed submission must not leave a run stranded"
        );
    }

    #[test]
    fn history_is_bounded_and_searchable_by_task_id() {
        let mut adapter = MeshAdapter::new().with_history_limit(3);
        for index in 0..6 {
            let mut envelope = envelope();
            envelope.task_id = Some(format!("task-{index}"));
            adapter.execute(&envelope).unwrap();
        }

        assert_eq!(adapter.history().count(), 3);
        assert!(adapter.trace("task-0").is_none(), "the oldest fell off");
        assert!(adapter.trace("task-5").is_some());
        assert_eq!(adapter.last_trace().unwrap().task_id, "task-5");
    }

    #[test]
    fn an_empty_mesh_refuses_to_pretend_it_did_the_work() {
        let mut adapter = MeshAdapter::empty();
        let plan = adapter.plan(&envelope()).unwrap();

        assert!(plan.stages.is_empty(), "nobody to route to");
        let receipt = adapter.finalize(&plan.run_id).unwrap();
        assert!(!receipt.verified);
        assert!(!receipt.accepted, "an unstaffed mesh escalates rather than claiming a pass");
        assert!(receipt.selected_agent.is_none());
    }

    // --- the decision head ------------------------------------------------

    #[test]
    fn a_receipt_names_the_model_that_proposed_the_action() {
        let mut adapter = MeshAdapter::new();
        let receipt = adapter.execute(&envelope()).unwrap();

        assert_eq!(receipt.decision_head.name, "native");
        assert!(receipt.decision_head.is_intact());
        assert!(!receipt.decision_head.is_policy());
        assert_eq!(receipt.decision_head.input_dim, 8);
        assert_eq!(receipt.decision_head.output_dim, 4);
    }

    #[test]
    fn a_mesh_running_the_wrong_head_says_so_on_every_receipt() {
        // A path that cannot load degrades to the built-in head rather than refusing to
        // start — but a caller acting on these decisions has to be able to tell.
        let mesh = NeuralMesh::default().with_backend(
            crate::inference::BackendKind::Onnx {
                model_path: "models/fixtures/definitely-not-here.onnx".into(),
            },
            hatcher_core::ModelSpec::native_default(),
        );
        let mut adapter = MeshAdapter::with_mesh(mesh);

        let receipt = adapter.execute(&envelope()).unwrap();
        assert!(!receipt.decision_head.is_intact());
        assert_eq!(receipt.decision_head.name, "native", "it fell back rather than dying");

        // The reason differs by build — a missing file with `--features onnx`, an
        // unavailable backend without it — but either way it has to reach the operator as
        // text, not as a bare boolean they have to interpret.
        let reason = receipt.decision_head.degraded.as_deref().unwrap();
        assert!(reason.contains("onnx"), "unhelpful degradation reason: {reason}");
    }

    #[test]
    fn swapping_the_decision_head_changes_the_mesh_commitment() {
        let intact = MeshAdapter::new();
        let degraded = MeshAdapter::with_mesh(NeuralMesh::default().with_backend(
            crate::inference::BackendKind::Onnx {
                model_path: "models/fixtures/definitely-not-here.onnx".into(),
            },
            hatcher_core::ModelSpec::native_default(),
        ));

        assert_eq!(
            intact.mesh.nodes, degraded.mesh.nodes,
            "identical cohorts, so only the head differs"
        );
        assert_ne!(
            intact.mesh.digest(),
            degraded.mesh.digest(),
            "a commitment that ignored the head would attest to state while saying nothing \
             about what that state was used to decide"
        );
    }

    #[test]
    fn metadata_rides_along_untouched() {
        let mut adapter = MeshAdapter::new();
        let mut envelope = envelope();
        envelope.metadata.insert("pr".into(), "4417".into());

        let receipt = adapter.execute(&envelope).unwrap();
        assert_eq!(receipt.metadata.get("pr").map(String::as_str), Some("4417"));
    }
}
