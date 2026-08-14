//! # HatcherLabs Agent Mesh Neural System — engine
//!
//! An adaptive, decentralized multi-agent intelligence framework. Individual agent
//! capability is multiplicative over intelligence, specialization, performance,
//! context, and memory; collective intelligence emerges from collaboration rather than
//! aggregation; trust evolves from outcomes; scheduling balances uncertainty, urgency,
//! and cost; and the whole mesh carries one global intelligence state that rises and
//! falls with what it actually accomplishes.
//!
//! ## Module map
//!
//! | Module | Role |
//! |---|---|
//! | [`equations`] | The ten update rules, as pure bounded functions |
//! | [`trust`] | The `T` and `W` matrices, and the self-organizing graph they form |
//! | [`mesh`] | Graph structure, GNN-style message passing, `Ω` bookkeeping, read models |
//! | [`router`] | Which agent executes which stage |
//! | [`learning`] | Credit assignment: outcomes into capability movement |
//! | [`pipeline`] | The ten-station execution loop and its sealed trace |
//! | [`inference`] | The decision head — native MLP or ONNX policy |
//!
//! ## Minimal use
//!
//! ```rust
//! use hatcher_core::TaskSpec;
//! use hatcher_neural::{pipeline, NeuralMesh};
//!
//! let mut mesh = NeuralMesh::default();
//! let task = TaskSpec::new("task-1", "ship the router", "rust")
//!     .with_features(vec![0.3, 0.7, 0.2, 0.9])
//!     .parsed_from_features();
//!
//! let trace = pipeline::run(&mut mesh, &task);
//! assert_eq!(trace.stages.len(), 10);
//! assert!(mesh.global.epoch == 1);
//! ```
//!
//! Run it a few dozen times and the interesting part shows up: hubs form, unreliable
//! agents stop receiving work, and `Ω` compounds or erodes depending on whether the
//! cohort is actually good enough for the work it is being given.

pub mod adapter;
pub mod equations;
pub mod inference;
pub mod learning;
pub mod mesh;
pub mod outcomes;
pub mod pipeline;
pub mod router;
pub mod trust;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use adapter::{MeshAdapter, RunStatus};
pub use equations::{OMEGA_CEILING, OMEGA_FLOOR};
pub use inference::{
    BackendKind, Decision, InferenceBackend, InferenceError, NativeBackend, SharedBackend,
};
pub use learning::StageOutcome;
pub use mesh::{default_cohort, NeuralMesh, MESSAGE_ROUNDS};
pub use outcomes::{
    MissingOutcome, OutcomeSource, ReportedOutcomes, ResolvedOutcome, SimulatedOutcomes,
    StageContext,
};
pub use pipeline::{
    run, run_batch, run_bound, run_with, run_with_outcomes, PipelineConfig, RunInputs,
};
pub use router::{rank, score_candidate, select};
pub use trust::{TrustGraph, TrustSettlement};

use hatcher_core::{HatcherRequest, HatcherResponse, MeshSimulation, PipelineTrace, TaskSpec};

/// Deprecated alias for the coefficient set, kept so existing callers keep compiling.
pub type MeshConfig = hatcher_core::MeshCoefficients;

impl NeuralMesh {
    /// Run a bridge request through the full pipeline and answer Hatcher.
    ///
    /// `accepted` means the mesh is willing to own the outcome — it is false exactly
    /// when the decision is `escalate`, because an escalation is the mesh declining.
    pub fn evaluate(&mut self, request: &HatcherRequest) -> HatcherResponse {
        let task = request.to_task();
        let trace = pipeline::run(self, &task);
        self.respond(&trace)
    }

    /// Run an already-parsed task and answer Hatcher.
    pub fn execute(&mut self, task: &TaskSpec) -> HatcherResponse {
        let trace = pipeline::run(self, task);
        self.respond(&trace)
    }

    fn respond(&self, trace: &PipelineTrace) -> HatcherResponse {
        let digest_prefix: String = trace.digest.chars().take(12).collect();
        HatcherResponse {
            accepted: trace.decision.action != "escalate",
            decision: trace.decision.clone(),
            trace_id: format!("trace-{digest_prefix}"),
            omega: trace.omega_after,
            intelligence: trace.intelligence,
            priority: trace.priority,
        }
    }

    /// Rehearse a bridge request without mutating the mesh.
    pub fn rehearse(&self, request: &HatcherRequest, steps: usize) -> MeshSimulation {
        self.simulate(&request.to_task(), steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, ExecutionMode};

    fn request() -> HatcherRequest {
        HatcherRequest::new(
            "agent-1",
            AgentRole::Orchestrator,
            ExecutionMode::Controlled,
        )
        .with_prompt("stress test the mesh")
        .with_features(vec![0.2, 0.4, 0.6, 0.8])
    }

    #[test]
    fn evaluate_answers_the_bridge_and_advances_the_mesh() {
        let mut mesh = NeuralMesh::default();
        let response = mesh.evaluate(&request());

        assert!(response.trace_id.starts_with("trace-"));
        assert!(response.decision.confidence >= 0.0 && response.decision.confidence <= 1.0);
        assert!(response.omega > 0.0);
        assert!(response.intelligence.total > 0.0);
        assert_eq!(mesh.global.epoch, 1);
        assert!(response.to_json().unwrap().contains("decision"));
    }

    #[test]
    fn acceptance_is_the_inverse_of_escalation() {
        let mut mesh = NeuralMesh::default();
        let response = mesh.evaluate(&request());
        assert_eq!(response.accepted, response.decision.action != "escalate");
    }

    #[test]
    fn rehearsal_leaves_the_mesh_untouched() {
        let mesh = NeuralMesh::default();
        let simulation = mesh.rehearse(&request(), 3);
        assert_eq!(simulation.steps.len(), 3);
        assert_eq!(mesh.global.epoch, 0);
    }

    #[test]
    fn the_deprecated_config_alias_still_resolves() {
        let coefficients = MeshConfig::default();
        assert!(coefficients.validate().is_ok());
    }
}
