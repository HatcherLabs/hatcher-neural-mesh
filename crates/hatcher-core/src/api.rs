//! The Hatcher bridge: request/response contracts plus the read models the
//! `hatcher-host-frontend` dashboard consumes.
//!
//! The frontend is a Next.js app that talks to its backend exclusively through a
//! typed `lib/api.ts` client and renders an agent through thirteen tabs. The view
//! models here are shaped to drop straight into the tabs that are actually about
//! intelligence: Overview, Config, Analytics, Logs, and Workflows.

use serde::{Deserialize, Serialize};

use crate::agent::{AgentRole, ExecutionMode};
use crate::coefficients::MeshCoefficients;
use crate::global::{OmegaRegime, OmegaSample};
use crate::graph::{MeshEdge, MeshIntelligence};
use crate::task::{PipelineTrace, PriorityScore, TaskSpec};

/// Minimal inference request kept for the original `/api/infer` contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    pub agent_id: String,
    pub prompt: String,
    pub features: Vec<f64>,
}

/// Minimal inference response for `/api/infer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub accepted: bool,
    pub action: String,
    pub confidence: f64,
    pub trace_id: String,
}

/// The decision the mesh emits for a request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralSignal {
    pub agent: String,
    pub intent: String,
    /// Calibrated confidence in `[0, 1]`.
    pub confidence: f64,
    /// One of `observe`, `delegate`, `stabilize`, `escalate`.
    pub action: String,
    pub rationale: String,
}

/// The four control actions the mesh can return to the Hatcher control plane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeshAction {
    /// Not enough signal — keep watching, do not act.
    Observe,
    /// Hand the work to a specific agent.
    Delegate,
    /// Accept and consolidate: the mesh is confident and coherent.
    Stabilize,
    /// Pressure is too high or trust too low — return to a human.
    Escalate,
}

impl MeshAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            MeshAction::Observe => "observe",
            MeshAction::Delegate => "delegate",
            MeshAction::Stabilize => "stabilize",
            MeshAction::Escalate => "escalate",
        }
    }
}

/// A request crossing the bridge from Hatcher into the Rust intelligence layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HatcherRequest {
    pub agent_id: String,
    pub role: AgentRole,
    pub execution_mode: ExecutionMode,
    pub prompt: String,
    pub features: Vec<f64>,
    /// Optional domain tag; defaults to `"general"` when absent.
    #[serde(default)]
    pub domain: Option<String>,
    /// Optional urgency in `[0, 1]`.
    #[serde(default)]
    pub urgency: Option<f64>,
}

impl HatcherRequest {
    pub fn new(
        agent_id: impl Into<String>,
        role: AgentRole,
        execution_mode: ExecutionMode,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            role,
            execution_mode,
            prompt: String::new(),
            features: Vec::new(),
            domain: None,
            urgency: None,
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }

    pub fn domain_or_default(&self) -> String {
        self.domain.clone().unwrap_or_else(|| "general".to_string())
    }

    /// Convert a bridge request into a parsed task, deriving `U`, `B`, `I`.
    pub fn to_task(&self) -> TaskSpec {
        let mut task = TaskSpec::new(
            format!("task-{}", self.agent_id),
            self.prompt.clone(),
            self.domain_or_default(),
        )
        .with_features(self.features.clone())
        .with_execution_mode(self.execution_mode)
        .parsed_from_features();
        task.urgency = self.urgency.unwrap_or(0.5).clamp(0.0, 1.0);
        task
    }
}

/// The response crossing back to Hatcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HatcherResponse {
    pub accepted: bool,
    pub decision: NeuralSignal,
    pub trace_id: String,
    /// `Ω` after the request was processed.
    pub omega: f64,
    pub intelligence: MeshIntelligence,
    pub priority: PriorityScore,
}

impl HatcherResponse {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// One row of the agent roster, for the Overview tab.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSummary {
    pub id: String,
    pub label: String,
    pub role: String,
    /// `A_i`
    pub capability: f64,
    /// The capability factor currently capping this agent.
    pub bottleneck: String,
    pub confidence: f64,
    /// `R_i`
    pub efficiency: f64,
    /// Mean trust flowing into this agent from the cohort.
    pub inbound_trust: f64,
    /// Mean trust this agent extends to the cohort.
    pub outbound_trust: f64,
    /// Number of live connections.
    pub degree: usize,
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    /// True when the cohort has effectively stopped routing to this agent.
    pub isolated: bool,
    /// True when this agent absorbs a disproportionate share of inbound trust.
    pub hub: bool,
}

/// Overview-tab read model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshOverview {
    pub omega: f64,
    pub regime: OmegaRegime,
    pub epoch: u64,
    pub intelligence: MeshIntelligence,
    /// Share of `A` that exists only because the agents are connected.
    pub emergence_ratio: f64,
    pub agent_count: usize,
    pub active_edge_count: usize,
    pub agents: Vec<AgentSummary>,
    pub hubs: Vec<String>,
    pub isolated: Vec<String>,
}

/// Analytics-tab read model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshAnalytics {
    pub omega: f64,
    /// Mean per-epoch change in `Ω` over the recent window.
    pub omega_slope: f64,
    pub series: Vec<OmegaSample>,
    pub intelligence: MeshIntelligence,
    pub mean_confidence: f64,
    pub mean_trust: f64,
    pub total_tasks: u64,
    pub success_rate: f64,
    pub agents: Vec<AgentSummary>,
}

/// Graph-tab read model: enough to draw the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshGraphView {
    pub nodes: Vec<AgentSummary>,
    pub edges: Vec<MeshEdge>,
    pub intelligence: MeshIntelligence,
    pub omega: f64,
}

/// Trust-matrix read model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustMatrixView {
    pub ids: Vec<String>,
    /// Row `i`, column `j` is `T_ij`.
    pub trust: Vec<Vec<f64>>,
    /// Row `i`, column `j` is `W_ij`.
    pub weight: Vec<Vec<f64>>,
}

/// Config-tab read/write model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfigView {
    pub coefficients: MeshCoefficients,
    pub agent_count: usize,
    pub execution_mode: ExecutionMode,
}

/// Body accepted by the task-submission endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSubmission {
    pub description: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub features: Vec<f64>,
    #[serde(default)]
    pub urgency: Option<f64>,
    #[serde(default)]
    pub execution_mode: Option<ExecutionMode>,
}

impl TaskSubmission {
    /// Turn a submitted body into a parsed task with a deterministic id.
    pub fn to_task(&self, sequence: u64) -> TaskSpec {
        let mut task = TaskSpec::new(
            format!("task-{sequence:04}"),
            self.description.clone(),
            self.domain.clone().unwrap_or_else(|| "general".to_string()),
        )
        .with_features(self.features.clone())
        .with_execution_mode(self.execution_mode.unwrap_or(ExecutionMode::Controlled))
        .parsed_from_features();
        task.urgency = self.urgency.unwrap_or(0.5).clamp(0.0, 1.0);
        task
    }
}

/// Response for a submitted task: the trace plus the resulting mesh headline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub trace: PipelineTrace,
    pub omega: f64,
    pub regime: OmegaRegime,
    pub intelligence: MeshIntelligence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_request_parses_into_a_task() {
        let request = HatcherRequest::new("agent-7", AgentRole::Explorer, ExecutionMode::Sandbox)
            .with_prompt("map the trust graph")
            .with_features(vec![0.2, 0.6, 0.4]);

        let task = request.to_task();
        assert_eq!(task.id, "task-agent-7");
        assert_eq!(
            task.domain, "general",
            "an absent domain falls back to general"
        );
        assert_eq!(task.execution_mode, ExecutionMode::Sandbox);
        assert!(task.uncertainty > 0.0);
    }

    #[test]
    fn submission_ids_are_sequence_stable() {
        let submission = TaskSubmission {
            description: "refactor the router".into(),
            domain: Some("rust".into()),
            features: vec![0.5, 0.5],
            urgency: Some(0.9),
            execution_mode: None,
        };
        let task = submission.to_task(12);
        assert_eq!(task.id, "task-0012");
        assert_eq!(task.domain, "rust");
        assert_eq!(task.urgency, 0.9);
        assert_eq!(task.execution_mode, ExecutionMode::Controlled);
    }

    #[test]
    fn urgency_is_clamped_into_range() {
        let submission = TaskSubmission {
            description: "panic".into(),
            domain: None,
            features: vec![],
            urgency: Some(9.0),
            execution_mode: None,
        };
        assert_eq!(submission.to_task(1).urgency, 1.0);
    }
}
