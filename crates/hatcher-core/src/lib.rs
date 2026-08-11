//! # HatcherLabs Agent Mesh Neural System — contracts
//!
//! `hatcher-core` is the typed contract layer for HAMNS: an adaptive, decentralized
//! multi-agent intelligence framework in which agent capability is multiplicative,
//! collective intelligence is emergent rather than additive, trust is learned from
//! outcomes, and the whole mesh carries a single global intelligence state.
//!
//! ## The five layers
//!
//! | Layer | Question it answers | Module |
//! |---|---|---|
//! | 1. Global intelligence | Is the mesh getting smarter? | [`global`] |
//! | 2. Agent capability | What is one agent worth? | [`agent`] |
//! | 3. Mesh intelligence | What is the cohort worth, connected? | [`graph`] |
//! | 4. Dynamic trust | Who should talk to whom? | [`graph`] |
//! | 5. Priority & resources | What runs next, and where? | [`task`], [`agent`] |
//!
//! ## The ten equations
//!
//! ```text
//! 1.  Global intelligence   Ω(t+1) = Ω(t) + α(L + E + C) − β(F + D)
//! 2.  Agent capability      A_i    = I_i · S_i · P_i · C_i · M_i
//! 3.  Mesh intelligence     A      = Σ A_i + γ Σ_{i≠j} A_i A_j W_ij
//! 4.  Trust evolution       T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij
//! 5.  Priority              P      = (U · B · I) / (C + τ)
//! 6.  Memory evolution      M_i(t+1) = M_i(t) + η K_i − δ R_i
//! 7.  Confidence            C_i(t+1) = C_i(t) + σ·success − ρ·error
//! 8.  Specialization        S_i(t+1) = S_i(t) + κ·experience − ω·obsolescence
//! 9.  Network plasticity    W_ij(t+1) = W_ij(t) + φ T_ij − ψ·latency
//! 10. Resource ratio        R_i    = P_i / (Energy_i + Latency_i)
//! ```
//!
//! The equations themselves live in `hatcher-neural`; this crate defines the state
//! they operate on, and seals that state into digest-backed [`envelope::Envelope`]s
//! so any mesh observation can be attested to by the Hatcher control plane.
//!
//! ## Integrating a real runtime
//!
//! [`contract`] is the surface an external agent runtime binds to: register an agent,
//! submit a task, receive a routing plan, report what actually happened, and get back a
//! decision with a digest that pins it. It is the only module a client needs, and it is
//! deliberately separate from the read models above — those describe the mesh to a
//! dashboard, this one lets something drive it.

pub mod agent;
pub mod api;
pub mod coefficients;
pub mod contract;
pub mod envelope;
pub mod global;
pub mod graph;
pub mod memory;
pub mod task;

pub use agent::{AgentNode, AgentRole, CapabilityVector, ExecutionMode, NodeTelemetry, ResourceProfile};
pub use api::{
    AgentSummary, ApiRequest, ApiResponse, HatcherRequest, HatcherResponse, MeshAction, MeshAnalytics,
    MeshConfigView, MeshGraphView, MeshOverview, NeuralSignal, TaskResult, TaskSubmission, TrustMatrixView,
};
pub use coefficients::{CoefficientError, MeshCoefficients};
pub use contract::{
    AgentRegistration, Attribution, ContractError, DecisionHead, ErrorClass, HeadCommitment, MeshReceipt,
    OutcomeProvenance, PlannedStage, RegistrationAck, RoutingPlan, RuntimeCalibration, StageOutcomeReport,
    StageReceipt, TaskEnvelope, CONTRACT_VERSION,
};
pub use envelope::{canonical_digest, fold_digests, Envelope, CANONICAL_CODEC, SCHEMA_VERSION};
pub use global::{GlobalState, OmegaDelta, OmegaLedger, OmegaRegime, OmegaSample};
pub use graph::{MeshEdge, MeshIntelligence, MeshSimulation, MeshState, MeshStepResult};
pub use memory::{MemoryGraph, MemoryRecord};
pub use task::{
    Assignment, PipelineStage, PipelineTrace, PriorityBand, PriorityScore, StageRecord, TaskConstraints,
    TaskSpec, DEFERRED_THRESHOLD, IMMEDIATE_THRESHOLD,
};

/// Shape and identity of an inference model bound to the mesh.
///
/// Used by the native engine for dimension checks and by the ONNX backend to
/// validate that a loaded graph matches what the mesh expects to feed it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub version: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl ModelSpec {
    pub fn new(name: impl Into<String>, version: impl Into<String>, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// The mesh's built-in decision head.
    ///
    /// Eight inputs — the canonical mesh feature layout, four mesh summary statistics
    /// followed by four task statistics — and four outputs, one logit per control
    /// action. Any model bound to the mesh must honour this contract.
    pub fn native_default() -> Self {
        Self::new("hamns-native", "2.0.0", 8, 12, 4)
    }
}

/// Clamp helper used across the equation set.
pub fn unit(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_clamps_and_absorbs_nan() {
        assert_eq!(unit(1.4), 1.0);
        assert_eq!(unit(-0.2), 0.0);
        assert_eq!(unit(f64::NAN), 0.0);
        assert_eq!(unit(0.5), 0.5);
    }

    #[test]
    fn native_model_spec_matches_the_decision_head() {
        let spec = ModelSpec::native_default();
        assert_eq!(spec.input_dim, 8, "four mesh statistics plus four task statistics");
        assert_eq!(spec.output_dim, 4, "one logit per MeshAction");
    }
}
